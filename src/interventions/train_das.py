"""Train a DAS (Distributed Alignment Search) rotation on counterfactual NLI data.

The training loop is intentionally minimal: we freeze the base LM and
learn *only* the orthogonal rotation parameters of the
:class:`pyvene.LowRankRotatedSpaceIntervention` (or
:class:`RotatedSpaceIntervention`) at the chosen site, with a vanilla
cross-entropy objective on the next-token logits at the answer
position. The "label" is the verbalizer token id corresponding to the
**high-level counterfactual label** computed by
:mod:`src.data.causal_model`.

Training signal in plain English
--------------------------------
For each batch we

1. take a base sequence (the prompt we want to intervene on),
2. take a source sequence (the prompt whose value of the target
   variable we want to import),
3. forward-pass both through the model, *swap* the rotated subspace at
   the configured site from source into base,
4. read the final-position logits, restrict to verbalizer tokens, and
   minimise CE against the gold counterfactual token.

If the rotated subspace really does encode the target variable, this
loss is minimisable; if it doesn't (wrong site, too-small ``d``, etc.),
the loss plateaus near chance and IIA stays at chance.

The function returns the wrapped :class:`pyvene.IntervenableModel` and
a list of ``{"epoch", "loss", "train_iia"}`` records. If ``log_path``
is given, the history is also dumped to JSON for later plotting.
"""

from __future__ import annotations

import json
import os
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import pyvene as pv

from ..metrics.logits import (
    LabelVerbalizer,
    _final_logits,
    decode_label,
)
from .das_config import das_config_meta
from .patching import _resolve_device, _resolve_verbalizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_das_device(device: Union[str, torch.device, None]) -> torch.device:
    """Resolve the DAS device, avoiding Apple MPS for rotated interventions.

    pyvene's rotated-space interventions call ``torch.linalg.householder_product``
    through the rotation layer. That operator is not implemented on MPS as of
    current PyTorch releases, so DAS must run on CUDA or CPU. Activation
    patching can still use MPS; only learned rotated interventions need this
    guard.
    """
    resolved = _resolve_device(device)
    if resolved.type == "mps":
        warnings.warn(
            "DAS rotated interventions require an operator that is not "
            "implemented on Apple MPS. Falling back to CPU for DAS training.",
            RuntimeWarning,
        )
        return torch.device("cpu")
    return resolved


def _format_unit_locations(positions: torch.Tensor) -> Union[int, List[int]]:
    """Convert a per-example LongTensor ``[B]`` of positions to the format
    accepted by :meth:`pyvene.IntervenableModel.forward`.

    - If all elements are equal, return a single int (most efficient).
    - Otherwise return a Python list of ints; pyvene broadcasts each
      element to the corresponding base example.
    """
    if not isinstance(positions, torch.Tensor):
        positions = torch.as_tensor(positions, dtype=torch.long)
    pl = positions.tolist()
    if len(pl) == 0:
        return 0
    if all(p == pl[0] for p in pl):
        return int(pl[0])
    return [int(p) for p in pl]


def infer_fixed_position(dataset, fixed_position: Optional[int] = None) -> int:
    """Return the single token position DAS should use for a dataset.

    DAS in this project is a *fixed-site* alignment: one layer, one
    component, one token position. pyvene treats a Python list of positions as
    "intervene on all of these positions" for each example, not "one position
    per example". Passing per-example lists can therefore multiply the hidden
    dimension (e.g. ``8 positions * 768 hidden = 6144``) and break low-rank
    rotations. We instead use one fixed integer position for the whole run.

    If ``fixed_position`` is not provided, we use the mode of the dataset's
    ``intervention_pos`` values and warn if positions are not uniform.
    """
    if fixed_position is not None:
        return int(fixed_position)

    positions: List[int] = []
    if hasattr(dataset, "examples"):
        for ex in dataset.examples:
            if getattr(ex, "intervention_pos", None) is not None:
                positions.append(int(ex.intervention_pos))

    if not positions:
        for i in range(len(dataset)):
            item = dataset[i]
            pos = item.get("intervention_pos", 0)
            if isinstance(pos, torch.Tensor):
                pos = int(pos.item())
            positions.append(int(pos))

    if not positions:
        return 0

    counts = Counter(positions)
    mode_pos, mode_count = counts.most_common(1)[0]
    if len(counts) > 1:
        warnings.warn(
            "DAS is a fixed-position intervention, but the dataset contains "
            f"multiple intervention positions {dict(counts)}. Using the most "
            f"common position {mode_pos} ({mode_count}/{len(positions)} examples). "
            "For stricter experiments, use a single template or pass "
            "`fixed_position=` explicitly.",
            RuntimeWarning,
        )
    return int(mode_pos)


def _collect_optim_params(intervenable: pv.IntervenableModel) -> List[Dict[str, Any]]:
    """Return optimizer param groups holding only intervention weights.

    We rely on pyvene's :meth:`get_trainable_parameters` when available;
    otherwise we hand-walk ``intervenable.interventions`` and pull each
    rotation's ``rotate_layer.parameters()`` (the canonical DAS rotation
    object). This matches the pattern in the pyvene Boundless-DAS tutorial.
    """
    trainable = list(intervenable.get_trainable_parameters())
    if trainable:
        return [{"params": trainable}]

    groups: List[Dict[str, Any]] = []
    for _, v in intervenable.interventions.items():
        # rotate_layer is the orthogonal-matrix wrapper used by
        # RotatedSpaceIntervention and LowRankRotatedSpaceIntervention.
        if hasattr(v, "rotate_layer"):
            groups.append({"params": list(v.rotate_layer.parameters())})
        else:
            groups.append({"params": list(v.parameters())})
    return groups


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class DASTrainOutput:
    """Lightweight container returned by :func:`train_das_alignment`."""

    intervenable: pv.IntervenableModel
    history: List[Dict[str, float]]
    meta: Dict[str, Any]


def train_das_alignment(
    model,
    tokenizer,
    train_cf_dataset,
    config: pv.IntervenableConfig,
    num_epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 8,
    device: Union[str, torch.device, None] = None,
    *,
    verbalizer: Optional[LabelVerbalizer] = None,
    log_path: Optional[str] = None,
    weight_decay: float = 0.0,
    progress: bool = True,
    fixed_position: Optional[int] = None,
) -> DASTrainOutput:
    """Train a DAS rotation on a counterfactual NLI dataset.

    Parameters
    ----------
    model:
        HuggingFace causal LM. The base parameters are frozen
        (``disable_model_gradients``); only the intervention's rotation
        matrix gets gradient updates.
    tokenizer:
        Tokenizer matching ``model``. Used only to build a default
        :class:`LabelVerbalizer` if ``verbalizer`` is not supplied.
    train_cf_dataset:
        A :class:`~src.data.counterfactual_pairs.CounterfactualDataset`,
        or any Dataset whose items expose the same keys
        (``base_input_ids``, ``source_input_ids``,
        ``base_attention_mask``, ``source_attention_mask``,
        ``counterfactual_label_id``, ``intervention_pos``).
    config:
        :class:`pyvene.IntervenableConfig` -- typically built by
        :func:`src.interventions.das_config.make_das_config`.
    num_epochs, lr, batch_size:
        Standard optimiser settings.
    device:
        Target device. Falls back to CPU if CUDA is requested but
        unavailable.
    verbalizer:
        Optional :class:`LabelVerbalizer`. Defaults to ``" yes" / " maybe"
        / " no"`` constructed from ``tokenizer``.
    log_path:
        If given, the training history is dumped to JSON at this path
        once training finishes. The parent directory is created if
        needed.
    weight_decay:
        AdamW weight decay (default 0.0 -- rotations don't usually want
        decay).
    progress:
        Show tqdm progress bars if True.
    fixed_position:
        Token position to intervene on for every example. DAS here is a
        fixed-site alignment, so this should usually be copied from the best
        activation-patching heatmap cell. If ``None``, we use the mode of
        ``train_cf_dataset``'s ``intervention_pos`` values.

    Returns
    -------
    :class:`DASTrainOutput`
        ``intervenable`` (trained), ``history`` (list of
        ``{"epoch", "loss", "train_iia"}`` records), and ``meta`` (a
        dict of config + training hyperparameters, useful for logging).
    """
    device = _resolve_das_device(device)
    model.to(device)
    model.eval()  # we never train the base LM

    verbalizer = _resolve_verbalizer(tokenizer, verbalizer)
    fixed_position = infer_fixed_position(train_cf_dataset, fixed_position)

    intervenable = pv.IntervenableModel(config, model)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()

    param_groups = _collect_optim_params(intervenable)
    if not param_groups:
        raise RuntimeError(
            "No trainable intervention parameters found. Did you pass a config "
            "for a learnable intervention (e.g. RotatedSpaceIntervention)?"
        )

    optimizer = torch.optim.Adam(
        [
            {"params": pg["params"], "lr": pg.get("lr", lr),
             "weight_decay": pg.get("weight_decay", weight_decay)}
            for pg in param_groups
        ]
    )

    # Cache the verbalizer token ids once.
    verb_ids = verbalizer.token_id_tensor(device=device)  # [num_labels]

    loader = DataLoader(train_cf_dataset, batch_size=batch_size, shuffle=True)

    history: List[Dict[str, float]] = []
    epoch_iter = range(num_epochs)
    if progress:
        epoch_iter = tqdm(epoch_iter, desc="epochs")

    for epoch in epoch_iter:
        epoch_loss_sum = 0.0
        epoch_loss_n = 0
        epoch_correct = 0
        epoch_total = 0

        step_iter = loader
        if progress:
            step_iter = tqdm(
                loader, desc=f"epoch {epoch}", leave=False, position=1
            )

        for batch in step_iter:
            base_inputs = {
                "input_ids": batch["base_input_ids"].to(device),
                "attention_mask": batch["base_attention_mask"].to(device),
            }
            source_inputs = {
                "input_ids": batch["source_input_ids"].to(device),
                "attention_mask": batch["source_attention_mask"].to(device),
            }
            cf_labels = batch["counterfactual_label_id"].to(device)        # [B]
            pos_arg = int(fixed_position)

            _, cf_out = intervenable(
                base_inputs,
                [source_inputs],
                {"sources->base": pos_arg},
            )

            # Cross-entropy on the full vocabulary, with the gold
            # verbalizer-token id as the target. Using full-vocab CE
            # (rather than CE restricted to the 3 verbalizer tokens)
            # keeps gradient signal on the rest of the vocab, which
            # tends to be more stable and matches the Boundless-DAS
            # tutorial.
            final_logits = _final_logits(
                cf_out.logits, base_inputs["attention_mask"]
            )  # [B, V]
            target_token_ids = verb_ids[cf_labels]  # [B]
            loss = F.cross_entropy(final_logits, target_token_ids)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = decode_label(
                    cf_out.logits, verbalizer,
                    attention_mask=base_inputs["attention_mask"],
                )
                epoch_correct += int((preds == cf_labels).sum().item())
                epoch_total += int(cf_labels.size(0))
                epoch_loss_sum += float(loss.item()) * int(cf_labels.size(0))
                epoch_loss_n += int(cf_labels.size(0))

            if progress and hasattr(step_iter, "set_postfix"):
                step_iter.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "iia": f"{epoch_correct / max(1, epoch_total):.2f}",
                })

        avg_loss = epoch_loss_sum / max(1, epoch_loss_n)
        train_iia = epoch_correct / max(1, epoch_total)
        history.append({
            "epoch": int(epoch),
            "loss": float(avg_loss),
            "train_iia": float(train_iia),
        })

    meta = {
        **das_config_meta(config),
        "num_epochs": int(num_epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "weight_decay": float(weight_decay),
        "fixed_position": int(fixed_position),
        "verbalizer": dict(verbalizer.label_to_string),
        "n_train_examples": int(len(train_cf_dataset)),
        "device": str(device),
    }

    if log_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({"history": history, "meta": meta}, f, indent=2)

    return DASTrainOutput(intervenable=intervenable, history=history, meta=meta)
