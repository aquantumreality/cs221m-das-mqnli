"""Evaluate a trained DAS intervenable on counterfactual NLI data.

Reports three things:

- ``factual_accuracy`` -- argmax accuracy of the *un-intervened* base
  run against the base example's gold (factual) NLI label. This is a
  sanity check: if the model can't even produce the base label
  unprompted, IIA numbers are hard to interpret.
- ``iia`` -- interchange-intervention accuracy: fraction of held-out
  pairs where the *patched* argmax matches the high-level counterfactual
  label predicted by the symbolic causal model.
- ``confusion`` -- 3x3 confusion matrix (``true_cf_label`` x
  ``pred_cf_label``) as a :class:`pandas.DataFrame`, useful for spotting
  asymmetric errors (e.g. "neutral collapses to contradiction").

Optionally, per-relation IIA is reported too -- handy because in our
lexical-NLI setup the four high-level relations (EQUIV / FORWARD /
REVERSE / DISJOINT) have systematically different difficulty.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import pyvene as pv

from ..data.causal_model import ID2LABEL
from ..metrics.iia import compute_iia, compute_iia_per_class
from ..metrics.logits import LabelVerbalizer, decode_label
from .train_das import _format_unit_locations
from .patching import _resolve_device, _resolve_verbalizer


def _confusion_matrix(
    gold: np.ndarray,
    pred: np.ndarray,
    labels: list,
) -> pd.DataFrame:
    """Counts-only 3x3 confusion table (no scikit-learn dependency)."""
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    for g, p in zip(gold.astype(np.int64), pred.astype(np.int64)):
        if 0 <= g < n and 0 <= p < n:
            cm[g, p] += 1
    return pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels],
    )


def evaluate_das_iia(
    intervenable: pv.IntervenableModel,
    dataset,
    tokenizer,
    device: Union[str, torch.device, None] = None,
    *,
    verbalizer: Optional[LabelVerbalizer] = None,
    batch_size: int = 8,
    label_names: Optional[list] = None,
) -> Dict[str, Any]:
    """Evaluate a (trained or untrained) DAS intervenable on a CF dataset.

    Parameters
    ----------
    intervenable:
        :class:`pyvene.IntervenableModel`. Both trained (DAS) and
        untrained (vanilla patching) intervenables work -- the function
        simply runs forward passes.
    dataset:
        :class:`~src.data.counterfactual_pairs.CounterfactualDataset` or
        compatible iterable of dicts.
    tokenizer:
        Used to build a default verbalizer if ``verbalizer`` is not
        supplied.
    device:
        Target device. Falls back to CPU when CUDA is unavailable.
    verbalizer:
        Optional :class:`LabelVerbalizer`.
    batch_size:
        Inner DataLoader batch size.
    label_names:
        Optional list of label-name strings in canonical order (length =
        number of labels). Defaults to ``verbalizer.labels``.

    Returns
    -------
    dict
        Keys:
        - ``factual_accuracy`` (float in [0, 1])
        - ``iia`` (float in [0, 1])
        - ``iia_per_class`` (dict label -> float)
        - ``confusion`` (:class:`pandas.DataFrame`, gold x pred counts)
        - ``n_examples`` (int)
        - ``base_preds`` / ``patched_preds`` / ``gold_cf_labels``
          (1-D numpy arrays of label ids, in dataset order; handy for
          downstream analysis and plotting).
    """
    device = _resolve_device(device)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    intervenable.model.eval()

    verbalizer = _resolve_verbalizer(tokenizer, verbalizer)
    if label_names is None:
        label_names = list(verbalizer.labels)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    base_preds_all: list = []
    base_labels_all: list = []
    patched_preds_all: list = []
    cf_labels_all: list = []

    with torch.no_grad():
        for batch in loader:
            base_inputs = {
                "input_ids": batch["base_input_ids"].to(device),
                "attention_mask": batch["base_attention_mask"].to(device),
            }
            source_inputs = {
                "input_ids": batch["source_input_ids"].to(device),
                "attention_mask": batch["source_attention_mask"].to(device),
            }
            base_labels = batch["base_label_id"].to(device)
            cf_labels = batch["counterfactual_label_id"].to(device)
            pos_arg = _format_unit_locations(batch["intervention_pos"])

            # Factual: run the underlying (un-intervened) model directly.
            base_out = intervenable.model(**base_inputs)
            base_preds = decode_label(
                base_out.logits, verbalizer,
                attention_mask=base_inputs["attention_mask"],
            )

            # Counterfactual: run the intervened model.
            _, cf_out = intervenable(
                base_inputs,
                [source_inputs],
                {"sources->base": pos_arg},
            )
            patched_preds = decode_label(
                cf_out.logits, verbalizer,
                attention_mask=base_inputs["attention_mask"],
            )

            base_preds_all.append(base_preds.detach().cpu().numpy())
            base_labels_all.append(base_labels.detach().cpu().numpy())
            patched_preds_all.append(patched_preds.detach().cpu().numpy())
            cf_labels_all.append(cf_labels.detach().cpu().numpy())

    base_preds_np = np.concatenate(base_preds_all) if base_preds_all else np.array([], dtype=np.int64)
    base_labels_np = np.concatenate(base_labels_all) if base_labels_all else np.array([], dtype=np.int64)
    patched_preds_np = np.concatenate(patched_preds_all) if patched_preds_all else np.array([], dtype=np.int64)
    cf_labels_np = np.concatenate(cf_labels_all) if cf_labels_all else np.array([], dtype=np.int64)

    factual_acc = compute_iia(base_preds_np, base_labels_np)
    iia = compute_iia(patched_preds_np, cf_labels_np)
    iia_per_class = compute_iia_per_class(
        patched_preds_np, cf_labels_np, class_names=label_names
    )
    confusion = _confusion_matrix(cf_labels_np, patched_preds_np, labels=label_names)

    return {
        "factual_accuracy": float(factual_acc),
        "iia": float(iia),
        "iia_per_class": iia_per_class,
        "confusion": confusion,
        "n_examples": int(cf_labels_np.shape[0]),
        "base_preds": base_preds_np,
        "patched_preds": patched_preds_np,
        "gold_cf_labels": cf_labels_np,
    }
