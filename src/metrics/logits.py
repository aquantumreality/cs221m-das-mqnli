"""Logit-difference and logit-recovery metrics.

For a 3-way NLI verbalizer (e.g. " yes" / " maybe" / " no") we read the
next-token logits at the final position of the prompt and reduce them to:

- ``label_logit_diff``: ``logit[correct_label] - logit[wrong_label]``
  averaged across distractors. This is the workhorse metric of
  Wang et al. 2022 ("Interpretability in the Wild") and the activation
  patching literature.
- ``logit_recovery``: how much of the *clean* logit-diff is recovered by
  a patched run, normalised to ``[0, 1]``:

    recovery = (LD_patched - LD_corrupted) / (LD_clean - LD_corrupted)

  recovery = 1 means the patch fully recovers the clean behaviour;
  recovery = 0 means it does nothing; values outside [0, 1] are possible
  (and informative) when patches over- or under-shoot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Verbalizer
# ---------------------------------------------------------------------------


@dataclass
class LabelVerbalizer:
    """Maps NLI labels to single-token ids in a given tokenizer.

    Examples
    --------
    >>> v = LabelVerbalizer.from_tokenizer(tok,
    ...     {"entailment": " yes", "neutral": " maybe", "contradiction": " no"})
    >>> v.token_ids
    {'entailment': 3763, 'neutral': 14373, 'contradiction': 645}
    """

    label_to_string: Dict[str, str]
    token_ids: Dict[str, int]

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        label_to_string: Dict[str, str],
    ) -> "LabelVerbalizer":
        token_ids: Dict[str, int] = {}
        for label, surface in label_to_string.items():
            ids = tokenizer(surface, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(
                    f"Verbalizer string {surface!r} for label {label!r} "
                    f"tokenizes to {len(ids)} tokens ({ids!r}); it must "
                    f"map to a single token. Try a different surface form."
                )
            token_ids[label] = int(ids[0])
        return cls(label_to_string=dict(label_to_string), token_ids=token_ids)

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(self.label_to_string.keys())

    def token_id_tensor(self, device=None) -> torch.Tensor:
        """Return token ids in canonical label order as a LongTensor."""
        ids = [self.token_ids[l] for l in self.labels]
        t = torch.tensor(ids, dtype=torch.long)
        return t.to(device) if device is not None else t


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _final_logits(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return logits at the last non-pad position. Shape ``[B, V]``."""
    if logits.dim() != 3:
        raise ValueError(f"Expected logits of shape [B, T, V], got {tuple(logits.shape)}")
    if attention_mask is None:
        return logits[:, -1, :]
    # Last index where attention_mask == 1.
    last_idx = attention_mask.long().sum(dim=-1) - 1
    last_idx = last_idx.clamp(min=0)
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    return logits[batch_idx, last_idx, :]


def decode_label(
    logits: torch.Tensor,
    verbalizer: LabelVerbalizer,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Argmax-decode an NLI label from full LM logits.

    Restricts the argmax to the verbalizer tokens, so the prediction is
    always a valid label id (entailment=0 / neutral=1 / contradiction=2,
    matching :data:`src.data.causal_model.LABEL2ID`).

    Returns
    -------
    LongTensor of shape ``[B]`` with label ids in the verbalizer's order.
    """
    final = _final_logits(logits, attention_mask)
    ids = verbalizer.token_id_tensor(device=final.device)  # [L]
    restricted = final[:, ids]  # [B, L]
    return restricted.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Logit difference
# ---------------------------------------------------------------------------


def label_logit_diff(
    logits: torch.Tensor,
    correct_label_ids: torch.Tensor,
    verbalizer: LabelVerbalizer,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Logit difference between correct and average-wrong label tokens.

    For each example ``i``,

        diff_i = logit[i, correct_i] - mean_{j != correct_i} logit[i, j]

    Parameters
    ----------
    logits:
        Full LM logits, shape ``[B, T, V]``.
    correct_label_ids:
        LongTensor of shape ``[B]`` with the gold label id in the
        *verbalizer* index space (0..len(labels)-1), matching the order
        of :attr:`LabelVerbalizer.labels`.
    verbalizer:
        :class:`LabelVerbalizer` for converting labels to token ids.
    attention_mask:
        Optional attention mask used to find the final non-pad position.
    reduction:
        ``"none"`` returns the per-example tensor;
        ``"mean"`` (default) returns a scalar tensor.

    Returns
    -------
    Tensor of shape ``[B]`` (reduction="none") or scalar (reduction="mean").
    """
    final = _final_logits(logits, attention_mask)  # [B, V]
    verb_ids = verbalizer.token_id_tensor(device=final.device)  # [L]
    restricted = final[:, verb_ids]  # [B, L]

    B, L = restricted.shape
    correct = correct_label_ids.to(restricted.device).long()
    if correct.shape != (B,):
        raise ValueError(
            f"correct_label_ids must have shape [B={B}], got {tuple(correct.shape)}"
        )

    correct_logit = restricted.gather(1, correct.unsqueeze(1)).squeeze(1)  # [B]

    # Mask out the correct entry, average the rest.
    mask = torch.ones_like(restricted, dtype=torch.bool)
    mask.scatter_(1, correct.unsqueeze(1), False)
    wrong_logits = restricted.masked_select(mask).view(B, L - 1)
    wrong_avg = wrong_logits.mean(dim=-1)

    diff = correct_logit - wrong_avg

    if reduction == "none":
        return diff
    if reduction == "mean":
        return diff.mean()
    raise ValueError(f"Unknown reduction {reduction!r}")


# ---------------------------------------------------------------------------
# Logit recovery
# ---------------------------------------------------------------------------


def logit_recovery(
    patched_diff: Union[torch.Tensor, float, np.ndarray],
    clean_diff: Union[torch.Tensor, float, np.ndarray],
    corrupted_diff: Union[torch.Tensor, float, np.ndarray],
    eps: float = 1e-8,
) -> Union[torch.Tensor, float]:
    """Normalised logit-difference recovery.

    .. math::
        \\mathrm{recovery}
            = \\frac{\\mathrm{LD}_{\\mathrm{patched}}
                    - \\mathrm{LD}_{\\mathrm{corrupted}}}
                   {\\mathrm{LD}_{\\mathrm{clean}}
                    - \\mathrm{LD}_{\\mathrm{corrupted}} + \\epsilon}

    A value of 1.0 means the patch fully restored the clean-run logit
    difference; 0.0 means it did nothing. The metric can fall outside
    ``[0, 1]`` (e.g. if the patch over-shoots, or if the model was
    *more* confident in the corrupted run than the clean run).

    Inputs can be scalars, NumPy arrays, or torch Tensors and are
    broadcast together. The return type matches the inputs (Tensor if
    any input is a Tensor).
    """

    def _is_tensor(x):
        return isinstance(x, torch.Tensor)

    if any(_is_tensor(x) for x in (patched_diff, clean_diff, corrupted_diff)):
        # Promote everything to tensors.
        def _t(x):
            if _is_tensor(x):
                return x
            return torch.as_tensor(x)

        p = _t(patched_diff)
        c = _t(clean_diff)
        cor = _t(corrupted_diff)
        return (p - cor) / (c - cor + eps)

    p = np.asarray(patched_diff, dtype=np.float64)
    c = np.asarray(clean_diff, dtype=np.float64)
    cor = np.asarray(corrupted_diff, dtype=np.float64)
    out = (p - cor) / (c - cor + eps)
    if out.ndim == 0:
        return float(out)
    return out
