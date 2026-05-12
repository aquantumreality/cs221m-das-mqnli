"""Interchange Intervention Accuracy (IIA).

IIA is the main metric advocated by the DAS / causal-abstraction line of
work (Geiger et al. 2021, 2023). Given a batch of interchange
interventions, it asks: *after patching the relevant low-level
representation from source into base, does the model's prediction match
the gold counterfactual label produced by the symbolic high-level model?*

If IIA is high, the low-level neural network is implementing the
high-level algorithm (at least for the chosen alignment). IIA == 1.0
means perfect causal abstraction; chance-level IIA means the alignment
explains nothing.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch


ArrayLike = Union[torch.Tensor, np.ndarray, Sequence[int]]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_iia(
    preds: ArrayLike,
    gold_labels: ArrayLike,
    *,
    mask: Optional[ArrayLike] = None,
) -> float:
    """Compute interchange intervention accuracy.

    Parameters
    ----------
    preds:
        Model predictions after the intervention. Shape ``[N]`` of int
        label ids.
    gold_labels:
        Gold counterfactual labels from the high-level causal model.
        Shape ``[N]`` of int label ids.
    mask:
        Optional boolean mask of shape ``[N]``. If given, only entries
        where the mask is true contribute to the metric. Useful to
        exclude examples whose base prediction was already wrong.

    Returns
    -------
    float
        IIA in ``[0, 1]``. Returns 0.0 if there are no valid examples.
    """
    preds = _to_numpy(preds).astype(np.int64).ravel()
    gold = _to_numpy(gold_labels).astype(np.int64).ravel()
    if preds.shape != gold.shape:
        raise ValueError(
            f"preds and gold_labels must have the same shape, "
            f"got {preds.shape} vs {gold.shape}"
        )

    if mask is not None:
        m = _to_numpy(mask).astype(bool).ravel()
        if m.shape != preds.shape:
            raise ValueError("mask must have the same shape as preds")
        preds = preds[m]
        gold = gold[m]

    if preds.size == 0:
        return 0.0
    return float((preds == gold).mean())


def compute_iia_per_class(
    preds: ArrayLike,
    gold_labels: ArrayLike,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Per-class IIA.

    Returns a dict mapping class name (or stringified id) to its accuracy.
    Classes with zero gold occurrences map to ``float('nan')``.
    """
    preds = _to_numpy(preds).astype(np.int64).ravel()
    gold = _to_numpy(gold_labels).astype(np.int64).ravel()
    if preds.shape != gold.shape:
        raise ValueError("preds and gold_labels must have the same shape")

    classes = np.unique(gold)
    out: Dict[str, float] = {}
    for c in classes:
        m = gold == c
        if m.sum() == 0:
            acc = float("nan")
        else:
            acc = float((preds[m] == gold[m]).mean())
        name = class_names[int(c)] if class_names is not None else str(int(c))
        out[name] = acc
    return out
