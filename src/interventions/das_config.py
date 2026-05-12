"""Build a Distributed Alignment Search (DAS) ``IntervenableConfig``.

This is the minimal "learned rotation" intervention used by DAS
(Geiger et al. 2023). Compared to the vanilla activation patching
baseline in :mod:`src.interventions.patching`, here we learn an
*orthogonal rotation* of the hidden-state subspace at a single
``(layer, component, position)`` site; the rotated subspace is what
gets swapped between source and base, so training pressure is what
finds the dimensions of the hidden state that causally implement the
target high-level variable.

We default to :class:`pyvene.LowRankRotatedSpaceIntervention` because
its API is the cleanest fit for "patch a ``d``-dim subspace at one
token":

- the rotation matrix is a single ``[hidden_size, low_rank_dimension]``
  parameter (a slice of an orthogonal matrix),
- *all* of the rotated subspace is swapped (no extra
  ``subspace_partition`` plumbing),
- the only trainable parameter is the rotation itself.

If a particular pyvene build doesn't expose ``LowRankRotatedSpaceIntervention``
we fall back to the full-rank :class:`pyvene.RotatedSpaceIntervention`
with a binary subspace partition ``[[0, d], [d, hidden]]`` and
intervene on subspace ``0``. The two formulations are mathematically
equivalent for our use case.
"""

from __future__ import annotations

from typing import Optional

import pyvene as pv


def _get_hidden_size(model) -> int:
    """Best-effort hidden size lookup across HF causal LM configs."""
    cfg = getattr(model, "config", None)
    for attr in ("hidden_size", "n_embd", "d_model"):
        if cfg is not None and hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise ValueError("Could not infer hidden size from model.config")


def _pick_intervention_class():
    """Return ``(cls, uses_low_rank_dimension_kwarg)``.

    We prefer :class:`LowRankRotatedSpaceIntervention` if pyvene exposes
    it; otherwise we fall back to the full-rank rotation.
    """
    if hasattr(pv, "LowRankRotatedSpaceIntervention"):
        return pv.LowRankRotatedSpaceIntervention, True
    if hasattr(pv, "RotatedSpaceIntervention"):
        return pv.RotatedSpaceIntervention, False
    raise ImportError(
        "Neither LowRankRotatedSpaceIntervention nor RotatedSpaceIntervention "
        "is available in your installed pyvene. Please upgrade pyvene."
    )


def make_das_config(
    model,
    layer: int,
    component: str = "block_output",
    intervention_dim: int = 4,
    unit: str = "pos",
    *,
    max_number_of_units: int = 1,
    force_full_rank: bool = False,
) -> pv.IntervenableConfig:
    """Build an :class:`IntervenableConfig` for DAS.

    Parameters
    ----------
    model:
        The HuggingFace causal LM that the intervention will be attached
        to. We need it only to infer ``hidden_size`` and ``type(model)``.
    layer:
        Transformer-block index at which to intervene.
    component:
        pyvene component name (default ``"block_output"`` -- the
        post-block residual stream, which is the canonical DAS site).
    intervention_dim:
        Dimensionality of the rotated subspace that DAS will swap. For
        our lexical-NLI task we have 4 lexical relations, so 4 is a
        sensible default; in general pick the smallest ``d`` that
        seems to work.
    unit:
        pyvene unit type. ``"pos"`` (default) addresses a single token
        position; ``"h"`` would address an attention head.
    max_number_of_units:
        Number of units swapped per intervention (default 1 -- patch a
        single token).
    force_full_rank:
        If True, use :class:`RotatedSpaceIntervention` with a binary
        subspace partition even if the low-rank variant is available.
        Useful when comparing implementations.

    Returns
    -------
    :class:`pyvene.IntervenableConfig`
        Configuration ready to be passed to
        :class:`pyvene.IntervenableModel`.

    Notes
    -----
    The trainable parameters created by pyvene live under
    ``intervenable.interventions[k].rotate_layer`` for every key ``k``.
    :func:`src.interventions.train_das.train_das_alignment` knows how to
    pull them out for the optimizer.
    """
    hidden = _get_hidden_size(model)
    if intervention_dim < 1 or intervention_dim > hidden:
        raise ValueError(
            f"intervention_dim={intervention_dim} must be in [1, hidden_size={hidden}]"
        )

    intervention_cls, low_rank_supported = _pick_intervention_class()
    if force_full_rank:
        if not hasattr(pv, "RotatedSpaceIntervention"):
            raise ImportError("RotatedSpaceIntervention not available in this pyvene build")
        intervention_cls = pv.RotatedSpaceIntervention
        low_rank_supported = False

    if low_rank_supported:
        rep = pv.RepresentationConfig(
            layer,
            component,
            unit,
            max_number_of_units,
            low_rank_dimension=intervention_dim,
        )
    else:
        # Full-rank rotation: split into a "target" subspace of size
        # ``intervention_dim`` and a "rest" subspace, and intervene on
        # subspace 0 (the target).
        rep = pv.RepresentationConfig(
            layer,
            component,
            unit,
            max_number_of_units,
            subspace_partition=[[0, intervention_dim], [intervention_dim, hidden]],
        )

    config = pv.IntervenableConfig(
        model_type=type(model),
        representations=[rep],
        intervention_types=intervention_cls,
    )
    # Stash some bookkeeping so downstream code can introspect without
    # having to re-derive everything.
    config.__dict__["_das_meta"] = {
        "layer": int(layer),
        "component": str(component),
        "intervention_dim": int(intervention_dim),
        "unit": str(unit),
        "max_number_of_units": int(max_number_of_units),
        "hidden_size": int(hidden),
        "intervention_class": intervention_cls.__name__,
        "low_rank_supported": bool(low_rank_supported),
    }
    return config


def das_config_meta(config: pv.IntervenableConfig) -> dict:
    """Return the bookkeeping dict :func:`make_das_config` attached to ``config``.

    Returns an empty dict if ``config`` wasn't built by ``make_das_config``.
    """
    return dict(config.__dict__.get("_das_meta", {}))
