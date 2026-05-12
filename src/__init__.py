"""CS221M DAS-for-controlled-NLI source package.

Subpackages
-----------
- ``data``: NLI templates, high-level causal model, counterfactual pair generation.
- ``models``: small HuggingFace causal LM loaders.
- ``interventions``: pyvene-based activation patching and DAS interventions.
- ``metrics``: IIA, logit-difference, logit-recovery.
- ``viz``: plotting helpers (patching heatmaps, etc.).
- ``utils``: misc utilities (seeding, devices, ...).
"""

__all__ = [
    "data",
    "models",
    "interventions",
    "metrics",
    "viz",
    "utils",
]
