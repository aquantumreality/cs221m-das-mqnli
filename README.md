# CS221M: Distributed Alignment Search for Controlled NLI

Reproduction-then-extension of **Distributed Alignment Search (DAS)**
(Geiger et al. 2023) on a controlled lexical NLI task, built on top of
[pyvene](https://github.com/stanfordnlp/pyvene).

## Project framing

- **Main method.** DAS via pyvene -- learn an orthogonal rotation of a
  hidden-state subspace so that interchange interventions on that
  subspace causally implement an intermediate variable of a symbolic
  high-level program.
- **Main task.** Controlled NLI on single-word lexical-entailment
  examples (`A dog is on the table.` / `A animal is on the table.`),
  with full control over premise word, hypothesis word, and surface
  template.
- **Baseline.** Activation patching: a layer x token-position heatmap of
  vanilla interchange interventions (no learned rotation), reduced by
  logit-difference recovery.
- **Main metric.** Interchange intervention accuracy (IIA). We also
  report logit-difference recovery as a secondary metric.

## Repo layout

```
src/
  data/          NLI templates, high-level causal model, counterfactual pairs
  models/        HuggingFace causal-LM loader
  interventions/ pyvene-based activation patching + DAS hooks
  metrics/       IIA, logit-diff, logit recovery
  viz/           heatmap plotting
  utils/         seeding, device selection
notebooks/       exploratory notebooks (DAS_Main_Introduction, Boundless_DAS, ...)
outputs/         saved figures, JSON metric dumps, checkpoints
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Smoke test (CPU is fine):

```python
from src.utils import set_seed
from src.models import load_causal_lm
from src.data import build_counterfactual_dataset
from src.metrics import LabelVerbalizer
from src.interventions import run_activation_patching_sweep
from src.viz import save_patching_heatmap

set_seed(0)
tok, model = load_causal_lm("distilgpt2", device="cpu")
ds = build_counterfactual_dataset(
    tok, target_variable="lexical_relation", n_examples=16, seed=0,
)
verb = LabelVerbalizer.from_tokenizer(
    tok, {"entailment": " yes", "neutral": " maybe", "contradiction": " no"}
)
result = run_activation_patching_sweep(model, ds, verbalizer=verb, metric="iia")
save_patching_heatmap(result, "outputs/patching_iia.png",
                      cmap="viridis", center=None, vmin=0.0, vmax=1.0)
```

## Conventions

- All scripts work on CPU with tiny data for debugging, and pick up CUDA
  if available.
- Label order is fixed: `entailment=0`, `neutral=1`, `contradiction=2`
  (see `src/data/causal_model.LABELS`).
- The high-level causal model in `src/data/causal_model.py` is the
  **ground truth** used to compute gold counterfactual labels; do not
  modify it without re-running the smoke tests.
