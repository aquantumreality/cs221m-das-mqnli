# CLAUDE.md

This repo is for a CS221M mechanistic interpretability project using pyvene.

Project goal:
- Reproduce the Distributed Alignment Search (DAS) causal-abstraction pipeline.
- Adapt it to controlled NLI / MQNLI-style experiments.
- Use activation patching as a coarse layer × token-position baseline.
- Main metric: interchange intervention accuracy (IIA), plus optional logit-difference recovery.

Important instructions:
- Do not restructure the repo without asking.
- Do not rewrite the main notebooks from scratch.
- Do not add large model checkpoints to Git.
- Keep experiments runnable in Colab first.
- Prefer small, debuggable changes.
- Keep scientific interpretation separate from engineering refactors.

Main notebooks:
- notebooks/MQNLI_DAS_experiments_updated.ipynb: main controlled NLI / MQNLI DAS notebook.
- notebooks/reference/01_das_original_hierarchical_equality.ipynb: original DAS reference.
- notebooks/reference/02_boundless_das_scaling.ipynb: Boundless DAS reference.

Immediate check-in goals:
1. Run MQNLI notebook in Colab.
2. Establish factual model accuracy.
3. Run DAS on QP_S as final-label sanity check.
4. Run DAS on NegP as first intermediate-variable target.
5. Save a small CSV/table/figure in outputs/results or outputs/figures.
