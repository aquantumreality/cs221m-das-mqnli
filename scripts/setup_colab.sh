#!/usr/bin/env bash
set -e

echo "Installing CS221M DAS MQNLI dependencies..."
pip install -q -r requirements-colab.txt

echo "Checking Python imports..."
python - <<'PY'
import torch
import transformers
import datasets
import numpy as np
import pandas as pd
import matplotlib
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("basic imports OK")
PY
