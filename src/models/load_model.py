"""Thin wrapper around HuggingFace ``AutoModelForCausalLM``.

We default to small models (``distilgpt2``, ``gpt2``) so the entire pipeline
can run on a laptop CPU when debugging. The function returns the tokenizer
and the model already moved onto the requested device and put in ``eval``
mode (we do *not* train the base LM in this project; only DAS intervention
parameters are learned).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


_DEFAULT_MODEL = "distilgpt2"


def load_causal_lm(
    model_name: str = _DEFAULT_MODEL,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load a HuggingFace causal LM + tokenizer.

    Parameters
    ----------
    model_name:
        Any HF Hub model id with a causal LM head, e.g. ``"distilgpt2"``,
        ``"gpt2"``, ``"gpt2-medium"``, ``"EleutherAI/pythia-160m"``.
    device:
        Destination device. If ``None``, picks CUDA if available, else CPU.
        Accepts strings (``"cuda"``, ``"cuda:0"``, ``"cpu"``) or
        :class:`torch.device` objects.
    dtype:
        Optional torch dtype to cast the model to. If ``None``, uses
        ``float32`` on CPU and ``float32`` on GPU as well (we keep things
        simple; if you want fp16 / bf16 pass it explicitly).
    cache_dir:
        Optional HF cache directory.

    Returns
    -------
    tokenizer, model
        - ``tokenizer``: a :class:`PreTrainedTokenizerBase` with a valid
          ``pad_token`` set (we fall back to ``eos_token`` if the base
          tokenizer doesn't define one, which is the case for GPT-2).
        - ``model``: a :class:`PreTrainedModel` in ``eval()`` mode on
          ``device``.

    Examples
    --------
    >>> tok, model = load_causal_lm("distilgpt2", device="cpu")
    >>> out = model(**tok("Hello", return_tensors="pt"))
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    # GPT-2 family has no pad token; use EOS as pad. This is safe for our
    # use case because we always compute logits at fixed positions and
    # never train the LM itself.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if cache_dir is not None:
        model_kwargs["cache_dir"] = cache_dir

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()
    # We never want to update the base LM in this project.
    for p in model.parameters():
        p.requires_grad_(False)

    return tokenizer, model
