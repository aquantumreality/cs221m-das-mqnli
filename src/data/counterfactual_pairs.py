"""Build (base, source) interchange-intervention pairs.

For each base example we pair it with one or more *source* examples and
compute the **gold counterfactual label** -- the label the model *should*
output after we transplant the value of some intermediate variable from
``source`` into ``base``.

Concretely, given a high-level intermediate variable ``V`` (e.g.
``lexical_relation`` or ``premise_word_identity``):

    counterfactual_label(base, source, V)
        = high_level.run(base_inputs,
                         interventions={V: high_level(source_inputs)[V]})

We materialize these tuples up front as a :class:`CounterfactualDataset`
which can be plugged directly into a DAS training loop or an IIA
evaluation loop. The tokenized fields (``base_input_ids``,
``source_input_ids``, ``intervention_pos``) are pre-computed so the
intervention loop just has to call the model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .causal_model import (
    LABEL2ID,
    LexicalCausalModel,
)
from .nli_templates import (
    DEFAULT_TEMPLATES,
    LEXICAL_PAIRS,
    NLITemplate,
    NLIExample,
    generate_examples,
)


# ---------------------------------------------------------------------------
# Single-example container
# ---------------------------------------------------------------------------


@dataclass
class CounterfactualExample:
    """One (base, source) interchange-intervention example.

    Fields
    ------
    base:
        :class:`NLIExample` for the base input.
    source:
        :class:`NLIExample` for the source input.
    target_variable:
        Name of the high-level intermediate variable being patched.
    base_label_id:
        Gold label id for the *unintervened* base example.
    source_label_id:
        Gold label id for the *unintervened* source example.
    counterfactual_label_id:
        Gold label id after patching ``target_variable`` from source
        into base, according to the high-level causal model. **This is
        the prediction target for IIA.**
    intervention_pos:
        Token index in the base sequence whose representation should be
        replaced. Computed by :func:`build_counterfactual_dataset`.
    """

    base: NLIExample
    source: NLIExample
    target_variable: str
    base_label_id: int
    source_label_id: int
    counterfactual_label_id: int
    intervention_pos: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["base"] = self.base.as_dict()
        d["source"] = self.source.as_dict()
        return d


# ---------------------------------------------------------------------------
# Position localisation
# ---------------------------------------------------------------------------


def _word_token_position(
    tokenizer,
    prompt: str,
    target_word: str,
    occurrence: int = 0,
) -> Optional[int]:
    """Return the token index of ``target_word`` inside ``prompt``.

    We tokenize the prompt with ``return_offsets_mapping=True`` and look
    for the first token whose character span begins inside the substring
    range of the requested occurrence of ``target_word``. If the
    tokenizer cannot give offsets (rare for HF fast tokenizers), we fall
    back to a slower decode-based search.
    """
    if tokenizer.is_fast:
        enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
        # Find the ``occurrence``-th occurrence of target_word in prompt.
        start = -1
        end = -1
        search_from = 0
        for _ in range(occurrence + 1):
            start = prompt.find(target_word, search_from)
            if start == -1:
                return None
            end = start + len(target_word)
            search_from = end
        for tok_idx, (a, b) in enumerate(offsets):
            if a == b:  # special token / empty span
                continue
            if a <= start < b or (start <= a < end):
                return tok_idx
        return None

    # Slow path: decode each prefix and find the first prefix that
    # contains the target word.
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    for i in range(len(ids)):
        decoded = tokenizer.decode(ids[: i + 1])
        if target_word in decoded:
            return i
    return None


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class CounterfactualDataset(Dataset):
    """Torch Dataset over pre-tokenized counterfactual examples.

    Each item is a dict with keys:
        - ``base_input_ids``  : LongTensor [T]
        - ``source_input_ids``: LongTensor [T]
        - ``base_attention_mask``, ``source_attention_mask``
        - ``base_label_id``           : LongTensor scalar
        - ``source_label_id``         : LongTensor scalar
        - ``counterfactual_label_id`` : LongTensor scalar (training target)
        - ``intervention_pos``        : LongTensor scalar
        - ``target_variable``         : str (collated as list)
    """

    def __init__(
        self,
        examples: Sequence[CounterfactualExample],
        tokenizer,
        max_length: Optional[int] = None,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        # Tokenize once and pad to the longest prompt across base+source
        # so positions align. This is the cleanest setup for fixed-position
        # interventions.
        all_prompts = [ex.base.prompt for ex in self.examples] + [
            ex.source.prompt for ex in self.examples
        ]
        enc = tokenizer(
            all_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        n = len(self.examples)
        self._base_input_ids = enc["input_ids"][:n]
        self._source_input_ids = enc["input_ids"][n:]
        self._base_attn = enc["attention_mask"][:n]
        self._source_attn = enc["attention_mask"][n:]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        pos = ex.intervention_pos if ex.intervention_pos is not None else 0
        return {
            "base_input_ids": self._base_input_ids[idx],
            "source_input_ids": self._source_input_ids[idx],
            "base_attention_mask": self._base_attn[idx],
            "source_attention_mask": self._source_attn[idx],
            "base_label_id": torch.tensor(ex.base_label_id, dtype=torch.long),
            "source_label_id": torch.tensor(ex.source_label_id, dtype=torch.long),
            "counterfactual_label_id": torch.tensor(
                ex.counterfactual_label_id, dtype=torch.long
            ),
            "intervention_pos": torch.tensor(pos, dtype=torch.long),
            "target_variable": ex.target_variable,
        }


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


# Which surface word each high-level intermediate variable corresponds to.
# This is the bridge between the symbolic causal model and the token
# positions in the prompt that we will intervene on.
_VARIABLE_TO_SOURCE_WORD: Dict[str, str] = {
    "premise_word_identity": "premise_word",
    "hypothesis_word_identity": "hypothesis_word",
    # ``lexical_relation`` is not localised to a single surface word at
    # the high level; we treat the *hypothesis* word position as the
    # default intervention site (it's where the model first has both
    # operands available) but callers can override.
    "lexical_relation": "hypothesis_word",
}


def build_counterfactual_dataset(
    tokenizer,
    *,
    target_variable: str = "lexical_relation",
    pairs: Sequence[Tuple[str, str, str]] = LEXICAL_PAIRS,
    templates: Sequence[NLITemplate] = DEFAULT_TEMPLATES,
    n_examples: int = 64,
    seed: int = 0,
    max_length: Optional[int] = None,
    intervention_word: Optional[str] = None,
    require_label_change: bool = False,
) -> CounterfactualDataset:
    """Generate a counterfactual dataset for interchange interventions.

    Parameters
    ----------
    tokenizer:
        HuggingFace tokenizer used to localise the intervention position.
    target_variable:
        Which intermediate variable in the high-level model to patch.
        Must be one of ``"premise_word_identity"``,
        ``"hypothesis_word_identity"``, ``"lexical_relation"``.
    pairs, templates:
        Lexical pairs and surface templates to sample base/source from.
    n_examples:
        Number of (base, source) pairs to generate.
    seed:
        RNG seed for reproducibility.
    max_length:
        Optional max sequence length passed to the tokenizer.
    intervention_word:
        Which surface word's token to patch. Defaults to the canonical
        position for ``target_variable`` (see ``_VARIABLE_TO_SOURCE_WORD``).
    require_label_change:
        If True, only keep pairs whose counterfactual label *differs*
        from the base label. This makes IIA evaluation strictly harder
        (random-guess baseline drops) and is the convention used in the
        original DAS paper.

    Returns
    -------
    :class:`CounterfactualDataset`
    """
    if target_variable not in _VARIABLE_TO_SOURCE_WORD:
        raise ValueError(
            f"target_variable {target_variable!r} not in "
            f"{sorted(_VARIABLE_TO_SOURCE_WORD)}"
        )
    intervention_word_key = intervention_word or _VARIABLE_TO_SOURCE_WORD[target_variable]

    rng = random.Random(seed)
    base_examples = generate_examples(pairs=pairs, templates=templates)
    if not base_examples:
        raise ValueError("No NLI examples were generated.")

    out: List[CounterfactualExample] = []
    attempts = 0
    max_attempts = n_examples * 20  # avoid infinite loops if filters are tight

    while len(out) < n_examples and attempts < max_attempts:
        attempts += 1
        base = rng.choice(base_examples)
        # Source must share the same surface template so positions align.
        candidate_sources = [
            ex for ex in base_examples if ex.template_name == base.template_name
        ]
        source = rng.choice(candidate_sources)

        # Compute the gold counterfactual label by querying the high-level
        # model with the intervention applied.
        causal = LexicalCausalModel(
            monotonicity=_template_monotonicity(base.template_name, templates)
        )

        # Build the intervention dict: replace the target variable in the
        # base trace with the source's value of that variable. For the
        # word-identity variables this means literally substituting one
        # word for another; for ``lexical_relation`` we read the source's
        # relation and inject it.
        source_trace = causal.run(
            premise_word=source.premise_word,
            hypothesis_word=source.hypothesis_word,
            context=source.template_name,
        )
        interventions = {target_variable: source_trace[target_variable]}
        cf_trace = causal.run(
            premise_word=base.premise_word,
            hypothesis_word=base.hypothesis_word,
            context=base.template_name,
            interventions=interventions,
        )
        cf_label_id = cf_trace["label_id"]

        if require_label_change and cf_label_id == base.label_id:
            continue

        # Localise the intervention position inside the base prompt.
        target_word = {
            "premise_word": base.premise_word,
            "hypothesis_word": base.hypothesis_word,
        }[intervention_word_key]
        # If premise_word == hypothesis_word (EQUIV bases) we still want
        # the *second* occurrence when the variable is the hypothesis
        # word.
        occurrence = 0
        if (
            intervention_word_key == "hypothesis_word"
            and base.premise_word == base.hypothesis_word
        ):
            occurrence = 1
        pos = _word_token_position(
            tokenizer, base.prompt, target_word, occurrence=occurrence
        )
        if pos is None:
            # Skip examples we can't localise. This should be rare with
            # the curated vocabulary above.
            continue

        out.append(
            CounterfactualExample(
                base=base,
                source=source,
                target_variable=target_variable,
                base_label_id=base.label_id,
                source_label_id=source.label_id,
                counterfactual_label_id=cf_label_id,
                intervention_pos=pos,
            )
        )

    if len(out) < n_examples:
        # Soft warning rather than a hard error: callers may have asked
        # for more examples than the (small) lexical bank can supply
        # under tight filters.
        import warnings

        warnings.warn(
            f"Only produced {len(out)} / {n_examples} counterfactual examples "
            f"after {attempts} attempts. Consider relaxing filters or "
            f"expanding LEXICAL_PAIRS.",
            RuntimeWarning,
        )

    return CounterfactualDataset(out, tokenizer=tokenizer, max_length=max_length)


def _template_monotonicity(
    template_name: str,
    templates: Sequence[NLITemplate],
) -> str:
    """Look up a template's monotonicity by name; default to ``"upward"``."""
    for t in templates:
        if t.name == template_name:
            return t.monotonicity
    return "upward"
