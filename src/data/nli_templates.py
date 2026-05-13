"""Controlled NLI template generator.

We synthesise tiny NLI examples by plugging single-word lexical items
into a fixed sentence frame. This gives us full control over the position
of the *content word* in the tokenized input, which is critical for
position-wise interventions like activation patching and DAS.

Each :class:`NLITemplate` defines:

- a ``premise_format`` containing the placeholder ``{word}``,
- a ``hypothesis_format`` containing the placeholder ``{word}``,
- an ``answer_prefix`` that nudges the model to emit a label token
  (we use a tiny verbalizer at decode time, e.g. " yes"/" no"/" maybe").

The default template is *upward-monotone* ("A X is on the table.") which
makes the relation->label table in :mod:`src.data.causal_model` directly
applicable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .causal_model import (
    LABEL2ID,
    LexicalCausalModel,
)


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

# Hand-curated lexical pairs covering the four relation types. Each entry is
# (premise_word, hypothesis_word, relation). These are intentionally chosen
# so that both words tokenize to a *single* GPT-2 BPE token when preceded
# by a space, which keeps the position-wise interventions clean.
LEXICAL_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    # EQUIV (identity; entailment in any monotonicity context)
    ("dog", "dog", "EQUIV"),
    ("cat", "cat", "EQUIV"),
    ("wolf", "wolf", "EQUIV"),
    ("fox", "fox", "EQUIV"),
    ("horse", "horse", "EQUIV"),
    ("robin", "robin", "EQUIV"),
    ("eagle", "eagle", "EQUIV"),
    ("car", "car", "EQUIV"),
    ("truck", "truck", "EQUIV"),
    ("boat", "boat", "EQUIV"),
    ("rose", "rose", "EQUIV"),
    ("hammer", "hammer", "EQUIV"),
    ("knife", "knife", "EQUIV"),
    # FORWARD (hyponym -> hypernym; entailment)
    # mammals
    ("dog", "mammal", "FORWARD"),
    ("dog", "animal", "FORWARD"),
    ("cat", "mammal", "FORWARD"),
    ("cat", "animal", "FORWARD"),
    ("wolf", "mammal", "FORWARD"),
    ("fox", "animal", "FORWARD"),
    ("horse", "mammal", "FORWARD"),
    ("cow", "animal", "FORWARD"),
    # birds
    ("robin", "bird", "FORWARD"),
    ("sparrow", "bird", "FORWARD"),
    ("eagle", "bird", "FORWARD"),
    ("owl", "animal", "FORWARD"),
    # vehicles
    ("car", "vehicle", "FORWARD"),
    ("truck", "vehicle", "FORWARD"),
    ("boat", "vehicle", "FORWARD"),
    ("plane", "vehicle", "FORWARD"),
    # plants
    ("rose", "flower", "FORWARD"),
    ("tulip", "flower", "FORWARD"),
    # tools
    ("hammer", "tool", "FORWARD"),
    ("saw", "tool", "FORWARD"),
    ("knife", "tool", "FORWARD"),
    ("drill", "tool", "FORWARD"),
    # REVERSE (hypernym -> hyponym; neutral in upward contexts)
    # mammals
    ("mammal", "dog", "REVERSE"),
    ("animal", "dog", "REVERSE"),
    ("mammal", "cat", "REVERSE"),
    ("animal", "cat", "REVERSE"),
    ("mammal", "wolf", "REVERSE"),
    ("animal", "fox", "REVERSE"),
    ("mammal", "horse", "REVERSE"),
    ("animal", "cow", "REVERSE"),
    # birds
    ("bird", "robin", "REVERSE"),
    ("bird", "sparrow", "REVERSE"),
    ("bird", "eagle", "REVERSE"),
    ("animal", "owl", "REVERSE"),
    # vehicles
    ("vehicle", "car", "REVERSE"),
    ("vehicle", "truck", "REVERSE"),
    ("vehicle", "boat", "REVERSE"),
    ("vehicle", "plane", "REVERSE"),
    # plants
    ("flower", "rose", "REVERSE"),
    ("flower", "tulip", "REVERSE"),
    # tools
    ("tool", "hammer", "REVERSE"),
    ("tool", "saw", "REVERSE"),
    ("tool", "knife", "REVERSE"),
    ("tool", "drill", "REVERSE"),
    # DISJOINT (contradiction in upward contexts).
    # We mix animals x vehicles, animals x tools, animals x plants,
    # vehicles x plants, vehicles x tools, plants x tools so the
    # rotation has to find a relation feature, not a category feature.
    ("dog", "car", "DISJOINT"),
    ("dog", "hammer", "DISJOINT"),
    ("cat", "truck", "DISJOINT"),
    ("cat", "knife", "DISJOINT"),
    ("wolf", "boat", "DISJOINT"),
    ("fox", "rose", "DISJOINT"),
    ("horse", "hammer", "DISJOINT"),
    ("cow", "plane", "DISJOINT"),
    ("robin", "saw", "DISJOINT"),
    ("sparrow", "car", "DISJOINT"),
    ("eagle", "knife", "DISJOINT"),
    ("owl", "drill", "DISJOINT"),
    ("car", "rose", "DISJOINT"),
    ("truck", "rose", "DISJOINT"),
    ("boat", "hammer", "DISJOINT"),
    ("plane", "rose", "DISJOINT"),
    ("rose", "truck", "DISJOINT"),
    ("rose", "knife", "DISJOINT"),
    ("tulip", "hammer", "DISJOINT"),
    ("hammer", "rose", "DISJOINT"),
    ("hammer", "robin", "DISJOINT"),
    ("saw", "horse", "DISJOINT"),
    ("knife", "cat", "DISJOINT"),
    ("drill", "owl", "DISJOINT"),
)


# ---------------------------------------------------------------------------
# Template dataclass
# ---------------------------------------------------------------------------


@dataclass
class NLITemplate:
    """A reusable NLI surface form.

    The format strings must each contain a single ``{word}`` placeholder.
    The ``answer_prefix`` is concatenated after the hypothesis so we can
    read a label from the next-token logits over the verbalizer tokens.
    """

    name: str
    premise_format: str = "A {word} is on the table."
    hypothesis_format: str = "A {word} is on the table."
    answer_prefix: str = " Answer:"
    # The verbalizer maps NLI labels to surface strings that should be
    # tokenizable to a single token after a leading space (e.g. " yes",
    # " no", " maybe" for GPT-2).
    verbalizer: Dict[str, str] = field(
        default_factory=lambda: {
            "entailment": " yes",
            "neutral": " maybe",
            "contradiction": " no",
        }
    )
    # Whether the carrier sentence is upward- or downward-monotone in the
    # premise/hypothesis word slot. Used to look up the right
    # relation->label table.
    monotonicity: str = "upward"

    def format_prompt(self, premise_word: str, hypothesis_word: str) -> str:
        """Materialize a full prompt for a (premise_word, hypothesis_word) pair."""
        premise = self.premise_format.format(word=premise_word)
        hypothesis = self.hypothesis_format.format(word=hypothesis_word)
        return f"{premise} {hypothesis}{self.answer_prefix}"


# A small bank of default templates. We start with a single upward-monotone
# frame; downward and quantified frames are stubs for later experiments.
DEFAULT_TEMPLATES: Tuple[NLITemplate, ...] = (
    NLITemplate(name="on_the_table"),
    NLITemplate(
        name="i_saw_a",
        premise_format="I saw a {word} yesterday.",
        hypothesis_format="I saw a {word} yesterday.",
    ),
)


# ---------------------------------------------------------------------------
# Example dataclass + generator
# ---------------------------------------------------------------------------


@dataclass
class NLIExample:
    """One controlled NLI example with full provenance."""

    template_name: str
    premise_word: str
    hypothesis_word: str
    lexical_relation: str
    label: str
    label_id: int
    prompt: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "template_name": self.template_name,
            "premise_word": self.premise_word,
            "hypothesis_word": self.hypothesis_word,
            "lexical_relation": self.lexical_relation,
            "label": self.label,
            "label_id": self.label_id,
            "prompt": self.prompt,
        }


def generate_examples(
    pairs: Sequence[Tuple[str, str, str]] = LEXICAL_PAIRS,
    templates: Sequence[NLITemplate] = DEFAULT_TEMPLATES,
    max_examples: Optional[int] = None,
) -> List[NLIExample]:
    """Materialize the cartesian product of ``pairs`` x ``templates``.

    Parameters
    ----------
    pairs:
        Iterable of ``(premise_word, hypothesis_word, relation)`` triples.
        Defaults to :data:`LEXICAL_PAIRS`.
    templates:
        Iterable of :class:`NLITemplate`. Defaults to :data:`DEFAULT_TEMPLATES`.
    max_examples:
        If given, truncate the output to at most this many examples.

    Returns
    -------
    list of :class:`NLIExample`.
    """
    out: List[NLIExample] = []
    for template in templates:
        # We respect the template's monotonicity when deriving labels.
        causal = LexicalCausalModel(monotonicity=template.monotonicity)
        for premise_word, hypothesis_word, expected_rel in pairs:
            trace = causal.run(
                premise_word=premise_word,
                hypothesis_word=hypothesis_word,
                context=template.name,
            )
            # Sanity-check that our hand-labeled relation agrees with the
            # symbolic model. If it doesn't, prefer the symbolic model and
            # warn loudly via assertion - this catches typos in
            # LEXICAL_PAIRS during development.
            assert trace["lexical_relation"] == expected_rel, (
                f"Relation mismatch for ({premise_word!r}, {hypothesis_word!r}): "
                f"hand-labeled {expected_rel!r} vs model {trace['lexical_relation']!r}"
            )
            prompt = template.format_prompt(premise_word, hypothesis_word)
            out.append(
                NLIExample(
                    template_name=template.name,
                    premise_word=premise_word,
                    hypothesis_word=hypothesis_word,
                    lexical_relation=trace["lexical_relation"],
                    label=trace["label"],
                    label_id=LABEL2ID[trace["label"]],
                    prompt=prompt,
                )
            )
            if max_examples is not None and len(out) >= max_examples:
                return out
    return out
