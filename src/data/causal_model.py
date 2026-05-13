"""High-level causal model of lexical entailment for controlled NLI.

This module formalises the **high-level** (algorithmic / symbolic) program
that we hope the low-level neural network implements. In the DAS framework
(Geiger et al. 2023), causal-abstraction analyses compare a *low-level*
model (a neural net) to a *high-level* model (a symbolic algorithm) by
performing matched interventions on both and checking whether the
outputs agree (the **interchange intervention accuracy**, IIA).

The high-level model implemented here is intentionally small. It captures
single-word lexical entailment under simple monotonicity-flavored
constructions:

    Premise:    "A {premise_word} is on the table."
    Hypothesis: "A {hypothesis_word} is on the table."

Given just the two content words, the lexical relation between them
(EQUIV / FORWARD / REVERSE / DISJOINT) determines the NLI label
(entailment / neutral / contradiction). For example:

    - dog,  dog    -> EQUIV     -> ENTAILMENT
    - dog,  animal -> FORWARD   -> ENTAILMENT  (hypernym)
    - animal, dog  -> REVERSE   -> NEUTRAL     (hyponym; loses info)
    - dog,  car    -> DISJOINT  -> CONTRADICTION

Variables in the high-level model
---------------------------------
Input variables:
    - ``premise_word``     : str
    - ``hypothesis_word``  : str
    - ``context``          : str (the carrier sentence frame; ignored
      semantically but kept around so interventions can swap it).

Intermediate variables:
    - ``premise_word_identity``    : str  (just the premise word)
    - ``hypothesis_word_identity`` : str  (just the hypothesis word)
    - ``lexical_relation``         : one of LEXICAL_RELATIONS

Output:
    - ``label`` : one of LABELS

This is exactly the kind of structure that DAS targets: each intermediate
variable corresponds to a candidate **subspace** in some hidden state of
the LM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Label / relation vocabulary
# ---------------------------------------------------------------------------

LABELS: Tuple[str, ...] = ("entailment", "neutral", "contradiction")
LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}
ID2LABEL: Dict[int, str] = {i: l for l, i in LABEL2ID.items()}

LEXICAL_RELATIONS: Tuple[str, ...] = ("EQUIV", "FORWARD", "REVERSE", "DISJOINT")

# Mapping from lexical relation -> NLI label, under an *upward-monotone*
# (positive) context like "A X is on the table." This is the table you would
# see in a NatLog-style system.
RELATION_TO_LABEL_UPWARD: Dict[str, str] = {
    "EQUIV": "entailment",
    "FORWARD": "entailment",   # hyponym -> hypernym preserves truth upward
    "REVERSE": "neutral",      # hypernym -> hyponym does not entail
    "DISJOINT": "contradiction",
}

# In a *downward-monotone* context like "No X is on the table." the
# direction flips. We keep this here so future experiments can swap it in.
RELATION_TO_LABEL_DOWNWARD: Dict[str, str] = {
    "EQUIV": "entailment",
    "FORWARD": "neutral",
    "REVERSE": "entailment",
    "DISJOINT": "contradiction",
}


# ---------------------------------------------------------------------------
# Lexical-knowledge base
# ---------------------------------------------------------------------------
# A *very* small ontology so we can ground the relations symbolically. In a
# real project you would swap this for WordNet or a similar resource; this
# minimal version makes unit tests trivial and keeps the high-level model
# fully transparent.

# Each entry maps a word to the set of words it is a hyponym of (i.e.
# things that are *more general* than it). EQUIV is the reflexive closure;
# DISJOINT is everything not in either hypernym chain.
_HYPERNYMS: Dict[str, Tuple[str, ...]] = {
    # animals - mammals
    "dog":    ("mammal", "animal", "creature"),
    "cat":    ("mammal", "animal", "creature"),
    "wolf":   ("mammal", "animal", "creature"),
    "fox":    ("mammal", "animal", "creature"),
    "horse":  ("mammal", "animal", "creature"),
    "cow":    ("mammal", "animal", "creature"),
    # animals - birds
    "robin":   ("bird",  "animal", "creature"),
    "sparrow": ("bird",  "animal", "creature"),
    "eagle":   ("bird",  "animal", "creature"),
    "owl":     ("bird",  "animal", "creature"),
    # animal hypernyms
    "mammal": ("animal", "creature"),
    "bird":   ("animal", "creature"),
    "animal": ("creature",),
    # vehicles
    "car":   ("vehicle", "object"),
    "truck": ("vehicle", "object"),
    "bike":  ("vehicle", "object"),
    "boat":  ("vehicle", "object"),
    "plane": ("vehicle", "object"),
    "ship":  ("vehicle", "object"),
    "vehicle": ("object",),
    # plants / flowers
    "rose":  ("flower", "plant"),
    "tulip": ("flower", "plant"),
    "flower": ("plant",),
    # tools
    "hammer": ("tool", "object"),
    "saw":    ("tool", "object"),
    "knife":  ("tool", "object"),
    "drill":  ("tool", "object"),
    "axe":    ("tool", "object"),
    "tool":   ("object",),
}


def _is_hyponym_of(a: str, b: str) -> bool:
    """Return True if ``a`` is a (transitive) hyponym of ``b``."""
    return b in _HYPERNYMS.get(a, ())


def lexical_relation(a: str, b: str) -> str:
    """Return the lexical relation between ``a`` and ``b``.

    The relation is reported from premise -> hypothesis: ``FORWARD`` means
    ``a`` entails ``b`` (a is more specific), ``REVERSE`` means ``b``
    entails ``a``, ``EQUIV`` means they're the same word, and ``DISJOINT``
    is the catch-all.
    """
    if a == b:
        return "EQUIV"
    if _is_hyponym_of(a, b):
        return "FORWARD"
    if _is_hyponym_of(b, a):
        return "REVERSE"
    return "DISJOINT"


# ---------------------------------------------------------------------------
# High-level program
# ---------------------------------------------------------------------------


@dataclass
class LexicalCausalModel:
    """Symbolic, fully-transparent high-level program for the NLI task.

    The model has three observable input variables (``premise_word``,
    ``hypothesis_word``, ``context``) and three intermediate variables
    (the two word identities and the lexical relation). The output is the
    NLI label.

    Interventions on intermediate variables are performed by passing the
    target variable name and the new value to :meth:`run`. Internally this
    short-circuits the relevant computation, exactly like an intervention
    in a structural causal model.

    Attributes
    ----------
    monotonicity:
        ``"upward"`` (default) or ``"downward"``. Selects which
        relation->label table to use, so the same lexical relation can map
        to different labels depending on the carrier frame.
    """

    monotonicity: str = "upward"

    # We carry the variable names as class attributes so external code
    # (DAS configs, intervention site enumerations) can reference them.
    INPUT_VARS: Tuple[str, ...] = field(
        default=("premise_word", "hypothesis_word", "context"), init=False
    )
    INTERMEDIATE_VARS: Tuple[str, ...] = field(
        default=(
            "premise_word_identity",
            "hypothesis_word_identity",
            "lexical_relation",
        ),
        init=False,
    )
    OUTPUT_VARS: Tuple[str, ...] = field(default=("label",), init=False)

    def _relation_to_label(self) -> Dict[str, str]:
        if self.monotonicity == "upward":
            return RELATION_TO_LABEL_UPWARD
        if self.monotonicity == "downward":
            return RELATION_TO_LABEL_DOWNWARD
        raise ValueError(f"Unknown monotonicity {self.monotonicity!r}")

    # ------------------------------------------------------------------
    # Forward pass with optional interventions
    # ------------------------------------------------------------------

    def run(
        self,
        premise_word: str,
        hypothesis_word: str,
        context: str = "",
        interventions: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Run the high-level model, optionally with interventions.

        Parameters
        ----------
        premise_word, hypothesis_word, context:
            The base input values.
        interventions:
            Optional dict mapping intermediate-variable name to a value
            that should *override* the value that would have been
            computed from inputs. Supported keys:
            ``premise_word_identity``, ``hypothesis_word_identity``,
            ``lexical_relation``.

        Returns
        -------
        dict
            A trace of every variable in the model, including the output
            label and label id. This makes it easy to inspect counterfactual
            behavior.
        """
        interventions = dict(interventions or {})

        # Intermediate values, possibly overwritten by interventions.
        pwi = interventions.pop(
            "premise_word_identity", premise_word
        )
        hwi = interventions.pop(
            "hypothesis_word_identity", hypothesis_word
        )
        rel = interventions.pop(
            "lexical_relation", lexical_relation(pwi, hwi)
        )

        if interventions:
            raise ValueError(
                f"Unknown intervention targets: {sorted(interventions)}"
            )

        rel_to_label = self._relation_to_label()
        label = rel_to_label[rel]

        return {
            "premise_word": premise_word,
            "hypothesis_word": hypothesis_word,
            "context": context,
            "premise_word_identity": pwi,
            "hypothesis_word_identity": hwi,
            "lexical_relation": rel,
            "label": label,
            "label_id": LABEL2ID[label],
        }

    # ------------------------------------------------------------------
    # Convenience batch APIs
    # ------------------------------------------------------------------

    def run_many(
        self,
        examples: Iterable[Dict[str, str]],
    ) -> List[Dict[str, object]]:
        """Batched :meth:`run` over an iterable of example dicts."""
        return [
            self.run(
                premise_word=ex["premise_word"],
                hypothesis_word=ex["hypothesis_word"],
                context=ex.get("context", ""),
            )
            for ex in examples
        ]


def run_high_level(
    premise_word: str,
    hypothesis_word: str,
    context: str = "",
    interventions: Optional[Dict[str, object]] = None,
    monotonicity: str = "upward",
) -> Dict[str, object]:
    """Functional wrapper around :class:`LexicalCausalModel`.

    Useful when you don't need a long-lived model object.
    """
    return LexicalCausalModel(monotonicity=monotonicity).run(
        premise_word=premise_word,
        hypothesis_word=hypothesis_word,
        context=context,
        interventions=interventions,
    )
