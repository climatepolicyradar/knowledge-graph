"""Tests for the keyword-expansion optimiser's scoring, masking and folding."""

import pytest

from knowledge_graph.classifier.auto_keyword_expansion import (
    EvalMode,
    _clean_keyword_lists,
    _keyword_classifier,
    _make_folds,
    _matched_passage_indices,
    _template_is_formattable,
    functional_confusion,
    masked_markdown,
    normalise_keyword,
    score_keyword_sets,
)
from knowledge_graph.classifier.keyword_expansion import (
    DEFAULT_KEYWORD_EXPANSION_PROMPT,
    sanitise_keywords,
)
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Greenhouse Gas", "greenhouse gas"),
        ("  GHG  ", "ghg"),
        ("greenhouse-gas", "greenhouse gas"),
        ("greenhouse–gas", "greenhouse gas"),  # en dash
        ("greenhouse   gas", "greenhouse gas"),
    ],
)
def test_normalise_keyword(raw, expected):
    assert normalise_keyword(raw) == expected


def test_score_keyword_sets_perfect_overlap():
    cm = score_keyword_sets(["pony", "Mare"], ["mare", "PONY"])
    assert cm.f_beta_score(beta=1.0) == 1.0
    assert cm.false_positives == 0
    assert cm.false_negatives == 0


def test_score_keyword_sets_disjoint():
    cm = score_keyword_sets(["car", "van"], ["horse", "pony"])
    assert cm.f_beta_score(beta=1.0) == 0.0
    assert cm.true_positives == 0


def test_score_keyword_sets_partial():
    # generated {pony, car}; gold {pony, mare} -> tp=1, fp=1 (car), fn=1 (mare)
    cm = score_keyword_sets(["pony", "car"], ["pony", "mare"])
    assert cm.true_positives == 1
    assert cm.false_positives == 1
    assert cm.false_negatives == 1
    assert cm.precision() == 0.5
    assert cm.recall() == 0.5


def _example_concept() -> Concept:
    return Concept(
        wikibase_id=WikibaseID("Q1"),
        preferred_label="horse",
        alternative_labels=["pony", "stallion", "mare"],
        description="A large four-legged mammal.",
        definition="An animal of the species Equus caballus.",
    )


@pytest.mark.parametrize("mode", list(EvalMode))
def test_masked_markdown_never_leaks_gold_labels(mode):
    concept = _example_concept()
    rendered = masked_markdown(concept, mode)
    for label in concept.alternative_labels:
        assert label not in rendered


def test_masked_markdown_mode_differences():
    concept = _example_concept()
    labels_hidden = masked_markdown(concept, EvalMode.LABELS_HIDDEN)
    minimal = masked_markdown(concept, EvalMode.MINIMAL)

    # LABELS_HIDDEN keeps description/definition; MINIMAL drops them
    assert concept.description in labels_hidden
    assert concept.definition in labels_hidden
    assert concept.description not in minimal
    assert concept.definition not in minimal


@pytest.mark.parametrize("mode", list(EvalMode))
def test_expander_prompt_never_contains_gold_labels(mode):
    """The fully-formatted prompt sent to the expander must not leak gold labels."""
    concept = _example_concept()
    prompt = DEFAULT_KEYWORD_EXPANSION_PROMPT.format(
        PREFERRED_LABEL=concept.preferred_label,
        CONCEPT_DESCRIPTION=masked_markdown(concept, mode),
    )
    for label in concept.alternative_labels:
        assert label not in prompt


def test_sanitise_keywords_drops_non_ascii_and_emoji():
    out = sanitise_keywords(["greenhouse gas", "激", "carbon ⚖️", "消消乐", "methane"])
    # CJK-only and emoji-containing tokens are removed
    assert "greenhouse gas" in out
    assert "methane" in out
    assert all(k.isascii() for k in out)
    assert "激" not in out
    assert "消消乐" not in out


def test_sanitise_keywords_splits_runons_and_dedupes():
    out = sanitise_keywords(["pony, mare; stallion", "pony", "foal\ngelding", "Mare"])
    # Run-ons split on comma/semicolon/newline; case-insensitive dedupe
    assert out == ["pony", "mare", "stallion", "foal", "gelding"]


def test_sanitise_keywords_drops_overlong_phrases():
    long_phrase = (
        "goal four of the sustainable development goals two thousand thirty agenda"
    )
    out = sanitise_keywords(["sea level rise", long_phrase])
    assert "sea level rise" in out
    assert long_phrase not in out


def test_template_is_formattable():
    assert _template_is_formattable(DEFAULT_KEYWORD_EXPANSION_PROMPT)
    # Missing a placeholder
    assert not _template_is_formattable("only {PREFERRED_LABEL}")
    # Unescaped stray brace breaks str.format
    assert not _template_is_formattable(
        "{PREFERRED_LABEL} {CONCEPT_DESCRIPTION} {oops}"
    )


def test_functional_confusion_math():
    cm = functional_confusion(
        generated_indices={0, 1}, human_indices={1, 2}, n_passages=5
    )
    assert cm.true_positives == 1  # passage 1 matched by both
    assert cm.false_positives == 1  # passage 0 matched by generated only
    assert cm.false_negatives == 1  # passage 2 matched by human only
    assert cm.true_negatives == 2  # passages 3, 4 matched by neither
    assert cm.precision() == 0.5
    assert cm.recall() == 0.5


def test_clean_keyword_lists_drops_preferred_dupes_and_overlap():
    concept = _example_concept()  # preferred "horse"
    pos, neg = _clean_keyword_lists(
        concept,
        positives=["horse", "Pony", "pony", "mare"],
        negatives=["mare", "wild horse", ""],
    )
    assert "horse" not in pos  # preferred label dropped
    assert pos == ["Pony", "mare"]  # case-insensitive dedupe, order preserved
    assert neg == ["wild horse"]  # "mare" overlaps positives -> dropped; empty dropped


def test_keyword_classifier_matches_expected_passages():
    concept = _example_concept()
    passages = ["a pony grazed", "the cat sat", "a horse ran", "mare and foal"]
    # Build a classifier from a restricted keyword set (preferred label is kept)
    classifier = _keyword_classifier(concept, positives=["pony"], negatives=[])
    matched = _matched_passage_indices(classifier, passages, batch_size=10)
    # "pony" -> passage 0; preferred label "horse" -> passage 2; "mare" not a keyword
    assert matched == {0, 2}
    concepts = [
        Concept(wikibase_id=WikibaseID(f"Q{i}"), preferred_label=f"concept {i}")
        for i in range(1, 13)
    ]
    k = 5
    folds = _make_folds(concepts, k=k, seed=42)

    assert len(folds) == k
    # Every concept appears exactly once across all folds
    flat = [c for fold in folds for c in fold]
    assert len(flat) == len(concepts)
    ids = {str(c.wikibase_id) for c in flat}
    assert ids == {str(c.wikibase_id) for c in concepts}

    # For each held-out fold, the train split (all other folds) is disjoint from it
    for held_out in folds:
        held_out_ids = {str(c.wikibase_id) for c in held_out}
        train_ids = {str(c.wikibase_id) for f in folds if f is not held_out for c in f}
        assert held_out_ids.isdisjoint(train_ids)
