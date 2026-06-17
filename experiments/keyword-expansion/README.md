# Keyword expansion

Evaluating (and self-improving) LLM generation of concept **alternative labels**,
using the concept store as ground truth.

## Idea

Every concept in the store has human-curated `alternative_labels`. We hide them, ask an
LLM to regenerate them from the concept's other fields, and score the overlap — then a
teacher LLM rewrites the expander's prompt to improve it (k-fold cross-validated to
avoid overfitting). The reusable library code lives in
[`knowledge_graph/classifier/auto_keyword_expansion.py`](../../knowledge_graph/classifier/auto_keyword_expansion.py)
and the expander itself in
[`knowledge_graph/classifier/keyword_expansion.py`](../../knowledge_graph/classifier/keyword_expansion.py).

## This directory

- `keyword_expansion_report.py` — single-pass report: generates keywords for a sample of
  concepts (both eval modes) and writes a per-concept/per-mode CSV comparing predicted vs
  actual labels with string-match TP/FP/FN.
- `keyword_expansion_predictions.csv` — latest output (50 concepts × 2 modes).

Run:

```bash
uv run python experiments/keyword-expansion/keyword_expansion_report.py --max-concepts 50
```

Columns: `concept_id, preferred_label, mode, predicted_keywords, actual_keywords,
TP_count, FP_count, FN_count, precision, recall, f1, TP, FP, FN` (TP/FP/FN hold the
keyword strings in each bucket; matching is on normalised form so case/separator variants
aren't counted as misses).

## Key findings so far

- **Aligning the prompt to the concept-store style guide** (synonyms/surface-variants
  only; no related/broader terms or misspellings) lifted held-out precision ~2.9× and cut
  over-generation from ~37 to ~6.5 keywords/concept.
- **Exact string match drastically understates quality.** A functional metric — build a
  `KeywordClassifier` from the generated keywords and compare which passages it matches
  against the human keywords (over the `build_dataset` sampled passages) — scored the same
  keyword sets at ~P=0.40 / R=0.62 vs ~F1=0.13 for string overlap.
- **Some "concepts" are entity catalogues or example bags, not synonym sets** (e.g.
  `state-owned enterprise` = ~916 company names). These are excluded via
  `EXCLUDED_CONCEPT_IDS` in the report script — they're the wrong task for keyword
  expansion and should be entity lists or LLM/BERT classifiers instead.
- After excluding those, ~68% of remaining false negatives are near-misses (right region,
  different surface form) rather than genuine vocabulary gaps — i.e. mostly a metric
  artifact, not an LLM capability ceiling.
