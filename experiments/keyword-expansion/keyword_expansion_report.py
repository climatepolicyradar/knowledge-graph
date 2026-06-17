"""
Generate keyword-expansion predictions for concepts and write a comparison CSV.

Compares predicted vs ground-truth alternative labels (string-match TP/FP/FN).

This is a single-pass report (no teacher/optimiser): it runs the keyword expander
once per concept per eval mode and scores the generated positive keywords against the
concept's curated alternative labels.

Usage:
    uv run python experiments/keyword-expansion/keyword_expansion_report.py --max-concepts 50
"""

import argparse
import asyncio
import csv
from pathlib import Path

from knowledge_graph.classifier.auto_keyword_expansion import (
    EvalMode,
    _load_eval_concepts,
    masked_markdown,
    normalise_keyword,
)
from knowledge_graph.classifier.keyword_expansion import KeywordExpansionClassifier

DEFAULT_OUTPUT = Path(__file__).parent / "keyword_expansion_predictions.csv"

# Concepts whose curated "alternative labels" are catalogues of named entities or
# enumerated examples rather than synonyms, so LLM keyword expansion is the wrong task
# for them and they distort the evaluation.
EXCLUDED_CONCEPT_IDS = {
    "Q1370",  # state-owned enterprise: ~916 specific company names/acronyms
}


def _string_match_breakdown(
    generated: list[str], gold: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """
    Split keywords into true positives, false positives and false negatives.

    Matching is on normalised form (case/separator-insensitive, consistent with the
    KeywordClassifier), but the original strings are returned for readability.

    :return: (tp_gold_labels, fp_generated_labels, fn_gold_labels)
    """
    gen_norm = {normalise_keyword(k) for k in generated if normalise_keyword(k)}
    gold_norm = {normalise_keyword(k) for k in gold if normalise_keyword(k)}

    tp = [k for k in gold if normalise_keyword(k) in gen_norm]
    fp = [
        k
        for k in generated
        if normalise_keyword(k) and normalise_keyword(k) not in gold_norm
    ]
    fn = [k for k in gold if normalise_keyword(k) not in gen_norm]
    return tp, fp, fn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-concepts", type=int, default=50)
    parser.add_argument("--min-alternative-labels", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument(
        "--model", type=str, default="openrouter:anthropic/claude-haiku-4.5"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[m.value for m in EvalMode],
        choices=[m.value for m in EvalMode],
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Additional concept IDs to exclude (merged with the built-in exclude list)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
    )
    args = parser.parse_args()

    modes = [EvalMode(m) for m in args.modes]
    excluded = EXCLUDED_CONCEPT_IDS | set(args.exclude)

    # Over-fetch so the exclude list doesn't shrink us below the requested count.
    concepts = asyncio.run(
        _load_eval_concepts(
            max_concepts=args.max_concepts + len(excluded),
            min_alternative_labels=args.min_alternative_labels,
            seed=args.seed,
            limit=args.limit,
        )
    )
    concepts = [c for c in concepts if str(c.wikibase_id) not in excluded][
        : args.max_concepts
    ]
    print(f"Generating predictions for {len(concepts)} concepts × {len(modes)} modes")

    # The expander's generation depends only on the prompt template, not the concept,
    # so a single instance (with the default style-guide prompt) is reused throughout.
    expander = KeywordExpansionClassifier(concept=concepts[0], model=args.model)

    fieldnames = [
        "concept_id",
        "preferred_label",
        "mode",
        "predicted_keywords",
        "actual_keywords",
        "TP_count",
        "FP_count",
        "FN_count",
        "precision",
        "recall",
        "f1",
        "TP",
        "FP",
        "FN",
    ]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, concept in enumerate(concepts, 1):
            for mode in modes:
                generated = expander._generate_keywords(
                    preferred_label=concept.preferred_label,
                    concept_description=masked_markdown(concept, mode),
                )["positive_keywords"]
                gold = concept.alternative_labels

                tp, fp, fn = _string_match_breakdown(generated, gold)
                precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0.0
                recall = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall)
                    else 0.0
                )

                writer.writerow(
                    {
                        "concept_id": str(concept.wikibase_id),
                        "preferred_label": concept.preferred_label,
                        "mode": mode.value,
                        "predicted_keywords": "; ".join(generated),
                        "actual_keywords": "; ".join(gold),
                        "TP_count": len(tp),
                        "FP_count": len(fp),
                        "FN_count": len(fn),
                        "precision": round(precision, 3),
                        "recall": round(recall, 3),
                        "f1": round(f1, 3),
                        "TP": "; ".join(tp),
                        "FP": "; ".join(fp),
                        "FN": "; ".join(fn),
                    }
                )
                f.flush()  # incremental write so nothing is lost if interrupted
            print(f"  [{i}/{len(concepts)}] {concept.preferred_label}")

    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
