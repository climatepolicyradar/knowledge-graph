r"""
LLM ensemble strategy bake-off for Q32 (Phase 1 of confidence-filtered HNM).

Phase 2 of the Q32 classifier work needs an LLM-disagreement signal to filter
the 10k-negative pool before hard-negative mining. Random-seed-only LLM
ensembling produces near-zero disagreement (pydantic-ai uses ``temperature=0``),
so this script tries four ensemble construction strategies on the 107-passage
Q32 Argilla gold set and reports which produces the most useful disagreement
signal per LLM call.

Strategies:

- ``seed_only`` (N=5)         — control. ``temperature=0``, varying ``random_seed``.
- ``temperature`` (N=5)       — fixed seed, varying ``temperature`` in
                                {0.0, 0.3, 0.5, 0.7, 0.9}. Tests whether higher-
                                temperature *level* produces disagreement.
- ``seed_at_temp_0.7`` (N=5)  — fixed ``temperature=0.7``, varying ``random_seed``.
                                Isolates whether seed variation produces
                                disagreement once we're off T=0 (separable from
                                temperature-level effects).
- ``prompt_rewording`` (N=4)  — fixed seed/temp, varying ``system_prompt_template``
                                (4 inline alternative phrasings).
- ``cross_model`` (N≤3)       — fixed seed/temp/prompt, varying ``model_name``
                                across vendors. Variants that 404 are dropped.

Headline metric: F1 lift at 80% retention (i.e. "drop the 20% most-disagreed-
on predictions; how much does F1 jump?"), normalised by total LLM calls.

Example::

    uv run python -m scripts.benchmarks.llm_ensemble_eval \
        --target-concept Q32 \
        --strategies seed_only,temperature,prompt_rewording,cross_model
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from knowledge_graph.classifier.large_language_model import (
    DEFAULT_SYSTEM_PROMPT,
    LLMClassifier,
    LLMClassifierPrompt,
)
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.labelling import label_passages_with_classifier
from scripts.active_learning.plot_ensemble_metrics import (
    calculate_cumulative_f1_curve,
    create_plots,
    create_predictions_dataframe,
)
from scripts.custom_concept_training.q32_train_llm import (
    CONCEPT_DEFINITION,
    get_concept_description,
    get_labelling_guidelines,
)
from scripts.custom_concept_training.q32_train_llm import (
    MODEL_NAME as DEFAULT_MODEL_NAME,
)
from scripts.get_concept import get_concept_async

console = Console()
app = typer.Typer()

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "llm_ensemble_eval_results"
RETENTION_RATE_HEADLINE = 0.80

SEED_VALUES = [1, 2, 3, 4, 5]
TEMPERATURE_VALUES = [0.0, 0.3, 0.5, 0.7, 0.9]
SEED_AT_TEMP_TEMPERATURE = 0.7
CROSS_MODEL_VARIANTS = [
    DEFAULT_MODEL_NAME,
    "openrouter:anthropic/claude-sonnet-4.5",
    "openrouter:openai/gpt-5",
]

# Four inline prompt rewordings for the ``prompt_rewording`` strategy. Each
# must contain ``{concept_description}`` (validated by LLMClassifierPrompt).
# They keep the same scope as DEFAULT_SYSTEM_PROMPT — XML <concept> tagging on
# input passages — but vary persona, ordering, and reasoning emphasis. The
# concept-specific labelling guidelines (Q32's normative criteria) are
# appended identically by LLMClassifierPrompt.format() in all variants.
PROMPT_VARIANT_EXPERT_ANNOTATOR = """
You are an expert annotator labelling policy passages for a concept-annotation dataset.

Review the concept you are annotating for:

<concept_description>
{concept_description}
</concept_description>

How to annotate:

1. Read the passage end-to-end, considering both direct mentions of the concept and indirect, paraphrased, or acronym-based references.
2. Wrap each occurrence with <concept>...</concept> tags.
3. If a passage has multiple occurrences, tag each one separately.
4. If the entire passage is about the concept without a specific phrase to highlight, wrap the whole passage in one <concept> tag.
5. Reproduce the input text exactly — character for character — adding only the <concept> tags.
6. If no occurrence matches the definition, return the passage unchanged.
7. Before finalising, re-read your annotations and confirm each tagged span has enough context to justify why it matches the concept.
"""

PROMPT_VARIANT_POLICY_ANALYST = """
You are a policy analyst working with documents from climate, nature and development contexts.
Your task is to identify passages or phrases that express a particular concept and mark them with XML tags.

The concept of interest:

<concept_description>
{concept_description}
</concept_description>

For each passage:

- Carefully consider how the concept might appear: directly, indirectly, through related terminology, acronyms, regional or sectoral jargon.
- Surround every match with <concept>...</concept> tags.
- If multiple distinct mentions appear, tag each one separately.
- If the passage as a whole expresses the concept but no single phrase does, wrap the whole passage.
- If no mention satisfies the definition, return the passage unchanged.
- Reproduce the input passage exactly, only adding tags. Preserve any typos or original formatting.
- Verify that every tagged span on its own would justify the match.
"""

PROMPT_VARIANT_RESEARCHER_EXAMPLES = """
You are a researcher coding text examples for a concept classifier. You must mark up mentions of a single concept using XML tags.

Concept under study:

<concept_description>
{concept_description}
</concept_description>

Worked process:

Step 1 — Read the entire passage carefully. Note any candidates for the concept, including indirect descriptions, acronyms, or domain-specific wording (climate, nature, development).
Step 2 — For each candidate, decide whether it satisfies the concept definition. If yes, wrap it in <concept>...</concept>.
Step 3 — If several non-overlapping mentions exist, tag each independently.
Step 4 — If the entire passage describes the concept without a specific phrase, wrap the whole passage in a single <concept> tag.
Step 5 — If no mention satisfies the definition, output the passage exactly as given.
Step 6 — Output rule: reproduce the input verbatim, only adding <concept> tags. Do not edit or correct the text otherwise.
Step 7 — Sanity-check: each tagged span should be self-contained enough to defend the match.
"""

PROMPT_VARIANT_CRITERIA_FIRST = """
Your job: mark up mentions of one specific concept in policy text using <concept>...</concept> XML tags. Follow the criteria precisely.

Criteria for a valid <concept> tag:
- The tagged span matches the concept definition below.
- Indirect references and paraphrases count if they clearly satisfy the definition.
- Acronyms or domain-specific phrases (climate, nature, development) count where contextually justified.
- Each tagged span must contain enough text to defend the match on its own.
- Multiple non-overlapping mentions in one passage must be tagged separately.
- A passage that is wholly about the concept without a specific phrase is tagged with one outer <concept> wrap.
- A passage with no qualifying mention is returned exactly as given, with no tags.

Output rules:
- Reproduce the input text exactly, character by character. Preserve typos and formatting.
- Add only <concept> tags. Do not edit, summarise, or paraphrase.

Concept definition:

<concept_description>
{concept_description}
</concept_description>
"""

PROMPT_VARIANTS: list[str] = [
    PROMPT_VARIANT_EXPERT_ANNOTATOR,
    PROMPT_VARIANT_POLICY_ANALYST,
    PROMPT_VARIANT_RESEARCHER_EXAMPLES,
    PROMPT_VARIANT_CRITERIA_FIRST,
]


@dataclass(frozen=True)
class Strategy:
    """
    One ensemble construction strategy and its per-variant kwarg overrides.

    Each entry in ``classifier_kwargs_per_variant`` is shallow-merged onto the
    base classifier kwargs (concept, model, prompt, seed, temperature) to
    instantiate one variant.
    """

    name: str
    classifier_kwargs_per_variant: list[dict[str, Any]] = field(default_factory=list)


def build_strategies(
    base_prompt_template: LLMClassifierPrompt,
) -> list[Strategy]:
    """Build the four strategies. Prompt variants inherit Q32 labelling guidelines."""
    prompt_variants = [
        LLMClassifierPrompt(
            system_prompt_template=p,
            labelling_guidelines=base_prompt_template.labelling_guidelines,
        )
        for p in PROMPT_VARIANTS
    ]
    return [
        Strategy(
            name="seed_only",
            classifier_kwargs_per_variant=[{"random_seed": s} for s in SEED_VALUES],
        ),
        Strategy(
            name="temperature",
            classifier_kwargs_per_variant=[
                {"temperature": t} for t in TEMPERATURE_VALUES
            ],
        ),
        Strategy(
            name="seed_at_temp_0.7",
            classifier_kwargs_per_variant=[
                {"random_seed": s, "temperature": SEED_AT_TEMP_TEMPERATURE}
                for s in SEED_VALUES
            ],
        ),
        Strategy(
            name="prompt_rewording",
            classifier_kwargs_per_variant=[
                {"system_prompt_template": p} for p in prompt_variants
            ],
        ),
        Strategy(
            name="cross_model",
            classifier_kwargs_per_variant=[
                {"model_name": m} for m in CROSS_MODEL_VARIANTS
            ],
        ),
    ]


async def fetch_q32_concept(wikibase_id: WikibaseID) -> Concept:
    """Fetch the concept plus its Argilla labelled passages, with Q32 overrides."""
    concept = await get_concept_async(
        wikibase_id=wikibase_id,
        include_recursive_has_subconcept=True,
        include_labels_from_subconcepts=True,
    )
    if not concept.labelled_passages:
        console.log(f"❌ {wikibase_id} has no labelled passages for evaluation")
        raise typer.Exit(1)

    if str(wikibase_id) == "Q32":
        concept.definition = CONCEPT_DEFINITION
        concept.description = get_concept_description()

    return concept


def run_variant(
    base_kwargs: dict[str, Any],
    variant_kwargs: dict[str, Any],
    passages: list[LabelledPassage],
    batch_size: int,
) -> list[LabelledPassage] | None:
    """Run one variant. Returns None if instantiation or labelling fails entirely."""
    merged_kwargs = {**base_kwargs, **variant_kwargs}
    try:
        clf = LLMClassifier(**merged_kwargs)
    except Exception as e:
        console.log(
            f"  ⚠️ Variant skipped ({variant_kwargs}): instantiation failed: {e}"
        )
        return None

    try:
        preds = label_passages_with_classifier(
            classifier=clf,
            labelled_passages=passages,
            batch_size=batch_size,
            show_progress=True,
        )
    except Exception as e:
        console.log(f"  ⚠️ Variant skipped ({variant_kwargs}): labelling failed: {e}")
        return None

    positive_count = sum(1 for p in preds if p.spans)
    console.log(
        f"  ✅ Variant {variant_kwargs}: {positive_count}/{len(preds)} positive"
    )
    return preds


def f1_at_retention(
    predictions_df: pd.DataFrame, retention_target: float
) -> tuple[float, float]:
    """
    Return (baseline_f1, f1_at_target_retention) using the Disagreement curve.

    Baseline F1 is F1 on all passages (retention=1.0). F1 at the target
    retention is the F1 score from the point on the cumulative curve whose
    retention rate is closest to ``retention_target``.
    """
    _, retention_rates, f1_scores = calculate_cumulative_f1_curve(
        predictions_df, "disagreement"
    )
    if not f1_scores:
        return float("nan"), float("nan")

    baseline_f1 = f1_scores[-1]
    diffs = [abs(r - retention_target) for r in retention_rates]
    idx = int(np.argmin(diffs))
    return baseline_f1, f1_scores[idx]


def summarise_strategy(
    strategy: Strategy,
    n_successful_variants: int,
    predictions_df: pd.DataFrame,
    n_passages: int,
) -> dict[str, Any]:
    """Per-strategy summary row for the comparison CSV / table."""
    total_calls = n_successful_variants * n_passages
    mean_disagreement = (
        float(predictions_df["disagreement"].mean())
        if len(predictions_df)
        else float("nan")
    )
    baseline_f1, f1_at_80 = f1_at_retention(predictions_df, RETENTION_RATE_HEADLINE)
    f1_lift = f1_at_80 - baseline_f1
    if f1_lift > 0:
        calls_per_f1_point = total_calls / (f1_lift * 100)
    else:
        calls_per_f1_point = float("nan")
    return {
        "strategy": strategy.name,
        "n_variants": n_successful_variants,
        "n_calls": total_calls,
        "mean_disagreement": mean_disagreement,
        "baseline_f1": baseline_f1,
        "f1_at_80pct_retention": f1_at_80,
        "f1_lift_at_80pct": f1_lift,
        "calls_per_f1_point_lift": calls_per_f1_point,
    }


def print_comparison_table(summary_rows: list[dict[str, Any]]) -> None:
    """Print the cross-strategy comparison table."""
    table = Table(
        title=f"LLM ensemble strategy bake-off — headline @ {int(RETENTION_RATE_HEADLINE * 100)}% retention",
        show_lines=True,
    )
    table.add_column("Strategy")
    table.add_column("N variants", justify="right")
    table.add_column("Calls", justify="right")
    table.add_column("Mean disagreement", justify="right")
    table.add_column("Baseline F1", justify="right")
    table.add_column(f"F1 @ {int(RETENTION_RATE_HEADLINE * 100)}%", justify="right")
    table.add_column("F1 lift", justify="right")
    table.add_column("Calls / F1-pt lift", justify="right")

    for row in summary_rows:
        f1_lift = row["f1_lift_at_80pct"]
        calls_per_f1 = row["calls_per_f1_point_lift"]
        table.add_row(
            row["strategy"],
            str(row["n_variants"]),
            str(row["n_calls"]),
            f"{row['mean_disagreement']:.4f}",
            f"{row['baseline_f1']:.4f}",
            f"{row['f1_at_80pct_retention']:.4f}",
            f"{f1_lift:+.4f}",
            f"{calls_per_f1:.1f}" if not np.isnan(calls_per_f1) else "—",
        )
    console.print(table)


def run_strategy(
    strategy: Strategy,
    base_kwargs: dict[str, Any],
    passages: list[LabelledPassage],
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any] | None:
    """Run one strategy end-to-end. Returns a summary row, or None if no variants succeeded."""
    console.rule(f"Strategy: {strategy.name}")

    predictions_per_variant: list[list[LabelledPassage]] = []
    for i, variant_kwargs in enumerate(strategy.classifier_kwargs_per_variant):
        console.log(
            f"Variant {i + 1}/{len(strategy.classifier_kwargs_per_variant)}: "
            f"{variant_kwargs}"
        )
        preds = run_variant(base_kwargs, variant_kwargs, passages, batch_size)
        if preds is not None:
            predictions_per_variant.append(preds)

    if len(predictions_per_variant) < 2:
        console.log(
            f"⚠️ Strategy {strategy.name}: only {len(predictions_per_variant)} "
            "variants succeeded — need ≥2 for disagreement. Skipping strategy."
        )
        return None

    strategy_dir = output_dir / strategy.name
    strategy_dir.mkdir(parents=True, exist_ok=True)

    df = create_predictions_dataframe(predictions_per_variant, passages)
    df["strategy"] = strategy.name
    df.to_csv(strategy_dir / "predictions.csv", index=False)

    # Build a representative classifier for the plot title — use the first
    # successful variant's kwargs.
    first_variant_kwargs = strategy.classifier_kwargs_per_variant[0]
    title_classifier = LLMClassifier(**{**base_kwargs, **first_variant_kwargs})
    create_plots(df, strategy_dir, title_classifier)

    summary = summarise_strategy(
        strategy=strategy,
        n_successful_variants=len(predictions_per_variant),
        predictions_df=df,
        n_passages=len(passages),
    )
    console.log(
        f"📊 {strategy.name}: mean_disagreement={summary['mean_disagreement']:.4f}, "
        f"baseline F1={summary['baseline_f1']:.4f}, "
        f"F1@80%={summary['f1_at_80pct_retention']:.4f}, "
        f"lift={summary['f1_lift_at_80pct']:+.4f}"
    )
    return summary


def parse_strategy_names(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@app.command()
def main(
    target_concept: Annotated[
        str,
        typer.Option(
            "--target-concept",
            help="Wikibase ID of the concept to bake off on. Currently only Q32 has "
            "the custom definition + labelling guidelines wired up.",
        ),
    ] = "Q32",
    strategies: Annotated[
        str,
        typer.Option(
            "--strategies",
            help="Comma-separated subset of strategies to run "
            "(seed_only, temperature, seed_at_temp_0.7, prompt_rewording, cross_model).",
        ),
    ] = "seed_only,temperature,seed_at_temp_0.7,prompt_rewording,cross_model",
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Where to write per-strategy outputs and the comparison CSV.",
        ),
    ] = DEFAULT_OUTPUT_DIR,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Batch size for label_passages_with_classifier (async fan-out).",
        ),
    ] = 16,
):
    """Bake off LLM ensemble strategies by disagreement signal per call."""
    console.log("LLM ensemble strategy bake-off (Phase 1 of confidence-filtered HNM)")

    selected = parse_strategy_names(strategies)
    valid_names = {
        "seed_only",
        "temperature",
        "seed_at_temp_0.7",
        "prompt_rewording",
        "cross_model",
    }
    if unknown := [s for s in selected if s not in valid_names]:
        raise typer.BadParameter(
            f"Unknown strategies: {unknown}. Valid: {sorted(valid_names)}"
        )

    target_id = WikibaseID(target_concept)
    concept = asyncio.run(fetch_q32_concept(target_id))
    console.log(
        f"Loaded {len(concept.labelled_passages)} labelled passages for {target_id}"
    )

    # Q32-specific labelling guidelines come from the Wikibase via
    # get_labelling_guidelines() — same path as q32_train_llm.train().
    labelling_guidelines = get_labelling_guidelines()
    base_prompt_template = LLMClassifierPrompt(
        system_prompt_template=DEFAULT_SYSTEM_PROMPT,
        labelling_guidelines=labelling_guidelines,
    )
    base_kwargs: dict[str, Any] = {
        "concept": concept,
        "model_name": DEFAULT_MODEL_NAME,
        "system_prompt_template": base_prompt_template,
        "random_seed": 42,
        "temperature": 0.0,
    }

    all_strategies = build_strategies(base_prompt_template)
    strategies_to_run = [s for s in all_strategies if s.name in selected]
    console.log(f"Running strategies: {[s.name for s in strategies_to_run]}")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for strategy in strategies_to_run:
        summary = run_strategy(
            strategy=strategy,
            base_kwargs=base_kwargs,
            passages=concept.labelled_passages,
            output_dir=output_dir,
            batch_size=batch_size,
        )
        if summary is not None:
            summary_rows.append(summary)

    if not summary_rows:
        console.log("❌ No strategies produced usable results.")
        raise typer.Exit(1)

    comparison_df = pd.DataFrame(summary_rows)
    comparison_path = output_dir / "strategy_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    console.log(f"💾 Strategy comparison saved to {comparison_path}")

    print_comparison_table(summary_rows)


if __name__ == "__main__":
    app()
