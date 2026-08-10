"""
Can an LLM ensemble auto-label our classifier evaluation sets? (FUS-164)

Runs a cross-family LLM ensemble over the gold-standard evaluation passages of the
concepts listed on FUS-163, and reports precision/recall/F1/support per concept plus a
macro average — at both passage and span level — so we can decide whether eval-set
labelling can be automated, semi-automated, or not automated.

Two families at opus-level sizing, five seeds each at ``temperature=0.7``. Every member's
predictions are cached to disk *per passage*, so we can compose ensembles later.

Concepts get ``DEFAULT_SYSTEM_PROMPT`` plus whatever is in the concept store — the tuned
labelling guidelines in ``scripts/custom_concept_training/configs/`` are not applied,
because eval-set labelling happens before prompt development.

Different ensembling methods are used at the passage- and span-level:

- passage level — the ensemble is positive if a majority of members are.
- span level — the ensemble's spans are the union of all members' spans, with
  overlapping spans merged

Two senses of "agreement" are deliberately kept apart, because conflating them is what
made an earlier version of this analysis unreadable:

- ``disagreement`` is between ensemble *members* (`Disagreement` in
  ``knowledge_graph.ensemble.metrics``). Gold is not involved. It is the axis along which
  passages are routed to a human.
- precision/recall/F1 are against *gold*. They score the labels a routing policy would
  assign, and they are the only metrics reported: on a negative-skewed task a
  true-negative-crediting metric lets a policy that assigns no positive labels at all
  score well while quietly discarding real mentions.

Usage::

    # check both families resolve on OpenRouter before spending 20k calls
    uv run python -m scripts.benchmarks.eval_set_autolabelling_experiment.cli probe

    # smoke test on the smallest concept, then the full sweep
    uv run python -m scripts.benchmarks.eval_set_autolabelling_experiment.cli predict \
        --concepts Q32
    uv run python -m scripts.benchmarks.eval_set_autolabelling_experiment.cli analyse \
        --concepts Q32

    # break the detailed views down by a different ensemble than the default
    uv run python -m scripts.benchmarks.eval_set_autolabelling_experiment.cli analyse \
        --headline mixed_n5

``predict`` skips members it has already cached, so it is safe to interrupt and rerun.
"""

import asyncio
from pathlib import Path
from typing import Annotated, Any, cast

import pandas as pd
import typer

from knowledge_graph.identifiers import WikibaseID
from scripts.benchmarks.eval_set_autolabelling_experiment.analysis import (
    agreement_f1_curves,
    automation_by_ensemble,
    disagreement_rows,
    for_console,
    headline_table,
    negative_side_table,
    plot_agreement_vs_f1,
    plot_automation_by_size,
    policy_table,
    policy_table_by_ensemble,
    print_table,
    recommendation,
    score_ensemble,
    smallest_automating_size,
    summarise_vote_splits,
    vote_splits_for_display,
)
from scripts.benchmarks.eval_set_autolabelling_experiment.config import (
    BY_ENSEMBLE_CONSOLE_COLUMNS,
    COMPLETENESS_COLUMNS,
    DEFAULT_CONCEPTS,
    DEFAULT_OUTPUT_DIR,
    FAMILIES,
    HEADLINE_ENSEMBLE,
    HEADLINE_SPAN_THRESHOLDS,
    NEGATIVE_SIDE_CONSOLE_COLUMNS,
    PASSAGE_LEVEL,
    POLICY_CONSOLE_COLUMNS,
    SEEDS,
    VOTE_SPLIT_CONSOLE_COLUMNS,
    console,
)
from scripts.benchmarks.eval_set_autolabelling_experiment.ensembles import (
    EnsembleMember,
    all_members,
    build_member_classifier,
    compose_possible_ensembles,
    concept_output_dir,
    ensemble_costs_per_passage,
    fetch_concept_and_gold,
    load_model_pricing,
    load_usable_concepts,
    run_member_with_caching,
    write_passages,
)

app = typer.Typer()


def parse_concepts(value: str) -> list[str]:
    """Parse a comma-separated list of Wikibase IDs."""
    ids = [v.strip() for v in value.split(",") if v.strip()]
    for wikibase_id in ids:
        WikibaseID(wikibase_id)
    return ids


@app.command()
def probe(
    wikibase_id: Annotated[
        str,
        typer.Option("--wikibase-id", help="Concept to probe with."),
    ] = "Q32",
):
    """
    Check that both model families resolve, on one passage each.

    Worth doing first: a model name that 404s would otherwise only surface part-way
    through a 20,000-call run.
    """

    fetched = asyncio.run(fetch_concept_and_gold(WikibaseID(wikibase_id)))
    if fetched is None:
        raise typer.Exit(1)
    concept, gold = fetched

    failures = []
    for family in FAMILIES:
        member: EnsembleMember = (family, SEEDS[0])
        try:
            classifier = build_member_classifier(concept, member)
            spans = classifier.predict(gold[0].text)
            console.log(f"✅ {FAMILIES[family]} responded ({len(spans)} spans)")
        except Exception as e:
            console.log(f"❌ {FAMILIES[family]} failed: {e}")
            failures.append(family)

    if failures:
        console.log(
            f"❌ Unavailable families: {failures}. Fix before running `predict`."
        )
        raise typer.Exit(1)

    console.log("✅ All families available")


@app.command()
def predict(
    concepts: Annotated[
        str,
        typer.Option(
            "--concepts",
            help="Comma-separated Wikibase IDs. Defaults to the FUS-163 concept list.",
        ),
    ] = ",".join(DEFAULT_CONCEPTS),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Where to cache per-member predictions."),
    ] = DEFAULT_OUTPUT_DIR,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Async fan-out width per member."),
    ] = 16,
    max_passages: Annotated[
        int | None,
        typer.Option(
            "--max-passages",
            help="Cap passages per concept. For smoke tests only — the experiment uses "
            "every gold passage.",
        ),
    ] = None,
):
    """
    Run every ensemble member over every concept's gold passages, caching as it goes.

    Safe to interrupt and rerun: each member caches per passage, so a rerun only requests
    the passages still missing.
    """

    wikibase_ids = parse_concepts(concepts)
    members: list[EnsembleMember] = all_members()

    console.log(
        f"Running {len(members)} members over {len(wikibase_ids)} concepts "
        f"({', '.join(FAMILIES.values())})"
    )

    for wikibase_id in wikibase_ids:
        console.rule(f"{wikibase_id}")
        concept_dir = concept_output_dir(output_dir, wikibase_id)
        gold_path = concept_dir / "gold.jsonl"

        fetched = asyncio.run(fetch_concept_and_gold(WikibaseID(wikibase_id)))
        if fetched is None:
            continue
        concept, gold = fetched

        if max_passages is not None:
            gold = gold[:max_passages]
            console.log(f"✂️  Capped to {len(gold)} passages")

        write_passages(gold_path, gold)

        for i, member in enumerate(members, start=1):
            family, seed = member
            run_member_with_caching(
                concept=concept,
                member=member,
                gold=gold,
                concept_dir=concept_dir,
                batch_size=batch_size,
                position=f"Member {i}/{len(members)}: {family} seed={seed}",
            )

    console.log(f"✅ Predictions cached under {output_dir}")


@app.command()
def analyse(
    concepts: Annotated[
        str,
        typer.Option(
            "--concepts",
            help="Comma-separated Wikibase IDs. Defaults to the FUS-163 concept list.",
        ),
    ] = ",".join(DEFAULT_CONCEPTS),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory holding the cached predictions."),
    ] = DEFAULT_OUTPUT_DIR,
    max_members: Annotated[
        int | None,
        typer.Option(
            "--max-members",
            help="Only consider ensembles of at most this many classifiers",
        ),
    ] = 5,
    headline_ensemble: Annotated[
        str,
        typer.Option(
            "--headline",
            help="Which ensemble the detailed per-concept, negative-side and vote-split "
            "views describe, by name (e.g. 'mixed_n5'). Every ensemble is scored and "
            "ranked regardless — this only picks the one the detailed views break down. "
            "An ensemble that the size cap excludes, or that has no cached predictions, "
            "is an error rather than grounds for substituting another.",
        ),
    ] = HEADLINE_ENSEMBLE,
):
    """Score every ensemble from the cached predictions."""

    wikibase_ids = parse_concepts(concepts)
    ensembles = compose_possible_ensembles(max_members)
    if not ensembles:
        console.log(f"❌ No ensembles with at most {max_members} members.")
        raise typer.Exit(1)

    within_cap = [c.name for c in ensembles]
    if headline_ensemble not in within_cap:
        composable = {c.name: c.size for c in compose_possible_ensembles()}
        if headline_ensemble in composable:
            console.log(
                f"❌ `{headline_ensemble}` needs {composable[headline_ensemble]} "
                f"classifiers but --max-members is {max_members}. Raise the cap, or pass "
                f"--headline naming one of: {', '.join(within_cap)}"
            )
        else:
            console.log(
                f"❌ `{headline_ensemble}` is not an ensemble this experiment composes. "
                f"Choose one of: {', '.join(composable)}"
            )
        raise typer.Exit(1)

    console.log(
        f"Scoring {len(ensembles)} ensembles over {len(wikibase_ids)} concepts"
        + (f", capped at {max_members} members" if max_members is not None else "")
    )

    # the only thing here that can touch the network, and only on the first run: after that
    # it is read from the cache alongside the predictions, so `analyse` stays offline
    prices_by_family = load_model_pricing(output_dir)
    costs_per_passage = (
        ensemble_costs_per_passage(ensembles, prices_by_family)
        if prices_by_family
        else None
    )
    if costs_per_passage is None:
        console.log("⚠️ No model pricing available — cost columns will be omitted")

    per_concept_path = output_dir / "per_concept_metrics.csv"

    all_rows: list[dict[str, Any]] = []
    all_disagreement: list[dict[str, Any]] = []

    usable, concept_completeness = load_usable_concepts(output_dir, wikibase_ids)
    if not usable:
        console.log(
            f"❌ No concept has all {len(all_members())} members cached, so there is "
            "no common set to score on. Finish `predict` and rerun."
        )
        raise typer.Exit(1)

    console.log(
        f"Scoring {len(usable)} of {len(wikibase_ids)} concepts: {', '.join(usable)}"
    )

    for wikibase_id, (gold, predictions_by_member) in usable.items():
        # every ensemble draws its members from the same (family, seed) grid every
        # concept in `usable` has cached, so there is nothing here to skip
        for ensemble in ensembles:
            all_rows.extend(
                score_ensemble(wikibase_id, ensemble, gold, predictions_by_member)
            )

            # every ensemble, not just the headline: the sizing question needs a
            # verdict per ensemble size, and re-aggregating cached spans costs no LLM calls
            all_disagreement.extend(
                disagreement_rows(wikibase_id, ensemble, gold, predictions_by_member)
            )

        # write after each concept so an interrupted analysis keeps its progress
        pd.DataFrame(all_rows).to_csv(per_concept_path, index=False)

    if not all_rows:
        console.log("❌ No cached predictions found. Run `predict` first.")
        raise typer.Exit(1)

    per_concept = pd.DataFrame(all_rows)

    # the headline has to have been scored, or the detailed views below would describe an
    # ensemble with no numbers. Reported as an error naming what *was* scored, so the fix
    # is a deliberate `--headline` rather than a silent substitution
    scored_names = set(per_concept["ensemble"])
    if headline_ensemble not in scored_names:
        console.log(
            f"❌ `{headline_ensemble}` was not scored on any concept — its members are "
            "missing from the cache. Run `predict` to fill them, or pass --headline naming "
            f"one of: {', '.join(sorted(scored_names))}"
        )
        raise typer.Exit(1)
    headline = next(c for c in ensembles if c.name == headline_ensemble)

    disagreement_df = pd.DataFrame(all_disagreement)
    by_ensemble = pd.DataFrame()
    curves = pd.DataFrame()
    if disagreement_df.empty:
        console.log("⚠️ No per-passage results — skipping the automation analysis")
        policies = pd.DataFrame()
        vote_splits = pd.DataFrame()
        n_decision_concepts = 0
    else:
        all_policies = policy_table_by_ensemble(disagreement_df)
        by_ensemble = automation_by_ensemble(all_policies, costs_per_passage)
        by_ensemble.to_csv(output_dir / "automation_by_ensemble.csv", index=False)
        plot_automation_by_size(by_ensemble, output_dir)

        curves = agreement_f1_curves(all_policies)
        curves.to_csv(output_dir / "agreement_vs_f1.csv", index=False)
        plot_agreement_vs_f1(curves, output_dir)

        headline_passages = cast(
            pd.DataFrame, disagreement_df[disagreement_df["ensemble"] == headline.name]
        )
        n_decision_concepts = int(cast(int, headline_passages["concept"].nunique()))
        vote_splits = summarise_vote_splits(headline_passages, headline.size)
        vote_splits.to_csv(output_dir / "vote_splits.csv", index=False)
        policies = policy_table(headline_passages)
        policies.to_csv(output_dir / "policy_table.csv", index=False)

    # reindexed rather than left to whatever keys the rows carry, so that a run excluding
    # nothing still has `excluded` and `exclusion_reason` columns — visibly empty
    completeness = pd.DataFrame(concept_completeness).reindex(
        columns=COMPLETENESS_COLUMNS
    )
    completeness.to_csv(output_dir / "data_completeness.csv", index=False)

    # the headline's own per-passage cost, which every per-concept table below is priced on
    headline_cost_per_passage = (
        costs_per_passage.get(headline.name) if costs_per_passage else None
    )

    if not by_ensemble.empty:
        print_table(
            for_console(by_ensemble, BY_ENSEMBLE_CONSOLE_COLUMNS),
            "Smallest ensemble that can automate",
        )
        console.print(smallest_automating_size(by_ensemble))

    print_table(
        headline_table(
            per_concept, PASSAGE_LEVEL, headline.name, headline_cost_per_passage
        ),
        f"Passage level ({headline.name})",
    )
    for threshold in HEADLINE_SPAN_THRESHOLDS:
        print_table(
            headline_table(
                per_concept,
                f"span@{threshold}",
                headline.name,
                headline_cost_per_passage,
            ),
            f"Span level (Jaccard ≥ {threshold}, {headline.name})",
        )

    negative_side = negative_side_table(per_concept, PASSAGE_LEVEL, headline.name)
    if not negative_side.empty:
        negative_side.to_csv(output_dir / "negative_side.csv", index=False)
        print_table(
            for_console(negative_side, NEGATIVE_SIDE_CONSOLE_COLUMNS),
            f"How much to trust a negative ({headline.name})",
        )

    if not policies.empty:
        print_table(
            for_console(policies, POLICY_CONSOLE_COLUMNS),
            f"Automation policies ({headline.name})",
        )
        print_table(
            for_console(
                vote_splits_for_display(vote_splits), VOTE_SPLIT_CONSOLE_COLUMNS
            ),
            "What each vote split buys",
        )
        console.print(recommendation(policies, n_decision_concepts, headline.name))

    console.log(f"💾 Results written to {output_dir}")


if __name__ == "__main__":
    app()
