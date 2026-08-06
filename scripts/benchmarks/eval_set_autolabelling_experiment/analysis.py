"""Turning cached predictions into scores, policies, tables and plots."""

import math
from pathlib import Path
from typing import Any, Callable, Iterable, cast

import matplotlib.pyplot as plt
import pandas as pd
from rich import box
from rich.table import Table

from knowledge_graph.ensemble.aggregation import MajorityVoteAggregator, UnionAggregator
from knowledge_graph.ensemble.metrics import Disagreement, MajorityVote
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.metrics import (
    ConfusionMatrix,
    count_passage_level_metrics,
    count_span_level_metrics,
)
from knowledge_graph.span import Span
from scripts.benchmarks.eval_set_autolabelling_experiment.config import (
    AGREEMENT_CURVE_COLUMNS,
    AUTOMATE_COVERAGE,
    AUTOMATE_F1,
    AUTOMATE_VERDICT,
    BELOW_BAR_VERDICT,
    COST_COLUMN,
    HEADLINE_ENSEMBLE,
    NOT_APPLICABLE,
    PASSAGE_LEVEL,
    SEMI_AUTOMATE_F1,
    SEMI_AUTOMATE_VERDICT,
    SPAN_AGREEMENT_THRESHOLDS,
    TOTAL_COST_COLUMN,
    UNANIMOUS_NEGATIVE_POLICY,
    UNANIMOUS_POSITIVE_POLICY,
    console,
)
from scripts.benchmarks.eval_set_autolabelling_experiment.ensembles import (
    EnsembleMember,
    NamedEnsemble,
    family_of,
)


def align_passages(
    gold: list[LabelledPassage],
    predictions_by_member: dict[EnsembleMember, dict[str, LabelledPassage]],
    members: Iterable[EnsembleMember],
) -> tuple[list[LabelledPassage], list[list[list[Span]]]]:
    """
    Line up gold passages with each member's spans, dropping any passage a member lacks.

    The metrics functions zip gold and predicted lists positionally, so alignment has to
    happen by passage id first.

    "A member" here means a member of *this ensemble*, which on its own would let two
    ensembles of the same concept be scored on different passages whenever a call
    failed. `analyse` closes that gap upstream rather than here: it hands in a gold list
    already restricted to the passages every member of the experiment holds, so this loop
    drops nothing and every ensemble sees the same passages. The drop stays in as a
    guard, not as the mechanism.

    :return: the gold passages retained, and per passage the spans from each member
    """

    members = list(members)
    aligned_gold: list[LabelledPassage] = []
    spans_per_passage: list[list[list[Span]]] = []

    for passage in gold:
        if any(passage.id not in predictions_by_member[m] for m in members):
            continue
        aligned_gold.append(passage)
        spans_per_passage.append(
            [predictions_by_member[m][passage.id].spans for m in members]
        )

    return aligned_gold, spans_per_passage


def build_predicted_passages(
    gold: list[LabelledPassage],
    spans_per_passage: list[list[list[Span]]],
) -> tuple[list[LabelledPassage], list[LabelledPassage]]:
    """
    Reduce each passage's per-member spans to the ensemble's prediction.

    Returns two views, because the two levels use different decision rules:

    - passage level: the union spans, but only if a majority of members found anything
    - span level: the union of every member's spans, however few members found them

    :return: (passage-level predictions, span-level predictions)
    """

    passage_aggregator = MajorityVoteAggregator()
    span_aggregator = UnionAggregator()

    passage_level: list[LabelledPassage] = []
    span_level: list[LabelledPassage] = []

    for passage, spans_per_member in zip(gold, spans_per_passage):
        passage_level.append(
            passage.model_copy(
                update={"spans": passage_aggregator(spans_per_member)}, deep=True
            )
        )
        span_level.append(
            passage.model_copy(
                update={"spans": span_aggregator(spans_per_member)}, deep=True
            )
        )

    return passage_level, span_level


def metrics_row(
    confusion_matrix: ConfusionMatrix,
    score_negative_class: bool = True,
    **identifiers: Any,
) -> dict[str, Any]:
    """
    Turn a confusion matrix into a results row.

    :param score_negative_class: whether `npv` and `specificity` mean anything for this
        matrix. False for the span level, where `count_span_level_metrics` counts false
        negatives per gold *span* but true negatives per empty *passage* — so both rates
        would divide one unit by another. P/R/F1 are unaffected: they never touch `TN`.
    """
    return {
        **identifiers,
        "precision": confusion_matrix.precision(),
        "recall": confusion_matrix.recall(),
        "f1": confusion_matrix.f1_score(),
        # the negative-class counterparts: how much a "no mention here" label can be
        # trusted, and how many of the genuine negatives it catches
        "npv": confusion_matrix.negative_predictive_value()
        if score_negative_class
        else math.nan,
        "specificity": confusion_matrix.specificity()
        if score_negative_class
        else math.nan,
        "support": confusion_matrix.support(),
        "true_positives": confusion_matrix.true_positives,
        "false_positives": confusion_matrix.false_positives,
        "true_negatives": confusion_matrix.true_negatives,
        "false_negatives": confusion_matrix.false_negatives,
    }


def score_ensemble(
    wikibase_id: str,
    ensemble: NamedEnsemble,
    gold: list[LabelledPassage],
    predictions_by_member: dict[EnsembleMember, dict[str, LabelledPassage]],
) -> list[dict[str, Any]]:
    """Score one ensemble on one concept, at passage and span level."""

    aligned_gold, spans_per_passage = align_passages(
        gold, predictions_by_member, ensemble.members
    )
    if not aligned_gold:
        return []

    passage_level, span_level = build_predicted_passages(
        aligned_gold, spans_per_passage
    )

    identifiers = {
        "concept": wikibase_id,
        "ensemble": ensemble.name,
        "n_members": ensemble.size,
    }

    rows = [
        metrics_row(
            count_passage_level_metrics(aligned_gold, passage_level),
            level=PASSAGE_LEVEL,
            **identifiers,
        )
    ]
    rows.extend(
        metrics_row(
            count_span_level_metrics(aligned_gold, span_level, threshold=threshold),
            score_negative_class=False,
            level=f"span@{threshold}",
            **identifiers,
        )
        for threshold in SPAN_AGREEMENT_THRESHOLDS
    )

    return rows


def disagreement_rows(
    wikibase_id: str,
    ensemble: NamedEnsemble,
    gold: list[LabelledPassage],
    predictions_by_member: dict[EnsembleMember, dict[str, LabelledPassage]],
) -> list[dict[str, Any]]:
    """
    Per-passage disagreement and correctness for one ensemble.

    This is what the automation recommendation rests on: how good the ensemble's labels
    are, as a function of how much its members disagreed.
    """

    aligned_gold, spans_per_passage = align_passages(
        gold, predictions_by_member, ensemble.members
    )

    disagreement = Disagreement()
    majority_vote = MajorityVote()

    rows = []
    for passage, spans_per_member in zip(aligned_gold, spans_per_passage):
        predicted_positive = majority_vote(spans_per_member) >= 0.5
        ground_truth_positive = bool(passage.spans)
        rows.append(
            {
                "concept": wikibase_id,
                "ensemble": ensemble.name,
                "n_members": ensemble.size,
                "passage_id": passage.id,
                "disagreement": float(disagreement(spans_per_member)),
                "n_positive_votes": sum(1 for s in spans_per_member if s),
                "predicted_positive": predicted_positive,
                "ground_truth_positive": ground_truth_positive,
            }
        )

    return rows


def confusion_matrix_for(passages: pd.DataFrame) -> ConfusionMatrix:
    """Build a passage-level confusion matrix from rows of the per-passage dataframe."""

    predicted = passages["predicted_positive"]
    gold = passages["ground_truth_positive"]

    return ConfusionMatrix(
        true_positives=int((predicted & gold).sum()),
        false_positives=int((predicted & ~gold).sum()),
        true_negatives=int((~predicted & ~gold).sum()),
        false_negatives=int((~predicted & gold).sum()),
    )


def assigns_positive_labels(confusion_matrix: ConfusionMatrix) -> bool:
    """
    Whether a subset contains any positive prediction at all.

    `ConfusionMatrix.precision` returns 0 rather than raising when nothing was predicted
    positive, so without this check a policy that assigns no positive labels would report
    F1 = 0.000 as though it had been scored and failed.
    """
    return (confusion_matrix.true_positives + confusion_matrix.false_positives) > 0


def assigns_negative_labels(confusion_matrix: ConfusionMatrix) -> bool:
    """
    Whether a subset contains any negative prediction at all.

    The mirror of `assigns_positive_labels`: `negative_predictive_value` also returns 0
    rather than raising, so without this check a subset that labels everything positive
    would report NPV = 0.000 as though its negatives had been scored and found wrong.
    """
    return (confusion_matrix.true_negatives + confusion_matrix.false_negatives) > 0


def macro(
    matrices: list[ConfusionMatrix], metric: Callable[[ConfusionMatrix], float]
) -> float:
    """
    Mean of one metric over the concepts whose matrices can define it.

    Callers filter `matrices` with `assigns_positive_labels` or `assigns_negative_labels`
    first, so a concept whose subset assigns no labels on the relevant side is left out of
    the mean rather than folded in as a zero — a concept that was never scored is not a
    concept that scored badly. If no concept can define the metric the result is `nan`,
    which is the honest answer and renders as `NOT_APPLICABLE`.
    """
    return sum(metric(cm) for cm in matrices) / len(matrices) if matrices else math.nan


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    Used to calculate confidence intervals for specific vote splits, as these have
    wildly varying sample sizes.
    """

    if n == 0:
        return (math.nan, math.nan)

    proportion = successes / n
    denominator = 1 + z**2 / n
    centre = (proportion + z**2 / (2 * n)) / denominator
    margin = (
        z * math.sqrt(proportion * (1 - proportion) / n + z**2 / (4 * n**2))
    ) / denominator

    return (max(0.0, centre - margin), min(1.0, centre + margin))


def summarise_vote_splits(
    disagreement_df: pd.DataFrame, n_members: int
) -> pd.DataFrame:
    """
    What each vote split buys, in precision and false negatives.

    Every passage in a bucket gets the same label, so only one of the two sides of the
    confusion matrix is informative, and which one depends on the side of the vote:

    - buckets the ensemble calls **positive** assign a positive label to everything, so
      recall within the bucket is trivially 1.0 and precision is the whole story;
    - buckets it calls **negative** assign no positive labels, so precision and F1 are
      undefined; what matters is the real mentions they discard.

    Both rates are reported with Wilson intervals, and every bucket from 0 to
    ``n_members`` gets a row even when empty — an absent bucket in an earlier version was
    indistinguishable from one that had been dropped.
    """

    total = len(disagreement_df)
    rows = []

    for votes in range(n_members + 1):
        bucket = disagreement_df[disagreement_df["n_positive_votes"] == votes]
        n_passages = len(bucket)

        # ties count as positive, matching `MajorityVote` and `MajorityVoteAggregator`
        ensemble_positive = 2 * votes >= n_members
        n_gold_positive = (
            int(bucket["ground_truth_positive"].sum()) if n_passages else 0
        )

        row: dict[str, Any] = {
            "votes": f"{votes}/{n_members}",
            "ensemble_label": ("positive" if ensemble_positive else "negative")
            if n_passages
            else None,
            "disagreement": 2 * min(votes, n_members - votes) / n_members,
            "n_passages": n_passages,
            "coverage": n_passages / total if total else math.nan,
            "precision": math.nan,
            "precision_ci_low": math.nan,
            "precision_ci_high": math.nan,
            "false_negatives": pd.NA,
            "missed_mention_rate": math.nan,
            "missed_ci_low": math.nan,
            "missed_ci_high": math.nan,
        }

        if n_passages and ensemble_positive:
            row["precision"] = n_gold_positive / n_passages
            row["precision_ci_low"], row["precision_ci_high"] = wilson_interval(
                n_gold_positive, n_passages
            )
        elif n_passages:
            row["false_negatives"] = n_gold_positive
            row["missed_mention_rate"] = n_gold_positive / n_passages
            row["missed_ci_low"], row["missed_ci_high"] = wilson_interval(
                n_gold_positive, n_passages
            )

        rows.append(row)

    vote_splits = pd.DataFrame(rows)
    vote_splits["false_negatives"] = vote_splits["false_negatives"].astype("Int64")
    return vote_splits


def automation_policies(
    disagreement_df: pd.DataFrame,
) -> list[tuple[str, float, pd.Series]]:
    """
    The candidate routing policies, as (name, threshold, mask) over the per-passage frame.

    Each policy auto-labels the passages it selects and routes the rest to a human. The
    nested ``disagreement <= t`` policies use only the disagreement values that actually
    occur — for a 9-member ensemble there are five, so resampling them onto a percentile
    grid (as the active-learning retention curve does) would manufacture ~96 duplicate
    points and hide how coarse the real axis is.

    The two directional policies split the unanimous subset, which otherwise averages an
    all-positive population together with an all-negative one. They are not points on the
    disagreement axis — both sit at disagreement 0 — so their threshold is ``nan``.
    """

    policies: list[tuple[str, float, pd.Series]] = [
        (
            f"disagreement <= {threshold:.3f}",
            float(threshold),
            disagreement_df["disagreement"] <= threshold,
        )
        for threshold in sorted(disagreement_df["disagreement"].unique())
    ]

    unanimous = disagreement_df["disagreement"] == 0
    predicted_positive = disagreement_df["predicted_positive"]
    policies.append(
        (UNANIMOUS_POSITIVE_POLICY, math.nan, unanimous & predicted_positive)
    )
    policies.append(
        (UNANIMOUS_NEGATIVE_POLICY, math.nan, unanimous & ~predicted_positive)
    )

    return policies


def policy_verdict(macro_f1: float, macro_coverage: float) -> str:
    """Apply the automation gates to one policy's macro numbers."""

    if pd.isna(macro_f1):
        return NOT_APPLICABLE
    if macro_f1 >= AUTOMATE_F1 and macro_coverage >= AUTOMATE_COVERAGE:
        return AUTOMATE_VERDICT
    if macro_f1 >= SEMI_AUTOMATE_F1:
        return SEMI_AUTOMATE_VERDICT
    return BELOW_BAR_VERDICT


def policy_table(disagreement_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score every candidate automation policy — the table the decision is read from.

    P/R/F1 are macro-averaged over concepts by `macro`, matching the headline per-concept
    tables, so one large evaluation set can't carry the verdict on its own. The pooled
    (micro) F1 sits alongside for comparison. If no concept assigns any positive label,
    the policy's P/R/F1 are undefined and `false_negatives` is what to read.

    `macro_npv` is the same treatment applied to the other side of the vote: how much a
    "no mention here" label from this policy can be trusted. It is what scores
    `unanimous negative only`, which by construction has no F1 at all.
    """

    passages_per_concept = disagreement_df.groupby("concept").size()
    total_passages = len(disagreement_df)

    rows = []
    for name, threshold, mask in automation_policies(disagreement_df):
        selected = cast(pd.DataFrame, disagreement_df[mask])

        coverages = []
        scored: list[ConfusionMatrix] = []
        scored_negative: list[ConfusionMatrix] = []
        for concept, group in selected.groupby("concept"):
            coverages.append(len(group) / passages_per_concept[concept])
            confusion_matrix = confusion_matrix_for(group)
            if assigns_positive_labels(confusion_matrix):
                scored.append(confusion_matrix)
            if assigns_negative_labels(confusion_matrix):
                scored_negative.append(confusion_matrix)

        # concepts with no selected passages still count as zero coverage
        missing_concepts = len(passages_per_concept) - len(coverages)
        coverages.extend([0.0] * missing_concepts)

        pooled = confusion_matrix_for(selected) if len(selected) else ConfusionMatrix()
        macro_coverage = sum(coverages) / len(coverages) if coverages else math.nan
        macro_f1 = macro(scored, ConfusionMatrix.f1_score)

        rows.append(
            {
                "policy": name,
                "disagreement_threshold": threshold,
                "n_passages": len(selected),
                "macro_coverage": macro_coverage,
                "macro_precision": macro(scored, ConfusionMatrix.precision),
                "macro_recall": macro(scored, ConfusionMatrix.recall),
                "macro_f1": macro_f1,
                "micro_f1": pooled.f1_score()
                if assigns_positive_labels(pooled)
                else math.nan,
                # the missed-mention rate is this subtracted from 1, so it isn't a column
                "macro_npv": macro(
                    scored_negative, ConfusionMatrix.negative_predictive_value
                ),
                "false_negatives": pooled.false_negatives,
                "n_human_remaining": total_passages - len(selected),
                "n_concepts": len(passages_per_concept),
                # false where the policy only ever labels positively, which makes its
                # recall 1.0 by construction rather than by merit
                "assigns_negatives": assigns_negative_labels(pooled),
                "verdict": policy_verdict(macro_f1, macro_coverage),
            }
        )

    return pd.DataFrame(rows)


def policy_table_by_ensemble(disagreement_df: pd.DataFrame) -> pd.DataFrame:
    """Score every policy for every ensemble, so each size can be read off."""

    frames = []
    for name, group in disagreement_df.groupby("ensemble", sort=False):
        table = policy_table(cast(pd.DataFrame, group))
        table.insert(0, "ensemble", str(name))
        table.insert(1, "n_members", int(group["n_members"].iloc[0]))
        table.insert(2, "family", family_of(str(name)))
        frames.append(table)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def automation_by_ensemble(
    policies: pd.DataFrame, costs_per_passage: dict[str, float] | None = None
) -> pd.DataFrame:
    """
    One row per ensemble: the best policy it achieves, and whether that automates.

    This is the "can N classifiers do the job, and how few will do?" view. Each ensemble
    is judged on its own best policy by macro F1, so a bigger ensemble is never penalised
    for a policy that happens to suit a smaller one.

    Only policies that label **both** classes are eligible. A positives-only policy cannot
    contain a false negative, so its recall is 1.0 by construction and its F1 beats any
    both-class policy of the same precision — it would win at every size and the table
    would rank an artefact. It also isn't automation of this task: it hands every negative
    passage back to a human, which is most of them. Those rows are dropped here rather than
    written anywhere, so `policy_table_by_ensemble` is where to look for them if the
    exclusion ever needs checking.

    :param costs_per_passage: per-passage USD cost keyed by ensemble name, from
        `ensemble_costs_per_passage`. Adds `TOTAL_COST_COLUMN`, which is what makes
        this a sizing table rather than a ranking: two ensembles can share a verdict and
        differ several-fold in price. `None` drops the column, for when pricing couldn't
        be fetched.
    """

    if policies.empty:
        return pd.DataFrame()

    scored = cast(
        pd.DataFrame,
        policies[policies["macro_f1"].notna() & policies["assigns_negatives"]],
    )
    if scored.empty:
        return pd.DataFrame()

    rows = []
    for _, group in scored.groupby("ensemble", sort=False):
        best = group.sort_values(["macro_f1", "macro_coverage"], ascending=False).iloc[
            0
        ]
        row = {
            "ensemble": best["ensemble"],
            "n_members": int(best["n_members"]),
            "family": best["family"],
            "n_concepts": int(best["n_concepts"]),
            "best_policy": best["policy"],
            "macro_coverage": best["macro_coverage"],
            "macro_precision": best["macro_precision"],
            "macro_recall": best["macro_recall"],
            "macro_f1": best["macro_f1"],
            "assigns_negatives": bool(best["assigns_negatives"]),
            "verdict": best["verdict"],
        }
        if costs_per_passage is not None:
            # every policy is a way of *reading* one full run of the ensemble, so the cost
            # is the whole evaluation set either way: the passages the policy auto-labels
            # plus the ones it hands to a human were all sent to every member. Routing
            # saves human time, not LLM spend.
            passages = int(best["n_passages"]) + int(best["n_human_remaining"])
            row[TOTAL_COST_COLUMN] = passages * costs_per_passage.get(
                str(best["ensemble"]), math.nan
            )
        rows.append(row)

    # best first, so the ensemble to beat is the top row. The sizing question is read off
    # `n_members` within it — `smallest_automating_size` re-sorts by size per family, so
    # the prose answer is unaffected by this ordering.
    # stable, so ensembles that tie on F1 stay in ensemble order rather than being
    # shuffled by the sort — several sizes scoring identically is a common outcome here
    return pd.DataFrame(rows).sort_values(
        "macro_f1", ascending=False, kind="stable", ignore_index=True
    )


def agreement_f1_curves(policies: pd.DataFrame) -> pd.DataFrame:
    """
    The agreement-vs-F1 curve for every ensemble, as tidy rows.

    Keeps only the nested ``disagreement <= t`` policies, which are the points of a curve:
    each one auto-labels everything its members agreed on to within ``t`` and hands the
    rest to a human, so the curve runs from the unanimous subset at ``t = 0`` to
    auto-labelling everything at the largest ``t``. The two directional unanimous policies
    are dropped — they both sit at disagreement 0, so they are a split of the leftmost
    point rather than points of their own.

    :param policies: the output of `policy_table_by_ensemble`
    """

    if policies.empty or "disagreement_threshold" not in policies.columns:
        return pd.DataFrame()

    curves = policies[policies["disagreement_threshold"].notna()]
    columns = [column for column in AGREEMENT_CURVE_COLUMNS if column in curves.columns]
    selected = cast(pd.DataFrame, curves[columns].copy())

    return selected.sort_values(
        by=["n_members", "ensemble", "disagreement_threshold"]
    ).reset_index(drop=True)


def passages_scored_per_concept(
    per_concept: pd.DataFrame, ensemble: str
) -> dict[str, int]:
    """
    How many gold passages each concept contributed to one ensemble's score.

    Read off the passage-level rows, where `support` *is* the passage count. It is not the
    same as the concept's gold set: `align_passages` drops any passage a member of this
    ensemble failed on, and the whole point of costing is to price what was actually
    sent to the models. Span-level rows can't supply it — their support counts gold spans.
    """

    rows = per_concept[
        (per_concept["ensemble"] == ensemble) & (per_concept["level"] == PASSAGE_LEVEL)
    ]
    return {
        str(concept): int(support)
        for concept, support in zip(rows["concept"], rows["support"])
    }


def headline_table(
    per_concept: pd.DataFrame,
    level: str,
    ensemble: str,
    cost_per_passage: float | None = None,
) -> pd.DataFrame:
    """
    Per-concept P/R/F1/support for one ensemble, with a macro row.

    :param cost_per_passage: what this ensemble costs to run over one passage, in USD,
        from `ensemble_cost_per_passage`. `None` drops the cost column entirely —
        pricing was unavailable, and a column of zeros would read as "this is free".
    """

    table = cast(
        pd.DataFrame,
        per_concept[
            (per_concept["ensemble"] == ensemble) & (per_concept["level"] == level)
        ][["concept", "precision", "recall", "f1", "support"]].copy(),
    )

    if table.empty:
        return table

    if cost_per_passage is not None:
        passages = passages_scored_per_concept(per_concept, ensemble)
        table[COST_COLUMN] = [
            passages.get(str(concept), math.nan) * cost_per_passage
            for concept in table["concept"]
        ]

    # name the row with its concept count, so a macro average over a partial set of
    # concepts can't be mistaken for one over all of them
    n_concepts = len(table)
    macro: dict[str, Any] = {
        "concept": f"macro average ({n_concepts} concepts)",
        "precision": table["precision"].mean(),
        "recall": table["recall"].mean(),
        "f1": table["f1"].mean(),
        "support": int(table["support"].sum()),
    }
    if COST_COLUMN in table.columns:
        # a mean, like every other cell in this row. `skipna=False`, because pandas would
        # otherwise average over just the priced concepts and present that as the figure
        # for all of them — a partial mean is not this row's mean. The whole-run budget is
        # `TOTAL_COST_COLUMN` in the sizing table.
        macro[COST_COLUMN] = table[COST_COLUMN].mean(skipna=False)

    return pd.concat([table, pd.DataFrame([macro])], ignore_index=True)


def negative_side_table(
    per_concept: pd.DataFrame, level: str, ensemble: str
) -> pd.DataFrame:
    """
    Per-concept reliability of the ensemble's *negative* labels, with a macro row.

    The mirror of `headline_table`, and the answer to "how many passages get no span
    prediction, and how much can that be trusted?". Recall doesn't answer it: recall is
    measured over gold positives, whereas what a labeller needs to know is what share of
    the passages handed back as empty actually contain a mention — `1 - npv`.

    A concept that assigns no negative labels at all is left out of the macro means, for
    the reason given on `macro`.
    """

    rows = cast(
        pd.DataFrame,
        per_concept[
            (per_concept["ensemble"] == ensemble) & (per_concept["level"] == level)
        ],
    )

    if rows.empty:
        return pd.DataFrame()

    predicted_negative = rows["true_negatives"] + rows["false_negatives"]
    table = pd.DataFrame(
        {
            "concept": rows["concept"],
            "predicted_negative": predicted_negative,
            "negative_share": predicted_negative / rows["support"],
            "npv": rows["npv"],
            "missed_mentions": rows["false_negatives"],
            "specificity": rows["specificity"],
        }
    )
    # undefined rather than 0.000 where the ensemble labelled nothing negative
    table.loc[predicted_negative.to_numpy() == 0, "npv"] = math.nan

    n_concepts = len(table)
    defined = table["npv"].notna()
    macro = pd.DataFrame(
        [
            {
                "concept": f"macro average ({n_concepts} concepts)",
                "predicted_negative": int(table["predicted_negative"].sum()),
                "negative_share": table["negative_share"].mean(),
                "npv": table.loc[defined, "npv"].mean()
                if bool(defined.any())
                else math.nan,
                "missed_mentions": int(table["missed_mentions"].sum()),
                "specificity": table["specificity"].mean(),
            }
        ]
    )

    return pd.concat([table, macro], ignore_index=True)


def cell_renderer(column: str, values: pd.Series) -> Callable[[Any], str]:
    """
    How one column's cells should be stringified for the console.

    Floats go to 3dp. Integer counts are left alone, so a count of 3 passages doesn't
    render as "3.000".

    Money is the exception to the 3dp rule, because dollars and scores don't share a
    scale: the same column holds a whole-experiment total in the hundreds and a single
    cheap ensemble's cost in cents, and 3dp would print the second as a rounding artefact
    of the first. Anything under a dollar keeps four decimal places so that "cheap" and
    "free" stay distinguishable. Matched on a substring rather than a suffix, because
    `for_console` renames the column to something shorter and the rule has to survive the
    rename.
    """

    if "usd" in column.lower():
        return lambda value: (
            f"${value:,.2f}" if abs(float(value)) >= 1 else f"${float(value):.4f}"
        )
    if pd.api.types.is_float_dtype(values):
        return lambda value: f"{value:.3f}"
    if pd.api.types.is_integer_dtype(values):
        return lambda value: str(int(value))
    return str


def format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stringify a dataframe for the console, one renderer per column.

    Anything missing becomes ``n/a`` — an undefined precision or F1 has to read as
    undefined rather than as a sentinel 0.000. `cell_renderer` decides the rest.
    """

    formatted = df.copy()
    for column in formatted.columns:
        values = cast(pd.Series, formatted[column])
        render = cell_renderer(str(column), values)
        # iterate the series rather than `.map`, which coerces a nullable Int64 column to
        # float to carry its missing values and would render 3 passages as "3.0"
        formatted[column] = [
            NOT_APPLICABLE if bool(pd.isna(value)) else render(value)
            for value in values
        ]

    return formatted


def vote_splits_for_display(vote_splits: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the vote-split table's interval bounds into one column each.

    The CSV keeps the bounds as separate numeric columns for plotting; a reader wants
    "0.755–0.913" in a single cell.
    """

    def interval(low: str, high: str) -> pd.Series:
        return pd.Series(
            [
                NOT_APPLICABLE
                if bool(pd.isna(lo)) or bool(pd.isna(hi))
                else f"{lo:.3f}–{hi:.3f}"
                for lo, hi in zip(vote_splits[low], vote_splits[high])
            ],
            index=vote_splits.index,
        )

    display = cast(
        pd.DataFrame,
        vote_splits[
            [
                "votes",
                "ensemble_label",
                "disagreement",
                "n_passages",
                "coverage",
                "precision",
            ]
        ].copy(),
    )
    display["precision_95_ci"] = interval("precision_ci_low", "precision_ci_high")
    display["false_negatives"] = vote_splits["false_negatives"]
    display["missed_mention_rate"] = vote_splits["missed_mention_rate"]
    display["missed_95_ci"] = interval("missed_ci_low", "missed_ci_high")

    return display


def for_console(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """
    Narrow and rename a table so it survives an 80-column terminal.

    Rich shrinks columns to fit the terminal, which turns a ten-column table into a grid of
    ellipses. The CSVs keep every column; the console gets the ones the decision turns
    on, under short headers.
    """

    available = {
        column: label for column, label in columns.items() if column in df.columns
    }
    narrowed = cast(pd.DataFrame, df[list(available)].copy())

    for column in ("policy", "best_policy"):
        if column in narrowed.columns:
            narrowed[column] = [
                str(policy).replace("disagreement <= ", "≤ ").replace(" only", "")
                for policy in narrowed[column]
            ]

    return cast(pd.DataFrame, narrowed.rename(columns=available))


def print_table(df: pd.DataFrame, title: str) -> None:
    """Print a dataframe to the console."""
    table = Table(title=title, box=box.SIMPLE, show_header=True)
    for column in df.columns:
        table.add_column(str(column))
    for row in format_for_display(df).itertuples(index=False):
        table.add_row(*[str(value) for value in row])
    console.print(table)


def plot_agreement_vs_f1(curves: pd.DataFrame, output_dir: Path) -> None:
    """
    One panel per ensemble: F1 against how much member disagreement is tolerated.

    The same reading as the active-learning ensemble plot
    (`scripts.active_learning.plot_ensemble_metrics.create_plots`) — F1 against the
    disagreement threshold, with the share of passages that buys on a secondary axis — but
    scored on this experiment's terms: macro F1 over concepts, on the labels the policy
    assigns.

    The x-axis carries only the disagreement values an ensemble can actually produce
    (``⌊n/2⌋ + 1`` of them), not a percentile grid, so a coarse curve looks coarse. An
    ``n=1`` ensemble has a single point at 0: it always agrees with itself, so there is no
    curve to read.
    """

    if curves.empty:
        return

    ensembles = list(dict.fromkeys(curves["ensemble"]))
    n_cols = min(3, len(ensembles))
    n_rows = math.ceil(len(ensembles) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.0 * n_cols, 4.2 * n_rows), squeeze=False
    )

    # a shared, zoomed y-range: on a 0–1 axis every curve here is a flat line, because the
    # whole population sits inside a few F1 points. Shared so the panels stay comparable —
    # a per-panel range would rescale each ensemble's noise to look like the same movement.
    scores = [value for value in curves["macro_f1"] if not bool(pd.isna(value))]
    y_limits = (min(scores) - 0.02, max(scores) + 0.02) if scores else (0.0, 1.0)

    for ax, ensemble in zip(axes.flat, ensembles):
        panel = cast(pd.DataFrame, curves[curves["ensemble"] == ensemble]).sort_values(
            "disagreement_threshold"
        )
        thresholds = panel["disagreement_threshold"].tolist()
        f1_scores = panel["macro_f1"].tolist()
        coverages = panel["macro_coverage"].tolist()

        ax.plot(
            thresholds,
            f1_scores,
            "b-",
            linewidth=2,
            marker="o",
            markersize=5,
            label="Macro F1",
        )

        # the rightmost point is "auto-label everything": the do-nothing baseline the rest
        # of the curve has to beat to be worth any human review at all
        baseline_f1 = f1_scores[-1]
        if not bool(pd.isna(baseline_f1)):
            ax.axhline(
                y=baseline_f1,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"No review = {baseline_f1:.3f}",
            )

        if bool(panel["macro_f1"].notna().any()):
            best = panel.loc[panel["macro_f1"].idxmax()]
            ax.scatter(
                best["disagreement_threshold"],
                best["macro_f1"],
                color="red",
                s=100,
                zorder=5,
                label=(
                    f"Best: {best['macro_f1']:.3f} "
                    f"(≤ {best['disagreement_threshold']:.2f}, "
                    f"covers {best['macro_coverage']:.0%})"
                ),
            )

        # coverage is monotonic in the threshold, so it maps onto the same axis; a single
        # point (n=1) has nothing to span, and matplotlib would collapse the limits
        finite_coverages = [value for value in coverages if not bool(pd.isna(value))]
        if len(set(finite_coverages)) > 1:
            coverage_axis = ax.twiny()
            coverage_axis.plot(coverages, f1_scores, alpha=0)
            coverage_axis.set_xlim(min(finite_coverages), max(finite_coverages))
            coverage_axis.set_xlabel("Coverage (share auto-labelled)", color="green")
            coverage_axis.tick_params(axis="x", labelcolor="green")

        n_members = int(panel["n_members"].iloc[0])
        ax.set_title(f"{ensemble} ({n_members} classifier(s))")
        ax.set_xlabel("Max member disagreement auto-labelled", color="blue")
        ax.set_ylabel("Macro F1 of the labels assigned")
        ax.tick_params(axis="x", labelcolor="blue")
        ax.set_ylim(*y_limits)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    for ax in axes.flat[len(ensembles) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "agreement_vs_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_automation_by_size(by_ensemble: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot the best achievable macro F1 against ensemble size, one line per family.

    This is the sizing figure: where a family's line crosses a gate is the smallest N that
    can automate, and where the line goes flat is the point past which extra classifiers
    change nothing.
    """

    if by_ensemble.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))

    for family, group in by_ensemble.groupby("family", sort=True):
        ordered = group.sort_values("n_members")
        ax.plot(
            ordered["n_members"],
            ordered["macro_f1"],
            marker="o",
            label=str(family),
        )
        for _, row in ordered.iterrows():
            if row["verdict"] == AUTOMATE_VERDICT:
                ax.scatter(
                    row["n_members"],
                    row["macro_f1"],
                    marker="o",
                    s=160,
                    facecolors="none",
                    edgecolors="green",
                    zorder=5,
                )

    ax.axhline(
        AUTOMATE_F1,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"automate ≥ {AUTOMATE_F1:.0%}",
    )
    ax.axhline(
        SEMI_AUTOMATE_F1,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"semi-automate ≥ {SEMI_AUTOMATE_F1:.0%}",
    )

    plotted = by_ensemble["macro_f1"].tolist() + [AUTOMATE_F1, SEMI_AUTOMATE_F1]
    finite = [value for value in plotted if not pd.isna(value)]
    ax.set_ylim(min(finite) - 0.04, max(finite) + 0.03)
    ax.set_xticks(sorted(set(by_ensemble["n_members"].tolist())))
    ax.set_xlabel("Ensemble size (number of classifiers)")
    ax.set_ylabel("Macro F1 of the best policy at that size")
    automating = (by_ensemble["verdict"] == AUTOMATE_VERDICT).any()
    ax.set_title(
        "How many classifiers does automation need?"
        + (
            "\n(ringed = clears the automate gate)"
            if automating
            else "\n(nothing reaches the automate gate on a both-class policy)"
        )
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "automation_by_size.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def find_policy(policies: pd.DataFrame, name: str) -> pd.Series | None:
    """Return one policy's row by name, or None if it isn't in the table."""
    matches = policies[policies["policy"] == name]
    return None if matches.empty else cast(pd.Series, matches.iloc[0])


def smallest_automating_size(by_ensemble: pd.DataFrame) -> str:
    """
    Report the fewest classifiers per family that reach each level of automation.

    Answers the sizing question directly: rather than ranking ensembles against each other,
    it asks of each family "at what point does adding classifiers stop changing the
    verdict?" — because the cheapest ensemble that clears the bar is the one to ship.
    """

    if by_ensemble.empty:
        return "No scored ensembles — cannot size one."

    lines = []
    for family, group in by_ensemble.groupby("family", sort=True):
        ordered = group.sort_values("n_members")
        parts = []
        for verdict in (AUTOMATE_VERDICT, SEMI_AUTOMATE_VERDICT):
            reaching = ordered[ordered["verdict"] == verdict]
            if not reaching.empty:
                first = reaching.iloc[0]
                parts.append(
                    f"**{verdict} from n={first['n_members']}** "
                    f"(macro F1 {first['macro_f1']:.3f} on `{first['best_policy']}`, "
                    f"covering {first['macro_coverage']:.1%})"
                )
        best = ordered.sort_values("macro_f1", ascending=False).iloc[0]
        if not parts:
            parts.append(
                f"never clears the bar — best is n={best['n_members']} at macro F1 "
                f"{best['macro_f1']:.3f}"
            )

        sizes = ordered["n_members"].tolist()
        f1s = ordered["macro_f1"].tolist()
        flat = len(set(round(value, 6) for value in f1s)) == 1 and len(f1s) > 1
        suffix = (
            f". Macro F1 is identical across n={min(sizes)}–{max(sizes)}, so the extra "
            "members change no majority vote"
            if flat
            else ""
        )
        lines.append(f"- `{family}`: {'; '.join(parts)}{suffix}")

    return "\n".join(lines)


def recommendation(
    policies: pd.DataFrame,
    n_concepts: int,
    headline_name: str = HEADLINE_ENSEMBLE,
) -> str:
    """
    Turn the policy table into the automate / semi-automate / don't verdict.

    Picks the best-scoring policy that clears a gate rather than assuming the unanimous
    subset is the only candidate — a looser threshold can score higher, because the
    passages it adds are not uniformly harder.

    Thresholds are fixed in this module's constants rather than chosen after seeing the
    results.

    :param headline_name: the ensemble the policies were scored on, named only so the
        no-results message can point at the one the caller actually asked for.
    """

    if policies.empty:
        return (
            f"No `{headline_name}` results — cannot assess automation. "
            "Run `predict` for every member first."
        )

    # both-class policies only, matching `automation_by_ensemble`: a positives-only
    # policy has recall 1.0 by construction, so ranking on F1 would always crown it, and it
    # leaves every negative passage to a human. Its numbers still appear in the asymmetry
    # bullet below and in the policy table.
    scored = cast(
        pd.DataFrame,
        policies[policies["macro_f1"].notna() & policies["assigns_negatives"]],
    )
    if scored.empty:
        return "No policy labels both classes — cannot assess automation of the whole task."

    for tier in (AUTOMATE_VERDICT, SEMI_AUTOMATE_VERDICT):
        candidates = cast(pd.DataFrame, scored[scored["verdict"] == tier])
        if not candidates.empty:
            break
    else:
        best = scored.sort_values("macro_f1", ascending=False).iloc[0]
        return (
            "**Don't automate**: no policy reaches the bar.\n\n"
            f"- Best was `{best['policy']}` at macro F1 {best['macro_f1']:.3f} "
            f"(semi-automate ≥ {SEMI_AUTOMATE_F1:.0%}), covering "
            f"{best['macro_coverage']:.1%} of passages\n"
            f"- Macro-averaged over {n_concepts} concept(s)"
        )

    best = candidates.sort_values(["macro_f1", "macro_coverage"], ascending=False).iloc[
        0
    ]
    # the widest policy that still routes something to a human. Without the filter this is
    # always "auto-label everything", which belongs below as the no-review baseline rather
    # than as a trade-off against it.
    reviewed = cast(pd.DataFrame, candidates[candidates["n_human_remaining"] > 0])
    widest = (
        reviewed.sort_values(["macro_coverage", "macro_f1"], ascending=False).iloc[0]
        if not reviewed.empty
        else None
    )

    if tier == AUTOMATE_VERDICT:
        verdict = f"**Automate** the `{best['policy']}` subset."
    else:
        verdict = (
            f"**Semi-automate**: pre-label the `{best['policy']}` subset in Argilla for "
            "human confirmation."
        )

    binds = "precision" if best["macro_precision"] <= best["macro_recall"] else "recall"
    lines = [
        f"{verdict}\n",
        f"- Macro F1 {best['macro_f1']:.3f} on the labels it assigns "
        f"(automate ≥ {AUTOMATE_F1:.0%}, semi-automate ≥ {SEMI_AUTOMATE_F1:.0%})",
        f"- Covers {best['macro_coverage']:.1%} of passages "
        f"(threshold for automating: {AUTOMATE_COVERAGE:.0%}), leaving "
        f"{best['n_human_remaining']} for a human",
        f"- **{binds.title()} binds**: precision {best['macro_precision']:.3f} vs recall "
        f"{best['macro_recall']:.3f}",
    ]

    if widest is not None and widest["policy"] != best["policy"]:
        lines.append(
            f"- Trade-off: `{widest['policy']}` also clears the bar and covers more "
            f"({widest['macro_coverage']:.1%}) at macro F1 {widest['macro_f1']:.3f}, "
            f"leaving only {widest['n_human_remaining']} for a human"
        )

    # auto-labelling everything is the do-nothing baseline: it is what the ensemble scores
    # with no human review at all, and the difference is what the review actually buys
    baseline = cast(pd.DataFrame, scored[scored["n_human_remaining"] == 0])
    if not baseline.empty:
        no_review = baseline.iloc[0]
        lines.append(
            f"- For comparison, auto-labelling *everything* scores macro F1 "
            f"{no_review['macro_f1']:.3f}, so routing the "
            f"{best['n_human_remaining']} most-disagreed-on passages to a human is worth "
            f"{best['macro_f1'] - no_review['macro_f1']:+.3f} F1"
        )

    unanimous_positive = find_policy(policies, UNANIMOUS_POSITIVE_POLICY)
    unanimous_negative = find_policy(policies, UNANIMOUS_NEGATIVE_POLICY)
    if unanimous_positive is not None and unanimous_negative is not None:
        lines.append(
            f"- The unanimous subset is asymmetric: unanimous *positive* scores macro F1 "
            f"{unanimous_positive['macro_f1']:.3f}, while unanimous *negative* assigns no "
            f"positive labels at all and discards "
            f"{unanimous_negative['false_negatives']} real mention(s), so F1 cannot "
            "score it"
        )
        # the negative side needs its own number, or the read is that F1 says nothing
        # about it and it therefore costs nothing
        if not bool(pd.isna(unanimous_negative["macro_npv"])):
            lines.append(
                f"- Those unanimous negatives are right {unanimous_negative['macro_npv']:.1%} "
                f"of the time (macro NPV), against macro precision "
                f"{best['macro_precision']:.3f} on the positives — so an empty passage is "
                + (
                    "the *less* trustworthy of the two labels"
                    if unanimous_negative["macro_npv"] < best["macro_precision"]
                    else "the more trustworthy of the two labels"
                )
            )

    lines.append(f"- Macro-averaged over {n_concepts} concept(s)")

    return "\n".join(lines)
