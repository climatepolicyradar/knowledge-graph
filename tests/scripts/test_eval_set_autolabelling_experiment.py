import pandas as pd
import pytest
import typer

from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.metrics import ConfusionMatrix
from knowledge_graph.span import Span
from scripts.benchmarks.eval_set_autolabelling_experiment import cli, ensembles
from scripts.benchmarks.eval_set_autolabelling_experiment.analysis import (
    agreement_f1_curves,
    automation_by_ensemble,
    format_for_display,
    headline_table,
    metrics_row,
    negative_side_table,
    policy_table,
    policy_table_by_ensemble,
    policy_verdict,
    summarise_vote_splits,
    wilson_interval,
)
from scripts.benchmarks.eval_set_autolabelling_experiment.config import (
    AUTOMATE_F1,
    AUTOMATE_VERDICT,
    BELOW_BAR_VERDICT,
    COST_COLUMN,
    ESTIMATED_COMPLETION_TOKENS_PER_PASSAGE,
    ESTIMATED_PROMPT_TOKENS_PER_PASSAGE,
    FAMILY_A,
    FAMILY_B,
    HEADLINE_ENSEMBLE,
    NOT_APPLICABLE,
    OPENROUTER_PRICING_CACHE_FILENAME,
    PASSAGE_LEVEL,
    SEMI_AUTOMATE_F1,
    SEMI_AUTOMATE_VERDICT,
    UNANIMOUS_NEGATIVE_POLICY,
    UNANIMOUS_POSITIVE_POLICY,
)
from scripts.benchmarks.eval_set_autolabelling_experiment.ensembles import (
    NamedEnsemble,
    TokenPrices,
    compose_possible_ensembles,
    ensemble_cost_per_passage,
    load_model_pricing,
    read_member_cache,
    run_member_with_caching,
)

CONCEPT_ID = WikibaseID("Q42")


@pytest.fixture
def gold() -> list[LabelledPassage]:
    return [
        LabelledPassage(id=f"p{i}", text=f"passage number {i}", spans=[])
        for i in range(4)
    ]


@pytest.fixture
def concept() -> Concept:
    return Concept(wikibase_id=CONCEPT_ID, preferred_label="just transition")


def stub_labelling(monkeypatch) -> list[list[str]]:
    """
    Replace the classifier and the labelling call with a stub.

    Returns a list that accumulates the texts the stub was asked to predict, one entry per
    call, so a test can assert which passages actually reached the LLM.
    """
    requested: list[list[str]] = []

    monkeypatch.setattr(
        ensembles, "build_member_classifier", lambda concept, member: object()
    )

    def fake_label(classifier, labelled_passages, batch_size, show_progress):
        requested.append([p.text for p in labelled_passages])
        return [
            passage.model_copy(
                update={
                    "spans": [
                        Span(
                            text=passage.text,
                            start_index=0,
                            end_index=7,
                            concept_id=CONCEPT_ID,
                            labellers=["stub"],
                        )
                    ]
                },
                deep=True,
            )
            for passage in labelled_passages
        ]

    monkeypatch.setattr(ensembles, "label_passages_with_classifier", fake_label)
    return requested


def test_whether_an_empty_cache_predicts_every_passage(
    monkeypatch, tmp_path, gold, concept
):
    """With nothing cached, the whole gold set goes to the LLM."""
    requested = stub_labelling(monkeypatch)

    run_member_with_caching(
        concept=concept,
        member=("opus", 1),
        gold=gold,
        concept_dir=tmp_path,
        batch_size=4,
        position="Member 1/1",
    )

    assert requested == [[p.text for p in gold]]
    assert set(read_member_cache(tmp_path, ("opus", 1))) == {"p0", "p1", "p2", "p3"}


def test_whether_a_rerun_only_requests_passages_missing_from_the_cache(
    monkeypatch, tmp_path, gold, concept
):
    """
    The point of caching per passage: a rerun covers only the gaps.

    Modelled here by the gold set growing between runs, which is the realistic case now
    that a failed call is cached as a negative rather than left out.
    """
    stub_labelling(monkeypatch)
    run_member_with_caching(
        concept=concept,
        member=("opus", 1),
        gold=gold[:2],
        concept_dir=tmp_path,
        batch_size=4,
        position="first run",
    )

    requested = stub_labelling(monkeypatch)
    run_member_with_caching(
        concept=concept,
        member=("opus", 1),
        gold=gold,
        concept_dir=tmp_path,
        batch_size=4,
        position="rerun with more gold",
    )

    assert requested == [["passage number 2", "passage number 3"]]
    assert set(read_member_cache(tmp_path, ("opus", 1))) == {"p0", "p1", "p2", "p3"}


def test_whether_a_complete_member_makes_no_further_calls(
    monkeypatch, tmp_path, gold, concept
):
    """A fully cached member is skipped, so reruns are cheap."""
    stub_labelling(monkeypatch)
    run_member_with_caching(
        concept=concept,
        member=("opus", 1),
        gold=gold,
        concept_dir=tmp_path,
        batch_size=4,
        position="first run",
    )

    requested = stub_labelling(monkeypatch)
    run_member_with_caching(
        concept=concept,
        member=("opus", 1),
        gold=gold,
        concept_dir=tmp_path,
        batch_size=4,
        position="rerun",
    )

    assert requested == []


def passages(
    concept: str, n_members: int, *splits: tuple[int, int, int]
) -> list[dict[str, object]]:
    """
    Build per-passage rows for the automation analysis.

    Each split is ``(positive votes, gold positives, gold negatives)``, which is enough to
    pin down every confusion-matrix cell: the vote count fixes the ensemble's label and the
    disagreement, and the two counts fix what gold said.
    """

    rows: list[dict[str, object]] = []
    for votes, gold_positives, gold_negatives in splits:
        for index in range(gold_positives + gold_negatives):
            rows.append(
                {
                    "concept": concept,
                    "passage_id": f"{concept}-{votes}-{index}",
                    "disagreement": 2 * min(votes, n_members - votes) / n_members,
                    "n_positive_votes": votes,
                    "predicted_positive": 2 * votes >= n_members,
                    "ground_truth_positive": index < gold_positives,
                }
            )
    return rows


def test_whether_a_policy_scores_its_subset_as_hand_computed():
    """
    Anchors the policy table against arithmetic done by hand.

    Unanimous: 3 passages labelled positive of which 2 are gold (TP=2, FP=1), plus one
    labelled negative that was gold (FN=1) — so P=R=2/3. Widening to every passage adds two
    false positives, dropping precision to 2/5.
    """
    df = pd.DataFrame(
        passages("Q1", 3, (3, 2, 1), (0, 1, 0), (2, 0, 2)),
    )

    policies = policy_table(df).set_index("policy")

    unanimous = policies.loc["disagreement <= 0.000"]
    assert unanimous["n_passages"] == 4
    assert unanimous["macro_precision"] == pytest.approx(2 / 3)
    assert unanimous["macro_recall"] == pytest.approx(2 / 3)
    assert unanimous["macro_f1"] == pytest.approx(2 / 3)
    assert unanimous["false_negatives"] == 1
    assert unanimous["n_human_remaining"] == 2

    everything = policies.loc["disagreement <= 0.667"]
    assert everything["macro_precision"] == pytest.approx(0.4)
    assert everything["macro_recall"] == pytest.approx(2 / 3)
    assert everything["n_human_remaining"] == 0


def test_whether_coverage_grows_with_the_disagreement_threshold():
    """The threshold policies are nested, so each one can only add passages."""
    df = pd.DataFrame(passages("Q1", 3, (3, 2, 1), (0, 1, 0), (2, 0, 2), (1, 1, 1)))

    thresholds = policy_table(df).sort_values(by=["policy"])
    nested = thresholds[thresholds["policy"].str.startswith("disagreement <=")]

    coverages = nested["macro_coverage"].tolist()
    assert coverages == sorted(coverages)
    assert coverages[-1] == pytest.approx(1.0)


def test_whether_a_policy_assigning_no_positive_labels_scores_as_undefined():
    """
    F1 cannot score a policy that never says positive, and 0.0 would be a lie.

    `ConfusionMatrix.precision` returns 0 rather than raising on an empty denominator, so
    without an explicit check this policy would read as a scored, failing 0.000 — and could
    then be compared against a gate.
    """
    df = pd.DataFrame(passages("Q1", 3, (3, 2, 1), (0, 3, 2)))

    negatives = policy_table(df).set_index("policy").loc[UNANIMOUS_NEGATIVE_POLICY]

    assert pd.isna(negatives["macro_precision"])
    assert pd.isna(negatives["macro_recall"])
    assert pd.isna(negatives["macro_f1"])
    assert pd.isna(negatives["micro_f1"])
    # the real mentions it would silently discard are what's left to read
    assert negatives["false_negatives"] == 3
    assert negatives["verdict"] == "n/a"


def test_whether_the_macro_average_differs_from_the_pooled_one():
    """
    The regression that motivated macro-averaging the verdict.

    A 10-passage concept at F1 0.947 and a 2-passage one at F1 0.667 must not be reported
    as the pooled 0.909, which is almost entirely the big concept's score.
    """
    df = pd.DataFrame(
        passages("Q1", 3, (3, 9, 1)) + passages("Q2", 3, (3, 1, 1)),
    )

    unanimous = policy_table(df).set_index("policy").loc["disagreement <= 0.000"]

    assert unanimous["macro_f1"] == pytest.approx((0.9473684 + 2 / 3) / 2, abs=1e-6)
    assert unanimous["micro_f1"] == pytest.approx(10 / 11, abs=1e-6)
    assert unanimous["macro_f1"] != pytest.approx(unanimous["micro_f1"])


def test_whether_empty_vote_splits_still_get_a_row():
    """An absent split must be visibly empty rather than silently missing."""
    df = pd.DataFrame(passages("Q1", 3, (3, 2, 1), (0, 1, 0)))

    vote_splits = summarise_vote_splits(df, 3).set_index("votes")

    assert list(vote_splits.index) == ["0/3", "1/3", "2/3", "3/3"]
    assert vote_splits.loc["1/3", "n_passages"] == 0
    assert vote_splits.loc["2/3", "n_passages"] == 0


def test_whether_each_vote_split_reports_only_its_informative_side():
    """
    Precision where the ensemble commits to a label, missed mentions where it doesn't.

    Recall inside a positive split is trivially 1.0 and precision inside a negative one is
    undefined, so reporting both everywhere would print numbers that mean nothing.
    """
    df = pd.DataFrame(passages("Q1", 3, (3, 2, 1), (0, 1, 3)))

    vote_splits = summarise_vote_splits(df, 3).set_index("votes")

    positive = vote_splits.loc["3/3"]
    assert positive["ensemble_label"] == "positive"
    assert positive["precision"] == pytest.approx(2 / 3)
    assert pd.isna(positive["false_negatives"])
    assert pd.isna(positive["missed_mention_rate"])

    negative = vote_splits.loc["0/3"]
    assert negative["ensemble_label"] == "negative"
    assert pd.isna(negative["precision"])
    assert negative["false_negatives"] == 1
    assert negative["missed_mention_rate"] == pytest.approx(0.25)


def test_whether_a_single_passage_split_gets_an_uninformative_interval():
    """The whole point of the intervals: a bare 1.000 off one passage must not look solid."""
    low, high = wilson_interval(1, 1)

    assert high == pytest.approx(1.0)
    assert low < 0.5

    for successes, n in [(0, 1), (1, 1), (3, 8), (67, 67), (0, 0)]:
        low, high = wilson_interval(successes, n)
        if n:
            assert 0.0 <= low <= high <= 1.0


@pytest.mark.parametrize(
    "macro_f1, macro_coverage, expected",
    [
        (AUTOMATE_F1 + 0.001, 0.9, AUTOMATE_VERDICT),
        (AUTOMATE_F1, 0.9, AUTOMATE_VERDICT),
        # clears the F1 gate but not the coverage one, so it can only be semi-automated
        (AUTOMATE_F1 + 0.001, 0.1, SEMI_AUTOMATE_VERDICT),
        (AUTOMATE_F1 - 0.001, 0.9, SEMI_AUTOMATE_VERDICT),
        (SEMI_AUTOMATE_F1, 0.9, SEMI_AUTOMATE_VERDICT),
        (SEMI_AUTOMATE_F1 - 0.001, 0.9, BELOW_BAR_VERDICT),
    ],
)
def test_whether_the_verdict_gates_are_applied_at_their_boundaries(
    macro_f1, macro_coverage, expected
):
    """Both gates are inclusive, and coverage only ever blocks full automation."""
    assert policy_verdict(macro_f1, macro_coverage) == expected


def test_whether_the_default_headline_survives_the_default_cap():
    """
    The advertised default has to be reachable, or the script picks the headline instead.

    An earlier pairing defaulted `--headline` to `mixed_n9` while `--max-members` defaulted
    to 5, so the request was unmeetable on every run and a fallback rule chose the headline
    — which is exactly the choice that should sit on the command line.
    """
    assert HEADLINE_ENSEMBLE in {c.name for c in compose_possible_ensembles(5)}


def test_whether_a_positives_only_policy_cannot_win_the_sizing_table():
    """
    The artefact this guard exists for.

    A policy that only labels positively has recall 1.0 by construction, so its F1 beats any
    both-class policy of equal precision and it would win at every ensemble size. Here the
    positives-only subset is perfect while the both-class one is not, so an unguarded search
    would pick it.
    """
    df = pd.DataFrame(
        passages("Q1", 3, (3, 4, 0), (0, 2, 2)),
    )
    df["ensemble"] = "opus_n3"
    df["n_members"] = 3

    policies = policy_table_by_ensemble(df)
    positives_only = policies.set_index("policy").loc[UNANIMOUS_POSITIVE_POLICY]
    assert positives_only["macro_f1"] == pytest.approx(1.0)
    assert not positives_only["assigns_negatives"]

    by_ensemble = automation_by_ensemble(policies)

    assert len(by_ensemble) == 1
    assert by_ensemble.iloc[0]["best_policy"] != UNANIMOUS_POSITIVE_POLICY
    assert by_ensemble.iloc[0]["assigns_negatives"]
    # the both-class policy misses the 2 real mentions the negative side discards
    assert by_ensemble.iloc[0]["macro_f1"] < 1.0


def test_whether_each_ensemble_gets_its_own_row_and_size():
    """The sizing question needs a verdict per ensemble size, not one for the headline."""
    df = pd.concat(
        [
            pd.DataFrame(passages("Q1", 3, (3, 3, 1), (0, 1, 2))).assign(
                ensemble="opus_n3", n_members=3
            ),
            pd.DataFrame(passages("Q1", 5, (5, 3, 1), (0, 1, 2))).assign(
                ensemble="opus_n5", n_members=5
            ),
        ],
        ignore_index=True,
    )

    by_ensemble = automation_by_ensemble(policy_table_by_ensemble(df))

    assert list(by_ensemble["ensemble"]) == ["opus_n3", "opus_n5"]
    assert list(by_ensemble["n_members"]) == [3, 5]
    assert set(by_ensemble["family"]) == {"opus"}


def test_whether_the_agreement_curve_holds_only_the_thresholds_an_ensemble_can_produce():
    """
    A curve point has to be a point on the disagreement axis.

    The two directional unanimous policies both sit at disagreement 0, so they are a split
    of the leftmost point rather than points of their own — plotting them would put two
    extra values at the same x. What is left is one point per disagreement value the
    ensemble can actually produce: ``⌊n/2⌋ + 1`` of them, so two for n=3 and three for n=5.
    """
    df = pd.concat(
        [
            pd.DataFrame(passages("Q1", 3, (3, 3, 1), (1, 1, 2), (0, 1, 2))).assign(
                ensemble="opus_n3", n_members=3
            ),
            pd.DataFrame(
                passages("Q1", 5, (5, 3, 1), (4, 1, 1), (2, 1, 1), (0, 1, 2))
            ).assign(ensemble="opus_n5", n_members=5),
        ],
        ignore_index=True,
    )

    curves = agreement_f1_curves(policy_table_by_ensemble(df))

    assert list(curves["ensemble"]) == ["opus_n3"] * 2 + ["opus_n5"] * 3
    assert bool(curves["disagreement_threshold"].notna().all())

    for _, panel in curves.groupby("ensemble"):
        thresholds = panel["disagreement_threshold"].tolist()
        assert thresholds == sorted(thresholds)
        assert thresholds[0] == pytest.approx(0.0)
        # the rightmost point is the no-review baseline: every passage auto-labelled
        assert panel["macro_coverage"].tolist()[-1] == pytest.approx(1.0)
        assert panel["n_human_remaining"].tolist()[-1] == 0


def test_whether_a_single_classifier_has_no_routing_signal():
    """
    A one-member ensemble always agrees with itself, so `disagreement` is always 0.

    That makes its unanimous subset the whole eval set at 100% coverage — no passage can be
    routed to a human. Worth pinning: it is why n=1 reads as maximum coverage rather than as
    a confident ensemble.
    """
    df = pd.DataFrame(passages("Q1", 1, (1, 3, 1), (0, 1, 2)))

    assert (df["disagreement"] == 0).all()

    policies = policy_table(df)
    unanimous = policies.set_index("policy").loc["disagreement <= 0.000"]
    assert unanimous["macro_coverage"] == pytest.approx(1.0)
    assert unanimous["n_human_remaining"] == 0


def test_whether_the_sizing_table_is_sorted_by_f1_descending():
    """The ensemble to beat should be the top row, not whichever family sorts first."""
    df = pd.concat(
        [
            pd.DataFrame(passages("Q1", 3, (3, 1, 3), (0, 1, 2))).assign(
                ensemble="opus_n3", n_members=3
            ),
            pd.DataFrame(passages("Q1", 3, (3, 3, 0), (0, 0, 3))).assign(
                ensemble="gemini_n3", n_members=3
            ),
        ],
        ignore_index=True,
    )

    by_ensemble = automation_by_ensemble(policy_table_by_ensemble(df))

    scores = by_ensemble["macro_f1"].tolist()
    assert scores == sorted(scores, reverse=True)
    # gemini's subset is perfect, opus' is not — so gemini leads despite sorting later
    assert by_ensemble.iloc[0]["ensemble"] == "gemini_n3"


def negative_side_rows(*concepts: tuple[str, int, int, int, int]) -> pd.DataFrame:
    """
    Build a per-concept metrics frame from hand-written confusion matrices.

    Each concept is ``(id, TP, FP, TN, FN)``. Goes through `metrics_row` rather than
    writing the columns out, so the test breaks if the row's shape drifts from what
    `negative_side_table` reads.
    """

    return pd.DataFrame(
        [
            metrics_row(
                ConfusionMatrix(
                    true_positives=tp,
                    false_positives=fp,
                    true_negatives=tn,
                    false_negatives=fn,
                ),
                concept=concept,
                ensemble="opus_n3",
                n_members=3,
                level=PASSAGE_LEVEL,
            )
            for concept, tp, fp, tn, fn in concepts
        ]
    )


def test_whether_predicted_negative_counts_match_the_confusion_matrix():
    """
    The "how many passages get no span" answer has to be every negative prediction.

    Reading it off `true_negatives` alone would quietly drop the missed mentions, which
    are exactly the passages the column exists to warn about.
    """
    per_concept = negative_side_rows(("Q1", 4, 2, 10, 4))

    table = negative_side_table(per_concept, PASSAGE_LEVEL, "opus_n3")
    concept = table.iloc[0]

    assert concept["predicted_negative"] == 14
    assert concept["negative_share"] == pytest.approx(14 / 20)
    assert concept["npv"] == pytest.approx(10 / 14)
    assert concept["missed_mentions"] == 4
    assert concept["specificity"] == pytest.approx(10 / 12)


def test_whether_npv_is_undefined_when_nothing_is_labelled_negative():
    """
    A concept the ensemble never calls negative has no negative labels to score.

    `ConfusionMatrix.negative_predictive_value` returns 0 rather than raising on an empty
    denominator, so without a guard this reads as a scored, maximally untrustworthy 0.000
    — and would then drag the macro average down.
    """
    per_concept = negative_side_rows(("Q1", 6, 4, 0, 0), ("Q2", 3, 1, 8, 2))

    table = negative_side_table(per_concept, PASSAGE_LEVEL, "opus_n3").set_index(
        "concept"
    )

    assert table.loc["Q1", "predicted_negative"] == 0
    assert pd.isna(table.loc["Q1", "npv"])
    # the macro row averages only the concept that has negatives to score
    assert table.iloc[-1]["npv"] == pytest.approx(8 / 10)


def test_whether_the_negative_side_is_scored_for_every_policy():
    """
    `unanimous negative only` has no F1 by construction, so NPV is all it can be judged on.

    Its subset is 2 gold-positive passages among 6 labelled negative, so two thirds of the
    passages it hands back as empty are genuinely empty.
    """
    df = pd.DataFrame(passages("Q1", 3, (3, 2, 1), (0, 2, 4)))

    negatives = policy_table(df).set_index("policy").loc[UNANIMOUS_NEGATIVE_POLICY]

    assert pd.isna(negatives["macro_f1"])
    assert negatives["macro_npv"] == pytest.approx(4 / 6)


def span_at(text: str) -> Span:
    """A span over the first word of a passage, which is all the metrics need."""
    return Span(
        text=text,
        start_index=0,
        end_index=7,
        concept_id=CONCEPT_ID,
        labellers=["stub"],
    )


def write_fake_cache(output_dir, concept, members, n_passages=12, n_gold_positive=6):
    """
    Write one concept's gold set plus a cached prediction file per named member.

    Members are deliberately not identical: every member calls the first
    ``n_gold_positive - 1`` passages positive, and only the odd seeds also call the last
    gold-positive one positive. That single split is what gives the ensembles a non-zero
    disagreement to route on — without it every policy would select every passage and the
    automation table would collapse to one row per ensemble with nothing to separate
    them.

    Members not listed get no file at all, which is how a concept the opus runs never
    reached is modelled.
    """

    concept_dir = ensembles.concept_output_dir(output_dir, concept)
    gold = [
        LabelledPassage(
            id=f"{concept}-p{index}",
            text=f"passage number {index} of {concept}",
            spans=[span_at(f"passage number {index} of {concept}")]
            if index < n_gold_positive
            else [],
        )
        for index in range(n_passages)
    ]
    ensembles.write_passages(concept_dir / "gold.jsonl", gold)

    for family, seed in members:
        predictions = [
            passage.model_copy(
                update={
                    "spans": [span_at(passage.text)]
                    if index < n_gold_positive - 1 + seed % 2
                    else []
                },
                deep=True,
            )
            for index, passage in enumerate(gold)
        ]
        ensembles.write_passages(
            ensembles.member_path(concept_dir, (family, seed)), predictions
        )


def write_three_concepts(tmp_path):
    """
    Two concepts every member covers, and one the opus run didn't finish.

    The shape the credit-limited run left behind, and the reason `analyse` needs a common
    concept set: `Q3` can be scored by every ensemble except the ones needing opus
    seeds 4 and 5.
    """

    every_member = ensembles.all_members()
    write_fake_cache(tmp_path, "Q1", every_member)
    write_fake_cache(tmp_path, "Q2", every_member)
    write_fake_cache(
        tmp_path,
        "Q3",
        [member for member in every_member if member not in {("opus", 4), ("opus", 5)}],
    )


def test_whether_every_ensemble_is_scored_on_the_same_concepts(tmp_path):
    """
    The comparison the sizing table exists for: every row over one common concept set.

    `Q3` has only three of opus' five seeds, so under per-ensemble skipping `opus_n5`
    was macro-averaged over two concepts while everything else got three — and a mean over
    two concepts cannot be compared with a mean over three, because the smaller row may
    simply have drawn the easier ones. Every row must now report the same `n_concepts`.
    """
    write_three_concepts(tmp_path)

    cli.analyse(concepts="Q1,Q2,Q3", output_dir=tmp_path, max_members=5)

    by_ensemble = pd.read_csv(tmp_path / "automation_by_ensemble.csv")
    # all eight ensembles under a cap of 5, and every one of them on Q1 and Q2 alone
    assert len(by_ensemble) == len(ensembles.compose_possible_ensembles(5))
    assert set(by_ensemble["n_concepts"]) == {2}

    per_concept = pd.read_csv(tmp_path / "per_concept_metrics.csv")
    assert set(per_concept["concept"]) == {"Q1", "Q2"}


def test_whether_every_ensemble_is_scored_on_the_same_passages(tmp_path):
    """
    The same bug one level down, at passage rather than concept granularity.

    `align_passages` drops a passage any member of *that ensemble* lacks, so one failed
    call put two ensembles of the same concept on different passages — `gemini_n3` on
    all 12 and anything containing the failed member on 11. Restricting gold to the
    passages every member holds makes it 11 for all of them, which is the only way the
    supports line up.
    """
    write_fake_cache(tmp_path, "Q1", ensembles.all_members())

    # one call failed for opus seed 5, the way a real interrupted run leaves things
    concept_dir = tmp_path / "Q1"
    cached = ensembles.read_passages(ensembles.member_path(concept_dir, ("opus", 5)))
    ensembles.write_passages(
        ensembles.member_path(concept_dir, ("opus", 5)), cached[:-1]
    )

    cli.analyse(concepts="Q1", output_dir=tmp_path, max_members=5)

    per_concept = pd.read_csv(tmp_path / "per_concept_metrics.csv")
    passage_rows = per_concept[per_concept["level"] == PASSAGE_LEVEL]
    # 12 gold passages, 1 held back by the failed call, so 11 for every ensemble
    assert set(passage_rows["support"]) == {11}

    completeness = pd.read_csv(tmp_path / "data_completeness.csv").set_index("concept")
    assert completeness.loc["Q1", "gold_passages"] == 12
    assert completeness.loc["Q1", "scored_passages"] == 11
    assert not completeness.loc["Q1", "excluded"]


def test_whether_an_excluded_concept_names_the_members_it_is_missing(tmp_path):
    """
    A dropped concept has to be loud in `data_completeness.csv`.

    Silently scoring nine concepts where ten were asked for is the failure this whole
    filter is correcting, so the exclusion and its reason have to be readable without
    re-running the command and watching the console.
    """
    write_three_concepts(tmp_path)

    cli.analyse(concepts="Q1,Q2,Q3", output_dir=tmp_path, max_members=5)

    completeness = pd.read_csv(tmp_path / "data_completeness.csv").set_index("concept")
    assert list(completeness.index) == ["Q1", "Q2", "Q3"]
    assert not completeness.loc["Q1", "excluded"]
    assert completeness.loc["Q3", "excluded"]
    assert completeness.loc["Q3", "members_cached"] == 8
    assert completeness.loc["Q3", "members_expected"] == 10
    for missing in ("opus seed=4", "opus seed=5"):
        assert missing in completeness.loc["Q3", "missing_members"]
        assert missing in completeness.loc["Q3", "exclusion_reason"]


def test_whether_no_concept_with_every_member_fails_rather_than_scoring_nothing(
    tmp_path,
):
    """
    With one family missing everywhere there is no common set, and no honest table.

    Falling through to an empty analysis would write out tables that look like a finding,
    so this exits the same way the "no cached predictions" path does — pointing at
    `predict` as the fix.
    """
    write_fake_cache(tmp_path, "Q1", [("gemini", seed) for seed in (1, 2, 3, 4, 5)])

    with pytest.raises(typer.Exit):
        cli.analyse(concepts="Q1", output_dir=tmp_path, max_members=5)


def test_whether_a_misspelled_headline_fails_instead_of_being_absorbed(tmp_path):
    """
    A name that isn't an ensemble at all is a typo, and gets its own message.

    Distinct from a real ensemble the size cap excludes: both exit, but a typo means
    the whole ensemble list is worth printing, whereas a capped-out request only needs the
    sizes currently on offer.
    """
    write_fake_cache(tmp_path, "Q1", ensembles.all_members())

    with pytest.raises(typer.Exit):
        cli.analyse(
            concepts="Q1",
            output_dir=tmp_path,
            max_members=5,
            headline_ensemble="mixed_n4",
        )


def test_whether_a_capped_out_headline_fails_rather_than_substituting(tmp_path):
    """
    An unmeetable headline is the caller's to resolve, not the script's to work around.

    `mixed_n9` is an ensemble this experiment can build, so asking for it under a cap of
    5 is a legitimate request that simply can't be met. Substituting the nearest available
    ensemble would put a different one under the detailed tables than the one named on the
    command line, so this exits and says which sizes are on offer instead.
    """
    write_fake_cache(tmp_path, "Q1", ensembles.all_members())

    with pytest.raises(typer.Exit):
        cli.analyse(
            concepts="Q1",
            output_dir=tmp_path,
            max_members=5,
            headline_ensemble="mixed_n9",
        )

    # it fails before any scoring, so nothing half-finished is left behind to be read
    assert not (tmp_path / "vote_splits.csv").exists()


# a deliberately round pair of prices, so every cost below can be checked by hand. The
# real ones are whatever OpenRouter is charging today, which is not something a test
# should be pinned to.
OPUS_PRICES = TokenPrices(prompt=1e-6, completion=1e-5)
GEMINI_PRICES = TokenPrices(prompt=2e-6, completion=2e-5)

PER_PASSAGE_OPUS = (
    ESTIMATED_PROMPT_TOKENS_PER_PASSAGE * 1e-6
    + ESTIMATED_COMPLETION_TOKENS_PER_PASSAGE * 1e-5
)
PER_PASSAGE_GEMINI = 2 * PER_PASSAGE_OPUS


def ensemble(name: str, *members: tuple[str, int]) -> NamedEnsemble:
    """A named ensemble from bare (family, seed) pairs."""
    return NamedEnsemble(name=name, members=tuple(members))


@pytest.fixture(autouse=True)
def offline_pricing(monkeypatch):
    """
    Keep the whole module off the network, including the end-to-end `analyse` tests.

    `analyse` fetches OpenRouter's price list the first time it runs against an output
    directory, and a unit test that reaches the internet is slow, flaky, and quietly
    dependent on what a model costs today. The tests that are *about* pricing set their
    own stub, which runs after this one and wins.
    """
    monkeypatch.setattr(
        ensembles,
        "fetch_model_pricing",
        lambda: {FAMILY_A: OPUS_PRICES, FAMILY_B: GEMINI_PRICES},
    )


def cost_rows(*concepts: tuple[str, int, int]) -> pd.DataFrame:
    """
    Per-concept metric rows at passage and span level, from ``(id, passages, gold spans)``.

    Goes through `metrics_row` so the frame has the shape `headline_table` really reads.
    Only `support` matters here, and it deliberately differs between the two levels: at
    passage level it counts passages, at span level it counts gold spans.
    """

    rows = []
    for concept, passages, gold_spans in concepts:
        for level, support in ((PASSAGE_LEVEL, passages), ("span@0", gold_spans)):
            rows.append(
                metrics_row(
                    ConfusionMatrix(
                        true_positives=1,
                        false_positives=0,
                        true_negatives=support - 1,
                        false_negatives=0,
                    ),
                    concept=concept,
                    ensemble="mixed_n5",
                    n_members=5,
                    level=level,
                )
            )
    return pd.DataFrame(rows)


def test_whether_an_unknown_cost_renders_as_not_applicable_rather_than_zero():
    """
    The fail-soft path, end to end: no price for a family means no number, not $0.00.

    A zero cost and an unknown one are opposite claims about an ensemble, and the whole
    reason `NOT_APPLICABLE` exists in this module is that a sentinel 0 reads as a measured
    result.
    """
    unpriced = ensemble_cost_per_passage(
        ensemble("mixed_n3", (FAMILY_A, 1), (FAMILY_A, 2), (FAMILY_B, 1)),
        {FAMILY_A: OPUS_PRICES},
    )

    table = headline_table(
        cost_rows(("Q1", 100, 40)), PASSAGE_LEVEL, "mixed_n5", cost_per_passage=unpriced
    )
    rendered = format_for_display(table)

    assert list(rendered[COST_COLUMN]) == [NOT_APPLICABLE, NOT_APPLICABLE]


def test_whether_a_failed_price_fetch_leaves_the_analysis_runnable(
    monkeypatch, tmp_path
):
    """
    `analyse` has to survive being run offline with no cached prices.

    It is the re-runnable, no-LLM-calls half of this experiment; a costing feature that
    could make it exit is a worse trade than losing the column.
    """

    def fail():
        raise RuntimeError("no network")

    monkeypatch.setattr(ensembles, "fetch_model_pricing", fail)

    assert load_model_pricing(tmp_path) == {}
    assert not (tmp_path / OPENROUTER_PRICING_CACHE_FILENAME).exists()
