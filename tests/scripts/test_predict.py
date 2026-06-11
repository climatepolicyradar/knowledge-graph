from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest

from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from scripts.predict import deduplicate_labelled_passages, run_prediction


def _passage(text: str, spans=None) -> LabelledPassage:
    return LabelledPassage(text=text, spans=spans or [])


def _make_mock_classifier(classifier_id: str = "clf-1") -> Mock:
    clf = Mock()
    clf.id = classifier_id
    clf.concept = Mock()
    clf.set_prediction_threshold = Mock(return_value=clf)
    return clf


def _label_passthrough(**kw):
    """Side-effect for label_passages_with_classifier that returns passages unchanged."""
    return list(kw["labelled_passages"])


def test_deduplicate_returns_empty_for_empty_list():
    assert deduplicate_labelled_passages([]) == []


def test_deduplicate_removes_exact_duplicates():
    passages = [_passage("hello"), _passage("hello"), _passage("world")]
    result = deduplicate_labelled_passages(passages)
    assert len(result) == 2
    assert {p.text for p in result} == {"hello", "world"}


def test_deduplicate_preserves_first_occurrence():
    p1 = _passage("same text")
    p2 = _passage("same text")
    result = deduplicate_labelled_passages([p1, p2])
    assert result[0] is p1


def test_deduplicate_preserves_order_of_unique_passages():
    texts = ["c", "a", "b"]
    passages = [_passage(t) for t in texts]
    result = deduplicate_labelled_passages(passages)
    assert [p.text for p in result] == texts


def test_deduplicate_all_unique_returns_all():
    passages = [_passage(f"text {i}") for i in range(5)]
    assert len(deduplicate_labelled_passages(passages)) == 5


@pytest.mark.asyncio
async def test_run_prediction_saves_output_jsonl(tmp_path):
    passages = [_passage(f"text {i}") for i in range(3)]
    mock_clf = _make_mock_classifier()

    with (
        patch("scripts.predict.wandb.init", return_value=nullcontext()),
        patch("scripts.predict.wandb.Api"),
        patch("scripts.predict.get_s3_client"),
        patch("scripts.predict.load_classifier_from_wandb", return_value=mock_clf),
        patch(
            "scripts.predict.label_passages_with_classifier",
            side_effect=_label_passthrough,
        ),
        patch("scripts.predict.predictions_dir", tmp_path),
    ):
        await run_prediction(
            wikibase_id=WikibaseID("Q787"),
            classifier_wandb_path="climatepolicyradar/Q787/abc123:v0",
            input_passages=passages,
            track_and_upload=False,
        )

    output_files = list(tmp_path.rglob("*.jsonl"))
    assert len(output_files) == 1


@pytest.mark.asyncio
async def test_run_prediction_deduplicates_when_flag_set(tmp_path):
    passages = [_passage("dup"), _passage("dup"), _passage("unique")]
    mock_clf = _make_mock_classifier()
    labelled_calls = []

    def capture_label(**kw):
        lp = list(kw["labelled_passages"])
        labelled_calls.extend(lp)
        return lp

    with (
        patch("scripts.predict.wandb.init", return_value=nullcontext()),
        patch("scripts.predict.wandb.Api"),
        patch("scripts.predict.get_s3_client"),
        patch("scripts.predict.load_classifier_from_wandb", return_value=mock_clf),
        patch(
            "scripts.predict.label_passages_with_classifier", side_effect=capture_label
        ),
        patch("scripts.predict.predictions_dir", tmp_path),
    ):
        await run_prediction(
            wikibase_id=WikibaseID("Q787"),
            classifier_wandb_path="climatepolicyradar/Q787/abc123:v0",
            input_passages=passages,
            track_and_upload=False,
            deduplicate_inputs=True,
        )

    texts_seen = [p.text for p in labelled_calls]
    assert texts_seen.count("dup") == 1
    assert "unique" in texts_seen


@pytest.mark.asyncio
async def test_run_prediction_skips_deduplication_when_flag_unset(tmp_path):
    passages = [_passage("dup"), _passage("dup")]
    mock_clf = _make_mock_classifier()
    labelled_calls = []

    def capture_label(**kw):
        lp = list(kw["labelled_passages"])
        labelled_calls.extend(lp)
        return lp

    with (
        patch("scripts.predict.wandb.init", return_value=nullcontext()),
        patch("scripts.predict.wandb.Api"),
        patch("scripts.predict.get_s3_client"),
        patch("scripts.predict.load_classifier_from_wandb", return_value=mock_clf),
        patch(
            "scripts.predict.label_passages_with_classifier", side_effect=capture_label
        ),
        patch("scripts.predict.predictions_dir", tmp_path),
    ):
        await run_prediction(
            wikibase_id=WikibaseID("Q787"),
            classifier_wandb_path="climatepolicyradar/Q787/abc123:v0",
            input_passages=passages,
            track_and_upload=False,
            deduplicate_inputs=False,
        )

    assert len(labelled_calls) == 2


@pytest.mark.asyncio
async def test_run_prediction_respects_limit(tmp_path):
    passages = [_passage(f"text {i}") for i in range(20)]
    mock_clf = _make_mock_classifier()
    labelled_calls = []

    def capture_label(**kw):
        lp = list(kw["labelled_passages"])
        labelled_calls.extend(lp)
        return lp

    with (
        patch("scripts.predict.wandb.init", return_value=nullcontext()),
        patch("scripts.predict.wandb.Api"),
        patch("scripts.predict.get_s3_client"),
        patch("scripts.predict.load_classifier_from_wandb", return_value=mock_clf),
        patch(
            "scripts.predict.label_passages_with_classifier", side_effect=capture_label
        ),
        patch("scripts.predict.predictions_dir", tmp_path),
    ):
        await run_prediction(
            wikibase_id=WikibaseID("Q787"),
            classifier_wandb_path="climatepolicyradar/Q787/abc123:v0",
            input_passages=passages,
            track_and_upload=False,
            limit=5,
        )

    assert len(labelled_calls) == 5


@pytest.mark.asyncio
async def test_run_prediction_sets_threshold_when_provided(tmp_path):
    mock_clf = _make_mock_classifier()

    with (
        patch("scripts.predict.wandb.init", return_value=nullcontext()),
        patch("scripts.predict.wandb.Api"),
        patch("scripts.predict.get_s3_client"),
        patch("scripts.predict.load_classifier_from_wandb", return_value=mock_clf),
        patch(
            "scripts.predict.label_passages_with_classifier",
            side_effect=_label_passthrough,
        ),
        patch("scripts.predict.predictions_dir", tmp_path),
    ):
        await run_prediction(
            wikibase_id=WikibaseID("Q787"),
            classifier_wandb_path="climatepolicyradar/Q787/abc123:v0",
            input_passages=[_passage("hello")],
            track_and_upload=False,
            prediction_threshold=0.9,
        )

    mock_clf.set_prediction_threshold.assert_called_once_with(0.9)


@pytest.mark.asyncio
async def test_run_prediction_raises_when_no_passage_source_provided(tmp_path):
    mock_clf = _make_mock_classifier()

    with (
        patch("scripts.predict.wandb.init", return_value=nullcontext()),
        patch("scripts.predict.wandb.Api"),
        patch("scripts.predict.get_s3_client"),
        patch("scripts.predict.load_classifier_from_wandb", return_value=mock_clf),
        patch(
            "scripts.predict.label_passages_with_classifier",
            side_effect=_label_passthrough,
        ),
        patch("scripts.predict.predictions_dir", tmp_path),
    ):
        with pytest.raises(ValueError, match="must be provided"):
            await run_prediction(
                wikibase_id=WikibaseID("Q787"),
                classifier_wandb_path="climatepolicyradar/Q787/abc123:v0",
                track_and_upload=False,
            )


@pytest.mark.asyncio
async def test_run_prediction_raises_when_both_path_sources_provided(tmp_path):
    mock_clf = _make_mock_classifier()

    with (
        patch("scripts.predict.wandb.init", return_value=nullcontext()),
        patch("scripts.predict.wandb.Api"),
        patch("scripts.predict.get_s3_client"),
        patch("scripts.predict.load_classifier_from_wandb", return_value=mock_clf),
        patch(
            "scripts.predict.label_passages_with_classifier",
            side_effect=_label_passthrough,
        ),
        patch("scripts.predict.predictions_dir", tmp_path),
    ):
        with pytest.raises(ValueError, match="cannot be defined"):
            await run_prediction(
                wikibase_id=WikibaseID("Q787"),
                classifier_wandb_path="climatepolicyradar/Q787/abc123:v0",
                labelled_passages_path=tmp_path / "some.jsonl",
                labelled_passages_wandb_path="climatepolicyradar/Q787/some-artifact:v0",
                track_and_upload=False,
            )
