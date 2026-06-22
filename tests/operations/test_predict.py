from contextlib import nullcontext
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.operations.predict import (
    deduplicate_labelled_passages,
    load_passages_from_snowflake,
    run_prediction,
)


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
        patch(
            "knowledge_graph.operations.predict.wandb.init", return_value=nullcontext()
        ),
        patch("knowledge_graph.operations.predict.wandb.Api"),
        patch("knowledge_graph.operations.predict.get_s3_client"),
        patch(
            "knowledge_graph.operations.predict.load_classifier_from_wandb",
            return_value=mock_clf,
        ),
        patch(
            "knowledge_graph.operations.predict.label_passages_with_classifier",
            side_effect=_label_passthrough,
        ),
        patch("knowledge_graph.operations.predict.predictions_dir", tmp_path),
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
        patch(
            "knowledge_graph.operations.predict.wandb.init", return_value=nullcontext()
        ),
        patch("knowledge_graph.operations.predict.wandb.Api"),
        patch("knowledge_graph.operations.predict.get_s3_client"),
        patch(
            "knowledge_graph.operations.predict.load_classifier_from_wandb",
            return_value=mock_clf,
        ),
        patch(
            "knowledge_graph.operations.predict.label_passages_with_classifier",
            side_effect=capture_label,
        ),
        patch("knowledge_graph.operations.predict.predictions_dir", tmp_path),
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
        patch(
            "knowledge_graph.operations.predict.wandb.init", return_value=nullcontext()
        ),
        patch("knowledge_graph.operations.predict.wandb.Api"),
        patch("knowledge_graph.operations.predict.get_s3_client"),
        patch(
            "knowledge_graph.operations.predict.load_classifier_from_wandb",
            return_value=mock_clf,
        ),
        patch(
            "knowledge_graph.operations.predict.label_passages_with_classifier",
            side_effect=capture_label,
        ),
        patch("knowledge_graph.operations.predict.predictions_dir", tmp_path),
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
        patch(
            "knowledge_graph.operations.predict.wandb.init", return_value=nullcontext()
        ),
        patch("knowledge_graph.operations.predict.wandb.Api"),
        patch("knowledge_graph.operations.predict.get_s3_client"),
        patch(
            "knowledge_graph.operations.predict.load_classifier_from_wandb",
            return_value=mock_clf,
        ),
        patch(
            "knowledge_graph.operations.predict.label_passages_with_classifier",
            side_effect=capture_label,
        ),
        patch("knowledge_graph.operations.predict.predictions_dir", tmp_path),
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
        patch(
            "knowledge_graph.operations.predict.wandb.init", return_value=nullcontext()
        ),
        patch("knowledge_graph.operations.predict.wandb.Api"),
        patch("knowledge_graph.operations.predict.get_s3_client"),
        patch(
            "knowledge_graph.operations.predict.load_classifier_from_wandb",
            return_value=mock_clf,
        ),
        patch(
            "knowledge_graph.operations.predict.label_passages_with_classifier",
            side_effect=_label_passthrough,
        ),
        patch("knowledge_graph.operations.predict.predictions_dir", tmp_path),
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
        patch(
            "knowledge_graph.operations.predict.wandb.init", return_value=nullcontext()
        ),
        patch("knowledge_graph.operations.predict.wandb.Api"),
        patch("knowledge_graph.operations.predict.get_s3_client"),
        patch(
            "knowledge_graph.operations.predict.load_classifier_from_wandb",
            return_value=mock_clf,
        ),
        patch(
            "knowledge_graph.operations.predict.label_passages_with_classifier",
            side_effect=_label_passthrough,
        ),
        patch("knowledge_graph.operations.predict.predictions_dir", tmp_path),
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
        patch(
            "knowledge_graph.operations.predict.wandb.init", return_value=nullcontext()
        ),
        patch("knowledge_graph.operations.predict.wandb.Api"),
        patch("knowledge_graph.operations.predict.get_s3_client"),
        patch(
            "knowledge_graph.operations.predict.load_classifier_from_wandb",
            return_value=mock_clf,
        ),
        patch(
            "knowledge_graph.operations.predict.label_passages_with_classifier",
            side_effect=_label_passthrough,
        ),
        patch("knowledge_graph.operations.predict.predictions_dir", tmp_path),
    ):
        with pytest.raises(ValueError, match="cannot be defined"):
            await run_prediction(
                wikibase_id=WikibaseID("Q787"),
                classifier_wandb_path="climatepolicyradar/Q787/abc123:v0",
                labelled_passages_path=tmp_path / "some.jsonl",
                labelled_passages_wandb_path="climatepolicyradar/Q787/some-artifact:v0",
                track_and_upload=False,
            )


def _make_fake_passages_df(n_rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TEXT_BLOCK_TEXT": [f"passage {i}" for i in range(n_rows)],
            "TEXT_BLOCK_TYPE": ["text"] * n_rows,
            "DOCUMENT_ID": [f"doc{i}" for i in range(n_rows)],
            "DOCUMENT_CONTENT_TYPE": ["Laws and Policies"] * n_rows,
            "DOCUMENT_NAME": [f"Doc {i}" for i in range(n_rows)],
            "DOCUMENT_SLUG": [f"doc-{i}" for i in range(n_rows)],
            "DOCUMENT_METADATA_TRANSLATED": [False] * n_rows,
            "DOCUMENT_METADATA_CORPUS_TYPE_NAME": ["Laws and Policies"] * n_rows,
            "DOCUMENT_METADATA_GEOGRAPHIES": ["[]"] * n_rows,
        }
    )


@pytest.fixture
def mock_passages_connection():
    """Patch the snowflake driver and PEM loader; yields the connect mock and cursor."""
    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = _make_fake_passages_df(n_rows=3)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    mock_private_key = MagicMock()
    mock_private_key.private_bytes.return_value = b"fake_der_bytes"

    with (
        patch("snowflake.connector.connect", return_value=mock_conn) as mock_connect,
        patch(
            "knowledge_graph.operations.snowflake.load_pem_private_key",
            return_value=mock_private_key,
        ),
    ):
        yield mock_connect, mock_cursor


def test_load_passages_returns_labelled_passages(mock_passages_connection):
    passages = load_passages_from_snowflake(["doc0", "doc1", "doc2"])

    assert len(passages) == 3
    assert all(isinstance(p, LabelledPassage) for p in passages)
    assert {p.text for p in passages} == {"passage 0", "passage 1", "passage 2"}
    # Metadata is carried through, with the passage text removed from it.
    assert passages[0].metadata["document_id"] == "doc0"
    assert "text_block.text" not in passages[0].metadata


def test_load_passages_parameterises_document_ids(mock_passages_connection):
    _, mock_cursor = mock_passages_connection
    document_ids = ["doc0", "doc1", "doc2"]

    load_passages_from_snowflake(document_ids)

    sql, params = mock_cursor.execute.call_args[0]
    # One %s placeholder per id, passed as query parameters (not interpolated).
    assert sql.count("%s") == len(document_ids)
    assert params == document_ids


def test_load_passages_uses_key_pair_when_credentials_provided(
    mock_passages_connection,
):
    mock_connect, _ = mock_passages_connection

    load_passages_from_snowflake(
        ["doc0"],
        snowflake_user="svc_user",
        snowflake_private_key="fake_pem_key",
        snowflake_account="test_account",
    )

    kwargs = mock_connect.call_args.kwargs
    assert "connection_name" not in kwargs
    assert kwargs["user"] == "svc_user"
    assert kwargs["account"] == "test_account"


def test_load_passages_falls_back_to_local_without_credentials(
    mock_passages_connection,
):
    mock_connect, _ = mock_passages_connection

    load_passages_from_snowflake(["doc0"])

    assert mock_connect.call_args.kwargs.get("connection_name") == "local_dev"
