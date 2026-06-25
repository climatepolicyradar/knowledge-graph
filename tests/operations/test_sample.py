from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from knowledge_graph.config import WANDB_ENTITY
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.operations.sample import run_sampling

CORPUS_TYPE_COL = "document_metadata.corpus_type_name"


def make_dataset(n=100, corpus_types=None):
    """Fake dataset with the columns that run_sampling expects."""
    return pd.DataFrame(
        {
            "text_block.text": [f"passage {i}" for i in range(n)],
            CORPUS_TYPE_COL: corpus_types
            or (["Laws and Policies"] * (n // 2) + ["Litigation"] * (n // 2)),
            "translated": [False] * n,
            "world_bank_region": ["Europe and Central Asia"] * n,
        }
    )


def _make_classifier(name: str) -> Mock:
    c = Mock()
    c.name = name
    c.id = f"{name}-id"
    c.fit.return_value = None
    c.set_prediction_threshold = Mock(return_value=c)
    c.predict = Mock(side_effect=lambda texts, **kwargs: [True] * len(texts))
    return c


@pytest.fixture
def patched_sample(tmp_path):
    """Patch all external dependencies so run_sampling can be called in unit tests."""
    mock_concept = Mock()
    mock_concept.id = "5d4xcy5g"
    mock_concept.wikibase_id = "Q787"

    mock_session = Mock()
    mock_session.get_concept.return_value = mock_concept

    kw = _make_classifier("KeywordClassifier")
    emb = _make_classifier("EmbeddingClassifier")

    with (
        patch(
            "knowledge_graph.operations.sample.WikibaseSession",
            return_value=mock_session,
        ),
        patch("knowledge_graph.operations.sample.KeywordClassifier", return_value=kw),
        patch(
            "knowledge_graph.operations.sample.EmbeddingClassifier", return_value=emb
        ),
        patch(
            "knowledge_graph.operations.sample.create_balanced_sample",
            side_effect=lambda df, sample_size, on_columns: df.head(sample_size),
        ),
        patch("knowledge_graph.operations.sample.processed_data_dir", tmp_path),
    ):
        yield mock_concept, mock_session, kw, emb


def test_run_sampling_include_filter_reduces_passages_passed_to_classifier(
    patched_sample,
):
    _, _, kw, _ = patched_sample
    dataset = make_dataset(n=100)  # 50 "Laws and Policies", 50 "Litigation"

    run_sampling(
        wikibase_id=WikibaseID("Q787"),
        dataset=dataset,
        corpus_types_include=["Laws and Policies"],
        track_and_upload=False,
    )

    texts_used = kw.predict.call_args[0][0]
    assert len(texts_used) == 50


def test_run_sampling_exclude_filter_removes_corpus_type_from_passages(patched_sample):
    _, _, kw, _ = patched_sample
    dataset = make_dataset(n=100)  # 50 of each

    run_sampling(
        wikibase_id=WikibaseID("Q787"),
        dataset=dataset,
        corpus_types_exclude=["Litigation"],
        track_and_upload=False,
    )

    texts_used = kw.predict.call_args[0][0]
    assert len(texts_used) == 50


def test_run_sampling_with_no_filter_uses_full_dataset(patched_sample):
    _, _, kw, _ = patched_sample
    dataset = make_dataset(n=80)

    run_sampling(
        wikibase_id=WikibaseID("Q787"),
        dataset=dataset,
        track_and_upload=False,
    )

    texts_used = kw.predict.call_args[0][0]
    assert len(texts_used) == 80


def test_run_sampling_truncates_dataset_to_max_size(patched_sample):
    _, _, kw, _ = patched_sample
    dataset = make_dataset(n=100)

    run_sampling(
        wikibase_id=WikibaseID("Q787"),
        dataset=dataset,
        max_size_to_sample_from=30,
        track_and_upload=False,
    )

    texts_used = kw.predict.call_args[0][0]
    assert len(texts_used) == 30


def test_run_sampling_does_not_truncate_when_dataset_is_smaller_than_max(
    patched_sample,
):
    _, _, kw, _ = patched_sample
    dataset = make_dataset(n=40)

    run_sampling(
        wikibase_id=WikibaseID("Q787"),
        dataset=dataset,
        max_size_to_sample_from=500_000,
        track_and_upload=False,
    )

    texts_used = kw.predict.call_args[0][0]
    assert len(texts_used) == 40


def test_run_sampling_applies_concept_override_to_attribute(patched_sample):
    mock_concept, _, _, _ = patched_sample
    mock_concept.description = "original"
    dataset = make_dataset(n=50)

    run_sampling(
        wikibase_id=WikibaseID("Q787"),
        dataset=dataset,
        concept_overrides={"description": "overridden"},
        track_and_upload=False,
    )

    assert mock_concept.description == "overridden"


def test_run_sampling_with_no_override_leaves_concept_unchanged(patched_sample):
    mock_concept, _, _, _ = patched_sample
    mock_concept.description = "original"
    dataset = make_dataset(n=50)

    run_sampling(
        wikibase_id=WikibaseID("Q787"),
        dataset=dataset,
        track_and_upload=False,
    )

    assert mock_concept.description == "original"


def test_run_sampling_returns_none_when_track_and_upload_is_false(patched_sample):
    result = run_sampling(
        wikibase_id=WikibaseID("Q787"),
        dataset=make_dataset(n=50),
        track_and_upload=False,
    )
    assert result is None


def test_run_sampling_returns_wandb_artifact_path_when_track_and_upload_is_true(
    patched_sample, tmp_path
):
    mock_concept, _, _, _ = patched_sample
    mock_concept.wikibase_id = WikibaseID("Q787")

    mock_artifact = Mock()
    mock_artifact.version = "v3"
    mock_artifact.wait = Mock()

    mock_run = MagicMock()
    mock_run.__enter__ = Mock(return_value=mock_run)
    mock_run.__exit__ = Mock(return_value=None)
    mock_run.summary = {}

    with (
        patch("wandb.init", return_value=mock_run),
        patch(
            "knowledge_graph.operations.sample.log_labelled_passages_artifact_to_wandb_run",
            return_value=mock_artifact,
        ),
    ):
        result = run_sampling(
            wikibase_id=WikibaseID("Q787"),
            dataset=make_dataset(n=50),
            track_and_upload=True,
        )

    assert result == f"{WANDB_ENTITY}/Q787/labelled-passages:v3"
