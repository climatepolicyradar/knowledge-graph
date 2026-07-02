from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import SecretStr

from flows.vibe_check import vibe_check_inference
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelling import ArgillaConfig
from knowledge_graph.wikibase import WikibaseConfig

N_PASSAGES = 10


def _make_passages_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text_block.text": [f"passage {i}" for i in range(n)],
            "document_id": ["doc_1"] * n,
            "document_name": ["Test Document"] * n,
            "document_slug": ["test-document"] * n,
            "family_slug": ["test-family"] * n,
            "translated": [False] * n,
            "publication_ts": ["2024-01-01"] * n,
            "document_metadata.corpus_type_name": ["corpus"] * n,
            "world_bank_region": ["ECA"] * n,
        }
    )


@dataclass
class VibeCheckExternals:
    """Handles to mocked external services used by the vibe check flow."""

    push_to_s3: MagicMock
    wikibase_session: MagicMock
    classifier: MagicMock


@pytest.fixture
def vibe_check_externals(test_config):
    concept = Concept(
        wikibase_id=WikibaseID("Q1"),
        preferred_label="test concept",
    )

    mock_classifier = MagicMock()
    mock_classifier.id = "test_classifier_id"
    mock_classifier.predict.return_value = [[] for _ in range(N_PASSAGES)]

    passages_df = _make_passages_df(N_PASSAGES)

    embedding_dim = 384
    rng = np.random.default_rng(42)
    embeddings = rng.random((N_PASSAGES, embedding_dim), dtype=np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    metadata = {"embedding_model_name": "test-model"}

    wikibase_config = WikibaseConfig(
        username="test",
        password=SecretStr("test"),
        url="https://test.wikibase.test",
    )
    argilla_config = ArgillaConfig(
        api_key=SecretStr("test"),
        url="https://test.argilla.test",
    )

    mock_s3_client = MagicMock()

    with (
        patch(
            "flows.vibe_check.Config.create",
            new_callable=AsyncMock,
            return_value=test_config,
        ),
        patch(
            "flows.vibe_check._set_up_training_environment",
            new_callable=AsyncMock,
            return_value=(
                test_config,
                wikibase_config,
                argilla_config,
                mock_s3_client,
            ),
        ),
        patch("flows.vibe_check.load_passages_dataset", return_value=passages_df),
        patch("flows.vibe_check.load_embeddings", return_value=embeddings),
        patch("flows.vibe_check.load_embeddings_metadata", return_value=metadata),
        patch("flows.vibe_check.WikibaseSession") as mock_wikibase_cls,
        patch("flows.vibe_check.SentenceTransformer") as mock_st_cls,
        patch(
            "flows.vibe_check.run_training",
            new_callable=AsyncMock,
            return_value=mock_classifier,
        ),
        patch("flows.vibe_check.push_object_bytes_to_s3") as push_mock,
        patch(
            "flows.vibe_check.get_bucket_name_from_ssm",
            return_value="test-bucket",
        ),
    ):
        mock_wikibase = MagicMock()
        mock_wikibase.get_concept_async = AsyncMock(return_value=concept)
        mock_wikibase_cls.return_value = mock_wikibase

        mock_st = MagicMock()
        concept_embedding = rng.random(embedding_dim).astype(np.float32)
        concept_embedding = concept_embedding / np.linalg.norm(concept_embedding)
        mock_st.encode.return_value = concept_embedding
        mock_st_cls.return_value = mock_st

        yield VibeCheckExternals(
            push_to_s3=push_mock,
            wikibase_session=mock_wikibase,
            classifier=mock_classifier,
        )


@pytest.mark.asyncio
async def test_vibe_check_inference(vibe_check_externals):
    results = await vibe_check_inference(wikibase_ids=["Q1"])

    assert len(results) == 1
    assert results[0]["status"] == "success"
    assert results[0]["concept_id"] == WikibaseID("Q1")
    assert results[0]["n_passages"] == N_PASSAGES
    assert vibe_check_externals.push_to_s3.call_count == 3


@pytest.mark.asyncio
async def test_vibe_check_returns_failed_status_when_concept_not_found(
    vibe_check_externals,
):
    vibe_check_externals.wikibase_session.get_concept_async = AsyncMock(
        side_effect=ValueError("concept not found")
    )

    results = await vibe_check_inference(wikibase_ids=["Q1"])

    assert len(results) == 1
    assert results[0]["status"] == "failed"
    assert "concept not found" in results[0]["error"]
    # Nothing should be uploaded when the concept can't be loaded.
    assert vibe_check_externals.push_to_s3.call_count == 0


@pytest.mark.asyncio
async def test_vibe_check_returns_failed_status_when_s3_upload_fails(
    vibe_check_externals,
):
    vibe_check_externals.push_to_s3.side_effect = RuntimeError("s3 upload failed")

    results = await vibe_check_inference(wikibase_ids=["Q1"])

    assert len(results) == 1
    assert results[0]["status"] == "failed"
    assert "s3 upload failed" in results[0]["error"]


@pytest.mark.asyncio
async def test_vibe_check_isolates_failures_across_multiple_concepts(
    vibe_check_externals,
):
    def _get_concept(wikibase_id):
        if wikibase_id == WikibaseID("Q2"):
            raise ValueError("Q2 not found")
        return Concept(wikibase_id=wikibase_id, preferred_label="ok concept")

    vibe_check_externals.wikibase_session.get_concept_async = AsyncMock(
        side_effect=_get_concept
    )

    results = await vibe_check_inference(wikibase_ids=["Q1", "Q2"])

    by_id = {r["concept_id"]: r for r in results}
    assert by_id[WikibaseID("Q1")]["status"] == "success"
    assert by_id[WikibaseID("Q2")]["status"] == "failed"
    assert "Q2 not found" in by_id[WikibaseID("Q2")]["error"]
