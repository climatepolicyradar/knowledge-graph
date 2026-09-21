import io
import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import SecretStr

from flows.vibe_check import process_single_concept, vibe_check_inference
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelling import ArgillaConfig
from knowledge_graph.wikibase import WikibaseConfig

N_PASSAGES = 10
EMBEDDING_DIM = 384


def _make_embedding_batch(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    embeddings = rng.random((n, dim), dtype=np.float32)
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def _make_single_embedding(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    embedding = rng.random(dim).astype(np.float32)
    return embedding / np.linalg.norm(embedding)


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


@pytest.fixture(autouse=True)
def no_retry_delay():
    """Keep the retries in `process_single_concept` from slowing the tests down."""
    with patch.object(process_single_concept, "retry_delay_seconds", 0):
        yield


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
    embeddings = _make_embedding_batch(N_PASSAGES, EMBEDDING_DIM, seed=42)

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
        mock_st.encode.return_value = _make_single_embedding(EMBEDDING_DIM, seed=43)
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
async def test_vibe_check_fails_the_run_when_concept_not_found(
    vibe_check_externals,
):
    vibe_check_externals.wikibase_session.get_concept_async = AsyncMock(
        side_effect=ValueError("concept not found")
    )

    with pytest.raises(RuntimeError, match="1/1 concepts failed to process: Q1"):
        await vibe_check_inference(wikibase_ids=["Q1"])

    # Nothing should be uploaded when the concept can't be loaded.
    assert vibe_check_externals.push_to_s3.call_count == 0


@pytest.mark.asyncio
async def test_vibe_check_fails_the_run_when_s3_upload_fails(
    vibe_check_externals,
):
    vibe_check_externals.push_to_s3.side_effect = RuntimeError("s3 upload failed")

    with pytest.raises(RuntimeError, match="1/1 concepts failed to process: Q1"):
        await vibe_check_inference(wikibase_ids=["Q1"])


@pytest.mark.asyncio
async def test_vibe_check_retries_a_failing_concept(vibe_check_externals):
    """A concept that fails is retried before the run is marked as failed."""
    get_concept = AsyncMock(side_effect=ValueError("transient failure"))
    vibe_check_externals.wikibase_session.get_concept_async = get_concept

    with pytest.raises(RuntimeError, match="1/1 concepts failed to process: Q1"):
        await vibe_check_inference(wikibase_ids=["Q1"])

    expected_attempts = 1 + (process_single_concept.retries or 0)
    assert get_concept.await_count == expected_attempts


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

    # The run fails because Q2 failed, but Q1 is still processed and uploaded
    with pytest.raises(RuntimeError, match="1/2 concepts failed to process: Q2"):
        await vibe_check_inference(wikibase_ids=["Q1", "Q2"])

    assert vibe_check_externals.push_to_s3.call_count == 3


VIBE_CHECK_BUCKET_NAME = "test-vibe-checker-bucket"


@pytest.fixture
def vibe_check_s3_ssm_environment(mock_s3_client, mock_ssm_client) -> None:
    """Seeds a real (moto) S3 bucket + SSM param with the files the flow reads for real."""
    mock_s3_client.create_bucket(
        Bucket=VIBE_CHECK_BUCKET_NAME,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
    )
    mock_ssm_client.put_parameter(
        Name="/vibe-checker/bucket-name",
        Value=VIBE_CHECK_BUCKET_NAME,
        Type="String",
    )

    passages_df = _make_passages_df(N_PASSAGES)
    feather_buffer = io.BytesIO()
    passages_df.to_feather(feather_buffer)
    mock_s3_client.put_object(
        Bucket=VIBE_CHECK_BUCKET_NAME,
        Key="passages_dataset.feather",
        Body=feather_buffer.getvalue(),
    )

    embeddings = _make_embedding_batch(N_PASSAGES, EMBEDDING_DIM, seed=42)
    embeddings_buffer = io.BytesIO()
    np.save(embeddings_buffer, embeddings)
    mock_s3_client.put_object(
        Bucket=VIBE_CHECK_BUCKET_NAME,
        Key="passages_embeddings.npy",
        Body=embeddings_buffer.getvalue(),
    )

    mock_s3_client.put_object(
        Bucket=VIBE_CHECK_BUCKET_NAME,
        Key="passages_embeddings_metadata.json",
        Body=json.dumps({"embedding_model_name": "test-model"}).encode("utf-8"),
    )


@dataclass
class VibeCheckEndToEndExternals:
    """Handles to the mocks left in the end-to-end test: real third-party services only."""

    wikibase_session: MagicMock
    classifier: MagicMock


@pytest.fixture
def vibe_check_end_to_end(vibe_check_s3_ssm_environment, test_config):
    mock_classifier = MagicMock()
    mock_classifier.id = "test_classifier_id"
    mock_classifier.predict.return_value = [[] for _ in range(N_PASSAGES)]

    with (
        patch(
            "flows.vibe_check.Config.create",
            new_callable=AsyncMock,
            return_value=test_config,
        ),
        # wandb is a genuine external service test_config would otherwise log into for real.
        patch("wandb.login"),
        patch("flows.vibe_check.WikibaseSession") as mock_wikibase_cls,
        patch("flows.vibe_check.SentenceTransformer") as mock_st_cls,
        patch(
            "flows.vibe_check.run_training",
            new_callable=AsyncMock,
            return_value=mock_classifier,
        ),
    ):
        mock_wikibase = MagicMock()
        mock_wikibase_cls.return_value = mock_wikibase

        mock_st = MagicMock()
        mock_st.encode.return_value = _make_single_embedding(EMBEDDING_DIM, seed=123)
        mock_st_cls.return_value = mock_st

        yield VibeCheckEndToEndExternals(
            wikibase_session=mock_wikibase,
            classifier=mock_classifier,
        )


@pytest.mark.asyncio
async def test_vibe_check_inference_end_to_end_isolates_failures_across_concepts(
    vibe_check_end_to_end,
    mock_s3_client,
):
    """Real S3/SSM wiring end-to-end: confirms Q2 failing doesn't block Q1's real S3 writes."""

    def _get_concept(wikibase_id):
        if wikibase_id == WikibaseID("Q2"):
            raise ValueError("Q2 not found")
        return Concept(wikibase_id=wikibase_id, preferred_label="ok concept")

    vibe_check_end_to_end.wikibase_session.get_concept_async = AsyncMock(
        side_effect=_get_concept
    )

    with pytest.raises(RuntimeError, match="1/2 concepts failed to process: Q2"):
        await vibe_check_inference(wikibase_ids=["Q1", "Q2"])

    objects = mock_s3_client.list_objects_v2(Bucket=VIBE_CHECK_BUCKET_NAME)
    keys = {obj["Key"] for obj in objects.get("Contents", [])}

    # Q1 succeeded, and its outputs were really written to S3...
    q1_prefix = f"Q1/{vibe_check_end_to_end.classifier.id}/"
    assert f"{q1_prefix}predictions.jsonl" in keys
    assert f"{q1_prefix}concept.json" in keys
    assert f"{q1_prefix}classifier.json" in keys

    # ...but Q2 failed before it got anywhere near S3.
    assert not any(key.startswith("Q2/") for key in keys)
