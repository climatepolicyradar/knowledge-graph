from io import BytesIO

import pytest
from botocore.exceptions import ClientError
from mypy_boto3_s3.client import S3Client

from flows.wikibase_to_s3 import (
    delete_from_s3,
    list_s3_concepts,
    upload_to_s3,
    wikibase_to_s3,
)
from src.concept import Concept
from src.identifiers import WikibaseID


def helper_get_concept_from_s3(
    mock_s3_client: S3Client, bucket_name: str, wikibase_id: str
) -> str:
    result = mock_s3_client.get_object(
        Bucket=bucket_name,
        Key=f"concepts/{wikibase_id}.json",
    )
    return result["Body"].read().decode("utf-8")


def helper_upload_concept_to_s3(
    mock_s3_client: S3Client, bucket_name: str, concept: Concept
):
    body = BytesIO(concept.model_dump_json().encode("utf-8"))
    key = f"concepts/{concept.wikibase_id}.json"
    mock_s3_client.put_object(
        Bucket=bucket_name, Key=key, Body=body, ContentType="application/json"
    )


def test_upload_to_s3(
    mock_s3_client, mock_cdn_bucket, test_wikibase_to_s3_config, mock_concepts
):
    mock_concept = mock_concepts[0]
    key = f"concepts/{mock_concept.wikibase_id}.json"

    # Ensure files are not there
    with pytest.raises(ClientError):
        mock_s3_client.head_object(Bucket=mock_cdn_bucket, Key=key)

    # Upload, now they should be there
    upload_to_s3(config=test_wikibase_to_s3_config, concept=mock_concept)
    assert mock_s3_client.head_object(Bucket=mock_cdn_bucket, Key=key)


def test_delete_from_s3(
    mock_s3_client, mock_cdn_bucket, test_wikibase_to_s3_config, mock_concepts
):
    mock_concept = mock_concepts[0]
    key = f"concepts/{mock_concept.wikibase_id}.json"

    # First upload the concept to S3
    helper_upload_concept_to_s3(mock_s3_client, mock_cdn_bucket, mock_concept)

    # Verify it exists
    assert mock_s3_client.head_object(Bucket=mock_cdn_bucket, Key=key)

    # Delete it
    delete_from_s3(
        config=test_wikibase_to_s3_config, concept_id=mock_concept.wikibase_id
    )

    # Verify it's gone
    with pytest.raises(ClientError):
        mock_s3_client.head_object(Bucket=mock_cdn_bucket, Key=key)


def test_list_s3_concepts(
    mock_s3_client, mock_cdn_bucket, mock_concepts, test_wikibase_to_s3_config
):
    for concept in mock_concepts:
        helper_upload_concept_to_s3(mock_s3_client, mock_cdn_bucket, concept)

    results = list_s3_concepts(config=test_wikibase_to_s3_config)
    assert set(results) == {
        "Q10",
        "Q20",
        "Q30",
    }


@pytest.mark.asyncio
async def test_wikibase_to_s3__empty_cdn_bucket(
    MockedWikibaseSession,
    mock_cdn_bucket,
    mock_prefect_slack_webhook,
    test_wikibase_to_s3_config,
    mock_s3_client,
):
    start = list_s3_concepts(config=test_wikibase_to_s3_config)
    assert len(start) == 0
    await wikibase_to_s3.fn(config=test_wikibase_to_s3_config)
    end = list_s3_concepts(config=test_wikibase_to_s3_config)
    assert len(end) == 10


@pytest.mark.asyncio
async def test_wikibase_to_s3__repeat_runs(
    MockedWikibaseSession,
    mock_cdn_bucket,
    mock_prefect_slack_webhook,
    test_wikibase_to_s3_config,
    mock_s3_client,
):
    start = list_s3_concepts(config=test_wikibase_to_s3_config)
    assert len(start) == 0
    await wikibase_to_s3.fn(config=test_wikibase_to_s3_config)
    await wikibase_to_s3.fn(config=test_wikibase_to_s3_config)
    end = list_s3_concepts(config=test_wikibase_to_s3_config)
    assert len(end) == 10


@pytest.mark.asyncio
async def test_wikibase_to_s3__overwrite_concept(
    MockedWikibaseSession,
    mock_prefect_slack_webhook,
    test_wikibase_to_s3_config,
    mock_s3_client,
    mock_concepts,
    mock_cdn_bucket,
):
    # Set up extra concept
    helper_upload_concept_to_s3(mock_s3_client, mock_cdn_bucket, mock_concepts[0])
    concept_before_overwrite = helper_get_concept_from_s3(
        mock_s3_client, test_wikibase_to_s3_config.cdn_bucket_name, "Q10"
    )
    await wikibase_to_s3.fn(config=test_wikibase_to_s3_config)
    concept_after_overwrite = helper_get_concept_from_s3(
        mock_s3_client, test_wikibase_to_s3_config.cdn_bucket_name, "Q10"
    )

    assert concept_before_overwrite != concept_after_overwrite


@pytest.mark.asyncio
async def test_wikibase_to_s3__trigger_deindexing_called(
    MockedWikibaseSession,
    mock_cdn_bucket,
    mock_prefect_slack_webhook,
    test_wikibase_to_s3_config,
    mock_s3_client,
    monkeypatch,
):
    test_wikibase_to_s3_config.trigger_deindexing = True

    # Set up extra concept in S3 that's not in Wikibase
    extra_concept_id = "Q999"
    extra_concept = Concept(
        wikibase_id=WikibaseID(extra_concept_id), preferred_label="Extra Concept"
    )
    helper_upload_concept_to_s3(
        mock_s3_client, test_wikibase_to_s3_config.cdn_bucket_name, extra_concept
    )

    # Verify it exists in S3
    s3_concepts_before = list_s3_concepts(config=test_wikibase_to_s3_config)
    assert extra_concept_id in s3_concepts_before

    # Mock the trigger_deindexing function to track if it's called
    trigger_deindexing_called = False

    async def mock_trigger_deindexing(extras_in_s3, _config):
        nonlocal trigger_deindexing_called
        trigger_deindexing_called = True
        assert extra_concept_id in extras_in_s3
        return None

    monkeypatch.setattr(
        "flows.wikibase_to_s3.trigger_deindexing", mock_trigger_deindexing
    )

    # Run the flow
    await wikibase_to_s3.fn(config=test_wikibase_to_s3_config)

    # Verify trigger_deindexing was called
    assert trigger_deindexing_called, "trigger_deindexing should have been called"

    # Verify the extra concept was removed from S3
    s3_concepts_after = list_s3_concepts(config=test_wikibase_to_s3_config)
    assert extra_concept_id not in s3_concepts_after
