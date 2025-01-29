from io import BytesIO

import pytest
from botocore.exceptions import ClientError
from mypy_boto3_s3.client import S3Client

from flows.wikibase_to_s3 import (
    list_s3_concepts,
    upload_to_s3,
    wikibase_to_s3,
)
from src.concept import Concept


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


def test_wikibase_to_s3__empty_cdn_bucket(
    MockedWikibaseSession,
    mock_cdn_bucket,
    mock_prefect_slack_webhook,
    test_wikibase_to_s3_config,
    mock_s3_client,
):
    start = list_s3_concepts(config=test_wikibase_to_s3_config)
    assert len(start) == 0
    wikibase_to_s3.fn(config=test_wikibase_to_s3_config)
    end = list_s3_concepts(config=test_wikibase_to_s3_config)
    assert len(end) == 10


def test_wikibase_to_s3__repeat_runs(
    MockedWikibaseSession,
    mock_cdn_bucket,
    mock_prefect_slack_webhook,
    test_wikibase_to_s3_config,
    mock_s3_client,
):
    start = list_s3_concepts(config=test_wikibase_to_s3_config)
    assert len(start) == 0
    wikibase_to_s3.fn(config=test_wikibase_to_s3_config)
    wikibase_to_s3.fn(config=test_wikibase_to_s3_config)
    end = list_s3_concepts(config=test_wikibase_to_s3_config)
    assert len(end) == 10


def test_wikibase_to_s3__extras_in_cdn_bucket(
    MockedWikibaseSession,
    mock_prefect_slack_webhook,
    test_wikibase_to_s3_config,
    mock_s3_client,
    mock_concepts,
    mock_cdn_bucket,
):
    mock_SlackWebhook, mock_prefect_slack_block = mock_prefect_slack_webhook
    # Set up extra concepts
    for concept in mock_concepts:
        helper_upload_concept_to_s3(mock_s3_client, mock_cdn_bucket, concept)

    # Run wikibase_to_s3 and find extras
    start = list_s3_concepts(config=test_wikibase_to_s3_config)
    assert set(start) == {"Q10", "Q20", "Q30"}

    expected_error_part = "2 concepts where found in S3 but where not part of the copy"
    with pytest.raises(ValueError, match=expected_error_part):
        wikibase_to_s3(config=test_wikibase_to_s3_config)

    # ensure failure hook was called
    mock_prefect_slack_block.notify.assert_called_once()
    message = mock_prefect_slack_block.notify.call_args.kwargs.get("body", "")
    assert expected_error_part in message

    # Check end count
    end = list_s3_concepts(config=test_wikibase_to_s3_config)
    assert len(end) == 12


def test_wikibase_to_s3__overwrite_concept(
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
    wikibase_to_s3.fn(config=test_wikibase_to_s3_config)
    concept_after_overwrite = helper_get_concept_from_s3(
        mock_s3_client, test_wikibase_to_s3_config.cdn_bucket_name, "Q10"
    )

    assert concept_before_overwrite != concept_after_overwrite
