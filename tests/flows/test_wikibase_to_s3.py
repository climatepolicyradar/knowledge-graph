import logging

import pytest
from botocore.exceptions import ClientError

from flows.wikibase_to_s3 import (
    list_s3_concepts,
    upload_to_s3,
    wikibase_to_s3,
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


def test_list_s3_concepts(mock_cdn_concepts, test_wikibase_to_s3_config):
    results = list_s3_concepts(config=test_wikibase_to_s3_config)
    assert set(results) == {
        "Q10",
        "Q20",
        "Q30",
    }


def test_wikibase_to_s3(
    caplog,
    MockedWikibaseSession,
    mock_cdn_concepts,
    test_wikibase_to_s3_config,
    mock_s3_client,
):
    caplog.set_level(logging.INFO)

    def get_concept_from_s3(wikibase_id):
        result = mock_s3_client.get_object(
            Bucket=test_wikibase_to_s3_config.cdn_bucket_name,
            Key=f"concepts/{wikibase_id}.json",
        )
        return result["Body"].read().decode("utf-8")

    # Concept count before running from test setup concepts
    start = list_s3_concepts(config=test_wikibase_to_s3_config)
    assert set(start) == {"Q10", "Q20", "Q30"}
    before_overwrite = get_concept_from_s3("Q10")

    # Run once adding new concepts from store
    assert "Extras: ['Q20', 'Q30']" not in caplog.text
    wikibase_to_s3.fn(config=test_wikibase_to_s3_config)
    run_with_new = list_s3_concepts(config=test_wikibase_to_s3_config)
    assert set(run_with_new) == set(
        start
        + [
            "Q10",
            "Q100",
            "Q1000",
            "Q1001",
            "Q1002",
        ]
    )
    assert "Extras: ['Q20', 'Q30']" in caplog.text

    # Rerun again, without new documents in the store
    wikibase_to_s3.fn(config=test_wikibase_to_s3_config)
    run_without_new = list_s3_concepts(config=test_wikibase_to_s3_config)
    assert set(run_with_new) == set(run_without_new)

    # Confirm overwritten concept has changed
    after_overwrite = get_concept_from_s3("Q10")
    assert before_overwrite != after_overwrite
