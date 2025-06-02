import json
from typing import Sequence
from unittest.mock import patch

import pytest
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.search_adaptors import VespaSearchAdapter
from vespa.io import VespaResponse

from flows.aggregate_inference_results import S3Uri
from flows.boundary import (
    DocumentImportId,
    get_document_passages_from_vespa__generator,
)
from flows.index_from_aggregate_results import index_aggregate_results_from_s3_to_vespa


@pytest.mark.asyncio
async def test_index_from_aggregated_inference_results(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    mock_s3_client,
    mock_bucket: str,
    mock_bucket_inference_results: list[str],
    s3_prefix_inference_results: str,
) -> None:
    """Test that we loaded the inference results from the mock bucket."""

    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        for file_key in mock_bucket_inference_results:
            response = mock_s3_client.get_object(Bucket=mock_bucket, Key=file_key)
            body = response["Body"].read().decode("utf-8")
            aggregated_inference_results = json.loads(body)
            s3_uri = S3Uri(bucket=mock_bucket, key=file_key)
            document_import_id = DocumentImportId(s3_uri.stem)

            # Get the original vespa passages
            passages_generator = get_document_passages_from_vespa__generator(
                document_import_id=document_import_id,
                vespa_connection_pool=vespa_connection_pool,
            )

            initial_responses = []
            async for vespa_passages in passages_generator:
                initial_responses.append(vespa_passages)

            # Index the aggregated inference results from S3 to Vespa
            await index_aggregate_results_from_s3_to_vespa(
                s3_uri=s3_uri,
                document_import_id=document_import_id,
                vespa_connection_pool=vespa_connection_pool,
            )

            # Get the final vespa passages
            final_passages_generator = get_document_passages_from_vespa__generator(
                document_import_id=document_import_id,
                vespa_connection_pool=vespa_connection_pool,
            )

            final_responses = []
            async for vespa_passages in final_passages_generator:
                final_responses.append(vespa_passages)

            # Assert the final responses are not the same as the original responses
            assert len(final_responses) == len(initial_responses)
            assert final_responses != initial_responses

            # Assert that for each passage that the only concepts present were those from the
            # aggregated inference results.
            for response in final_responses:
                for text_block_id in response.keys():
                    text_block_response = response[text_block_id]

                    vespa_passage: VespaPassage = text_block_response[1]
                    vespa_passage_concepts: Sequence[VespaConcept] = (
                        vespa_passage.concepts if vespa_passage.concepts else []
                    )
                    # When parent concepts is empty we are loading it as None from Vespa
                    # as opposed to an empty list.
                    for concept in vespa_passage_concepts:
                        if concept.parent_concepts is None:
                            concept.parent_concepts = []

                    expected_concepts_json: Sequence[dict] = (
                        aggregated_inference_results[text_block_id]
                    )
                    expected_concepts: Sequence[VespaConcept] = [
                        VespaConcept.model_validate(concept)
                        for concept in expected_concepts_json
                    ]

                    assert len(vespa_passage_concepts) == len(expected_concepts), (
                        f"Passage {text_block_id} has {len(vespa_passage_concepts)} "
                        f"concepts, expected {len(expected_concepts)}."
                    )
                    for concept in vespa_passage_concepts:
                        assert concept in expected_concepts, (
                            f"Concept {concept} not found in expected concepts for passage "
                            f"{text_block_id}."
                        )


@pytest.mark.asyncio
async def test_index_from_aggregated_inference_results__error_handling(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    mock_s3_client,
    mock_bucket: str,
    mock_bucket_inference_results: list[str],
    s3_prefix_inference_results: str,
) -> None:
    """Test that we loaded the inference results from the mock bucket."""

    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        file_keys = mock_bucket_inference_results
        for file_key in file_keys:
            s3_uri = S3Uri(bucket=mock_bucket, key=file_key)

            # Mock this function response _update_vespa_passage_concepts to simulate an error
            with patch(
                "flows.index_from_aggregate_results._update_vespa_passage_concepts"
            ) as mock_update_vespa_passage_concepts:
                mock_update_vespa_passage_concepts.return_value = VespaResponse(
                    status_code=500,
                    operation_type="update",
                    json={"error": "Mocked error"},
                    url="http://mocked-vespa-url",
                )

                # Index the aggregated inference results from S3 to Vespa
                with pytest.raises(ValueError) as excinfo:
                    await index_aggregate_results_from_s3_to_vespa(
                        s3_uri=s3_uri,
                        document_import_id=DocumentImportId(s3_uri.stem),
                        vespa_connection_pool=vespa_connection_pool,
                    )

                assert "Mocked error" in str(excinfo.value)
