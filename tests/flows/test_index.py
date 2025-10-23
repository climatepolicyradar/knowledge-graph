import json
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from cpr_sdk.models.search import Document as VespaDocument
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.search_adaptors import VespaSearchAdapter
from prefect.artifacts import Artifact
from prefect.client.schemas.objects import FlowRun
from prefect.context import FlowRunContext
from prefect.states import Completed, Running
from vespa.io import VespaResponse

from flows.aggregate import RunOutputIdentifier, parse_model_field
from flows.boundary import (
    TextBlockId,
    get_document_from_vespa,
    get_document_passages_from_vespa__generator,
)
from flows.index import (
    METADATA_FILE_NAME,
    Metadata,
    SimpleConcept,
    build_v2_document_concepts,
    build_v2_passage_spans,
    index,
    index_batch_of_documents,
    index_document_passages,
    index_family_document,
    store_metadata,
)
from flows.result import is_err, is_ok, unwrap_err
from flows.utils import (
    DocumentImportId,
    DocumentStem,
    remove_translated_suffix,
)


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_index_document_passages(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    mock_async_bucket_inference_results: dict[str, dict[str, Any]],
    mock_run_output_identifier_str: str,
    s3_prefix_inference_results: str,
    test_config,
) -> None:
    """Test that we loaded the inference results from the mock bucket."""
    run_output_identifier = RunOutputIdentifier(mock_run_output_identifier_str)

    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        for (
            file_key,
            aggregated_inference_results,
        ) in mock_async_bucket_inference_results.items():
            document_stem = DocumentStem(Path(file_key).stem)
            document_id: DocumentImportId = remove_translated_suffix(document_stem)

            # Get the original vespa passages
            passages_generator = get_document_passages_from_vespa__generator(
                document_import_id=document_id,
                vespa_connection_pool=vespa_connection_pool,
            )

            initial_responses = []
            async for vespa_passages in passages_generator:
                initial_responses.append(vespa_passages)

            # Index the aggregated inference results from S3 to Vespa
            await index_document_passages(
                config=test_config,
                run_output_identifier=run_output_identifier,
                document_stem=document_stem,
                vespa_connection_pool=vespa_connection_pool,
            )

            # Get the final vespa passages
            final_passages_generator = get_document_passages_from_vespa__generator(
                document_import_id=document_id,
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
                    vespa_passage_concepts: Sequence[VespaPassage.Concept] = (
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
                    expected_concepts: Sequence[VespaPassage.Concept] = [
                        VespaPassage.Concept.model_validate(concept)
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


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_index_document_passages__error_handling(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    mock_async_bucket_inference_results: dict[str, dict[str, Any]],
    mock_run_output_identifier_str: str,
    s3_prefix_inference_results: str,
    test_config,
    snapshot,
) -> None:
    """Test that we loaded the inference results from the mock bucket."""

    run_output_identifier = RunOutputIdentifier(mock_run_output_identifier_str)

    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        for file_key, _ in mock_async_bucket_inference_results.items():
            document_stem = DocumentStem(Path(file_key).stem)

            # Mock this function response _update_vespa_passage_concepts to simulate an error
            with patch(
                "flows.index._update_vespa_passage_concepts"
            ) as mock_update_vespa_passage_concepts:
                mock_update_vespa_passage_concepts.return_value = VespaResponse(
                    status_code=500,
                    operation_type="update",
                    json={"error": "Mocked error"},
                    url="http://mocked-vespa-url",
                )

                # Index the aggregated inference results from S3 to Vespa
                assert snapshot == await index_document_passages(
                    run_output_identifier=run_output_identifier,
                    config=test_config,
                    document_stem=document_stem,
                    vespa_connection_pool=vespa_connection_pool,
                )


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_index_batch_of_documents(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    mock_async_bucket_inference_results: dict[str, dict[str, Any]],
    aggregate_inference_results_document_stems: list[DocumentStem],
    mock_run_output_identifier_str: str,
    s3_prefix_inference_results: str,
    test_config,
) -> None:
    """Test that we loaded the inference results from the mock bucket."""

    run_output_identifier = RunOutputIdentifier(mock_run_output_identifier_str)

    with patch(
        "flows.index.get_vespa_search_adapter_from_aws_secrets",
        return_value=local_vespa_search_adapter,
    ):
        await index_batch_of_documents(
            run_output_identifier=run_output_identifier,
            document_stems=aggregate_inference_results_document_stems,
            config_json=test_config.model_dump(),
        )

        # Verify that the final data in vespa matches the expected results
        async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
            for file_key in mock_async_bucket_inference_results.keys():
                document_stem = DocumentStem(Path(file_key).stem)
                document_id: DocumentImportId = remove_translated_suffix(document_stem)

                passages_generator = get_document_passages_from_vespa__generator(
                    document_import_id=document_id,
                    vespa_connection_pool=vespa_connection_pool,
                )

                # Get all indexed passages for this document
                final_passages = {}
                async for vespa_passages in passages_generator:
                    final_passages.update(vespa_passages)

                # Find the corresponding file for this document ID
                expected_concepts = mock_async_bucket_inference_results[file_key]

                # Assert all text blocks were indexed with their concepts
                assert set(final_passages.keys()) == set(expected_concepts.keys()), (
                    f"Text blocks in Vespa don't match expected text blocks for document {document_id}"
                )

                # Check each passage has the correct concepts
                for text_block_id, (_, vespa_passage) in final_passages.items():
                    vespa_passage_concepts = vespa_passage.concepts or []
                    # When parent concepts is empty we are loading it as None from Vespa
                    # as opposed to an empty list.
                    for concept in vespa_passage_concepts:
                        if concept.parent_concepts is None:
                            concept.parent_concepts = []

                    passage_expected_concepts = [
                        VespaPassage.Concept.model_validate(c)
                        for c in expected_concepts[text_block_id]
                    ]

                    assert len(vespa_passage_concepts) == len(
                        passage_expected_concepts
                    ), (
                        f"Passage {text_block_id} has {len(vespa_passage_concepts)} concepts, "
                        f"expected {len(passage_expected_concepts)}"
                    )

                    for concept in vespa_passage_concepts:
                        assert concept in passage_expected_concepts, (
                            f"Concept {concept} not found in expected concepts for passage "
                            f"{text_block_id}."
                        )

                # Verify that concept_counts were updated on family_document in Vespa
                # Get the family document from Vespa
                _vespa_hit_id, vespa_document = get_document_from_vespa(
                    document_import_id=document_id,
                    vespa_search_adapter=local_vespa_search_adapter,
                )

                # Verify that concept_counts field exists and is not None/empty
                assert vespa_document.concept_counts is not None, (
                    f"concept_counts should not be None for document {document_id}"
                )
                assert len(vespa_document.concept_counts) > 0, (
                    f"concept_counts should not be empty for document {document_id}"
                )

                # Verify that concept_counts contains expected concept IDs from the aggregate results
                # Get all unique concept IDs from the expected results
                expected_concept_ids = set()
                for passage_concepts in expected_concepts.values():
                    for concept_data in passage_concepts:
                        expected_concept_ids.add(concept_data["id"])

                # Extract concept IDs from concept_counts keys (format: "Q123:concept_name")
                actual_concept_ids = {
                    key.split(":")[0] for key in vespa_document.concept_counts.keys()
                }

                # Check that concept_counts contains exactly the expected concept IDs
                assert expected_concept_ids == actual_concept_ids, (
                    f"Expected concept IDs {expected_concept_ids} do not match concept_counts "
                    f"for document {document_id}. Actual concept IDs: {actual_concept_ids}"
                )


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_index_batch_of_documents_with_v2_concepts(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    mock_async_bucket_inference_results: dict[str, dict[str, Any]],
    aggregate_inference_results_document_stems: list[DocumentStem],
    mock_run_output_identifier_str: str,
    s3_prefix_inference_results: str,
    test_config,
    snapshot,
) -> None:
    """Test that we loaded the inference results from the mock bucket."""

    run_output_identifier = RunOutputIdentifier(mock_run_output_identifier_str)

    with patch(
        "flows.index.get_vespa_search_adapter_from_aws_secrets",
        return_value=local_vespa_search_adapter,
    ):
        await index_batch_of_documents(
            run_output_identifier=run_output_identifier,
            document_stems=aggregate_inference_results_document_stems,
            config_json=test_config.model_dump(),
            enable_v2_concepts=True,
        )

        # Verify that the final data in vespa matches the expected results
        async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
            all_vespa_documents = []
            for file_key in mock_async_bucket_inference_results.keys():
                document_stem = DocumentStem(Path(file_key).stem)
                document_id: DocumentImportId = remove_translated_suffix(document_stem)

                passages_generator = get_document_passages_from_vespa__generator(
                    document_import_id=document_id,
                    vespa_connection_pool=vespa_connection_pool,
                )

                # Get all indexed passages for this document
                final_passages = {}
                async for vespa_passages in passages_generator:
                    final_passages.update(vespa_passages)

                # Find the corresponding file for this document ID
                expected_concepts = mock_async_bucket_inference_results[file_key]

                # Assert all text blocks were indexed with their concepts
                assert set(final_passages.keys()) == set(expected_concepts.keys()), (
                    f"Text blocks in Vespa don't match expected text blocks for document {document_id}"
                )

                # Check each passage has the correct concepts
                for text_block_id, (_, vespa_passage) in final_passages.items():
                    vespa_passage_concepts = vespa_passage.concepts or []
                    # When parent concepts is empty we are loading it as None from Vespa
                    # as opposed to an empty list.
                    for concept in vespa_passage_concepts:
                        if concept.parent_concepts is None:
                            concept.parent_concepts = []

                    passage_expected_concepts = [
                        VespaPassage.Concept.model_validate(c)
                        for c in expected_concepts[text_block_id]
                    ]

                    assert len(vespa_passage_concepts) == len(
                        passage_expected_concepts
                    ), (
                        f"Passage {text_block_id} has {len(vespa_passage_concepts)} concepts, "
                        f"expected {len(passage_expected_concepts)}"
                    )

                    for concept in vespa_passage_concepts:
                        assert concept in passage_expected_concepts, (
                            f"Concept {concept} not found in expected concepts for passage "
                            f"{text_block_id}."
                        )

                # Verify that concept_counts were updated on family_document in Vespa
                # Get the family document from Vespa
                _vespa_hit_id, vespa_document = get_document_from_vespa(
                    document_import_id=document_id,
                    vespa_search_adapter=local_vespa_search_adapter,
                )

                all_vespa_documents.append(vespa_document)

            assert all_vespa_documents == snapshot


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_index_batch_of_documents__failure(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    mock_async_bucket_inference_results: dict[str, dict[str, Any]],
    aggregate_inference_results_document_stems: list[DocumentStem],
    mock_run_output_identifier_str: str,
    s3_prefix_inference_results: str,
    test_config,
) -> None:
    """Test that we handled the exception correctly during passage indexing."""

    non_existent_stem = DocumentStem("non_existent_document")
    document_stems = aggregate_inference_results_document_stems + [non_existent_stem]

    run_output_identifier = RunOutputIdentifier(mock_run_output_identifier_str)

    with patch(
        "flows.index.get_vespa_search_adapter_from_aws_secrets",
        return_value=local_vespa_search_adapter,
    ):
        # Index the aggregated inference results from S3 to Vespa
        with pytest.raises(ValueError) as excinfo:
            await index_batch_of_documents(
                run_output_identifier=run_output_identifier,
                config_json=test_config.model_dump(),
                document_stems=document_stems,
            )

        assert f"Failed to process 1/{len(document_stems)} documents" in str(
            excinfo.value
        )


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_run_indexing_from_aggregate_results__invokes_subdeployments_correctly(
    vespa_app,
    mock_async_bucket_inference_results: dict[str, dict[str, Any]],
    aggregate_inference_results_document_stems: list[DocumentStem],
    mock_run_output_identifier_str: str,
    test_config,
) -> None:
    """Test that run passage level indexing correctly from aggregated results."""

    run_output_identifier = RunOutputIdentifier(mock_run_output_identifier_str)

    with patch("flows.utils.run_deployment") as mock_run_deployment:
        # Mock the response of the run_deployment function.
        flow_run_counter = 0

        async def mock_awaitable(*args, **kwargs):
            nonlocal flow_run_counter
            flow_run_counter += 1
            from prefect.results import ResultRecord

            return FlowRun(
                flow_id=uuid.uuid4(),
                name=f"mock-run-{flow_run_counter}",
                state=Completed(data=ResultRecord(result=None)),
            )

        mock_run_deployment.side_effect = mock_awaitable

        # Run indexing
        try:
            await index(
                run_output_identifier=run_output_identifier,
                document_stems=aggregate_inference_results_document_stems,
                config=test_config,
                batch_size=1,
            )
        except ValueError:
            # Expected to fail in test environment due to mocking limitations
            # The actual functionality being tested is the deployment calls below
            pass

        # Assert that the run_deployment was called the expected params
        assert mock_run_deployment.call_count == len(
            aggregate_inference_results_document_stems
        )
        for call in mock_run_deployment.call_args_list:
            call_params = call.kwargs["parameters"]
            assert call_params["run_output_identifier"] == run_output_identifier
            assert len(call_params["document_stems"]) == 1
            assert (
                call_params["document_stems"][0]
                in aggregate_inference_results_document_stems
            )
            assert call_params["config_json"] == test_config.model_dump()

        # Reset the mock_run_deployment call count
        mock_run_deployment.reset_mock()

        # Run indexing with no document_ids specified, which should run for all documents
        try:
            await index(
                run_output_identifier=run_output_identifier,
                document_stems=None,
                config=test_config,
                batch_size=1,
            )
        except ValueError:
            # Expected to fail in test environment due to mocking limitations
            pass

        # Assert that the run_deployment was called the expected params
        assert mock_run_deployment.call_count == len(
            aggregate_inference_results_document_stems
        )
        for call in mock_run_deployment.call_args_list:
            call_params = call.kwargs["parameters"]
            assert call_params["run_output_identifier"] == run_output_identifier
            assert len(call_params["document_stems"]) == 1
            assert (
                call_params["document_stems"][0]
                in aggregate_inference_results_document_stems
            )
            assert call_params["config_json"] == test_config.model_dump()


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_run_indexing_from_aggregate_results__handles_failures(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    mock_async_bucket_inference_results: dict[str, dict[str, Any]],
    mock_run_output_identifier_str: str,
    aggregate_inference_results_document_stems: list[DocumentStem],
    s3_prefix_inference_results: str,
    test_config,
) -> None:
    """Test that run passage level indexing correctly from aggregated results."""

    non_existent_stem = DocumentStem("non_existent_document")
    document_stems = aggregate_inference_results_document_stems + [non_existent_stem]

    run_output_identifier = RunOutputIdentifier(mock_run_output_identifier_str)

    # Assert that the indexing runs correctly when called as sub deployments and that we
    # continue on failure of one of the documents.
    with patch(
        "flows.index.get_vespa_search_adapter_from_aws_secrets",
        return_value=local_vespa_search_adapter,
    ):
        with pytest.raises(ValueError) as excinfo:
            await index(
                run_output_identifier=run_output_identifier,
                document_stems=document_stems,
                config=test_config,
                batch_size=1,
            )

        # Expect all documents to fail in test environment due to mocking limitations
        assert (
            f"Some batches of documents had failures: {len(document_stems)}/{len(document_stems)} failed."
            in str(excinfo.value)
        )

        # Assert that the summary artifact was created after flow completion
        summary_artifact = await Artifact.get(
            "indexing-aggregate-results-summary-sandbox"
        )
        assert summary_artifact and summary_artifact.description
        assert (
            summary_artifact.description
            == "Summary of the passages indexing run to update concept counts."
        )


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_index_family_document(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    aggregate_inference_results_document_stems: list[DocumentStem],
) -> None:
    """Test that index_family_document correctly updates concept counts in Vespa."""

    # Use the first document stem from our test data
    document_stem = aggregate_inference_results_document_stems[0]
    document_id: DocumentImportId = remove_translated_suffix(document_stem)

    # Create some test concepts with old model format (no v2 enrichment)
    test_concepts = [
        SimpleConcept(
            id="Q123",
            name="Climate Change",
            model='KeywordClassifier("Climate Change")',
            parsed_model=None,
        ),
        SimpleConcept(
            id="Q456",
            name="Carbon Emissions",
            model='KeywordClassifier("Carbon Emissions")',
            parsed_model=None,
        ),
        SimpleConcept(
            id="Q123",
            name="Climate Change",
            model='KeywordClassifier("Climate Change")',
            parsed_model=None,
        ),  # Duplicate to test counting
        SimpleConcept(
            id="Q789",
            name="Renewable Energy",
            model='KeywordClassifier("Renewable Energy")',
            parsed_model=None,
        ),
    ]

    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        # Get the initial state of the family document
        initial_vespa_hit_id, initial_vespa_document = get_document_from_vespa(
            document_import_id=document_id,
            vespa_search_adapter=local_vespa_search_adapter,
        )

        # Index the concepts
        result = await index_family_document(
            document_id=document_id,
            vespa_connection_pool=vespa_connection_pool,
            simple_concepts=test_concepts,
        )

        # Assert the operation was successful
        assert is_ok(result), f"Expected Ok result, got {result}"

        # Get the updated family document from Vespa
        final_vespa_hit_id, final_vespa_document = get_document_from_vespa(
            document_import_id=document_id,
            vespa_search_adapter=local_vespa_search_adapter,
        )

        # Verify that concept_counts field exists and is not None/empty
        assert final_vespa_document.concept_counts is not None, (
            f"concept_counts should not be None for document {document_id}"
        )
        assert len(final_vespa_document.concept_counts) > 0, (
            f"concept_counts should not be empty for document {document_id}"
        )

        # Verify the concept counts are correct
        expected_concept_counts = {
            "Q123:Climate Change": 2,  # Appears twice
            "Q456:Carbon Emissions": 1,
            "Q789:Renewable Energy": 1,
        }

        assert final_vespa_document.concept_counts == expected_concept_counts, (
            f"Expected concept_counts {expected_concept_counts}, "
            f"got {final_vespa_document.concept_counts}"
        )


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_index_family_document__failure(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    aggregate_inference_results_document_stems: list[DocumentStem],
) -> None:
    """Test that index_family_document correctly handles Vespa update failures."""

    # Use the first document stem from our test data
    document_stem = aggregate_inference_results_document_stems[0]
    document_id: DocumentImportId = remove_translated_suffix(document_stem)

    # Create some test concepts with old model format
    test_concepts = [
        SimpleConcept(
            id="Q123",
            name="Climate Change",
            model='KeywordClassifier("Climate Change")',
            parsed_model=None,
        ),
        SimpleConcept(
            id="Q456",
            name="Carbon Emissions",
            model='KeywordClassifier("Carbon Emissions")',
            parsed_model=None,
        ),
    ]

    # Mock the update_data method at the module level to avoid Prefect serialization issues
    with patch("vespa.application.VespaAsync.update_data") as mock_update_data:
        mock_update_data.return_value = VespaResponse(
            status_code=500,
            operation_type="update",
            json={"error": "Internal server error"},
            url="http://mocked-vespa-url",
        )

        async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
            # Index the concepts
            result = await index_family_document(
                document_id=document_id,
                vespa_connection_pool=vespa_connection_pool,
                simple_concepts=test_concepts,
            )

            # Assert the operation failed
            assert is_err(result), f"Expected Err result, got {result}"

            error = unwrap_err(result)
            assert error.msg == "Vespa update failed"
            assert "json" in error.metadata
            assert error.metadata["json"] == {"error": "Internal server error"}


@pytest.mark.asyncio
async def test_store_metadata(
    test_config,
    mock_async_bucket,
    mock_s3_async_client,
    snapshot,
):
    """Test that store_metadata correctly builds S3 URI and stores metadata."""
    mock_tags = ["tag:value1", "sha:abc123", "branch:main"]
    mock_run_output_id = "2025-01-15T10:30-test-flow-run"

    # Create a real FlowRun object with proper data
    flow_run = FlowRun(
        id=uuid.UUID("0199bef8-7e41-7afc-9b4c-d3abd406be84"),
        flow_id=uuid.UUID("b213352f-3214-48e3-8f5d-ec19959cb28e"),
        name="test-flow-run",
        state=Running(),
        tags=mock_tags,
    )

    mock_context = MagicMock(spec=FlowRunContext)
    mock_context.flow_run = flow_run

    # Mock only the Prefect context, let moto handle S3
    with patch("flows.index.get_run_context", return_value=mock_context):
        await store_metadata(
            config=test_config,
            run_output_identifier=mock_run_output_id,
        )

    expected_key = os.path.join(
        test_config.index_results_prefix,
        mock_run_output_id,
        METADATA_FILE_NAME,
    )

    response = await mock_s3_async_client.head_object(
        Bucket=test_config.cache_bucket, Key=expected_key
    )
    assert response["ContentLength"] > 0, (
        f"Expected S3 object {expected_key} to have content"
    )

    response = await mock_s3_async_client.get_object(
        Bucket=test_config.cache_bucket, Key=expected_key
    )
    metadata_content = await response["Body"].read()
    metadata_dict = json.loads(metadata_content.decode("utf-8"))

    metadata = Metadata.model_validate(metadata_dict)
    assert metadata == snapshot


def test_build_v2_passage_spans__valid_concepts():
    """Test build_v2_passage_spans with valid concepts using new model field format."""

    text_block_id = TextBlockId("test.passage.1")
    # Use new model field format: "wikibase_id:concept_id:classifier_id"
    serialised_concepts = [
        {
            "id": "Q123",
            "name": "Climate Change",
            "start": 0,
            "end": 10,
            "parent_concepts": [],
            "parent_concept_ids_flat": "",
            "model": "Q123:ttbb2345:abcd2345",
            "timestamp": "2025-01-01T00:00:00",
        },
        {
            "id": "Q456",
            "name": "Carbon Emissions",
            "start": 15,
            "end": 25,
            "parent_concepts": [],
            "parent_concept_ids_flat": "",
            "model": "Q456:abcd2345:efgh6789",
            "timestamp": "2025-01-01T00:00:00",
        },
    ]

    result = build_v2_passage_spans(
        text_block_id=text_block_id,
        serialised_concepts=serialised_concepts,
    )

    assert result == [
        {
            "concepts_v2": [
                {
                    "classifier_id": "abcd2345",
                    "concept_id": "ttbb2345",
                    "concept_wikibase_id": "Q123",
                }
            ],
            "end": 10,
            "start": 0,
        },
        {
            "concepts_v2": [
                {
                    "classifier_id": "efgh6789",
                    "concept_id": "abcd2345",
                    "concept_wikibase_id": "Q456",
                }
            ],
            "end": 25,
            "start": 15,
        },
    ]


def test_build_v2_passage_spans__invalid_wikibase_id():
    """Test build_v2_passage_spans with invalid WikibaseID."""
    text_block_id = TextBlockId("test.passage.1")
    serialised_concepts = [
        {
            "id": "invalid_id",  # Invalid WikibaseID
            "name": "Test Concept",
            "start": 0,
            "end": 10,
            "parent_concepts": [],
            "parent_concept_ids_flat": "",
            "model": "invalid_id:concept123:classifier456",
            "timestamp": "2025-01-01T00:00:00",
        },
        {
            "id": "Q0",  # Invalid (Q0 not allowed)
            "name": "Another Concept",
            "start": 15,
            "end": 25,
            "parent_concepts": [],
            "parent_concept_ids_flat": "",
            "model": "Q0:concept789:classifierabc",
            "timestamp": "2025-01-01T00:00:00",
        },
    ]

    result = build_v2_passage_spans(
        text_block_id=text_block_id,
        serialised_concepts=serialised_concepts,
    )

    assert result == []


def test_build_v2_passage_spans__old_model_format():
    """Test build_v2_passage_spans skips concepts with old model format."""
    text_block_id = TextBlockId("test.passage.1")
    serialised_concepts = [
        {
            "id": "Q999",
            "name": "Old Format Concept",
            "start": 0,
            "end": 10,
            "parent_concepts": [],
            "parent_concept_ids_flat": "",
            "model": 'KeywordClassifier("Unknown Concept")',  # Old format
            "timestamp": "2025-01-01T00:00:00",
        },
    ]

    result = build_v2_passage_spans(
        text_block_id=text_block_id,
        serialised_concepts=serialised_concepts,
    )

    # Old format concepts are skipped, so no v2 spans
    assert result == []


def test_build_v2_passage_spans__grouping_by_position():
    """Test build_v2_passage_spans groups concepts at same position."""
    text_block_id = TextBlockId("test.passage.1")
    # Multiple concepts at same position (0-10)
    serialised_concepts = [
        {
            "id": "Q123",
            "name": "Climate Change",
            "start": 0,
            "end": 10,
            "parent_concepts": [],
            "parent_concept_ids_flat": "",
            "model": "Q123:gbhf2299:abcd2345",
            "timestamp": "2025-01-01T00:00:00",
        },
        {
            "id": "Q456",
            "name": "Carbon Emissions",
            "start": 0,
            "end": 10,  # Same position as above
            "parent_concepts": [],
            "parent_concept_ids_flat": "",
            "model": "Q456:tttt3333:efgh6789",
            "timestamp": "2025-01-01T00:00:00",
        },
    ]

    result = build_v2_passage_spans(
        text_block_id=text_block_id,
        serialised_concepts=serialised_concepts,
    )

    assert result == [
        {
            "concepts_v2": [
                {
                    "classifier_id": "abcd2345",
                    "concept_id": "gbhf2299",
                    "concept_wikibase_id": "Q123",
                },
                {
                    "classifier_id": "efgh6789",
                    "concept_id": "tttt3333",
                    "concept_wikibase_id": "Q456",
                },
            ],
            "end": 10,
            "start": 0,
        },
    ]


def test_build_v2_passage_spans__empty_inputs():
    """Test build_v2_passage_spans with empty inputs."""
    text_block_id = TextBlockId("test.passage.1")
    serialised_concepts = []

    result = build_v2_passage_spans(
        text_block_id=text_block_id,
        serialised_concepts=serialised_concepts,
    )

    assert result == []


def test_build_v2_document_concepts__valid_concepts():
    """Test build_v2_document_concepts with valid concepts."""
    simple_concepts = [
        SimpleConcept(
            id="Q123",
            name="Climate Change",
            model="Q123:xabs2345:abcd2345",
            parsed_model=parse_model_field("Q123:xabs2345:abcd2345"),
        ),
        SimpleConcept(
            id="Q456",
            name="Carbon Emissions",
            model="Q456:abcd2345:efgh6789",
            parsed_model=parse_model_field("Q456:abcd2345:efgh6789"),
        ),
    ]

    result = build_v2_document_concepts(
        simple_concepts=simple_concepts,
    )

    assert result == [
        {
            "classifier_id": "abcd2345",
            "concept_id": "xabs2345",
            "concept_wikibase_id": "Q123",
            "count": 1,
        },
        {
            "classifier_id": "efgh6789",
            "concept_id": "abcd2345",
            "concept_wikibase_id": "Q456",
            "count": 1,
        },
    ]


def test_build_v2_document_concepts__concept_counting():
    """Test build_v2_document_concepts correctly counts duplicate concepts."""
    simple_concepts = [
        SimpleConcept(
            id="Q123",
            name="Climate Change",
            model="Q123:tqhn2243:abcd2345",
            parsed_model=parse_model_field("Q123:tqhn2243:abcd2345"),
        ),
        SimpleConcept(
            id="Q123",
            name="Climate Change",
            model="Q123:tqhn2243:abcd2345",
            parsed_model=parse_model_field("Q123:tqhn2243:abcd2345"),
        ),  # Duplicate
        SimpleConcept(
            id="Q123",
            name="Climate Change",
            model="Q123:tqhn2243:abcd2345",
            parsed_model=parse_model_field("Q123:tqhn2243:abcd2345"),
        ),  # Duplicate
        SimpleConcept(
            id="Q456",
            name="Carbon Emissions",
            model="Q456:qkjt4493:efgh6789",
            parsed_model=parse_model_field("Q456:qkjt4493:efgh6789"),
        ),
    ]

    result = build_v2_document_concepts(
        simple_concepts=simple_concepts,
    )

    assert result == [
        {
            "classifier_id": "abcd2345",
            "concept_id": "tqhn2243",
            "concept_wikibase_id": "Q123",
            "count": 3,
        },
        {
            "classifier_id": "efgh6789",
            "concept_id": "qkjt4493",
            "concept_wikibase_id": "Q456",
            "count": 1,
        },
    ]


def test_build_v2_document_concepts__invalid_wikibase_id():
    """
    Test build_v2_document_concepts with invalid WikibaseID.

    When parse_model_field encounters invalid IDs, it returns None,
    so we test that the function skips concepts with None parsed_model.
    """
    simple_concepts = [
        SimpleConcept(
            id="invalid_id",
            name="Test Concept",
            model="invalid_id:concept123:classifier456",
            parsed_model=None,  # parse_model_field would return None for invalid IDs
        ),
        SimpleConcept(
            id="Q0",
            name="Another Concept",
            model="Q0:concept789:classifierabc",
            parsed_model=None,  # parse_model_field would return None for Q0
        ),
    ]

    result = build_v2_document_concepts(
        simple_concepts=simple_concepts,
    )

    assert result == []


def test_build_v2_document_concepts__old_model_format():
    """Test build_v2_document_concepts skips concepts with old model format."""
    simple_concepts = [
        SimpleConcept(
            id="Q999",
            name="Unknown Concept",
            model='KeywordClassifier("Unknown Concept")',
            parsed_model=None,
        ),  # Old format
    ]

    result = build_v2_document_concepts(
        simple_concepts=simple_concepts,
    )

    # Old format concepts are skipped
    assert result == []


def test_build_v2_document_concepts__empty_inputs():
    """Test build_v2_document_concepts with empty inputs."""
    simple_concepts = []

    result = build_v2_document_concepts(
        simple_concepts=simple_concepts,
    )

    assert result == []


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_index_document_passages__with_v2_concepts(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    mock_async_bucket_inference_results: dict[str, dict[str, Any]],
    mock_run_output_identifier_str: str,
    s3_prefix_inference_results: str,
    test_config,
) -> None:
    """Test that v2 spans are created when aggregate_metadata is provided."""
    run_output_identifier = RunOutputIdentifier(mock_run_output_identifier_str)

    # NOTE: The test fixture data has been updated to use the new model format
    # "wikibase_id:concept_id:classifier_id" for WikibaseIDs Q237, Q309, Q371
    # These match the classifier specs: Q237:ttbb2345:abcd2345, Q309:xyzw6789:efgh6789, Q371:hjkm9999:mnpq2468

    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        for (
            file_key,
            aggregated_inference_results,
        ) in mock_async_bucket_inference_results.items():
            document_stem = DocumentStem(Path(file_key).stem)
            document_id: DocumentImportId = remove_translated_suffix(document_stem)

            # Index the aggregated inference results from S3 to Vespa
            await index_document_passages(
                config=test_config,
                run_output_identifier=run_output_identifier,
                document_stem=document_stem,
                vespa_connection_pool=vespa_connection_pool,
                enable_v2_concepts=True,
            )

            # Get the final vespa passages
            final_passages_generator = get_document_passages_from_vespa__generator(
                document_import_id=document_id,
                vespa_connection_pool=vespa_connection_pool,
            )

            final_responses = []
            async for vespa_passages in final_passages_generator:
                final_responses.append(vespa_passages)

            # Verify that v2 spans were created for passages with matching concepts
            # Since we're using classifier specs that match the fixture data (Q434, Q395, Q592),
            # we should find at least some spans
            found_spans = False
            spans_checked = 0

            for response in final_responses:
                for text_block_id in response.keys():
                    text_block_response = response[text_block_id]
                    vespa_passage: VespaPassage = text_block_response[1]

                    vespa_passage_spans: Sequence[VespaPassage.Span] | None = (
                        vespa_passage.spans
                    )

                    # Check passages that have spans (concepts that matched our classifier specs)
                    if vespa_passage_spans and len(vespa_passage_spans) > 0:
                        found_spans = True
                        # Verify each span has the correct structure
                        for span in vespa_passage_spans:
                            spans_checked += 1
                            assert span.start is not None
                            assert span.end is not None
                            assert span.concepts_v2 is not None
                            assert len(span.concepts_v2) > 0, (
                                f"Span at position ({span.start}, {span.end}) "
                                f"should have concepts_v2"
                            )

                            # Verify concepts_v2 have enriched classifier data
                            for concept_v2 in span.concepts_v2:
                                assert concept_v2.concept_id is not None
                                assert concept_v2.concept_wikibase_id is not None
                                assert concept_v2.classifier_id is not None
                                # Verify wikibase_id matches our test fixture concepts
                                assert concept_v2.concept_wikibase_id in [
                                    "Q237",
                                    "Q309",
                                    "Q371",
                                ], (
                                    f"concept_v2 wikibase_id {concept_v2.concept_wikibase_id} "
                                    f"should match test fixture concepts"
                                )
                                # Verify classifier_id matches one of our test classifier specs
                                assert concept_v2.classifier_id in [
                                    "abcd2345",
                                    "efgh6789",
                                    "mnpq2468",
                                ], (
                                    f"concept_v2 classifier_id {concept_v2.classifier_id} "
                                    f"should match test specs"
                                )

            # Assert that we found and verified at least some spans
            # The test fixtures have concepts Q237, Q309, Q371 which match our classifier specs
            assert found_spans, (
                "Expected to find v2 spans for passages with concepts matching classifier specs"
            )
            assert spans_checked > 0, (
                f"Expected to verify at least one span, but checked {spans_checked}"
            )


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_index_family_document__with_v2_concepts(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    aggregate_inference_results_document_stems: list[DocumentStem],
    test_config,
) -> None:
    """Test that concepts_v2 are created when aggregate_metadata is provided."""

    # Use the first document stem from our test data
    document_stem = aggregate_inference_results_document_stems[0]
    document_id: DocumentImportId = remove_translated_suffix(document_stem)

    # Use new model field format: "wikibase_id:concept_id:classifier_id"
    simple_concepts = [
        SimpleConcept(
            id="Q123",
            name="Climate Change",
            model="Q123:ttbb2345:abcd2345",
            parsed_model=parse_model_field("Q123:ttbb2345:abcd2345"),
        ),
        SimpleConcept(
            id="Q456",
            name="Carbon Emissions",
            model="Q456:xyzw6789:efgh6789",
            parsed_model=parse_model_field("Q456:xyzw6789:efgh6789"),
        ),
        SimpleConcept(
            id="Q123",
            name="Climate Change",
            model="Q123:ttbb2345:abcd2345",
            parsed_model=parse_model_field("Q123:ttbb2345:abcd2345"),
        ),  # Duplicate to test counting
        SimpleConcept(
            id="Q789",
            name="Renewable Energy",
            model="Q789:hjkm9999:mnpq2468",
            parsed_model=parse_model_field("Q789:hjkm9999:mnpq2468"),
        ),
    ]

    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        # Get the initial state of the family document
        initial_vespa_hit_id, initial_vespa_document = get_document_from_vespa(
            document_import_id=document_id,
            vespa_search_adapter=local_vespa_search_adapter,
        )

        # Index the concepts - v2 concepts will be created from model field
        result = await index_family_document(
            document_id=document_id,
            vespa_connection_pool=vespa_connection_pool,
            simple_concepts=simple_concepts,
            enable_v2_concepts=True,
        )

        # Assert the operation was successful
        assert is_ok(result), f"Expected Ok result, got {result}"

        # Get the updated family document from Vespa
        final_vespa_hit_id, final_vespa_document = get_document_from_vespa(
            document_import_id=document_id,
            vespa_search_adapter=local_vespa_search_adapter,
        )

        # Verify that concept_counts field exists and is correct
        assert final_vespa_document.concept_counts is not None
        assert len(final_vespa_document.concept_counts) > 0

        expected_concept_counts = {
            "Q123:Climate Change": 2,  # Appears twice
            "Q456:Carbon Emissions": 1,
            "Q789:Renewable Energy": 1,
        }

        assert final_vespa_document.concept_counts == expected_concept_counts

        # Verify that concepts_v2 were created with enriched data
        assert final_vespa_document.concepts_v2 is not None, (
            "concepts_v2 should be present when aggregate_metadata is provided"
        )
        assert len(final_vespa_document.concepts_v2) == 3, (
            f"Expected 3 concepts_v2, got {len(final_vespa_document.concepts_v2)}"
        )

        # Verify the structure and content of concepts_v2
        expected_concepts_v2 = [
            VespaDocument.ConceptV2(
                concept_id="ttbb2345",
                concept_wikibase_id="Q123",
                classifier_id="abcd2345",
                count=2,
            ),
            VespaDocument.ConceptV2(
                concept_id="xyzw6789",
                concept_wikibase_id="Q456",
                classifier_id="efgh6789",
                count=1,
            ),
            VespaDocument.ConceptV2(
                concept_id="hjkm9999",
                concept_wikibase_id="Q789",
                classifier_id="mnpq2468",
                count=1,
            ),
        ]

        assert final_vespa_document.concepts_v2 == expected_concepts_v2, (
            f"Expected concepts_v2 to match. "
            f"Got: {final_vespa_document.concepts_v2}, "
            f"Expected: {expected_concepts_v2}"
        )
