import uuid
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

import pytest
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.search_adaptors import VespaSearchAdapter
from prefect.artifacts import Artifact
from prefect.client.schemas.objects import FlowRun
from prefect.states import Completed
from vespa.io import VespaResponse

from flows.aggregate_inference_results import RunOutputIdentifier
from flows.boundary import (
    DocumentImportId,
    DocumentStem,
    get_document_from_vespa,
    get_document_passages_from_vespa__generator,
)
from flows.index_from_aggregate_results import (
    SimpleConcept,
    index_aggregate_results_for_batch_of_documents,
    index_document_passages,
    index_family_document,
    run_indexing_from_aggregate_results,
)
from flows.result import is_err, is_ok, unwrap_err
from flows.utils import remove_translated_suffix


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_index_document_passages(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    mock_s3_client,
    mock_bucket: str,
    mock_bucket_inference_results: dict[str, dict[str, Any]],
    mock_run_output_identifier_str: str,
    s3_prefix_inference_results: str,
    test_aggregate_config,
) -> None:
    """Test that we loaded the inference results from the mock bucket."""

    run_output_identifier = RunOutputIdentifier(mock_run_output_identifier_str)

    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        for (
            file_key,
            aggregated_inference_results,
        ) in mock_bucket_inference_results.items():
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
                config=test_aggregate_config,
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


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_index_document_passages__error_handling(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    mock_s3_client,
    mock_bucket: str,
    mock_bucket_inference_results: dict[str, dict[str, Any]],
    mock_run_output_identifier_str: str,
    s3_prefix_inference_results: str,
    test_aggregate_config,
    snapshot,
) -> None:
    """Test that we loaded the inference results from the mock bucket."""

    run_output_identifier = RunOutputIdentifier(mock_run_output_identifier_str)

    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        for file_key, _ in mock_bucket_inference_results.items():
            document_stem = DocumentStem(Path(file_key).stem)

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
                assert snapshot == await index_document_passages(
                    run_output_identifier=run_output_identifier,
                    config=test_aggregate_config,
                    document_stem=document_stem,
                    vespa_connection_pool=vespa_connection_pool,
                )


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_index_aggregate_results_for_batch_of_documents(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    mock_s3_client,
    mock_bucket: str,
    mock_bucket_inference_results: dict[str, dict[str, Any]],
    aggregate_inference_results_document_stems: list[DocumentStem],
    mock_run_output_identifier_str: str,
    s3_prefix_inference_results: str,
    test_aggregate_config,
) -> None:
    """Test that we loaded the inference results from the mock bucket."""

    run_output_identifier = RunOutputIdentifier(mock_run_output_identifier_str)

    with patch(
        "flows.index_from_aggregate_results.get_vespa_search_adapter_from_aws_secrets",
        return_value=local_vespa_search_adapter,
    ):
        await index_aggregate_results_for_batch_of_documents(
            run_output_identifier=run_output_identifier,
            document_stems=aggregate_inference_results_document_stems,
            config_json=test_aggregate_config.to_json(),
        )

        # Verify that the final data in vespa matches the expected results
        async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
            for file_key in mock_bucket_inference_results.keys():
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
                expected_concepts = mock_bucket_inference_results[file_key]

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
                        VespaConcept.model_validate(c)
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
async def test_index_aggregate_results_for_batch_of_documents__failure(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    mock_s3_client,
    mock_bucket: str,
    mock_bucket_inference_results: dict[str, dict[str, Any]],
    aggregate_inference_results_document_stems: list[DocumentStem],
    mock_run_output_identifier_str: str,
    s3_prefix_inference_results: str,
    test_aggregate_config,
) -> None:
    """Test that we handled the exception correctly during passage indexing."""

    non_existent_stem = DocumentStem("non_existent_document")
    document_stems = aggregate_inference_results_document_stems + [non_existent_stem]

    run_output_identifier = RunOutputIdentifier(mock_run_output_identifier_str)

    with patch(
        "flows.index_from_aggregate_results.get_vespa_search_adapter_from_aws_secrets",
        return_value=local_vespa_search_adapter,
    ):
        # Index the aggregated inference results from S3 to Vespa
        with pytest.raises(ValueError) as excinfo:
            await index_aggregate_results_for_batch_of_documents(
                run_output_identifier=run_output_identifier,
                config_json=test_aggregate_config.to_json(),
                document_stems=document_stems,
            )

        assert f"Failed to process 1/{len(document_stems)} documents" in str(
            excinfo.value
        )


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_run_indexing_from_aggregate_results__invokes_subdeployments_correctly(
    vespa_app,
    mock_s3_client,
    mock_bucket_inference_results: dict[str, dict[str, Any]],
    aggregate_inference_results_document_stems: list[DocumentStem],
    mock_run_output_identifier_str: str,
    test_aggregate_config,
) -> None:
    """Test that run passage level indexing correctly from aggregated restuls."""

    run_output_identifier = RunOutputIdentifier(mock_run_output_identifier_str)

    with patch(
        "flows.index_from_aggregate_results.run_deployment"
    ) as mock_run_deployment:
        # Mock the response of the run_deployment function.
        flow_run_counter = 0

        async def mock_awaitable(*args, **kwargs):
            nonlocal flow_run_counter
            flow_run_counter += 1
            return FlowRun(
                flow_id=uuid.uuid4(),
                name=f"mock-run-{flow_run_counter}",
                state=Completed(),
            )

        mock_run_deployment.side_effect = mock_awaitable

        # Run indexing
        await run_indexing_from_aggregate_results(
            run_output_identifier=run_output_identifier,
            document_stems=aggregate_inference_results_document_stems,
            config=test_aggregate_config,
            batch_size=1,
        )

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
            assert call_params["config_json"] == test_aggregate_config.to_json()

        # Reset the mock_run_deployment call count
        mock_run_deployment.reset_mock()

        # Run indexing with no document_ids specified, which should run for all documents
        await run_indexing_from_aggregate_results(
            run_output_identifier=run_output_identifier,
            document_stems=None,
            config=test_aggregate_config,
            batch_size=1,
        )

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
            assert call_params["config_json"] == test_aggregate_config.to_json()


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_run_indexing_from_aggregate_results__handles_failures(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    mock_s3_client,
    mock_bucket: str,
    mock_bucket_inference_results: dict[str, dict[str, Any]],
    mock_run_output_identifier_str: str,
    aggregate_inference_results_document_stems: list[DocumentStem],
    s3_prefix_inference_results: str,
    test_aggregate_config,
) -> None:
    """Test that run passage level indexing correctly from aggregated restuls."""

    non_existent_stem = DocumentStem("non_existent_document")
    document_stems = aggregate_inference_results_document_stems + [non_existent_stem]

    run_output_identifier = RunOutputIdentifier(mock_run_output_identifier_str)

    # Assert that the indexing runs correctly when called as sub deployments and that we
    # continue on failure of one of the documents.
    with patch(
        "flows.index_from_aggregate_results.get_vespa_search_adapter_from_aws_secrets",
        return_value=local_vespa_search_adapter,
    ):
        with pytest.raises(ValueError) as excinfo:
            await run_indexing_from_aggregate_results(
                run_output_identifier=run_output_identifier,
                document_stems=document_stems,
                config=test_aggregate_config,
                batch_size=1,
            )

            assert (
                f"Some batches of documents had failures: 1/{len(document_stems)} failed."
                in str(excinfo.value)
            )

            # Assert that the summary artifact was created
            summary_artifact = await Artifact.get("Aggregate Indexing Summary")
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

    # Create some test concepts
    test_concepts = [
        SimpleConcept(id="Q123", name="Climate Change"),
        SimpleConcept(id="Q456", name="Carbon Emissions"),
        SimpleConcept(id="Q123", name="Climate Change"),  # Duplicate to test counting
        SimpleConcept(id="Q789", name="Renewable Energy"),
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

    # Create some test concepts
    test_concepts = [
        SimpleConcept(id="Q123", name="Climate Change"),
        SimpleConcept(id="Q456", name="Carbon Emissions"),
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
