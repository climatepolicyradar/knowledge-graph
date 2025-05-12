import json
from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.s3 import _s3_object_read_text
from cpr_sdk.search_adaptors import VespaSearchAdapter
from prefect.logging import disable_run_logger
from vespa.io import VespaResponse

from flows.boundary import (
    ConceptModel,
    Operation,
    TextBlockId,
    VespaDataId,
    get_document_passages_from_vespa,
    op_to_fn,
    run_partial_updates_of_concepts_for_document_passages,
    update_concepts_on_existing_vespa_concepts,
    update_feed_result_callback,
    update_s3_with_update_concepts_counts,
)
from flows.index import (
    CONCEPTS_COUNTS_PREFIX_DEFAULT,
    Config,
    index_labelled_passages_from_s3_to_vespa,
)
from scripts.cloud import ClassifierSpec
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage


@pytest.mark.asyncio
@pytest.mark.vespa
async def test_run_partial_updates_of_concepts_for_document_passages(
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
    mock_bucket,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files: list[str],
    labelled_passage_fixture_ids: list[str],
    mock_bucket_labelled_passages,
    mock_s3_client,
) -> None:
    """Test that we can run partial updates of concepts for document passages."""
    document_fixture = labelled_passage_fixture_files[0]
    document_import_id = labelled_passage_fixture_ids[0]
    document_object_uri = (
        f"s3://{mock_bucket}/{s3_prefix_labelled_passages}/{document_fixture}"
    )

    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        # Confirm that the example concepts are not in the document passages
        initial_passages = await get_document_passages_from_vespa(
            document_import_id=document_import_id,
            text_blocks_ids=["1570", "1273", "1052"],
            vespa_connection_pool=vespa_connection_pool,
        )

    # Make the lists' orders deterministic for comparisons
    s = partial(sorted, key=lambda x: x[0])

    assert s(initial_passages) == s(
        [
            (
                "id:doc_search:document_passage::CCLW.executive.10014.4470.1052",
                VespaPassage(
                    family_name=None,
                    family_description=None,
                    family_source=None,
                    family_import_id=None,
                    family_slug=None,
                    family_category=None,
                    family_publication_ts=None,
                    family_geography=None,
                    family_geographies=None,
                    document_import_id=None,
                    document_slug=None,
                    document_languages=None,
                    document_content_type=None,
                    document_cdn_object=None,
                    document_source_url=None,
                    corpus_type_name=None,
                    corpus_import_id=None,
                    metadata=None,
                    concepts=[
                        VespaConcept(
                            id="concept_2_2",
                            name="sectors",
                            parent_concepts=[
                                {"name": "Q2-name", "id": "Q2"},
                                {"name": "Q3-name", "id": "Q3"},
                            ],
                            parent_concept_ids_flat="Q2,Q3,",
                            model="sectors_model",
                            end=11,
                            start=0,
                            timestamp=datetime(2024, 9, 26, 16, 15, 39, 817896),
                        ),
                        VespaConcept(
                            id="concept_2_2",
                            name="environment",
                            parent_concepts=[
                                {"name": "Q2-name", "id": "Q2"},
                                {"name": "Q3-name", "id": "Q3"},
                            ],
                            parent_concept_ids_flat="Q2,Q3,",
                            model="environment_model",
                            end=31,
                            start=15,
                            timestamp=datetime(2024, 9, 26, 16, 15, 39, 817896),
                        ),
                    ],
                    relevance=None,
                    rank_features=None,
                    concept_counts=None,
                    text_block="Environmental protection provisions integrated into sectoral policy documents (energy, agriculture, industry, trade, transport, constructions and public health)",
                    text_block_id="1052",
                    text_block_type="BlockType.TABLE_CELL",
                    text_block_page=73,
                    text_block_coords=[
                        (529.3079833984375, 133.9416046142578),
                        (613.2672119140625, 133.9416046142578),
                        (613.2672119140625, 314.8775939941406),
                        (529.3079833984375, 314.8775939941406),
                    ],
                ),
            ),
            (
                "id:doc_search:document_passage::CCLW.executive.10014.4470.1273",
                VespaPassage(
                    family_name=None,
                    family_description=None,
                    family_source=None,
                    family_import_id=None,
                    family_slug=None,
                    family_category=None,
                    family_publication_ts=None,
                    family_geography=None,
                    family_geographies=None,
                    document_import_id=None,
                    document_slug=None,
                    document_languages=None,
                    document_content_type=None,
                    document_cdn_object=None,
                    document_source_url=None,
                    corpus_type_name=None,
                    corpus_import_id=None,
                    metadata=None,
                    concepts=[
                        VespaConcept(
                            id="concept_1_1",
                            name="just transition",
                            parent_concepts=[
                                {"name": "Q1-name", "id": "Q1"},
                                {"name": "Q2-name", "id": "Q2"},
                            ],
                            parent_concept_ids_flat="Q1,Q2,",
                            model="just transition_model",
                            end=20,
                            start=1,
                            timestamp=datetime(2024, 9, 26, 16, 15, 39, 817896),
                        ),
                        VespaConcept(
                            id="concept_1_1",
                            name="sectors",
                            parent_concepts=[
                                {"name": "Q1-name", "id": "Q1"},
                                {"name": "Q2-name", "id": "Q2"},
                            ],
                            parent_concept_ids_flat="Q1,Q2,",
                            model="sectors_model",
                            end=31,
                            start=18,
                            timestamp=datetime(2024, 9, 26, 16, 15, 39, 817896),
                        ),
                    ],
                    relevance=None,
                    rank_features=None,
                    concept_counts=None,
                    text_block="5",
                    text_block_id="1273",
                    text_block_type="BlockType.TABLE_CELL",
                    text_block_page=79,
                    text_block_coords=[
                        (528.7536010742188, 494.80560302734375),
                        (613.2528076171875, 494.80560302734375),
                        (613.2528076171875, 509.731201171875),
                        (528.7536010742188, 509.731201171875),
                    ],
                ),
            ),
            (
                "id:doc_search:document_passage::CCLW.executive.10014.4470.1570",
                VespaPassage(
                    family_name=None,
                    family_description=None,
                    family_source=None,
                    family_import_id=None,
                    family_slug=None,
                    family_category=None,
                    family_publication_ts=None,
                    family_geography=None,
                    family_geographies=None,
                    document_import_id=None,
                    document_slug=None,
                    document_languages=None,
                    document_content_type=None,
                    document_cdn_object=None,
                    document_source_url=None,
                    corpus_type_name=None,
                    corpus_import_id=None,
                    metadata=None,
                    concepts=[
                        VespaConcept(
                            id="concept_1723_1723",
                            name="floods",
                            parent_concepts=[
                                {"name": "Q1723-name", "id": "Q1723"},
                                {"name": "Q1724-name", "id": "Q1724"},
                            ],
                            parent_concept_ids_flat="Q1723,Q1724,",
                            model="floods_model",
                            end=15,
                            start=0,
                            timestamp=datetime(2024, 9, 26, 16, 15, 39, 817896),
                        ),
                        VespaConcept(
                            id="concept_1723_1723",
                            name="just transition",
                            parent_concepts=[
                                {"name": "Q1723-name", "id": "Q1723"},
                                {"name": "Q1724-name", "id": "Q1724"},
                            ],
                            parent_concept_ids_flat="Q1723,Q1724,",
                            model="just transition_model",
                            end=34,
                            start=16,
                            timestamp=datetime(2024, 9, 26, 16, 15, 39, 817896),
                        ),
                    ],
                    relevance=None,
                    rank_features=None,
                    concept_counts=None,
                    text_block="7",
                    text_block_id="1570",
                    text_block_type="BlockType.TABLE_CELL",
                    text_block_page=86,
                    text_block_coords=[
                        (691.6176147460938, 495.1296081542969),
                        (784.0223999023438, 494.41680908203125),
                        (784.0223999023438, 509.349609375),
                        (691.6176147460938, 509.349609375),
                    ],
                ),
            ),
        ]
    )

    # Confirm that we can add the example concepts to the document passages
    #
    # The model names should be different. It's to do with bad fixture setup by us.
    test_counts = Counter(
        {
            ConceptModel(
                wikibase_id=WikibaseID("Q10015"),
                model_name='KeywordClassifier("professional services sector")',
            ): 2,
            ConceptModel(
                wikibase_id=WikibaseID("Q10014"),
                model_name='KeywordClassifier("professional services sector")',
            ): 1,
        }
    )

    (
        merge_serialise_concepts_cb,
        vespa_response_handler_cb,
        concepts_counts_updater_cb,
    ) = op_to_fn(Operation.INDEX)

    assert (
        test_counts
        == await run_partial_updates_of_concepts_for_document_passages.fn(
            document_importer=(document_import_id, document_object_uri),
            vespa_search_adapter=local_vespa_search_adapter,
            cache_bucket=mock_bucket,
            concepts_counts_prefix=CONCEPTS_COUNTS_PREFIX_DEFAULT,
            merge_serialise_concepts_cb=merge_serialise_concepts_cb,
            vespa_response_handler_cb=vespa_response_handler_cb,
            concepts_counts_updater_cb=concepts_counts_updater_cb,
        )
    )
    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        updated_passages = await get_document_passages_from_vespa(
            document_import_id=document_import_id,
            text_blocks_ids=None,
            vespa_connection_pool=vespa_connection_pool,
        )
    updated_concepts = [
        concept
        for _, passage in updated_passages
        if passage.concepts
        for concept in passage.concepts
    ]

    # Original + fixture (.tests/flows/fixtures/CCLW.executive.10014.4470.json)
    assert len(updated_concepts) == 3663

    test_counts_serialised = {str(k): v for k, v in test_counts.items()}

    result = mock_s3_client.get_object(
        Bucket=mock_bucket,
        Key="concepts_counts/Q788/v4/CCLW.executive.10014.4470.json",
    )
    assert test_counts_serialised == json.loads(result["Body"].read().decode("utf-8"))


@pytest.mark.asyncio
async def test_index_labelled_passages_from_s3_to_vespa_doesnt_allow_latest(
    mock_bucket,
    mock_bucket_labelled_passages,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files,
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
) -> None:
    classifier_spec = ClassifierSpec(name="Q788", alias="latest")
    document_ids = [
        Path(labelled_passage_fixture_file).stem
        for labelled_passage_fixture_file in labelled_passage_fixture_files
    ]
    config = Config(
        cache_bucket=mock_bucket,
        vespa_search_adapter=local_vespa_search_adapter,
        as_deployment=False,
    )

    with pytest.raises(ValueError, match="`latest` is not allowed"):
        await index_labelled_passages_from_s3_to_vespa(
            classifier_specs=[classifier_spec],
            document_ids=document_ids,
            config=config,
        )


@pytest.mark.asyncio
@pytest.mark.vespa
async def test_index_labelled_passages_from_s3_to_vespa_with_document_ids_with_config(
    mock_bucket,
    mock_bucket_labelled_passages,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files,
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
) -> None:
    initial_passages_response = local_vespa_search_adapter.client.query(
        yql="select * from document_passage where true"
    )
    initial_concepts_count = sum(
        len(hit["fields"]["concepts"]) for hit in initial_passages_response.hits
    )

    classifier_spec = ClassifierSpec(name="Q788", alias="v4")
    document_ids = [
        Path(labelled_passage_fixture_file).stem
        for labelled_passage_fixture_file in labelled_passage_fixture_files
    ]
    config = Config(
        cache_bucket=mock_bucket,
        vespa_search_adapter=local_vespa_search_adapter,
        as_deployment=False,
    )

    await index_labelled_passages_from_s3_to_vespa(
        classifier_specs=[classifier_spec],
        document_ids=document_ids,
        config=config,
    )

    final_passages_response = local_vespa_search_adapter.client.query(
        yql="select * from document_passage where concepts.id contains \"Q\""
    )
    final_concepts_count = sum(
        len(hit["fields"]["concepts"]) for hit in final_passages_response.hits
    )

    assert initial_concepts_count < final_concepts_count
    # Original + fixture (.tests/flows/fixtures/*.json)
    assert final_concepts_count == 3935


@pytest.mark.asyncio
@pytest.mark.vespa
@pytest.mark.skip(reason="cannot test due to run_deployment usage")
async def test_index_labelled_passages_from_s3_to_vespa_with_document_ids_with_default_config(
    mock_bucket,
    mock_bucket_labelled_passages,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files,
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
) -> None:
    with (
        patch("flows.index.get_prefect_job_variable", return_value=mock_bucket),
        patch(
            "flows.boundary.get_vespa_search_adapter_from_aws_secrets",
            return_value=local_vespa_search_adapter,
        ),
        disable_run_logger(),
    ):
        initial_passages_response = local_vespa_search_adapter.client.query(
            yql="select * from document_passage where true"
        )
        initial_concepts_count = sum(
            len(hit["fields"]["concepts"]) for hit in initial_passages_response.hits
        )

        classifier_spec = ClassifierSpec(name="Q788", alias="v4")
        document_ids = [
            Path(labelled_passage_fixture_file).stem
            for labelled_passage_fixture_file in labelled_passage_fixture_files
        ]

        await index_labelled_passages_from_s3_to_vespa(
            classifier_specs=[classifier_spec],
            document_ids=document_ids,
        )

        final_passages_response = local_vespa_search_adapter.client.query(
            yql="select * from document_passage where true"
        )
        final_concepts_count = sum(
            len(hit["fields"]["concepts"]) for hit in final_passages_response.hits
        )

    assert initial_concepts_count < final_concepts_count
    # Original + fixture (.tests/flows/fixtures/*.json)
    assert final_concepts_count == 3935


def test_update_concepts_on_existing_vespa_concepts(
    example_vespa_concepts: list[VespaConcept],
) -> None:
    """Test that we can retrieve the updated passage concepts dict."""
    for concept in example_vespa_concepts:
        # Test we can add a concept to the passage concepts that doesn't already
        # exist.
        updated_passage_concepts = update_concepts_on_existing_vespa_concepts(
            passage=VespaPassage(
                text_block="Test text.",
                text_block_id="1",
                text_block_type="Text",
                concepts=[],
            ),
            concepts=[concept],
        )
        assert len(updated_passage_concepts) == 1
        assert updated_passage_concepts[0] == concept.model_dump(mode="json")

        # Test that we can remove old model concepts from the passage concepts and
        # add the new one.
        updated_passage_concepts = update_concepts_on_existing_vespa_concepts(
            passage=VespaPassage(
                text_block="Test text.",
                text_block_id="1",
                text_block_type="Text",
                concepts=[
                    VespaConcept(
                        id="1",
                        name="extreme weather",
                        parent_concepts=[{"name": "weather", "id": "Q123"}],
                        parent_concept_ids_flat="Q123,",
                        model=concept.model,  # Ensure the models are the same
                        end=100,
                        start=0,
                        timestamp=datetime.now(),
                    )
                ],
            ),
            concepts=[concept],
        )
        assert len(updated_passage_concepts) == 1
        assert updated_passage_concepts[0] == concept.model_dump(mode="json")

        # Test that we can add new concepts and retain concepts from other models
        updated_passage_concepts = update_concepts_on_existing_vespa_concepts(
            passage=VespaPassage(
                text_block="Test text.",
                text_block_id="1",
                text_block_type="Text",
                concepts=[
                    VespaConcept(
                        id="1",
                        name="extreme weather",
                        parent_concepts=[{"name": "weather", "id": "Q123"}],
                        parent_concept_ids_flat="Q123,",
                        model="non-existent model",  # Ensure the models are NOT the same
                        end=100,
                        start=0,
                        timestamp=datetime.now(),
                    )
                ],
            ),
            concepts=[concept],
        )
        assert len(updated_passage_concepts) == 2
        assert concept.model_dump(mode="json") in updated_passage_concepts


def test_update_feed_result_callback():
    failures: list[VespaResponse] = []
    concepts_counts: Counter[ConceptModel] = Counter()
    grouped_concepts: dict[TextBlockId, list[VespaConcept]] = {
        "18593": [
            VespaConcept(
                id="Q100",
                name="sectors",
                parent_concepts=[
                    {"name": "Q2-name", "id": "Q200"},
                    {"name": "Q3-name", "id": "Q300"},
                ],
                parent_concept_ids_flat="Q200,Q300,",
                model="sectors_model",
                end=11,
                start=0,
                timestamp=datetime(2024, 9, 26, 16, 15, 39, 817896),
            ),
        ]
    }
    response: VespaResponse = VespaResponse(
        json={},
        status_code=200,
        url="test-url",
        operation_type="update",
    )
    data_id: VespaDataId = "UNFCCC.party.1062.0.18593"

    assert (
        update_feed_result_callback(
            failures=failures,
            concepts_counts=concepts_counts,
            grouped_concepts=grouped_concepts,
            response=response,
            data_id=data_id,
        )
        is None
    )

    assert concepts_counts == Counter(
        {ConceptModel(wikibase_id=WikibaseID("Q100"), model_name="sectors_model"): 1}
    )
    assert failures == []


def test_update_feed_result_callback_not_successful_response():
    failures: list[VespaResponse] = []
    concepts_counts: Counter[ConceptModel] = Counter()
    grouped_concepts: dict[TextBlockId, list[VespaConcept]] = {}
    response: VespaResponse = VespaResponse(
        json={},
        status_code=403,
        url="test-url",
        operation_type="update",
    )
    data_id: VespaDataId = "UNFCCC.party.1062.0.18593"

    assert (
        update_feed_result_callback(
            failures=failures,
            concepts_counts=concepts_counts,
            grouped_concepts=grouped_concepts,
            response=response,
            data_id=data_id,
        )
        is None
    )

    assert concepts_counts == Counter()
    assert failures == [response]


@pytest.mark.asyncio
async def test_update_s3_with_update_concepts_counts(
    mock_bucket,
    mock_s3_client,
):
    document_import_id = "CCLW.executive.10014.4470"
    document_object_uri = (
        f"s3://{mock_bucket}/labelled_passages/Q787/v4/{document_import_id}.json"
    )
    document_importer = (document_import_id, document_object_uri)
    concepts_counter = Counter(
        {
            ConceptModel(
                wikibase_id=WikibaseID("Q123"),
                model_name='KeywordClassifier("concept1")',
            ): 1,
        }
    )
    document_labelled_passages: list[LabelledPassage] = []

    assert (
        await update_s3_with_update_concepts_counts(
            document_importer=document_importer,
            concepts_counts=concepts_counter,
            cache_bucket=mock_bucket,
            concepts_counts_prefix=CONCEPTS_COUNTS_PREFIX_DEFAULT,
            document_labelled_passages=document_labelled_passages,
        )
        is None
    )

    assert json.loads(
        _s3_object_read_text(
            s3_path=f"s3://{mock_bucket}/{CONCEPTS_COUNTS_PREFIX_DEFAULT}/Q787/v4/{document_import_id}.json"
        )
    ) == {"Q123:concept1": 1}
