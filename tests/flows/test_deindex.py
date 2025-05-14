import json
import os
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from functools import partial
from io import BytesIO
from typing import Any

import pytest
import vespa
import vespa.application
from botocore.exceptions import ClientError
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.s3 import S3_PATTERN, _s3_object_read_text
from cpr_sdk.search_adaptors import VespaSearchAdapter
from vespa.io import VespaResponse

import flows.count_family_document_concepts as count_family_document_concepts
from flows.boundary import (
    ConceptModel,
    DocumentImporter,
    DocumentImportId,
    DocumentObjectUri,
    Operation,
    TextBlockId,
    VespaHitId,
    get_data_id_from_vespa_hit_id,
    get_document_passage_from_vespa,
    op_to_fn,
    remove_concepts_from_existing_vespa_concepts,
    remove_result_callback,
    run_partial_updates_of_concepts_for_document_passages,
    s3_paths_or_s3_prefixes,
    serialise_concepts_counts,
    update_concepts_on_existing_vespa_concepts,
    update_s3_with_all_successes,
    update_s3_with_latest_concepts_counts,
    update_s3_with_some_successes,
)
from flows.deindex import (
    CONCEPTS_COUNTS_PREFIX_DEFAULT,
    DEFAULT_DEINDEXING_TASK_BATCH_SIZE,
    DEFAULT_DOCUMENTS_BATCH_SIZE,
    Config,
    cleanups_by_s3,
    deindex_labelled_passages_from_s3_to_vespa,
    find_all_classifier_specs_for_latest,
    search_s3_for_aliases,
)
from flows.inference import DOCUMENT_TARGET_PREFIX_DEFAULT, serialise_labels
from flows.utils import _s3_object_write_bytes, _s3_object_write_text
from scripts.cloud import AwsEnv, ClassifierSpec
from src.concept import Concept
from src.exceptions import PartialUpdateError
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.span import Span


def test_remove_concepts_from_existing_vespa_concepts():
    existing_concepts = [
        VespaConcept(
            id="Q123",
            name="concept1",
            model='KeywordClassifier("concept1")',
            start=0,
            end=10,
            timestamp=datetime.now(),
        ),
        VespaConcept(
            id="Q456",
            name="concept2",
            model='KeywordClassifier("concept2")',
            start=20,
            end=30,
            timestamp=datetime.now(),
        ),
        VespaConcept(
            id="Q789",
            name="concept3",
            model='KeywordClassifier("concept3")',
            start=40,
            end=50,
            timestamp=datetime.now(),
        ),
    ]

    concepts_to_remove = [
        VespaConcept(
            id="Q456",
            name="concept2",
            model='KeywordClassifier("concept2")',
            start=20,
            end=30,
            timestamp=datetime.now(),
        ),
    ]

    passage = VespaPassage(
        text_block="Test text.",
        text_block_id="1",
        text_block_type="Text",
        concepts=existing_concepts,
    )

    result = remove_concepts_from_existing_vespa_concepts(passage, concepts_to_remove)

    assert len(result) == 2
    assert result[0].get("id") == "Q123"
    assert result[1].get("id") == "Q789"

    result_empty = remove_concepts_from_existing_vespa_concepts(passage, [])
    assert len(result_empty) == 3
    assert result_empty[0].get("id") == "Q123"
    assert result_empty[1].get("id") == "Q456"
    assert result_empty[2].get("id") == "Q789"

    non_matching_concepts = [
        VespaConcept(
            id="Q999",
            name="non-matching",
            model='KeywordClassifier("non-matching")',
            start=0,
            end=10,
            timestamp=datetime.now(),
        ),
    ]
    result_non_matching = remove_concepts_from_existing_vespa_concepts(
        passage, non_matching_concepts
    )
    assert len(result_non_matching) == 3
    assert result_non_matching[0].get("id") == "Q123"
    assert result_non_matching[1].get("id") == "Q456"
    assert result_non_matching[2].get("id") == "Q789"


def test_remove_concepts_from_existing_vespa_concepts_empty():
    existing_concepts = [
        VespaConcept(
            id="Q123",
            name="concept1",
            model='KeywordClassifier("concept1")',
            start=0,
            end=10,
            timestamp=datetime.now(),
        ),
        VespaConcept(
            id="Q456",
            name="concept2",
            model='KeywordClassifier("concept2")',
            start=20,
            end=30,
            timestamp=datetime.now(),
        ),
        VespaConcept(
            id="Q789",
            name="concept3",
            model='KeywordClassifier("concept3")',
            start=40,
            end=50,
            timestamp=datetime.now(),
        ),
    ]

    passage = VespaPassage(
        text_block="Test text.",
        text_block_id="1",
        text_block_type="Text",
        concepts=existing_concepts,
    )

    result_empty = remove_concepts_from_existing_vespa_concepts(passage, [])
    assert len(result_empty) == 3
    assert result_empty[0].get("id") == "Q123"
    assert result_empty[1].get("id") == "Q456"
    assert result_empty[2].get("id") == "Q789"


def test_remove_concepts_from_existing_vespa_concepts_non_matching():
    existing_concepts = [
        VespaConcept(
            id="Q123",
            name="concept1",
            model='KeywordClassifier("concept1")',
            start=0,
            end=10,
            timestamp=datetime.now(),
        ),
        VespaConcept(
            id="Q456",
            name="concept2",
            model='KeywordClassifier("concept2")',
            start=20,
            end=30,
            timestamp=datetime.now(),
        ),
        VespaConcept(
            id="Q789",
            name="concept3",
            model='KeywordClassifier("concept3")',
            start=40,
            end=50,
            timestamp=datetime.now(),
        ),
    ]

    passage = VespaPassage(
        text_block="Test text.",
        text_block_id="1",
        text_block_type="Text",
        concepts=existing_concepts,
    )

    non_matching_concepts = [
        VespaConcept(
            id="Q999",
            name="non-matching",
            model='KeywordClassifier("non-matching")',
            start=0,
            end=10,
            timestamp=datetime.now(),
        ),
    ]
    result_non_matching = remove_concepts_from_existing_vespa_concepts(
        passage, non_matching_concepts
    )
    assert len(result_non_matching) == 3
    assert result_non_matching[0].get("id") == "Q123"
    assert result_non_matching[1].get("id") == "Q456"
    assert result_non_matching[2].get("id") == "Q789"


def test_update_s3_with_all_successes(
    mock_bucket,
    mock_s3_client,
):
    """Test that update_s3_with_all_successes correctly deletes both S3 objects."""
    document_import_id = "CCLW.executive.10014.4470"

    document_object_uri = (
        f"s3://{mock_bucket}/labelled_passages/Q787/v4/{document_import_id}.json"
    )

    expected_concepts_counts_key = (
        f"{CONCEPTS_COUNTS_PREFIX_DEFAULT}/Q787/v4/{document_import_id}.json"
    )
    expected_labelled_passages_key = (
        f"labelled_passages/Q787/v4/{document_import_id}.json"
    )

    # Put some objects to be deleted
    body = BytesIO(json.dumps({}).encode("utf-8"))
    mock_s3_client.put_object(
        Bucket=mock_bucket,
        Key=expected_concepts_counts_key,
        Body=body,
        ContentType="application/json",
    )
    mock_s3_client.put_object(
        Bucket=mock_bucket,
        Key=expected_labelled_passages_key,
        Body=body,
        ContentType="application/json",
    )

    assert (
        update_s3_with_all_successes(
            document_object_uri=document_object_uri,
            cache_bucket=mock_bucket,
            concepts_counts_prefix=CONCEPTS_COUNTS_PREFIX_DEFAULT,
        )
        is None
    )

    # Ensure the objects were deleted
    with pytest.raises(ClientError):
        mock_s3_client.head_object(Bucket=mock_bucket, Key=expected_concepts_counts_key)
    with pytest.raises(ClientError):
        mock_s3_client.head_object(
            Bucket=mock_bucket, Key=expected_labelled_passages_key
        )


@pytest.mark.vespa
def test_update_s3_with_some_successes(
    mock_bucket: str,
):
    document_import_id = "CCLW.executive.10014.4470"

    document_object_uri = (
        f"s3://{mock_bucket}/labelled_passages/Q787/v4/{document_import_id}.json"
    )

    concept_model_1 = ConceptModel(
        wikibase_id=WikibaseID("Q123"),
        model_name='KeywordClassifier("professional services sector")',
    )

    concept_model_1 = ConceptModel(
        wikibase_id=WikibaseID("Q456"), model_name='KeywordClassifier("agriculture")'
    )

    concepts_to_keep = Counter({concept_model_1: 3})

    document_labelled_passages: list[LabelledPassage] = [
        LabelledPassage(
            id="lp1",
            text="once upon a time",
            spans=[
                Span(
                    text="once",
                    start_index=0,
                    end_index=1,
                    concept_id=WikibaseID("Q123"),
                ),
                Span(
                    text="upon",
                    start_index=0,
                    end_index=1,
                    concept_id=WikibaseID("Q456"),
                ),
            ],
        ),
        LabelledPassage(
            id="lp2",
            text="lorem ipsum",
            spans=[
                Span(
                    text="ipsum",
                    start_index=0,
                    end_index=1,
                    concept_id=WikibaseID("Q789"),
                ),
            ],
        ),
    ]

    assert (
        update_s3_with_some_successes(
            document_object_uri,
            concepts_to_keep,
            document_labelled_passages,
            mock_bucket,
            CONCEPTS_COUNTS_PREFIX_DEFAULT,
        )
        is None
    )

    assert json.loads(
        _s3_object_read_text(
            s3_path=f"s3://{mock_bucket}/labelled_passages/Q787/v4/{document_import_id}.json"
        )
    ) == [
        '{"id":"lp1","text":"once upon a time","spans":[{"text":"upon","start_index":0,"end_index":1,"concept_id":"Q456","labellers":[],"timestamps":[],"id":"bpp4juku","labelled_text":"u"}],"metadata":{}}',
        '{"id":"lp2","text":"lorem ipsum","spans":[],"metadata":{}}',
    ]
    assert json.loads(
        _s3_object_read_text(
            s3_path=f"s3://{mock_bucket}/concepts_counts/Q787/v4/{document_import_id}.json"
        )
    ) == {"Q456:agriculture": 3}


@pytest.mark.asyncio
async def test_update_s3_with_latest_concepts_counts_all_success(
    mock_bucket,
    mock_s3_client,
):
    document_import_id = "CCLW.executive.10014.4470"
    document_object_uri = (
        f"s3://{mock_bucket}/labelled_passages/Q787/v4/{document_import_id}.json"
    )
    document_importer = (document_import_id, document_object_uri)

    expected_concepts_counts_key = (
        f"{CONCEPTS_COUNTS_PREFIX_DEFAULT}/Q787/v4/{document_import_id}.json"
    )
    expected_labelled_passages_key = (
        f"labelled_passages/Q787/v4/{document_import_id}.json"
    )

    # Put some objects to be deleted
    body = BytesIO(json.dumps({}).encode("utf-8"))
    mock_s3_client.put_object(
        Bucket=mock_bucket,
        Key=expected_concepts_counts_key,
        Body=body,
        ContentType="application/json",
    )
    mock_s3_client.put_object(
        Bucket=mock_bucket,
        Key=expected_labelled_passages_key,
        Body=body,
        ContentType="application/json",
    )

    concepts_counter = Counter(
        {
            ConceptModel(
                wikibase_id=WikibaseID("Q123"),
                model_name='KeywordClassifier("concept1")',
            ): 0,
        }
    )

    assert (
        await update_s3_with_latest_concepts_counts(
            document_importer=document_importer,
            concepts_counts=concepts_counter,
            cache_bucket=mock_bucket,
            concepts_counts_prefix=CONCEPTS_COUNTS_PREFIX_DEFAULT,
            document_labelled_passages=[],  # Not used in this branch
        )
        is None
    )

    # Ensure the objects were deleted
    with pytest.raises(ClientError):
        mock_s3_client.head_object(Bucket=mock_bucket, Key=expected_concepts_counts_key)
    with pytest.raises(ClientError):
        mock_s3_client.head_object(
            Bucket=mock_bucket, Key=expected_labelled_passages_key
        )


@pytest.mark.asyncio
async def test_update_s3_with_latest_concepts_counts_some_success(
    mock_bucket,
    mock_s3_client,
):
    document_import_id = "CCLW.executive.10014.4470"
    document_object_uri = (
        f"s3://{mock_bucket}/labelled_passages/Q787/v4/{document_import_id}.json"
    )
    document_importer = (document_import_id, document_object_uri)

    document_labelled_passages: list[LabelledPassage] = [
        LabelledPassage(
            id="lp1",
            text="once upon a time",
            spans=[
                Span(
                    text="once",
                    start_index=0,
                    end_index=1,
                    concept_id=WikibaseID("Q123"),
                ),
                Span(
                    text="upon",
                    start_index=0,
                    end_index=1,
                    concept_id=WikibaseID("Q456"),
                ),
            ],
        ),
        LabelledPassage(
            id="lp2",
            text="lorem ipsum",
            spans=[
                Span(
                    text="ipsum",
                    start_index=0,
                    end_index=1,
                    concept_id=WikibaseID("Q789"),
                ),
            ],
        ),
    ]

    concepts_counter = Counter(
        {
            ConceptModel(
                wikibase_id=WikibaseID("Q123"),
                model_name='KeywordClassifier("concept1")',
            ): 1,
            ConceptModel(
                wikibase_id=WikibaseID("Q456"),
                model_name='KeywordClassifier("concept2")',
            ): 0,
        }
    )

    assert (
        await update_s3_with_latest_concepts_counts(
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
            s3_path=f"s3://{mock_bucket}/labelled_passages/Q787/v4/{document_import_id}.json"
        )
    ) == [
        '{"id":"lp1","text":"once upon a time","spans":[{"text":"once","start_index":0,"end_index":1,"concept_id":"Q123","labellers":[],"timestamps":[],"id":"z55va9eh","labelled_text":"o"}],"metadata":{}}',
        '{"id":"lp2","text":"lorem ipsum","spans":[],"metadata":{}}',
    ]
    assert json.loads(
        _s3_object_read_text(
            s3_path=f"s3://{mock_bucket}/concepts_counts/Q787/v4/{document_import_id}.json"
        )
    ) == {"Q123:concept1": 1}


async def partial_update_text_block(
    text_block: tuple[VespaHitId, VespaPassage],
    concepts: list[VespaConcept],  # A possibly empty list
    vespa_connection_pool: vespa.application.VespaAsync,
    update_function: Callable[[VespaPassage, list[VespaConcept]], list[dict[str, Any]]],
):
    """Partial update a singular text block and its concepts."""
    document_passage_id, document_passage = text_block[0], text_block[1]

    data_id = get_data_id_from_vespa_hit_id(document_passage_id)

    serialised_concepts = update_function(document_passage, concepts)

    response: VespaResponse = await vespa_connection_pool.update_data(
        schema="document_passage",
        namespace="doc_search",
        data_id=data_id,
        fields={"concepts": serialised_concepts},
    )

    if not response.is_successful():
        raise PartialUpdateError(data_id, response.get_status_code())

    return None


@pytest.mark.asyncio
@pytest.mark.vespa
async def test_run_partial_updates_of_concepts_for_document_passages(
    mock_bucket: str,
    mock_s3_client,
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
):
    document_import_id_remove: DocumentImportId = "CCLW.executive.10014.4470"

    document_object_uri_remove: DocumentObjectUri = f"s3://{mock_bucket}/{DOCUMENT_TARGET_PREFIX_DEFAULT}/Q760/v4/{document_import_id_remove}.json"
    document_importer_remove: DocumentImporter = (
        document_import_id_remove,
        document_object_uri_remove,
    )

    document_import_id_keep: DocumentImportId = "CCLW.executive.4934.1571"

    document_object_uri_keep: DocumentObjectUri = f"s3://{mock_bucket}/{DOCUMENT_TARGET_PREFIX_DEFAULT}/Q787/v4/{document_import_id_keep}.json"
    _document_importer_keep: DocumentImporter = (
        document_import_id_keep,
        document_object_uri_keep,
    )

    # S3 state before
    # Document passages

    # To be removed
    labelled_passages_remove: list[LabelledPassage] = [
        LabelledPassage(
            id="1570",
            text="National Council for Sustainable Development of the Kyrgyz Republic",
            spans=[
                Span(
                    text="National Council for Sustainable Development of the Kyrgyz Republic",
                    start_index=0,
                    end_index=10,
                    concept_id=WikibaseID("Q760"),
                    labellers=['KeywordClassifier("nuclear sector")'],
                    timestamps=["2021-09-29T14:00:00.000Z"],
                )
            ],
            metadata={
                "concept": {
                    "preferred_label": "nuclear sector",
                    "alternative_labels": [
                        "nuclear energy",
                        "nuclear power",
                        "nuclear industry",
                        "atomic energy",
                        "nuclear technology",
                        "nuclear reactor",
                        "nuclear fuel",
                        "radiation protection",
                        "nuclear engineering",
                        "uranium processing",
                    ],
                    "negative_labels": [],
                    "description": "Activities related to the development, production, and management of nuclear energy and technology.",
                    "wikibase_id": "Q25285",
                    "subconcept_of": ["Q11434"],
                    "has_subconcept": [],
                    "related_concepts": ["Q177", "Q7397", "Q8090"],
                    "definition": None,
                    "labelled_passages": [],
                }
            },
        ),
    ]
    serialised_labelled_passages_remove = serialise_labels(labelled_passages_remove)

    # It's setup, now write it to S3 to be available
    _s3_object_write_bytes(
        s3_uri=document_object_uri_remove,
        bytes=serialised_labelled_passages_remove,
    )

    # To be kept
    labelled_passages_keep: list[LabelledPassage] = [
        LabelledPassage(
            id="197",
            text="Some random text on climate change.",
            spans=[
                Span(
                    text="Some random text on climate change.",
                    start_index=10,
                    end_index=11,
                    concept_id=WikibaseID("Q787"),
                    labellers=['KeywordClassifier("forestry sector")'],
                    timestamps=["2021-09-29T14:00:00.000Z"],
                )
            ],
            metadata={
                "concept": {
                    "preferred_label": "forestry sector",
                    "alternative_labels": [
                        "forest pest",
                        "wood industry",
                        "forest industry",
                        "silviculture",
                        "forest management",
                        "forestry",
                        "forestry sector",
                        "forest fire prevention",
                        "logging",
                        "lumber",
                        "prevention of forest fires",
                        "timber",
                    ],
                    "negative_labels": [],
                    "description": "Activities that relate to the production of goods and services from forests.",
                    "wikibase_id": "Q787",
                    "subconcept_of": ["Q709"],
                    "has_subconcept": [],
                    "related_concepts": ["Q7", "Q5", "Q4"],
                    "definition": None,
                    "labelled_passages": [],
                }
            },
        ),
    ]
    serialised_labelled_passages_keep = serialise_labels(labelled_passages_keep)

    # It's setup, now write it to S3 to be available
    _s3_object_write_bytes(
        s3_uri=document_object_uri_keep,
        bytes=serialised_labelled_passages_keep,
    )

    # Concepts counts
    concepts_counts_remove: Counter[ConceptModel] = Counter(
        {
            ConceptModel(
                wikibase_id=WikibaseID("Q760"),
                model_name='KeywordClassifier("nuclear sector")',
            ): 1
        }
    )
    serialised_concepts_counts_remove = serialise_concepts_counts(
        concepts_counts_remove
    )
    concepts_counts_remove_uri = f"s3://{mock_bucket}/{CONCEPTS_COUNTS_PREFIX_DEFAULT}/Q760/v4/{document_import_id_remove}.json"

    # It's setup, now write it to S3 to be available
    _s3_object_write_text(
        s3_uri=concepts_counts_remove_uri,
        text=serialised_concepts_counts_remove,
    )

    concepts_counts_keep: Counter[ConceptModel] = Counter(
        {
            ConceptModel(
                wikibase_id=WikibaseID("Q787"),
                model_name='KeywordClassifier("forestry sector")',
            ): 1,
        }
    )
    serialised_concepts_counts_keep = serialise_concepts_counts(concepts_counts_keep)
    concepts_counts_keep_uri = f"s3://{mock_bucket}/{CONCEPTS_COUNTS_PREFIX_DEFAULT}/Q787/v4/{document_import_id_keep}.json"

    # It's setup, now write it to S3 to be available
    _s3_object_write_text(
        s3_uri=concepts_counts_keep_uri,
        text=serialised_concepts_counts_keep,
    )

    # Vespa state before
    # Document passages
    hit_id_1, passage_remove_1_pre = get_document_passage_from_vespa(
        text_block_id=labelled_passages_remove[0].id,
        document_import_id=document_import_id_remove,
        vespa_search_adapter=local_vespa_search_adapter,
    )

    _hit_id, passage_keep_1_pre = get_document_passage_from_vespa(
        text_block_id=labelled_passages_keep[0].id,
        document_import_id=document_import_id_keep,
        vespa_search_adapter=local_vespa_search_adapter,
    )

    # Add the concepts expected to be removed, to the document passage
    # in the local Vespa instance for the test.
    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        assert (
            await partial_update_text_block(
                text_block=(hit_id_1, passage_remove_1_pre),
                concepts=[
                    VespaConcept(
                        id="Q760",
                        name="nuclear sector",
                        model='KeywordClassifier("nuclear sector")',
                        start=0,
                        end=10,
                        timestamp=datetime.fromisoformat(
                            "2021-09-29T14:00:00.000Z".replace("Z", "+00:00")
                        ),
                    ),
                ],
                vespa_connection_pool=vespa_connection_pool,
                update_function=update_concepts_on_existing_vespa_concepts,
            )
            is None
        )

    test_counts = Counter(
        {
            ConceptModel(
                wikibase_id=WikibaseID("Q760"),
                model_name='KeywordClassifier("nuclear sector")',
            ): 0
        }
    )

    (
        merge_serialise_concepts_cb,
        vespa_response_handler_cb,
        concepts_counts_updater_cb,
    ) = op_to_fn(Operation.DEINDEX)

    assert (
        test_counts
        == await run_partial_updates_of_concepts_for_document_passages.fn(
            document_importer=document_importer_remove,
            cache_bucket=mock_bucket,
            concepts_counts_prefix=CONCEPTS_COUNTS_PREFIX_DEFAULT,
            merge_serialise_concepts_cb=merge_serialise_concepts_cb,
            vespa_response_handler_cb=vespa_response_handler_cb,
            concepts_counts_updater_cb=concepts_counts_updater_cb,
            vespa_search_adapter=local_vespa_search_adapter,
        )
    )

    # The S3 and Vespa state are checked, so that any removals only
    # affect the 1 expected document.

    # S3 state after:
    # Document passages
    with pytest.raises(ClientError):
        s3_match = S3_PATTERN.match(document_object_uri_remove)
        key = s3_match.group("prefix")
        mock_s3_client.head_object(
            Bucket=mock_bucket,
            Key=key,
        )

    s3_match = S3_PATTERN.match(document_object_uri_keep)
    key = s3_match.group("prefix")
    mock_s3_client.head_object(
        Bucket=mock_bucket,
        Key=key,
    )

    # Concepts counts
    with pytest.raises(ClientError):
        s3_match = S3_PATTERN.match(concepts_counts_remove_uri)
        key = s3_match.group("prefix")
        mock_s3_client.head_object(
            Bucket=mock_bucket,
            Key=key,
        )

    assert Counter(
        {Concept(preferred_label="forestry sector", wikibase_id=WikibaseID("Q787")): 1}
    ) == await count_family_document_concepts.load_parse_concepts_counts(
        concepts_counts_keep_uri
    )

    # Vespa state after:
    _hit_id, passage_remove_1_post = get_document_passage_from_vespa(
        text_block_id=labelled_passages_remove[0].id,
        document_import_id=document_import_id_remove,
        vespa_search_adapter=local_vespa_search_adapter,
    )

    assert passage_remove_1_pre == passage_remove_1_post

    _hit_id, passage_keep_1_post = get_document_passage_from_vespa(
        text_block_id=labelled_passages_keep[0].id,
        document_import_id=document_import_id_keep,
        vespa_search_adapter=local_vespa_search_adapter,
    )

    assert passage_keep_1_pre == passage_keep_1_post


@pytest.mark.asyncio
@pytest.mark.vespa
@pytest.mark.flaky_on_ci
async def test_deindex_labelled_passages_from_s3_to_vespa(
    mock_bucket: str,
    mock_s3_client,
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
):
    primary_alias = "v4"
    cleanup_1_alias = "v3"
    cleanup_2_alias = "v1"

    document_import_id_remove: DocumentImportId = "CCLW.executive.10014.4470"

    document_object_uri_remove: DocumentObjectUri = f"s3://{mock_bucket}/{DOCUMENT_TARGET_PREFIX_DEFAULT}/Q760/{primary_alias}/{document_import_id_remove}.json"
    document_object_uri_remove_cleanup_1: DocumentObjectUri = f"s3://{mock_bucket}/{DOCUMENT_TARGET_PREFIX_DEFAULT}/Q760/{cleanup_1_alias}/{document_import_id_remove}.json"
    document_object_uri_remove_cleanup_2: DocumentObjectUri = f"s3://{mock_bucket}/{DOCUMENT_TARGET_PREFIX_DEFAULT}/Q760/{cleanup_2_alias}/{document_import_id_remove}.json"

    document_import_id_keep: DocumentImportId = "CCLW.executive.4934.1571"

    document_object_uri_keep: DocumentObjectUri = f"s3://{mock_bucket}/{DOCUMENT_TARGET_PREFIX_DEFAULT}/Q787/{primary_alias}/{document_import_id_keep}.json"
    _document_importer_keep: DocumentImporter = (
        document_import_id_keep,
        document_object_uri_keep,
    )

    # S3 state before
    # Document passages

    # To be removed
    labelled_passages_remove: list[LabelledPassage] = [
        LabelledPassage(
            id="1570",
            text="National Council for Sustainable Development of the Kyrgyz Republic",
            spans=[
                Span(
                    text="National Council for Sustainable Development of the Kyrgyz Republic",
                    start_index=0,
                    end_index=10,
                    concept_id=WikibaseID("Q760"),
                    labellers=['KeywordClassifier("nuclear sector")'],
                    timestamps=["2021-09-29T14:00:00.000Z"],
                )
            ],
            metadata={
                "concept": {
                    "preferred_label": "nuclear sector",
                    "alternative_labels": [
                        "nuclear energy",
                        "nuclear power",
                        "nuclear industry",
                        "atomic energy",
                        "nuclear technology",
                        "nuclear reactor",
                        "nuclear fuel",
                        "radiation protection",
                        "nuclear engineering",
                        "uranium processing",
                    ],
                    "negative_labels": [],
                    "description": "Activities related to the development, production, and management of nuclear energy and technology.",
                    "wikibase_id": "Q25285",
                    "subconcept_of": ["Q11434"],
                    "has_subconcept": [],
                    "related_concepts": ["Q177", "Q7397", "Q8090"],
                    "definition": None,
                    "labelled_passages": [],
                }
            },
        ),
    ]
    serialised_labelled_passages_remove = serialise_labels(labelled_passages_remove)

    # It's setup, now write it to S3 to be available
    _s3_object_write_bytes(
        s3_uri=document_object_uri_remove,
        bytes=serialised_labelled_passages_remove,
    )
    _s3_object_write_bytes(
        s3_uri=document_object_uri_remove_cleanup_1,
        bytes=serialised_labelled_passages_remove,
    )
    _s3_object_write_bytes(
        s3_uri=document_object_uri_remove_cleanup_2,
        bytes=serialised_labelled_passages_remove,
    )

    # To be kept
    labelled_passages_keep: list[LabelledPassage] = [
        LabelledPassage(
            id="197",
            text="Some random text on climate change.",
            spans=[
                Span(
                    text="Some random text on climate change.",
                    start_index=10,
                    end_index=11,
                    concept_id=WikibaseID("Q787"),
                    labellers=['KeywordClassifier("forestry sector")'],
                    timestamps=["2021-09-29T14:00:00.000Z"],
                )
            ],
            metadata={
                "concept": {
                    "preferred_label": "forestry sector",
                    "alternative_labels": [
                        "forest pest",
                        "wood industry",
                        "forest industry",
                        "silviculture",
                        "forest management",
                        "forestry",
                        "forestry sector",
                        "forest fire prevention",
                        "logging",
                        "lumber",
                        "prevention of forest fires",
                        "timber",
                    ],
                    "negative_labels": [],
                    "description": "Activities that relate to the production of goods and services from forests.",
                    "wikibase_id": "Q787",
                    "subconcept_of": ["Q709"],
                    "has_subconcept": [],
                    "related_concepts": ["Q7", "Q5", "Q4"],
                    "definition": None,
                    "labelled_passages": [],
                }
            },
        ),
    ]
    serialised_labelled_passages_keep = serialise_labels(labelled_passages_keep)

    # It's setup, now write it to S3 to be available
    _s3_object_write_bytes(
        s3_uri=document_object_uri_keep,
        bytes=serialised_labelled_passages_keep,
    )

    # Concepts counts
    concepts_counts_remove: Counter[ConceptModel] = Counter(
        {
            ConceptModel(
                wikibase_id=WikibaseID("Q760"),
                model_name='KeywordClassifier("nuclear sector")',
            ): 1
        }
    )
    serialised_concepts_counts_remove = serialise_concepts_counts(
        concepts_counts_remove
    )
    concepts_counts_remove_uri = f"s3://{mock_bucket}/{CONCEPTS_COUNTS_PREFIX_DEFAULT}/Q760/{primary_alias}/{document_import_id_remove}.json"
    concepts_counts_remove_uri_cleanup_1 = f"s3://{mock_bucket}/{CONCEPTS_COUNTS_PREFIX_DEFAULT}/Q760/{cleanup_1_alias}/{document_import_id_remove}.json"
    concepts_counts_remove_uri_cleanup_2 = f"s3://{mock_bucket}/{CONCEPTS_COUNTS_PREFIX_DEFAULT}/Q760/{cleanup_2_alias}/{document_import_id_remove}.json"

    # It's setup, now write it to S3 to be available
    _s3_object_write_text(
        s3_uri=concepts_counts_remove_uri,
        text=serialised_concepts_counts_remove,
    )
    _s3_object_write_text(
        s3_uri=concepts_counts_remove_uri_cleanup_1,
        text=serialised_concepts_counts_remove,
    )
    _s3_object_write_text(
        s3_uri=concepts_counts_remove_uri_cleanup_2,
        text=serialised_concepts_counts_remove,
    )

    concepts_counts_keep: Counter[ConceptModel] = Counter(
        {
            ConceptModel(
                wikibase_id=WikibaseID("Q787"),
                model_name='KeywordClassifier("forestry sector")',
            ): 1,
        }
    )
    serialised_concepts_counts_keep = serialise_concepts_counts(concepts_counts_keep)
    concepts_counts_keep_uri = f"s3://{mock_bucket}/{CONCEPTS_COUNTS_PREFIX_DEFAULT}/Q787/{primary_alias}/{document_import_id_keep}.json"

    # It's setup, now write it to S3 to be available
    _s3_object_write_text(
        s3_uri=concepts_counts_keep_uri,
        text=serialised_concepts_counts_keep,
    )

    # Vespa state before
    # Document passages
    hit_id_1, passage_remove_1_pre = get_document_passage_from_vespa(
        text_block_id=labelled_passages_remove[0].id,
        document_import_id=document_import_id_remove,
        vespa_search_adapter=local_vespa_search_adapter,
    )

    _hit_id, passage_keep_1_pre = get_document_passage_from_vespa(
        text_block_id=labelled_passages_keep[0].id,
        document_import_id=document_import_id_keep,
        vespa_search_adapter=local_vespa_search_adapter,
    )

    # Add the concepts expected to be removed, to the document passage
    # in the local Vespa instance for the test.
    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        assert (
            await partial_update_text_block(
                text_block=(hit_id_1, passage_remove_1_pre),
                concepts=[
                    VespaConcept(
                        id="Q760",
                        name="nuclear sector",
                        model='KeywordClassifier("nuclear sector")',
                        start=0,
                        end=10,
                        timestamp=datetime.fromisoformat(
                            "2021-09-29T14:00:00.000Z".replace("Z", "+00:00")
                        ),
                    ),
                ],
                vespa_connection_pool=vespa_connection_pool,
                update_function=update_concepts_on_existing_vespa_concepts,
            )
            is None
        )

    # Run the function
    config = Config(
        cache_bucket=mock_bucket,
        vespa_search_adapter=local_vespa_search_adapter,
        as_deployment=False,
    )

    await deindex_labelled_passages_from_s3_to_vespa(
        classifier_specs=[ClassifierSpec(name="Q760", alias="latest")],
        document_ids=[document_import_id_remove],
        config=config,
    )

    # The S3 and Vespa state are checked, so that any removals only
    # affect the 1 expected document.

    # S3 state after:
    # Document passages
    with pytest.raises(ClientError):
        s3_match = S3_PATTERN.match(document_object_uri_remove)
        key = s3_match.group("prefix")
        mock_s3_client.head_object(
            Bucket=mock_bucket,
            Key=key,
        )
        s3_match_cleanup_1 = S3_PATTERN.match(document_object_uri_remove_cleanup_1)
        key_cleanup_1 = s3_match_cleanup_1.group("prefix")
        mock_s3_client.head_object(
            Bucket=mock_bucket,
            Key=key_cleanup_1,
        )
        s3_match_cleanup_2 = S3_PATTERN.match(document_object_uri_remove_cleanup_2)
        key_cleanup_2 = s3_match_cleanup_2.group("prefix")
        mock_s3_client.head_object(
            Bucket=mock_bucket,
            Key=key_cleanup_2,
        )

    with pytest.raises(ClientError):
        s3_match = S3_PATTERN.match(document_object_uri_remove)
        key = s3_match.group("prefix")
        mock_s3_client.head_object(
            Bucket=mock_bucket,
            Key=key,
        )
        s3_match_cleanup_1 = S3_PATTERN.match(document_object_uri_remove_cleanup_1)
        key_cleanup_1 = s3_match_cleanup_1.group("prefix")
        mock_s3_client.head_object(
            Bucket=mock_bucket,
            Key=key_cleanup_1,
        )

    s3_match = S3_PATTERN.match(document_object_uri_keep)
    key = s3_match.group("prefix")
    mock_s3_client.head_object(
        Bucket=mock_bucket,
        Key=key,
    )

    # Concepts counts
    with pytest.raises(ClientError):
        s3_match = S3_PATTERN.match(concepts_counts_remove_uri)
        key = s3_match.group("prefix")
        mock_s3_client.head_object(
            Bucket=mock_bucket,
            Key=key,
        )

    assert Counter(
        {Concept(preferred_label="forestry sector", wikibase_id=WikibaseID("Q787")): 1}
    ) == await count_family_document_concepts.load_parse_concepts_counts(
        concepts_counts_keep_uri
    )

    # Vespa state after:
    _hit_id, passage_remove_1_post = get_document_passage_from_vespa(
        text_block_id=labelled_passages_remove[0].id,
        document_import_id=document_import_id_remove,
        vespa_search_adapter=local_vespa_search_adapter,
    )

    assert passage_remove_1_pre == passage_remove_1_post

    _hit_id, passage_keep_1_post = get_document_passage_from_vespa(
        text_block_id=labelled_passages_keep[0].id,
        document_import_id=document_import_id_keep,
        vespa_search_adapter=local_vespa_search_adapter,
    )

    assert passage_keep_1_pre == passage_keep_1_post


def put_empty_document_artifact(
    s3_client,
    bucket: str,
    prefix: str,
    concept: WikibaseID,
    alias: str,
    document_id: str,
) -> None:
    body = BytesIO(json.dumps({}).encode("utf-8"))

    key = str(os.path.join(prefix, concept, alias, f"{document_id}.json"))
    print(f"putting empty document artifact at bucket: `{bucket}`, key: `{key}`")

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


@pytest.mark.parametrize(
    "deindex,maintained,primaries,cleanups,exception",
    [
        (
            [
                ClassifierSpec(name="Q200", alias="latest"),
                ClassifierSpec(name="Q300", alias="latest"),
                ClassifierSpec(name="Q400", alias="v6"),
            ],
            [
                ClassifierSpec(name="Q100", alias="v5"),
                ClassifierSpec(name="Q200", alias="v9"),
                ClassifierSpec(name="Q300", alias="v2"),
                ClassifierSpec(name="Q400", alias="v6"),
                ClassifierSpec(name="Q500", alias="v13"),
            ],
            [
                ClassifierSpec(name="Q200", alias="v9"),
                ClassifierSpec(name="Q300", alias="v2"),
                ClassifierSpec(name="Q400", alias="v6"),
            ],
            [
                ClassifierSpec(name="Q200", alias="v5"),
                ClassifierSpec(name="Q200", alias="v6"),
                ClassifierSpec(name="Q300", alias="v1"),
            ],
            None,
        ),
        (
            [
                ClassifierSpec(name="Q400", alias="v6"),
            ],
            [],
            [],
            [],
            "classifier spec. name='Q400' alias='v6' was not found in the maintained list",
        ),
        (
            [
                ClassifierSpec(name="Q400", alias="v6"),
                ClassifierSpec(name="Q400", alias="v6"),
            ],
            [
                ClassifierSpec(name="Q400", alias="v6"),
            ],
            [],
            [],
            "already have name='Q400' alias='v6' as a primary",
        ),
    ],
)
def test_find_all_classifier_specs_for_latest(
    mock_bucket,
    mock_s3_client,
    deindex: list[ClassifierSpec],
    maintained: list[ClassifierSpec],
    primaries: list[ClassifierSpec],
    cleanups: list[ClassifierSpec],
    exception: str | None,
):
    document_id = "CCLW.executive.10014.4470"

    put = partial(
        put_empty_document_artifact,
        s3_client=mock_s3_client,
        bucket=mock_bucket,
        prefix=DOCUMENT_TARGET_PREFIX_DEFAULT,
        document_id=document_id,
    )

    [
        put(
            concept=WikibaseID(
                classifier_spec.name,
            ),
            alias=classifier_spec.alias,
        )
        for classifier_spec in maintained
    ]

    [
        put(
            concept=WikibaseID(
                classifier_spec.name,
            ),
            alias=classifier_spec.alias,
        )
        for classifier_spec in cleanups
    ]

    if exception is not None:
        with pytest.raises(ValueError, match=exception):
            classifier_specs_primaries, classifier_specs_cleanups = (
                find_all_classifier_specs_for_latest(
                    to_deindex=deindex,
                    being_maintained=maintained,
                    cache_bucket=mock_bucket,
                    document_source_prefix=DOCUMENT_TARGET_PREFIX_DEFAULT,
                    s3_client=mock_s3_client,
                )
            )
    else:
        classifier_specs_primaries, classifier_specs_cleanups = (
            find_all_classifier_specs_for_latest(
                to_deindex=deindex,
                being_maintained=maintained,
                cache_bucket=mock_bucket,
                document_source_prefix=DOCUMENT_TARGET_PREFIX_DEFAULT,
                s3_client=mock_s3_client,
            )
        )

        assert set(classifier_specs_primaries) == set(primaries)
        assert set(classifier_specs_cleanups) == set(cleanups)


@pytest.mark.parametrize(
    "aliases,expected,exception",
    [
        (
            ["v2", "v3", "latest", "v7"],
            ["v2", "v3", "v7"],
            None,
        ),
        (
            ["v9", "v3"],
            ["v3", "v9"],
            None,
        ),
        (
            [],
            [],
            "found 0 aliases for concept Q100",
        ),
    ],
)
def test_search_s3_for_aliases(
    mock_bucket: str,
    mock_s3_client,
    aliases: list[str],
    expected: list[str],
    exception: str | None,
):
    concept = WikibaseID("Q100")
    document_id = "CCLW.executive.10014.4470"

    put = partial(
        put_empty_document_artifact,
        s3_client=mock_s3_client,
        bucket=mock_bucket,
        prefix=DOCUMENT_TARGET_PREFIX_DEFAULT,
        concept=concept,
        document_id=document_id,
    )

    [put(alias=alias) for alias in aliases]

    if exception is not None:
        with pytest.raises(ValueError, match=exception):
            search_s3_for_aliases(
                concept=concept,
                cache_bucket=mock_bucket,
                document_source_prefix=DOCUMENT_TARGET_PREFIX_DEFAULT,
                s3_client=mock_s3_client,
            )
    else:
        # Compare sets to ignore order
        assert set(
            search_s3_for_aliases(
                concept=concept,
                cache_bucket=mock_bucket,
                document_source_prefix=DOCUMENT_TARGET_PREFIX_DEFAULT,
                s3_client=mock_s3_client,
            )
        ) == set(expected)


@pytest.mark.asyncio
async def test_cleanups_by_s3(
    mock_bucket: str,
    mock_s3_client,
):
    aws_env = AwsEnv.sandbox
    document_id = "CCLW.executive.10014.4470"
    document_ids: list[str] = [document_id]
    concept = WikibaseID("Q100")
    classifier_specs_cleanup: list[ClassifierSpec] = [
        ClassifierSpec(name=str(concept), alias="v1"),
        ClassifierSpec(name=str(concept), alias="v2"),
    ]

    put = partial(
        put_empty_document_artifact,
        s3_client=mock_s3_client,
        bucket=mock_bucket,
        concept=concept,
        document_id=document_id,
    )

    # Primary to be left alone
    primary_alias = "v9"
    put(alias=primary_alias, prefix=DOCUMENT_TARGET_PREFIX_DEFAULT)
    put(alias=primary_alias, prefix=CONCEPTS_COUNTS_PREFIX_DEFAULT)

    # To be cleaned-up
    for classifier_spec in classifier_specs_cleanup:
        put(alias=classifier_spec.alias, prefix=DOCUMENT_TARGET_PREFIX_DEFAULT)
        put(alias=classifier_spec.alias, prefix=CONCEPTS_COUNTS_PREFIX_DEFAULT)

    s3_accessor_cleanups = s3_paths_or_s3_prefixes(
        classifier_specs=classifier_specs_cleanup,
        document_ids=document_ids,
        cache_bucket=mock_bucket,
        prefix=DOCUMENT_TARGET_PREFIX_DEFAULT,
    )

    assert (
        await cleanups_by_s3.fn(
            aws_env=aws_env,
            s3_prefixes=s3_accessor_cleanups.prefixes,
            s3_paths=s3_accessor_cleanups.paths,
            cache_bucket=mock_bucket,
            batch_size=DEFAULT_DOCUMENTS_BATCH_SIZE,
            cleanups_task_batch_size=DEFAULT_DEINDEXING_TASK_BATCH_SIZE,
            concepts_counts_prefix=CONCEPTS_COUNTS_PREFIX_DEFAULT,
            as_deployment=False,
        )
        is None
    )

    for classifier_spec in classifier_specs_cleanup:
        with pytest.raises(ClientError):
            key = str(
                os.path.join(
                    DOCUMENT_TARGET_PREFIX_DEFAULT,
                    concept,
                    classifier_spec.alias,
                    f"{document_id}.json",
                )
            )
            mock_s3_client.head_object(Bucket=mock_bucket, Key=key)

    key = str(
        os.path.join(
            DOCUMENT_TARGET_PREFIX_DEFAULT,
            concept,
            primary_alias,
            f"{document_id}.json",
        )
    )
    mock_s3_client.head_object(Bucket=mock_bucket, Key=key)


def test_remove_result_callback():
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
    text_block_id: TextBlockId = "18593"

    assert (
        remove_result_callback(
            concepts_counts=concepts_counts,
            grouped_concepts=grouped_concepts,
            response=response,
            text_block_id=text_block_id,
        )
        is None
    )

    assert concepts_counts == Counter(
        {ConceptModel(wikibase_id=WikibaseID("Q100"), model_name="sectors_model"): 0}
    )


def test_remove_result_callback_not_successful_response():
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
        status_code=403,
        url="test-url",
        operation_type="update",
    )
    text_block_id: TextBlockId = "18593"

    assert (
        remove_result_callback(
            concepts_counts=concepts_counts,
            grouped_concepts=grouped_concepts,
            response=response,
            text_block_id=text_block_id,
        )
        is None
    )

    assert concepts_counts == Counter(
        {ConceptModel(wikibase_id=WikibaseID("Q100"), model_name="sectors_model"): 1}
    )
