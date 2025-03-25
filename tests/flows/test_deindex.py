import json
from collections import Counter
from datetime import datetime
from io import BytesIO
from typing import Sequence

import pytest
from botocore.exceptions import ClientError
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.s3 import S3_PATTERN, _s3_object_read_text
from cpr_sdk.search_adaptors import VespaSearchAdapter

import flows.boundary as boundary
import flows.count_family_document_concepts as count_family_document_concepts
from flows.boundary import (
    ConceptModel,
    DocumentImporter,
    DocumentImportId,
    DocumentObjectUri,
    calculate_concepts_counts_from_results,
    get_document_passage_from_vespa,
    get_document_passages_from_vespa,
    partial_update_text_block,
    remove_concepts_from_existing_vespa_concepts,
    run_partial_updates_of_concepts_for_document_passages__removal,
    serialise_concepts_counts,
    update_concepts_on_existing_vespa_concepts,
    update_s3_with_all_successes,
    update_s3_with_latest_concepts_counts,
    update_s3_with_some_successes,
)
from flows.deindex import (
    CONCEPTS_COUNTS_PREFIX_DEFAULT,
    Config,
    deindex_labelled_passages_from_s3_to_vespa,
)
from flows.inference import DOCUMENT_TARGET_PREFIX_DEFAULT, serialise_labels
from flows.utils import _s3_object_write_bytes, _s3_object_write_text
from scripts.cloud import ClassifierSpec
from src.concept import Concept
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


@pytest.mark.asyncio
@pytest.mark.vespa
async def test_partial_update_text_block_with_removal(
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
):
    document_import_id = "CCLW.executive.10014.4470"

    # Confirm that the concepts to remove are in the document passages
    initial_passages = get_document_passages_from_vespa(
        document_import_id=document_import_id,
        vespa_search_adapter=local_vespa_search_adapter,
    )

    try:
        first_passage_with_concepts = next(
            passage for _, passage in initial_passages if passage.concepts
        )
    except StopIteration:
        raise ValueError("no concepts found in any passages, check the fixtures")

    assert first_passage_with_concepts.concepts
    assert len(first_passage_with_concepts.concepts) >= 2, "must be at least 2 concepts"

    # Get a slice of 1 concept
    concepts_to_remove: Sequence[VespaConcept] = first_passage_with_concepts.concepts[
        0:1
    ]
    concepts_to_keep: Sequence[VespaConcept] = first_passage_with_concepts.concepts[1:]

    assert (
        await partial_update_text_block(
            text_block_id=first_passage_with_concepts.text_block_id,
            document_import_id=document_import_id,
            concepts=list(concepts_to_remove),
            vespa_search_adapter=local_vespa_search_adapter,
            update_function=remove_concepts_from_existing_vespa_concepts,
        )
        is None
    )

    _hit_id, updated_passage = get_document_passage_from_vespa(
        text_block_id=first_passage_with_concepts.text_block_id,
        document_import_id=document_import_id,
        vespa_search_adapter=local_vespa_search_adapter,
    )

    assert updated_passage.concepts == concepts_to_keep


@pytest.mark.asyncio
@pytest.mark.vespa
async def test_partial_update_text_block_with_empty(
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
):
    document_import_id = "CCLW.executive.10014.4470"

    # Confirm that the concepts to remove are in the document passages
    initial_passages = get_document_passages_from_vespa(
        document_import_id=document_import_id,
        vespa_search_adapter=local_vespa_search_adapter,
    )

    try:
        first_passage_with_concepts = next(
            passage for _, passage in initial_passages if passage.concepts
        )
    except StopIteration:
        raise ValueError("no concepts found in any passages, check the fixtures")

    assert first_passage_with_concepts.concepts
    assert len(first_passage_with_concepts.concepts) >= 2, "must be at least 2 concepts"

    assert (
        await partial_update_text_block(
            text_block_id=first_passage_with_concepts.text_block_id,
            document_import_id=document_import_id,
            concepts=[],
            vespa_search_adapter=local_vespa_search_adapter,
            update_function=remove_concepts_from_existing_vespa_concepts,
        )
        is None
    )

    _hit_id, updated_passage = get_document_passage_from_vespa(
        text_block_id=first_passage_with_concepts.text_block_id,
        document_import_id=document_import_id,
        vespa_search_adapter=local_vespa_search_adapter,
    )

    assert updated_passage.concepts == first_passage_with_concepts.concepts


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


@pytest.mark.parametrize(
    "results,batch,expected_counts",
    [
        # Case: All successful updates (no exceptions)
        (
            [None, None],
            [
                (
                    "text_block_1",
                    [
                        VespaConcept(
                            id="Q123",
                            name="concept1",
                            model='KeywordClassifier("concept1")',
                            start=0,
                            end=10,
                            timestamp=datetime.now(),
                        )
                    ],
                ),
                (
                    "text_block_2",
                    [
                        VespaConcept(
                            id="Q456",
                            name="concept2",
                            model='KeywordClassifier("concept2")',
                            start=0,
                            end=10,
                            timestamp=datetime.now(),
                        )
                    ],
                ),
            ],
            Counter(
                {
                    ConceptModel(
                        wikibase_id=WikibaseID("Q123"),
                        model_name='KeywordClassifier("concept1")',
                    ): 0,
                    ConceptModel(
                        wikibase_id=WikibaseID("Q456"),
                        model_name='KeywordClassifier("concept2")',
                    ): 0,
                }
            ),
        ),
        # Case: One failed update (with exception)
        (
            [None, Exception("Update failed")],
            [
                (
                    "text_block_1",
                    [
                        VespaConcept(
                            id="Q123",
                            name="concept1",
                            model='KeywordClassifier("concept1")',
                            start=0,
                            end=10,
                            timestamp=datetime.now(),
                        )
                    ],
                ),
                (
                    "text_block_2",
                    [
                        VespaConcept(
                            id="Q456",
                            name="concept2",
                            model='KeywordClassifier("concept2")',
                            start=0,
                            end=10,
                            timestamp=datetime.now(),
                        )
                    ],
                ),
            ],
            Counter(
                {
                    ConceptModel(
                        wikibase_id=WikibaseID("Q123"),
                        model_name='KeywordClassifier("concept1")',
                    ): 0,
                    ConceptModel(
                        wikibase_id=WikibaseID("Q456"),
                        model_name='KeywordClassifier("concept2")',
                    ): 1,
                }
            ),
        ),
        # Case: Multiple concepts in one text block
        (
            [None],
            [
                (
                    "text_block_1",
                    [
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
                    ],
                )
            ],
            Counter(
                {
                    ConceptModel(
                        wikibase_id=WikibaseID("Q123"),
                        model_name='KeywordClassifier("concept1")',
                    ): 0,
                    ConceptModel(
                        wikibase_id=WikibaseID("Q456"),
                        model_name='KeywordClassifier("concept2")',
                    ): 0,
                }
            ),
        ),
        # Case: All failed updates
        (
            [Exception("Update failed"), Exception("Another failure")],
            [
                (
                    "text_block_1",
                    [
                        VespaConcept(
                            id="Q123",
                            name="concept1",
                            model='KeywordClassifier("concept1")',
                            start=0,
                            end=10,
                            timestamp=datetime.now(),
                        )
                    ],
                ),
                (
                    "text_block_2",
                    [
                        VespaConcept(
                            id="Q456",
                            name="concept2",
                            model='KeywordClassifier("concept2")',
                            start=0,
                            end=10,
                            timestamp=datetime.now(),
                        )
                    ],
                ),
            ],
            Counter(
                {
                    ConceptModel(
                        wikibase_id=WikibaseID("Q123"),
                        model_name='KeywordClassifier("concept1")',
                    ): 1,
                    ConceptModel(
                        wikibase_id=WikibaseID("Q456"),
                        model_name='KeywordClassifier("concept2")',
                    ): 1,
                }
            ),
        ),
        # Case: Empty batch
        ([], [], Counter()),
    ],
)
def test_calculate_concepts_counts_from_results(
    results,
    batch,
    expected_counts,
):
    """Test that subtract_concepts_counts correctly processes results and batch data."""
    actual_counts = calculate_concepts_counts_from_results(results, batch)

    assert actual_counts == expected_counts


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
            id="p_37_b_6",
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
    _hit_id, passage_remove_1_pre = get_document_passage_from_vespa(
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
    assert (
        await boundary.partial_update_text_block(
            text_block_id=labelled_passages_remove[0].id,
            document_import_id=document_import_id_remove,
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
            vespa_search_adapter=local_vespa_search_adapter,
            update_function=update_concepts_on_existing_vespa_concepts,
        )
        is None
    )

    # Run the function
    assert (
        await run_partial_updates_of_concepts_for_document_passages__remove(
            document_importer=document_importer_remove,
            cache_bucket=mock_bucket,
            concepts_counts_prefix=CONCEPTS_COUNTS_PREFIX_DEFAULT,
            vespa_search_adapter=local_vespa_search_adapter,
        )
        is None
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
async def test_deindex_labelled_passages_from_s3_to_vespa(
    mock_bucket: str,
    mock_s3_client,
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
):
    document_import_id_remove: DocumentImportId = "CCLW.executive.10014.4470"

    document_import_id_remove: DocumentImportId = "CCLW.executive.10014.4470"

    document_object_uri_remove: DocumentObjectUri = f"s3://{mock_bucket}/{DOCUMENT_TARGET_PREFIX_DEFAULT}/Q760/v4/{document_import_id_remove}.json"

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
            id="p_37_b_6",
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
    _hit_id, passage_remove_1_pre = get_document_passage_from_vespa(
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
    assert (
        await boundary.partial_update_text_block(
            text_block_id=labelled_passages_remove[0].id,
            document_import_id=document_import_id_remove,
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
            vespa_search_adapter=local_vespa_search_adapter,
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
        classifier_specs=[ClassifierSpec(name="Q760", alias="v4")],
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
