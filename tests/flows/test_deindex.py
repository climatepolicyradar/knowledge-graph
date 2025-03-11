import json
from collections import Counter
from datetime import datetime
from io import BytesIO

import pytest
from botocore.exceptions import ClientError
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.s3 import _s3_object_read_text
from cpr_sdk.search_adaptors import VespaSearchAdapter

from flows.deindex import (
    CONCEPTS_COUNTS_PREFIX_DEFAULT,
    ConceptModel,
    calculate_concepts_counts_from_results,
    get_document_passage_from_vespa,
    get_document_passages_from_vespa,
    partial_update_text_block,
    run_partial_updates_of_concepts_for_document_passages,
    update_s3_with_all_successes,
    update_s3_with_latest_concepts_counts,
    update_s3_with_some_successes,
)
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.span import Span


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

    assert len(first_passage_with_concepts.concepts) >= 2, "must be at least 2 concepts"

    # Get a slice of 1 concept
    concepts_to_remove: list[VespaConcept] = first_passage_with_concepts.concepts[0:1]
    concepts_to_keep: list[VespaConcept] = first_passage_with_concepts.concepts[1:]

    assert (
        await partial_update_text_block(
            text_block_id=first_passage_with_concepts.text_block_id,
            document_import_id=document_import_id,
            concepts=concepts_to_remove,
            vespa_search_adapter=local_vespa_search_adapter,
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

    assert len(first_passage_with_concepts.concepts) >= 2, "must be at least 2 concepts"

    assert (
        await partial_update_text_block(
            text_block_id=first_passage_with_concepts.text_block_id,
            document_import_id=document_import_id,
            concepts=[],
            vespa_search_adapter=local_vespa_search_adapter,
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
    mock_s3_client,
    local_vespa_search_adapter: VespaSearchAdapter,
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

    expected_concepts_counts_key = (
        f"{CONCEPTS_COUNTS_PREFIX_DEFAULT}/Q787/v4/{document_import_id}.json"
    )
    expected_labelled_passages_key = (
        f"labelled_passages/Q787/v4/{document_import_id}.json"
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
async def test_run_partial_updates_of_concepts_for_document_passages(
    mock_bucket,
    mock_s3_client,
    mock_bucket_labelled_passages,
    labelled_passage_fixture_ids,
    labelled_passage_fixture_files,
    s3_prefix_labelled_passages,
    local_vespa_search_adapter: VespaSearchAdapter,
):
    document_import_id = labelled_passage_fixture_ids[0]
    document_object_uri = f"s3://{mock_bucket}/{s3_prefix_labelled_passages}/{labelled_passage_fixture_files[0]}"
    document_importer = (document_import_id, document_object_uri)

    # TODO: Get S3 state before
    # - Concepts counts
    # - Labelled passages

    # TODO: Get Vespa state before
    # - Document passages

    assert (
        await run_partial_updates_of_concepts_for_document_passages(
            document_importer=document_importer,
            cache_bucket=mock_bucket,
            concepts_counts_prefix=CONCEPTS_COUNTS_PREFIX_DEFAULT,
            vespa_search_adapter=local_vespa_search_adapter,
        )
        is None
    )

    # TODO: Test Vespa state after
    # - Document passages

    # TODO: Test S3 state after
    # - Concepts counts
    # - Labelled passages


# TODO:
# @pytest.mark.asyncio
# async def test_run_partial_updates_of_concepts_for_batch(
#     mock_bucket,
#     mock_s3_client,
# ):
#     await run_partial_updates_of_concepts_for_batch()


# TODO: deindex_by_s3
# TODO: deindex_labelled_passages_from_s3_to_vespa
