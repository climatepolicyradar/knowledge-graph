import datetime
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pydantic
import pytest
import yaml
from cpr_sdk.models.search import Concept as VespaConcept
from prefect import flow

from flows.aggregate_inference_results import (
    aggregate_inference_results,
    build_run_output_prefix,
    combine_labelled_passages,
    get_all_labelled_passages_for_one_document,
    validate_passages_are_same_except_concepts,
)
from scripts.cloud import ClassifierSpec
from src.labelled_passage import LabelledPassage
from src.span import Span


def test_aggregate_inference_results(
    mock_bucket_labelled_passages_large, test_aggregate_config
):
    keys, bucket, s3_client = mock_bucket_labelled_passages_large

    document_ids = [
        "CCLW.executive.10061.4515",
        "CPR.document.i00000549.n0000",
        "UNFCCC.non-party.467.0",
        "UNFCCC.party.492.0",
    ]

    with tempfile.TemporaryDirectory() as spec_dir:
        # Write the concept specs to a YAML file
        temp_spec_dir = Path(spec_dir)
        concept_specs = ["Q123:v4", "Q223:v3", "Q218:v5", "Q767:v3", "Q1286:v3"]
        spec_file = temp_spec_dir / "sandbox.yaml"
        with open(spec_file, "w") as f:
            yaml.dump(concept_specs, f)

        with patch("scripts.update_classifier_spec.SPEC_DIR", temp_spec_dir):
            run_reference = aggregate_inference_results(
                document_ids, test_aggregate_config
            )

            all_collected_ids = []
            collected_ids_for_document = []

            for document_id in document_ids:
                s3_path = os.path.join(run_reference, f"{document_id}.json")
                try:
                    response = s3_client.get_object(Bucket=bucket, Key=s3_path)
                    data = response["Body"].read().decode("utf-8")
                    data = json.loads(data)
                except s3_client.exceptions.NoSuchKey:
                    pytest.fail(f"Unable to find output file: {s3_path}")
                except json.JSONDecodeError:
                    pytest.fail(f"Unable to deserialise output for: {document_id}")
                except Exception as e:
                    pytest.fail(f"Unexpected error: {e}")

                wikibase_ids = [
                    concept_spec.split(":")[0] for concept_spec in concept_specs
                ]

                document_inference_output = list(data.values())
                for concepts in document_inference_output:
                    for concept in concepts:
                        try:
                            vespa_concept = VespaConcept.model_validate(concept)
                            collected_ids_for_document.append(vespa_concept.id)
                        except pydantic.ValidationError as e:
                            pytest.fail(
                                f"Unable to deserialise concept: {concept} with error: {e}"
                            )

                assert len(collected_ids_for_document) > 0, (
                    f"No concepts found for document: {document_id}"
                )
                all_collected_ids.extend(collected_ids_for_document)

            assert set(all_collected_ids) == set(wikibase_ids), (
                f"Outputted: {set(all_collected_ids)} which doesnt match those in the specs: {set(wikibase_ids)}"
            )
            COUNT = 329
            assert len(all_collected_ids) == COUNT, (
                f"Expected {COUNT} concepts to be outputted, found: {len(all_collected_ids)}"
            )


def test_build_run_output_prefix():
    @flow()
    def fake_flow():
        return build_run_output_prefix("test-prefix")

    prefix = fake_flow()
    # From https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html
    s3_unsupported_chars = r'\^`>{[<%#"}]~|'

    for char in s3_unsupported_chars:
        assert char not in prefix, f"Unsupported char found in prefix: {prefix}"

    assert "None" not in prefix
    assert "test-prefix" in prefix


def test_get_all_labelled_passages_for_one_document(
    mock_bucket_labelled_passages_large, test_aggregate_config
):
    document_id = "CCLW.executive.10061.4515"
    classifier_specs = [
        ClassifierSpec(name="Q218", alias="v5"),
        ClassifierSpec(name="Q767", alias="v3"),
        ClassifierSpec(name="Q1286", alias="v3"),
    ]
    labelled_passages = get_all_labelled_passages_for_one_document(
        document_id, classifier_specs, test_aggregate_config
    )
    assert len(labelled_passages) == len(classifier_specs)


def test_validate_passages_are_same_except_concepts():
    passages: list[LabelledPassage] = []
    for i in range(10):
        span_one = Span(text=f"unique spans: {i}", start_index=0, end_index=4)
        span_two = Span(text=f"unique spans: {i} {i}", start_index=0, end_index=4)
        passage = LabelledPassage(
            id="1",
            text="id and text should match across identical labelled passages",
            spans=[span_one, span_two],
        )
        passages.append(passage)

    validate_passages_are_same_except_concepts(passages)

    with pytest.raises(ValueError):
        passages.append(
            LabelledPassage(
                id="2",
                text="Imagine if we messed up and drew in a different passage",
                spans=[Span(text="span", start_index=0, end_index=4)],
            )
        )
        validate_passages_are_same_except_concepts(passages)


def test_combine_labelled_passages(concept):
    base_span = Span(
        text="The",
        start_index=0,
        end_index=3,
        concept_id=None,
        labellers=['KeywordClassifier("greenhouse gas")'],
        timestamps=[datetime.datetime(2025, 2, 24, 18, 42, 12, 677997)],
    )

    Q218_concept = concept.model_copy(update={"id": "Q218"})
    Q218_span = base_span.model_copy(update={"concept_id": "Q218"})

    Q767_concept = concept.model_copy(update={"id": "Q767"})
    Q767_span = base_span.model_copy(update={"concept_id": "Q767"})

    Q1286_concept = concept.model_copy(update={"id": "Q1286"})
    Q1286_span = base_span.model_copy(update={"concept_id": "Q1286"})

    labelled_passages = {
        "Q218:v5": [
            LabelledPassage(
                id="b0",
                text="The",
                spans=[Q218_span, Q218_span],
                metadata={"concept": Q218_concept},
            ),
            LabelledPassage(
                id="b1", text="The", spans=[], metadata={"concept": Q218_concept}
            ),
            LabelledPassage(
                id="b2", text="The", spans=[], metadata={"concept": Q218_concept}
            ),
        ],
        "Q767:v3": [
            LabelledPassage(
                id="b0",
                text="The",
                spans=[Q767_span],
                metadata={"concept": Q767_concept},
            ),
            LabelledPassage(
                id="b1",
                text="The",
                spans=[Q767_span],
                metadata={"concept": Q767_concept},
            ),
            LabelledPassage(
                id="b2", text="The", spans=[], metadata={"concept": Q767_concept}
            ),
        ],
        "Q1286:v3": [
            LabelledPassage(
                id="b0",
                text="The",
                spans=[Q1286_span],
                metadata={"concept": Q1286_concept},
            ),
            LabelledPassage(
                id="b1",
                text="The",
                spans=[Q1286_span],
                metadata={"concept": Q1286_concept},
            ),
            LabelledPassage(
                id="b2",
                text="The",
                spans=[Q1286_span, Q1286_span, Q1286_span],
                metadata={"concept": Q1286_concept},
            ),
        ],
    }

    # Happy case
    combined_labelled_passages = combine_labelled_passages(labelled_passages)
    assert len(combined_labelled_passages) == 3
    assert len(combined_labelled_passages["b0"]) == 4
    assert len(combined_labelled_passages["b1"]) == 2
    assert len(combined_labelled_passages["b2"]) == 3

    # different length lists
    labelled_passages["Q218:v5"].append(
        LabelledPassage(
            id="b3", text="The", spans=[Q218_span], metadata={"concept": Q218_concept}
        )
    )
    with pytest.raises(ValueError):
        combine_labelled_passages(labelled_passages)

    # passage mismatch
    labelled_passages["Q218:v5"][0].text = "Different text"
    with pytest.raises(ValueError):
        combine_labelled_passages(labelled_passages)
