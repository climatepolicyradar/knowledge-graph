import os
from pathlib import Path

from flows.index import s3_obj_generator


def test_s3_obj_generator(
    mock_bucket,
    mock_ssm_client,
    mock_bucket_concepts,
    s3_prefix_concepts,
    concept_fixture_files,
) -> None:
    """Test the s3 object generator."""
    s3_gen = s3_obj_generator(os.path.join("s3://", mock_bucket, s3_prefix_concepts))
    s3_files = list(s3_gen)
    assert len(s3_files) == len(concept_fixture_files)

    expected_keys = [
        f"{s3_prefix_concepts}/{Path(f).stem}" for f in concept_fixture_files
    ]
    s3_files_keys = [file[0].replace(".json", "") for file in s3_files]
    assert sorted(s3_files_keys) == sorted(expected_keys)


# TODO Test that we successfully confirm if a TextBlock object exists in Vespa


# TODO Test that we successfully index a Concept object into Vespa
# def test_index_concepts_from_s3_to_vespa(
#     test_config, mock_classifiers_dir, mock_bucket, mock_bucket_documents
# ):
#     doc_ids = [Path(doc_file).stem for doc_file in mock_bucket_documents]
#     with prefect_test_harness():
#         classifier_inference(
#             classifier_spec=[("Q788", "latest")],
#             document_ids=doc_ids,
#             config=test_config,
#         )
