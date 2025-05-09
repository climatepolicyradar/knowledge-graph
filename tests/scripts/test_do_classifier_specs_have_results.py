import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.do_classifier_specs_have_results import (
    Result,
    check_classifier_specs,
    check_single_spec,
    collect_file_names,
    write_result,
)

# the flows conftest wouldn't be in scope here (yet?) but it does have what we need
from tests.flows.conftest import *  # noqa: F403


def test_check_classifier_specs(mock_bucket, capsys):
    check_classifier_specs(
        aws_env="sandbox",
        bucket_name=mock_bucket,
        max_workers=4,
        write_file_names=False,
    )
    stdout, stderr = capsys.readouterr()
    assert "to_process: " in stdout
    assert (
        "Checking sandbox classifier specs in test_bucket/labelled_passages" in stdout
    )
    assert stderr == ""


def test_check_single_spec(
    mock_bucket, mock_bucket_labelled_passages, s3_prefix_labelled_passages
):
    _, name, alias = s3_prefix_labelled_passages.split("/")
    classifier_spec = f"{name}:{alias}"
    result = check_single_spec(bucket_name=mock_bucket, classifier_spec=classifier_spec)
    assert result.path_exists
    assert result.classifier_spec == classifier_spec


def test_collect_file_names(
    mock_bucket,
    s3_prefix_labelled_passages,
    mock_bucket_labelled_passages,
    labelled_passage_fixture_ids,
):
    file_names = collect_file_names(
        bucket_name=mock_bucket, prefix=s3_prefix_labelled_passages
    )
    expected_file_names = [f"{doc_id}.json" for doc_id in labelled_passage_fixture_ids]
    assert set(file_names) == set(expected_file_names)


def test_write_result():
    test_file_name = "test-file-name"
    with TemporaryDirectory() as temp_dir:
        result = Result(
            path_exists=True,
            classifier_spec="test-classifier-spec",
            file_names=[test_file_name],
        )
        path = write_result(
            result=result,
            start_time="YYYY-MM-DD",
            parent_dir=Path(temp_dir),
            aws_env="sandbox",
        )
        assert path.read_text() == json.dumps([test_file_name])
        assert path.parent.name == "sandbox"
