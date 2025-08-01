import uuid
from unittest.mock import patch

import pytest
from argilla import Dataset, Record, Settings, SpanQuestion, TextField, TextQuestion
from dotenv import find_dotenv, load_dotenv

from src.labelling import ArgillaSession

load_dotenv(find_dotenv())
pytest.skip(
    reason="These tests actually create a dataset in Argilla when run, so skipping",
    allow_module_level=True,
)


def test_combine_datasets():
    """
    Tests the combine dataset function

    NOTE: this creates a dataset in the Argilla default workspace. This was mostly for understanding, and illustrating
    the functionality in v2, but do not run unless you delete the resulting dataset afterwards named:
    "combined-dataset1-dataset2"
    """
    dataset1 = Dataset(
        name="dataset1",
        settings=Settings(
            fields=[TextField(name="field1")],  # type: ignore
            questions=[
                SpanQuestion(name="question1", field="field1", labels=["yes", "no"])
            ],  # type: ignore
        ),
    )
    dataset2 = Dataset(
        name="dataset2",
        settings=Settings(
            fields=[TextField(name="field1")],  # type: ignore
            questions=[
                SpanQuestion(name="question1", field="field1", labels=["yes", "no"])
            ],  # type: ignore
        ),
    )

    records = {
        "dataset1": [
            Record(
                id=uuid.uuid4(),
                fields={"field1": "text1"},
                metadata={"metadata1": "metadata1"},
            ),
            Record(
                id=uuid.uuid4(),
                fields={"field1": "text2"},
                metadata={"metadata2": "metadata2"},
            ),
        ],
        "dataset2": [
            Record(
                id=uuid.uuid4(),
                fields={"field1": "text3"},
                metadata={"metadata3": "metadata3"},
            ),
            Record(
                id=uuid.uuid4(),
                fields={"field1": "text4"},
                metadata={"metadata4": "metadata4"},
            ),
        ],
    }

    with patch(
        "tests.test_labelling.DatasetRecordsIterator._list",
        lambda self: records[self.__dataset.name],
    ):
        argilla = ArgillaSession()
        combine_dataset = argilla.combine_datasets(dataset1, dataset2)

        assert combine_dataset.name == "combined-dataset1-dataset2"
        assert combine_dataset.settings.fields == [TextField(name="field1")]
        assert combine_dataset.settings.questions == [
            SpanQuestion(name="question1", field="field1", labels=["yes", "no"])
        ]
        assert len(list(combine_dataset.records)) == 4
        assert list(combine_dataset.records)[0].fields == {"field1": "text1"}
        assert list(combine_dataset.records)[0].metadata == {"metadata1": "metadata1"}
        assert list(combine_dataset.records)[-1].fields == {"field1": "text4"}
        assert list(combine_dataset.records)[-1].metadata == {"metadata4": "metadata4"}


def test__assert_datasets_of_the_same_type_errors():
    argilla = ArgillaSession()
    dataset1 = Dataset(
        name="dataset1",
        settings=Settings(
            fields=[TextField(name="field1")],  # type: ignore
            questions=[
                SpanQuestion(name="question1", field="field1", labels=["yes", "no"]),
                TextQuestion(name="question2"),
            ],  # type: ignore
        ),
    )
    dataset2 = Dataset(
        name="dataset2",
        settings=Settings(
            fields=[TextField(name="field1")],  # type: ignore
            questions=[
                SpanQuestion(name="question1", field="field1", labels=["yes", "no"])
            ],  # type: ignore
        ),
    )

    with pytest.raises(ValueError):
        argilla._assert_datasets_of_the_same_type(dataset1, dataset2)


def test__assert_datasets_of_the_same_type():
    argilla = ArgillaSession()
    dataset1 = Dataset(
        name="dataset1",
        settings=Settings(
            fields=[TextField(name="field1")],  # type: ignore
            questions=[
                SpanQuestion(name="question1", field="field1", labels=["yes", "no"])
            ],  # type: ignore
        ),
    )
    dataset2 = Dataset(
        name="dataset2",
        settings=Settings(
            fields=[TextField(name="field1")],  # type: ignore
            questions=[
                SpanQuestion(name="question1", field="field1", labels=["yes", "no"])
            ],  # type: ignore
        ),
    )

    assert argilla._assert_datasets_of_the_same_type(dataset1, dataset2) is None
