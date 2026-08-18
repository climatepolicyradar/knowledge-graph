from unittest.mock import MagicMock, patch

import pytest

from knowledge_graph.identifiers import Identifier, WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.operations.extend_dataset import (
    NotEnoughNewPassagesError,
    extend_dataset_locally,
    run_extend_dataset,
)

WIKIBASE_ID = WikibaseID("Q2419")


def make_passages(n: int) -> list[LabelledPassage]:
    return [LabelledPassage(text=f"passage {i}", spans=[]) for i in range(n)]


class CountingRecords:
    """
    Stands in for `dataset.records`, counting how many times it is iterated.

    Each iteration of the real thing is a fresh paginated HTTP fetch, so the count is what
    lets us assert the records are only pulled once.
    """

    def __init__(self, texts: list[str]):
        self.texts = texts
        self.access_count = 0

    def __iter__(self):
        """Count this iteration, then yield a mock record per text."""
        self.access_count += 1
        return iter([MagicMock(fields={"text": text}) for text in self.texts])


@pytest.fixture
def mock_logger():
    """
    Patch the module's get_logger.

    caplog is unreliable here: get_logger() can return a Prefect run logger which doesn't
    propagate to caplog's root handler. See tests/test_large_language_model.py for the
    same workaround.
    """
    with patch(
        "knowledge_graph.operations.extend_dataset.get_logger"
    ) as mock_get_logger:
        yield mock_get_logger.return_value


def logged(mock_logger, level: str) -> list[str]:
    """Messages passed to the given log level."""
    return [str(call.args[0]) for call in getattr(mock_logger, level).call_args_list]


@pytest.fixture
def mock_argilla():
    """Patch ArgillaSession, exposing the class, the session and its dataset."""
    with patch(
        "knowledge_graph.operations.extend_dataset.ArgillaSession"
    ) as argilla_cls:
        session = argilla_cls.return_value
        dataset = MagicMock()
        dataset.name = str(WIKIBASE_ID)
        dataset.records = CountingRecords([])
        session.get_dataset.return_value = dataset
        yield {"cls": argilla_cls, "session": session, "dataset": dataset}


def test_deduplicates_against_text_already_in_argilla(mock_argilla):
    """Passages whose text is already a record are dropped."""
    mock_argilla["dataset"].records = CountingRecords(
        [f"passage {i}" for i in range(4)]
    )

    added = run_extend_dataset(
        wikibase_id=WIKIBASE_ID, labelled_passages=make_passages(10), limit=None
    )

    assert [p.text for p in added] == [f"passage {i}" for i in range(4, 10)]
    kwargs = mock_argilla["session"].add_labelled_passages.call_args.kwargs
    assert kwargs["labelled_passages"] == added


def test_limit_is_applied_after_deduplication(mock_argilla):
    """
    The load-bearing regression test.

    `limit` counts *new* passages, not inputs. An implementation that sliced to the limit
    before deduplicating would push nothing here, because the first 5 inputs are all
    already in Argilla.
    """
    mock_argilla["dataset"].records = CountingRecords(
        [f"passage {i}" for i in range(5)]
    )

    added = run_extend_dataset(
        wikibase_id=WIKIBASE_ID, labelled_passages=make_passages(10), limit=3
    )

    assert len(added) == 3
    assert {p.text for p in added} <= {f"passage {i}" for i in range(5, 10)}


def test_warns_and_adds_fewer_when_not_enough_new_passages(mock_argilla, mock_logger):
    """Short of the limit, it adds what it has rather than failing."""
    mock_argilla["dataset"].records = CountingRecords(
        [f"passage {i}" for i in range(8)]
    )

    added = run_extend_dataset(
        wikibase_id=WIKIBASE_ID, labelled_passages=make_passages(10), limit=50
    )

    assert len(added) == 2
    assert any(
        "Only 2 new passages available" in message
        for message in logged(mock_logger, "warning")
    )


def test_raises_when_raise_on_insufficient_passages_and_not_enough_new(mock_argilla):
    """With raise_on_insufficient_passages, a shortfall fails loudly and pushes nothing."""
    mock_argilla["dataset"].records = CountingRecords(
        [f"passage {i}" for i in range(8)]
    )

    with pytest.raises(NotEnoughNewPassagesError, match="Only 2"):
        run_extend_dataset(
            wikibase_id=WIKIBASE_ID,
            labelled_passages=make_passages(10),
            limit=50,
            raise_on_insufficient_passages=True,
        )

    mock_argilla["session"].add_labelled_passages.assert_not_called()


def test_does_nothing_when_every_passage_is_already_in_argilla(mock_argilla):
    """No new passages means no pointless round-trip to Argilla."""
    mock_argilla["dataset"].records = CountingRecords(
        [f"passage {i}" for i in range(10)]
    )

    added = run_extend_dataset(
        wikibase_id=WIKIBASE_ID, labelled_passages=make_passages(10), limit=50
    )

    assert added == []
    mock_argilla["session"].add_labelled_passages.assert_not_called()


def test_forwards_credentials_workspace_and_suggestion_model(mock_argilla):
    """Credentials build the session; workspace and suggestions reach Argilla."""
    suggestion_model = MagicMock()

    run_extend_dataset(
        wikibase_id=WIKIBASE_ID,
        labelled_passages=make_passages(3),
        workspace="my-workspace",
        limit=None,
        suggestion_model=suggestion_model,
        argilla_api_url="https://argilla.test",
        argilla_api_key="key-123",
    )

    mock_argilla["cls"].assert_called_once_with(
        api_url="https://argilla.test", api_key="key-123"
    )
    mock_argilla["session"].get_dataset.assert_called_once_with(
        WIKIBASE_ID, workspace="my-workspace"
    )
    kwargs = mock_argilla["session"].add_labelled_passages.call_args.kwargs
    assert kwargs["workspace"] == "my-workspace"
    assert kwargs["suggestion_model"] is suggestion_model


def test_reads_raw_records_not_submitted_responses(mock_argilla):
    """
    Dedup must not go via get_labelled_passages.

    That method returns only records with *submitted* responses, so passages still awaiting
    annotation would look absent and get pushed a second time.
    """
    run_extend_dataset(
        wikibase_id=WIKIBASE_ID, labelled_passages=make_passages(3), limit=None
    )

    mock_argilla["session"].get_labelled_passages.assert_not_called()


def test_dataset_records_are_fetched_once(mock_argilla):
    """Each iteration is a paginated HTTP fetch, so once is all we get."""
    records = CountingRecords([f"passage {i}" for i in range(3)])
    mock_argilla["dataset"].records = records

    run_extend_dataset(
        wikibase_id=WIKIBASE_ID, labelled_passages=make_passages(10), limit=None
    )

    assert records.access_count == 1


def test_final_count_is_computed_not_re_read(mock_argilla, mock_logger):
    """A stale read-back after the write must not change the reported total."""
    mock_argilla["dataset"].records = CountingRecords(
        [f"passage {i}" for i in range(3)]
    )

    def go_stale(**kwargs):
        mock_argilla["dataset"].records = CountingRecords([])
        return mock_argilla["dataset"]

    mock_argilla["session"].add_labelled_passages.side_effect = go_stale

    run_extend_dataset(
        wikibase_id=WIKIBASE_ID, labelled_passages=make_passages(10), limit=2
    )

    assert any(
        "Dataset now contains 5 distinct passages" in message
        for message in logged(mock_logger, "info")
    )


def test_extend_dataset_locally_raises_without_a_sample_file(tmp_path, mock_argilla):
    """The error must point at the sample script."""
    with patch(
        "knowledge_graph.operations.extend_dataset.processed_data_dir", tmp_path
    ):
        with pytest.raises(FileNotFoundError, match="Run the sample script"):
            extend_dataset_locally(wikibase_id=WIKIBASE_ID)

    mock_argilla["cls"].assert_not_called()


def test_extend_dataset_locally_reads_the_sample_file(tmp_path, mock_argilla):
    """The local JSONL is parsed and its passages forwarded."""
    sampled_dir = tmp_path / "sampled_passages"
    sampled_dir.mkdir(parents=True)
    passages = make_passages(4)
    (sampled_dir / f"{WIKIBASE_ID}.jsonl").write_text(
        "\n".join(p.model_dump_json() for p in passages), encoding="utf-8"
    )

    with patch(
        "knowledge_graph.operations.extend_dataset.processed_data_dir", tmp_path
    ):
        added = extend_dataset_locally(wikibase_id=WIKIBASE_ID, limit=None)

    assert [p.text for p in added] == [p.text for p in passages]


def test_deduplicates_by_passage_id_not_raw_text(mock_argilla):
    """Deduplicate keys on LabelledPassage.id."""
    mock_argilla["dataset"].records = CountingRecords(["passage 0"])

    added = run_extend_dataset(
        wikibase_id=WIKIBASE_ID,
        labelled_passages=[
            LabelledPassage(text="passage 0", spans=[]),
            LabelledPassage(text="passage 0 ", spans=[]),
        ],
        limit=None,
    )

    assert [lp.text for lp in added] == ["passage 0 "]


def test_does_not_refetch_records_when_given_existing_passage_ids(mock_argilla):
    """Each iteration of dataset.records is an HTTP fetch, so the caller can skip it."""
    records = CountingRecords([f"passage {i}" for i in range(4)])
    mock_argilla["dataset"].records = records
    known: set[str] = {Identifier.generate(f"passage {i}") for i in range(4)}

    added = run_extend_dataset(
        wikibase_id=WIKIBASE_ID,
        labelled_passages=make_passages(10),
        existing_passage_ids=known,
        limit=None,
    )

    assert records.access_count == 0
    assert len(added) == 6


def test_fetches_records_once_when_not_given_existing_passage_ids(mock_argilla):
    records = CountingRecords([f"passage {i}" for i in range(4)])
    mock_argilla["dataset"].records = records

    run_extend_dataset(
        wikibase_id=WIKIBASE_ID, labelled_passages=make_passages(10), limit=None
    )

    assert records.access_count == 1
