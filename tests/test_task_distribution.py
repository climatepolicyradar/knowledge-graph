import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.argilla_v2 import ArgillaSession

session = ArgillaSession()


@st.composite
def mock_dataset_strategy(draw):
    return draw(st.builds(object))


@st.composite
def dataset_list_strategy(draw):
    num_datasets = draw(st.integers(min_value=2, max_value=10))
    return draw(
        st.lists(mock_dataset_strategy(), min_size=num_datasets, max_size=num_datasets)
    )


@st.composite
def labeller_list_strategy(draw):
    return draw(st.lists(st.text(min_size=1), min_size=2, max_size=10, unique=True))


@given(datasets=dataset_list_strategy(), labeller_names=labeller_list_strategy())
def test_whether_distributor_returns_a_generator_of_correct_length(
    datasets: list, labeller_names: list[str]
):
    result = list(session._distribute_labelling_projects(datasets, labeller_names))
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
    assert len(result) == len(datasets) * min(len(labeller_names), 2)


@given(
    datasets=dataset_list_strategy(),
    min_labellers=st.integers(min_value=2, max_value=5),
    extra_labellers=st.lists(st.text(min_size=1), min_size=0, max_size=5),
)
def test_whether_distributor_returns_a_generator_of_correct_length_with_min_labellers(
    datasets: list, min_labellers: int, extra_labellers: list[str]
):
    # Ensure we have at least min_labellers
    labeller_names = [f"labeller_{i}" for i in range(min_labellers)] + extra_labellers

    result = list(
        session._distribute_labelling_projects(datasets, labeller_names, min_labellers)
    )

    assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
    assert len(result) == len(datasets) * min_labellers


@given(datasets=dataset_list_strategy(), labellers=labeller_list_strategy())
def test_whether_distributor_raises_error_with_insufficient_labellers(
    datasets: list, labellers: list[str]
):
    with pytest.raises(ValueError):
        list(
            session._distribute_labelling_projects(
                datasets=datasets,
                labellers=labellers,
                min_labellers=len(labellers) + 1,
            )
        )


@settings(max_examples=10)
@given(
    datasets=dataset_list_strategy(),
    labeller_names=labeller_list_strategy(),
    min_labellers=st.integers(min_value=2, max_value=5),
    extra_labellers=st.lists(st.text(min_size=1), min_size=0, max_size=5),
)
def test_whether_distributor_assigns_correct_number_of_labellers(
    datasets: list,
    labeller_names: list[str],
    min_labellers: int,
    extra_labellers: list[str],
):
    labeller_names = labeller_names = [
        f"labeller_{i}" for i in range(min_labellers)
    ] + extra_labellers

    result = list(
        session._distribute_labelling_projects(datasets, labeller_names, min_labellers)
    )

    # Check whether the total number of assignments is correct
    assert len(result) == len(datasets) * min_labellers

    # Check whether each dataset is assigned to the correct number of labellers
    for dataset, _ in result:
        assert (
            len([labeller for _, labeller in result if _ == dataset]) >= min_labellers
        )

    # if there are enough of them, check whether all labellers are used
    if len(labeller_names) >= min_labellers:
        used_labellers = set(labeller for _, labeller in result)
        assert used_labellers.issubset(set(labeller_names))


# Test for even distribution of work
@given(datasets=dataset_list_strategy(), labeller_names=labeller_list_strategy())
def test_whether_distributor_assigns_tasks_evenly_between_labellers(
    datasets: list, labeller_names: list[str]
):
    result = list(session._distribute_labelling_projects(datasets, labeller_names))

    labeller_counts = {labeller: 0 for labeller in labeller_names}
    for _, labeller in result:
        labeller_counts[labeller] += 1

    # Check if the difference between max and min assignments is at most 1
    assert max(labeller_counts.values()) - min(labeller_counts.values()) <= 1


def test_whether_distributor_returns_empty_list_on_empty_input():
    datasets = []
    labellers = ["Alice", "Bob"]
    result = list(session._distribute_labelling_projects(datasets, labellers))
    assert len(result) == 0
