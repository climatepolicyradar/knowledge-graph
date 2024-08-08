from itertools import cycle
from typing import Any, Generator, Tuple


def distribute_labelling_projects(
    datasets: list, labellers: list[str], min_labellers: int = 2
) -> Generator[Tuple[Any, str], None, None]:
    """
    Distribute labelling projects to labellers.

    For efficient labelling, tasks should be distributed such that each dataset is
    labelled by at least `min_labellers` labellers, and each labeller is assigned to a
    minimal number of datasets.

    :param list[] datasets: datasets to distribute among labellers
    :param list[str] labellers: list of labellers
    :param int min_labellers: minimum number of labellers per dataset, defaults to 2
    :return Generator[Tuple[Any, str], None, None]: a generator of tuples containing
        the dataset and the labeller assigned to it
    """
    if len(labellers) < min_labellers:
        raise ValueError(
            "number of items in labellers must be greater than or equal to min_labellers"
        )

    labeller_cycle = cycle(labellers)
    for dataset in datasets:
        for _ in range(min_labellers):
            yield dataset, next(labeller_cycle)
