from itertools import cycle, product

import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TimeRemainingColumn,
)


def sample_balanced_dataset(
    df: pd.DataFrame,
    sample_size: int,
    columns: list[str],
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Sample a balanced dataset from a dataframe based on values in the specified columns

    The function will sample `sample_size` rows from the dataframe `df` such that the
    resulting sample is balanced over the values in the specified `columns`. For example,
    if `columns=["A", "B"]`, the function will sample `sample_size` rows such that the
    number of rows with each unique combination of values in columns "A" and "B" is
    approximately equal. The function will sample without replacement, so the number of
    unique combinations of values in the specified columns must be less than the number
    of rows in the dataframe.

    Note that the results will probably not be _perfectly_ balanced, as filtering for
    some combinations of values in the supplied `columns` is likely to result in a
    small/empty dataset in most real-world datasets. However, in most cases the
    results should be much more balanced than the original dataset.

    :param pd.DataFrame df: The dataframe to sample from
    :param int sample_size: The number of rows to sample
    :param list[str] columns: The columns to balance the sample over
    :param bool show_progress: If True, show a progress bar. Default is True
    :return pd.DataFrame: A balanced sample of the dataframe
    """
    missing_columns = set(columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} are not in the dataframe")

    if len(columns) == 0:
        raise ValueError("At least one column must be specified")

    if sample_size > len(df):
        raise ValueError(
            "Sample size should be less than the number of rows in the dataframe"
        )

    # get all the unique values for each column
    categorical_column_values = {
        column: [
            value
            for value in df[column].unique()
            if value != "None" and value is not None
        ]
        for column in columns
    }

    # get all the combinations of values
    combination_values = list(
        product(*[values for values in categorical_column_values.values()])
    )

    # create a dictionary of the combinations
    combinations = [
        {column: value for column, value in zip(columns, combination)}
        for combination in combination_values
    ]

    # sample without replacement from df over each combination until we have a sample of sample_size
    balanced_sample_dataframe = pd.DataFrame()

    # first set up a cache for the rows which match each of our constraints so that we don't
    # need to run a fresh query each time, and can instead sample from the cached rows
    matching_rows_cache = {}

    if show_progress:
        progress_bar = Progress(
            "[progress.description]{task.description}",
            TaskProgressColumn(),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            transient=True,
        )
        # set up a progress bar to visualise the sampling process
        progress_task = progress_bar.add_task(
            description="Sampling a more balanced dataset", total=sample_size
        )
        progress_bar.start()

    # cycle through the combinations until we have a dataset of the desired size
    combination_cycle = cycle(combinations)

    while len(balanced_sample_dataframe) < sample_size:
        combination = next(combination_cycle)
        matching_rows = matching_rows_cache.get(str(combination), None)
        if matching_rows is None:
            matching_rows = df.copy()
            for column, value in combination.items():
                matching_rows = matching_rows[matching_rows[column] == value]
            matching_rows_cache[str(combination)] = matching_rows

        if len(matching_rows) == 0:
            continue

        else:
            sampled_row = matching_rows.sample(1)
            matching_rows_cache[str(combination)] = matching_rows.drop(
                sampled_row.index
            )
            balanced_sample_dataframe = pd.concat(
                [balanced_sample_dataframe, sampled_row]
            )
            if show_progress:
                progress_bar.update(progress_task, advance=1)  # pyright: ignore

    balanced_sample_dataframe = balanced_sample_dataframe.reset_index(drop=True)

    if show_progress:
        progress_bar.stop()  # pyright: ignore

    return balanced_sample_dataframe
