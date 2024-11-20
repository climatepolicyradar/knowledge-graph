import warnings

import numpy as np
import pandas as pd


def split_evenly(n: int, k: int) -> list[int]:
    """Split n into k values that sum to n, distributing remainder at the end"""
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


def create_balanced_sample(
    df: pd.DataFrame,
    sample_size: int,
    on_columns: list[str],
    min_samples_per_combination: int = 1,
) -> pd.DataFrame:
    """
    Sample a balanced dataset from a dataframe with improved performance and balance.

    :param pd.DataFrame df: The dataframe to sample from
    :param int sample_size: The number of rows to sample
    :param list[str] columns: The columns to balance the sample over
    :param bool show_progress: If True, show a progress bar. Default is True
    :param int min_samples_per_combination: Minimum samples required per combination to be included
    :return pd.DataFrame: A balanced sample of the dataframe
    """
    # Input validation
    missing_columns = set(on_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} are not in the dataframe")
    if len(on_columns) == 0:
        raise ValueError("At least one column must be specified")
    if sample_size > len(df):
        raise ValueError(
            "Sample size should be less than or equal to the number of rows in the dataframe"
        )

    # Drop any rows where the column values are None or "None"
    df = df.dropna(subset=on_columns)
    df = df[~df[on_columns].eq("None").any(axis=1)]

    # Pre-compute categorical values and create category codes for faster filtering
    for col in on_columns:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = pd.Categorical(df[col])

    # Create a compound key for faster grouping
    df["_group_key"] = df[on_columns].apply(lambda x: "_".join(x.astype(str)), axis=1)

    # Get group sizes and filter out rare combinations
    group_sizes = df.groupby("_group_key").size()
    valid_groups = group_sizes[group_sizes >= min_samples_per_combination]

    if len(valid_groups) == 0:
        raise ValueError(
            "No valid combinations found with the specified minimum samples"
        )

    # Calculate the optimal number of samples per group to get as close to sample_size
    # as possible
    samples_per_group = np.floor(sample_size / len(valid_groups))

    # Sample from each group
    group_dfs: list[pd.DataFrame] = []

    for group in valid_groups.index:
        group_df = df[df["_group_key"] == group]
        n_samples = min(int(samples_per_group), len(group_df))

        sampled_indices = np.random.choice(
            group_df.index, size=n_samples, replace=False
        )
        group_dfs.append(df.loc[sampled_indices])

    result = pd.concat(group_dfs)

    # Handle any remaining samples needed to reach sample_size
    missing_rows = sample_size - len(result)
    if missing_rows > 0:
        remaining_df = df[~df.index.isin(result.index)]
        if len(remaining_df) > 0:
            try:
                remaining_indices = np.random.choice(
                    remaining_df.index,
                    size=missing_rows,
                    replace=False,
                )
                result = pd.concat([result, df.loc[remaining_indices]])
            except ValueError as e:
                warnings.warn(f"Warning: Error sampling remaining rows: {e}")
        else:
            warnings.warn(
                "Warning: Not enough remaining rows to reach the desired sample size"
            )
    elif missing_rows < 0:
        try:
            result = result.sample(sample_size)
        except ValueError as e:
            warnings.warn(f"Warning: Error reducing the sample size: {e}")

    # Combine and clean up
    result = result.reset_index(drop=True)
    result = result.drop("_group_key", axis=1)

    return result
