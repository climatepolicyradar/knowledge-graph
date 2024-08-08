from collections import defaultdict
from typing import List, Optional

import pandas as pd


class Sampler:
    """A class to sample a dataset with control over the target distribution of specified columns."""

    def __init__(
        self,
        sample_size: int,
        stratified_columns: List[str] = [],
        equal_columns: List[str] = [],
    ):
        """
        Initialize the Sampler.

        :param List[str] stratified_columns: The columns to stratify the sample on.
        :param List[str] equal_columns: The columns to keep the distribution equal.
        :param int sample_size: The number of samples to take.
        """
        self.stratified_columns = list(stratified_columns) if stratified_columns else []
        self.equal_columns = equal_columns
        self.sample_size = int(sample_size)

    def _validate_dataset_and_config(self, dataset: pd.DataFrame) -> None:
        """
        Validate the dataset and configuration.

        :param Dataset dataset: The dataset to validate against the configuration.
        """
        missing_columns = set(self.stratified_columns + self.equal_columns) - set(
            dataset.columns
        )
        if missing_columns:
            raise ValueError(
                f"Columns {missing_columns} are not present in the dataset."
            )
        if self.sample_size > len(dataset):
            raise ValueError(
                "sample_size cannot be larger than the number of samples in the dataset."
            )

    def sample(
        self,
        dataset: pd.DataFrame,
        reference_dataset: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Sample a dataset with control over the target distribution of specified columns.

        :param Dataset dataset: The dataset to sample from
        :param Optional[Dataset] reference_dataset: The dataset to use as a reference
        for the target distribution. Samples will be taken from the dataset to match
        the distribution of the reference_dataset. If None, the target distribution
        will be based on the primary dataset.
        :return Dataset: The sampled dataset.
        """
        self._validate_dataset_and_config(dataset)

        dataset = dataset.fillna("")
        ref_dataset: pd.DataFrame = (
            reference_dataset.fillna("") if reference_dataset is not None else dataset
        )
        self._validate_dataset_and_config(ref_dataset)

        if self.stratified_columns:
            # Create the strata based on stratified columns
            strata = ref_dataset[self.stratified_columns].apply(tuple, axis=1)  # noqa
            strata_proportions = strata.value_counts(normalize=True)

            # Calculate target counts for each stratum
            target_sample_counts = defaultdict(int)
            for stratum, count in strata_proportions.items():
                target_sample_counts[stratum] = int(count * self.sample_size)

            # Ensure the total count is exactly the sample size
            total_count = sum(target_sample_counts.values())
            if total_count < self.sample_size:
                # Add the difference to the largest stratum
                max_stratum = max(
                    target_sample_counts, key=lambda x: target_sample_counts[x]
                )
                target_sample_counts[max_stratum] += self.sample_size - total_count
            elif total_count > self.sample_size:
                # Subtract the difference from the largest stratum
                max_stratum = max(
                    target_sample_counts, key=lambda x: target_sample_counts[x]
                )
                target_sample_counts[max_stratum] -= total_count - self.sample_size

            # Create the final sample DataFrame
            samples = []
            for stratum, count in target_sample_counts.items():
                stratum_df = dataset[strata == stratum]

                if self.equal_columns:
                    # Calculate the number of groups in self.equal_columns
                    equal_columns = (
                        self.equal_columns[0]
                        if len(self.equal_columns) == 1
                        else self.equal_columns
                    )

                    num_groups = stratum_df[equal_columns].nunique()
                    samples_per_group = int(count // num_groups)
                    for _, group_df in stratum_df.groupby(equal_columns):
                        self.sample_size_for_group = min(
                            samples_per_group, len(group_df)
                        )
                        sampled_group = group_df.sample(
                            self.sample_size_for_group, replace=False
                        )
                        samples.append(sampled_group)
                else:
                    # If equal_columns is empty, sample directly from stratum_df
                    sampled_stratum = stratum_df.sample(count, replace=False)
                    samples.append(sampled_stratum)

            sample_df = pd.concat(samples).reset_index(drop=True)

            # If the size is still not correct, adjust by dropping or sampling extra rows
            if len(sample_df) < self.sample_size:
                additional_sample = dataset[
                    ~dataset.index.isin(sample_df.index)
                ].sample(self.sample_size - len(sample_df), replace=False)
                sample_df = pd.concat([sample_df, additional_sample])

        elif self.equal_columns:
            # In this case, we are not stratifying the sample but keeping the
            # distribution of the equal_columns equal. We do so by sampling from each
            # group in the equal_columns separately and then concatenating the samples.
            # Any reference_dataset can be ignored in this case.
            n_groups = len(dataset[self.equal_columns].nunique())
            samples = []
            grouper = (
                self.equal_columns[0]
                if len(self.equal_columns) == 1
                else self.equal_columns
            )
            for _, group_df in dataset.groupby(grouper, dropna=False):
                samples.append(
                    group_df.sample(self.sample_size // n_groups, replace=False)
                )

            sample_df = pd.concat(samples).reset_index(drop=True)

            # If the sampled set is too small, add more from the leftover rows
            if len(sample_df) < self.sample_size:
                additional_sample = dataset[
                    ~dataset.index.isin(sample_df.index)
                ].sample(self.sample_size - len(sample_df), replace=False)
                sample_df = pd.concat([sample_df, additional_sample])

        else:
            # In this case, we are not stratifying or equalizing any columns so a random
            # sample can be taken directly from the dataframe
            sample_df = dataset

        # If the size is too large, sample randomly from the set before returning
        sample_df = sample_df.sample(self.sample_size, replace=False)

        return sample_df
