from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from src.concept import WikibaseID


class SamplingConfig(BaseModel):
    """A class to hold the sampling configuration."""

    stratified_columns: list[str] = Field(
        default_factory=list,
        description="Sampled passages will be sampled from these columns according to the distribution of the reference dataset.",
    )
    equal_columns: list[str] = Field(
        default_factory=list,
        description="Sampled passages will be sampled from these columns with an equal number of samples from each group.",
    )
    sample_size: int = Field(
        ...,
        description="The number of samples to take.",
        examples=[1000],
        ge=1,
    )
    negative_proportion: float = Field(
        default=0,
        description="The proportion of negative samples to take.",
        examples=[0.2],
        ge=0,
        le=1,
    )
    wikibase_ids: list[WikibaseID] = Field(
        default_factory=list,
        description="The Wikibase IDs of the concepts to sample.",
        examples=["Q42"],
    )
    labellers: list[str] = Field(
        default_factory=list,
        description="The usernames of the labellers to sample.",
        examples=["alice", "bob"],
    )

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "SamplingConfig":
        """
        Load the sampling configuration from a YAML file.

        :param Union[str, Path] file_path: The path to the YAML file.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        if file_path.suffix != ".yaml":
            raise ValueError("File must be a YAML file.")

        with open(file_path, "r") as file:
            config = yaml.safe_load(file)

        return cls(**config)


class Sampler:
    """A class to sample a dataset with control over the target distribution of specified columns."""

    def __init__(self, config: SamplingConfig):
        """
        Initialize the Sampler.

        :param SamplingConfig config: The sampling configuration.
        """
        self.config = config

    def _validate_dataset_and_config(self, dataset: pd.DataFrame) -> None:
        """
        Validate the dataset and configuration.

        :param Dataset dataset: The dataset to validate against the configuration.
        """
        missing_columns = set(
            self.config.stratified_columns + self.config.equal_columns
        ) - set(dataset.columns)
        if missing_columns:
            raise ValueError(
                f"Columns {missing_columns} are not present in the dataset."
            )
        if self.config.sample_size > len(dataset):
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

        if self.config.stratified_columns:
            # Create the strata based on stratified columns
            strata = ref_dataset[self.config.stratified_columns].apply(tuple, axis=1)  # noqa
            strata_proportions = strata.value_counts(normalize=True)

            # Calculate target counts for each stratum
            target_sample_counts = defaultdict(int)
            for stratum, count in strata_proportions.items():
                target_sample_counts[stratum] = int(count * self.config.sample_size)

            # Ensure the total count is exactly the sample size
            total_count = sum(target_sample_counts.values())
            if total_count < self.config.sample_size:
                # Add the difference to the largest stratum
                max_stratum = max(
                    target_sample_counts, key=lambda x: target_sample_counts[x]
                )
                target_sample_counts[max_stratum] += (
                    self.config.sample_size - total_count
                )
            elif total_count > self.config.sample_size:
                # Subtract the difference from the largest stratum
                max_stratum = max(
                    target_sample_counts, key=lambda x: target_sample_counts[x]
                )
                target_sample_counts[max_stratum] -= (
                    total_count - self.config.sample_size
                )

            # Create the final sample DataFrame
            samples = []
            for stratum, count in target_sample_counts.items():
                stratum_df = dataset.loc[strata == stratum]

                if self.config.equal_columns:
                    # Calculate the number of groups in self.config.equal_columns
                    equal_columns = (
                        self.config.equal_columns[0]
                        if len(self.config.equal_columns) == 1
                        else self.config.equal_columns
                    )

                    num_groups = stratum_df[equal_columns].nunique()
                    if num_groups > 0:
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
                        # If num_groups is 0, skip this stratum and move on to the next one
                        continue
                else:
                    # If equal_columns is empty, sample directly from stratum_df
                    sampled_stratum = stratum_df.sample(count, replace=False)
                    samples.append(sampled_stratum)

            sample_df = pd.concat(samples).reset_index(drop=True)

            # If the size is still not correct, adjust by dropping or sampling extra rows
            if len(sample_df) < self.config.sample_size:
                additional_sample = dataset[
                    ~dataset.index.isin(sample_df.index)
                ].sample(self.config.sample_size - len(sample_df), replace=False)
                sample_df = pd.concat([sample_df, additional_sample])

        elif self.config.equal_columns:
            # In this case, we are not stratifying the sample but keeping the
            # distribution of the equal_columns equal. We do so by sampling from each
            # group in the equal_columns separately and then concatenating the samples.
            # Any reference_dataset can be ignored in this case.
            n_groups = len(dataset[self.config.equal_columns].nunique())
            samples = []
            grouper = (
                self.config.equal_columns[0]
                if len(self.config.equal_columns) == 1
                else self.config.equal_columns
            )
            for _, group_df in dataset.groupby(grouper, dropna=False):
                samples.append(
                    group_df.sample(self.config.sample_size // n_groups, replace=False)
                )

            sample_df = pd.concat(samples).reset_index(drop=True)

            # If the sampled set is too small, add more from the leftover rows
            if len(sample_df) < self.config.sample_size:
                additional_sample = dataset[
                    ~dataset.index.isin(sample_df.index)
                ].sample(self.config.sample_size - len(sample_df), replace=False)
                sample_df = pd.concat([sample_df, additional_sample])

        else:
            # In this case, we are not stratifying or equalizing any columns so a random
            # sample can be taken directly from the dataframe
            sample_df = dataset

        # If the size is too large, sample randomly from the set before returning
        sample_df = sample_df.sample(self.config.sample_size, replace=False)

        return sample_df
