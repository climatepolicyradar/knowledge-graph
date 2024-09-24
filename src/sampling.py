from collections import defaultdict
from typing import Optional

import pandas as pd

from scripts.config import EQUAL_COLUMNS, STRATIFIED_COLUMNS
from src.labelled_passage import LabelledPassage


class Sampler:
    """A class to sample a dataset with control over the target distribution of specified columns."""

    def __init__(
        self,
        stratified_columns: list[str] = STRATIFIED_COLUMNS,
        equal_columns: list[str] = EQUAL_COLUMNS,
    ):
        """
        Initialise a sampler

        :param list[str] stratified_columns: Sampled passages will be sampled from
        these columns according to the distribution of the reference dataset
        :param list[str] equal_columns: Sampled passages will be sampled from these
        columns with an equal number of samples from each group
        :param int sample_size: The number of samples to take
        """
        self.stratified_columns = stratified_columns
        self.equal_columns = equal_columns

    @staticmethod
    def _labelled_passages_to_dataframe(
        labelled_passages: list[LabelledPassage],
    ) -> pd.DataFrame:
        """
        Convert a list of labelled passages to a DataFrame.

        The incoming labelled passages should be unlabelled at this point, ie they should
        contain no spans (only text and metadata). The metadata will be flattened and
        included in the DataFrame for sampling.

        :param list[LabelledPassage] labelled_passages: The labelled passages to convert.
        :return DataFrame: The labelled passages as a DataFrame.
        """
        return pd.DataFrame(
            [
                {
                    "text": labelled_passage.text,
                    **labelled_passage.metadata,
                }
                for labelled_passage in labelled_passages
            ]
        ).fillna("")

    @staticmethod
    def dataframe_to_labelled_passages(
        dataframe: pd.DataFrame,
    ) -> list[LabelledPassage]:
        """
        Convert a DataFrame to a list of labelled passages.

        As the input dataframe is expected to be the output of a sampling operation, it
        will contain only text and flattened metadata (no spans). Any non-text columns
        in the dataset will be separated from the text and included in the metadata
        field of the LabelledPassage.

        :param pd.DataFrame dataframe: The DataFrame to convert.
        :return list[LabelledPassage]: The labelled passages.
        """
        data = []
        for _, row in dataframe.iterrows():
            text = row.pop("text")
            data.append(
                LabelledPassage(text=text, spans=[], metadata=row.astype(str).to_dict())
            )
        return data

    def _validate_dataset_and_config(self, dataframe: pd.DataFrame) -> None:
        """
        Validate the dataframe and configuration.

        :param Dataset dataframe: The dataframe to validate against the configuration.
        """
        missing_columns = set(self.stratified_columns + self.equal_columns) - set(
            dataframe.columns
        )
        if missing_columns:
            raise ValueError(
                f"Columns {missing_columns} are not present in the dataframe."
            )

    def sample(
        self,
        sample_size: int,
        dataset: list[LabelledPassage],
        reference_dataset: Optional[pd.DataFrame] = None,
        # TODO: The dataset and reference_dataset types should both be made List[LabelledPassage] in a future PR
    ) -> list[LabelledPassage]:
        """
        Sample a dataset with control over the target distribution of specified columns.

        :param int sample_size: The number of samples to take
        :param list[LabelledPassage] dataset: The dataset to sample from
        :param Optional[pd.DataFrame] reference_dataset: The dataset to use as a
        reference for the target distribution. Samples will be taken from the dataset
        to match the distribution of the reference_dataset. If None, the target
        distribution will be based on the primary dataset.
        :return list[LabelledPassage]: The sampled dataset.
        """
        dataframe = self._labelled_passages_to_dataframe(dataset)
        self._validate_dataset_and_config(dataframe)

        if reference_dataset is not None:
            self._validate_dataset_and_config(reference_dataset)
        else:
            reference_dataset = dataframe

        # reassure pyright that reference_dataset is a DataFrame
        dataframe: pd.DataFrame = dataframe
        reference_dataframe: pd.DataFrame = reference_dataset

        if sample_size < 1:
            raise ValueError("sample_size must be a positive integer.")
        if sample_size > len(dataframe):
            raise ValueError(
                "sample_size cannot be larger than the number of samples in the dataset."
            )

        if self.stratified_columns:
            # Create the strata based on stratified columns
            strata = reference_dataframe[self.stratified_columns].apply(tuple, axis=1)  # noqa
            strata_proportions = strata.value_counts(normalize=True)

            # Calculate target counts for each stratum
            target_sample_counts = defaultdict(int)
            for stratum, count in strata_proportions.items():
                target_sample_counts[stratum] = int(count * sample_size)

            # Ensure the total count is exactly the sample size
            total_count = sum(target_sample_counts.values())
            if total_count < sample_size:
                # Add the difference to the largest stratum
                max_stratum = max(
                    target_sample_counts, key=lambda x: target_sample_counts[x]
                )
                target_sample_counts[max_stratum] += sample_size - total_count
            elif total_count > sample_size:
                # Subtract the difference from the largest stratum
                max_stratum = max(
                    target_sample_counts, key=lambda x: target_sample_counts[x]
                )
                target_sample_counts[max_stratum] -= total_count - sample_size

            # Create the final sample DataFrame
            samples = []
            for stratum, count in target_sample_counts.items():
                stratum_df = dataframe.loc[strata == stratum]

                if self.equal_columns:
                    # Calculate the number of groups in self.equal_columns
                    equal_columns = (
                        self.equal_columns[0]
                        if len(self.equal_columns) == 1
                        else self.equal_columns
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

            sample_df: pd.DataFrame = pd.concat(samples).reset_index(drop=True)

            # If the size is still not correct, adjust by dropping or sampling extra rows
            if len(sample_df) < sample_size:
                additional_sample = dataframe[
                    ~dataframe.index.isin(sample_df.index)
                ].sample(sample_size - len(sample_df), replace=False)
                sample_df = pd.concat([sample_df, additional_sample])

        elif self.equal_columns:
            # In this case, we are not stratifying the sample but keeping the
            # distribution of the equal_columns equal. We do so by sampling from each
            # group in the equal_columns separately and then concatenating the samples.
            # Any reference_dataset can be ignored in this case.
            n_groups = len(dataframe[self.equal_columns].nunique())
            samples = []
            grouper = (
                self.equal_columns[0]
                if len(self.equal_columns) == 1
                else self.equal_columns
            )
            for _, group_df in dataframe.groupby(grouper, dropna=False):
                samples.append(group_df.sample(sample_size // n_groups, replace=False))

            sample_df = pd.concat(samples).reset_index(drop=True)

            # If the sampled set is too small, add more from the leftover rows
            if len(sample_df) < sample_size:
                additional_sample = dataframe[
                    ~dataframe.index.isin(sample_df.index)
                ].sample(sample_size - len(sample_df), replace=False)
                sample_df = pd.concat([sample_df, additional_sample])

        else:
            # In this case, we are not stratifying or equalizing any columns so a random
            # sample can be taken directly from the dataframe
            sample_df = dataframe

        # If the size is too large, sample randomly from the set before returning
        sample_df = sample_df.sample(sample_size, replace=False)

        return self.dataframe_to_labelled_passages(sample_df)
