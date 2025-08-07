import os
import uuid
from datetime import datetime
from typing import Optional

from argilla import (
    Argilla,
    Dataset,
    FloatMetadataProperty,
    Record,
    Settings,
    SpanQuestion,
    TermsMetadataProperty,
    TextField,
    Workspace,
)
from dotenv import find_dotenv, load_dotenv

from src.labelled_passage import LabelledPassage
from src.wikibase import Concept

load_dotenv(find_dotenv())


class ArgillaSession:
    """A session for interacting with Argilla"""

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.client = Argilla(
            api_key=api_key or os.getenv("ARGILLA_API_KEY"),
            api_url=api_url or os.getenv("ARGILLA_API_URL"),
        )

    def get_all_datasets(self, workspace_name: str) -> list[Dataset]:
        """
        Get all datasets in a workspace.

        :param Workspace workspace: The workspace to get the datasets from
        :return list[Dataset]: A list of datasets
        """
        datasets = []
        workspace = self.client.workspaces(name=workspace_name)
        assert isinstance(workspace, Workspace)
        for dataset in workspace.datasets:
            datasets.append(dataset)
        return datasets

    @staticmethod
    def _concept_to_dataset_name(concept: Concept) -> str:
        """Extracts the dataset name from a concept"""
        if not concept.wikibase_id:
            raise ValueError("Concept has no Wikibase ID")
        return concept.wikibase_id

    def labelled_passages_to_dataset(
        self,
        labelled_passages: list[LabelledPassage],
        concept: Concept,
        workspace: Workspace,
    ) -> Dataset:
        """
        Convert a list of LabelledPassages into an Argilla kDataset.

        :param list[LabelledPassage] labelled_passages: The labelled passages to convert
        :param Concept concept: The concept being annotated
        :return Dataset: An Argilla Dataset, ready to be pushed
        """
        dataset = Dataset(
            name=self._concept_to_dataset_name(concept),
            settings=Settings(
                guidelines="Highlight the entity if it is present in the text",
                fields=[
                    TextField(name="text", title="Text", use_markdown=True),
                ],
                questions=[
                    SpanQuestion(
                        name="entities",
                        labels={str(concept.wikibase_id): concept.preferred_label},
                        field="text",
                        required=True,
                        allow_overlapping=False,
                    )
                ],
                # Argilla has the following regex for metadata field names: {"pattern":"^(?=.*[a-z0-9])[a-z0-9_-]+$"}
                # changing the dots to hyphens.
                # Also, it doesn't allow capital characters, so lowercasing everything
                metadata=[
                    TermsMetadataProperty("text_block-text_block_id"),
                    TermsMetadataProperty("text_block-language"),
                    TermsMetadataProperty("text_block-type"),
                    FloatMetadataProperty("text_block-type_confidence"),
                    FloatMetadataProperty("text_block-page_number"),
                    TermsMetadataProperty("text_block-coords"),
                    TermsMetadataProperty("document_id"),
                    TermsMetadataProperty("document_name"),
                    TermsMetadataProperty("document_source_url"),
                    TermsMetadataProperty("document_content_type"),
                    TermsMetadataProperty("document_md5_sum"),
                    TermsMetadataProperty("languages"),
                    TermsMetadataProperty("translated"),
                    TermsMetadataProperty("has_valid_text"),
                    TermsMetadataProperty("pipeline_metadata"),
                    TermsMetadataProperty("document_metadata-name"),
                    TermsMetadataProperty("document_metadata-document_title"),
                    TermsMetadataProperty("document_metadata-description"),
                    TermsMetadataProperty("document_metadata-import_id"),
                    TermsMetadataProperty("document_metadata-slug"),
                    TermsMetadataProperty("document_metadata-family_import_id"),
                    TermsMetadataProperty("document_metadata-family_slug"),
                    TermsMetadataProperty("document_metadata-publication_ts"),
                    TermsMetadataProperty("document_metadata-date"),
                    TermsMetadataProperty("document_metadata-source_url"),
                    TermsMetadataProperty("document_metadata-download_url"),
                    TermsMetadataProperty("document_metadata-corpus_import_id"),
                    TermsMetadataProperty("document_metadata-corpus_type_name"),
                    TermsMetadataProperty("document_metadata-collection_title"),
                    TermsMetadataProperty("document_metadata-collection_summary"),
                    TermsMetadataProperty("document_metadata-type"),
                    TermsMetadataProperty("document_metadata-source"),
                    TermsMetadataProperty("document_metadata-category"),
                    TermsMetadataProperty("document_metadata-geography"),
                    TermsMetadataProperty("document_metadata-geographies"),
                    TermsMetadataProperty("document_metadata-languages"),
                    TermsMetadataProperty("document_metadata-metadata"),
                    TermsMetadataProperty("document_description"),
                    TermsMetadataProperty("document_cdn_object"),
                    TermsMetadataProperty("document_slug"),
                    TermsMetadataProperty("pdf_data-md5sum"),
                    TermsMetadataProperty("pdf_data_page_metadata-dimensions"),
                    TermsMetadataProperty("pdf_data_page_metadata-page_number"),
                    TermsMetadataProperty("_html_data-detected_title"),
                    TermsMetadataProperty("_html_data-detected_date"),
                    TermsMetadataProperty("_html_data-has_valid_text"),
                    TermsMetadataProperty("pipeline_metadata-parser_metadata"),
                    TermsMetadataProperty("text_block-index"),
                    TermsMetadataProperty("world_bank_region"),
                    TermsMetadataProperty("corpus_type_name"),
                    TermsMetadataProperty("geographies"),
                ],
            ),
            workspace=workspace,
            client=self.client,
        )

        dataset.id = uuid.uuid4()
        dataset.create()

        records = [
            Record(
                fields={"text": passage.text},
                metadata=self._reformat_metadata_keys(passage.metadata),
            )
            for passage in labelled_passages
        ]

        dataset.records.log(records)

        return dataset

    @staticmethod
    def _reformat_metadata_keys(metadata: dict) -> dict:
        """Changes dots to hyphens in the key name"""
        # Dropping this, as it can't be serialised into the metadata field by Argilla...
        metadata.pop("KeywordClassifier", None)
        metadata.pop("EmbeddingClassifier", None)
        return {key.replace(".", "-").lower(): value for key, value in metadata.items()}

    def dataset_to_labelled_passages(self, dataset: Dataset) -> list[LabelledPassage]:
        """
        Convert an Argilla Dataset into a list of LabelledPassages.

        :param kDataset dataset: The Argilla Dataset to convert
        :return list[LabelledPassage]: A list of LabelledPassage objects
        """
        return [
            LabelledPassage.from_argilla_record(record, self.client)
            for record in dataset.records
        ]

    @staticmethod
    def _is_between_timestamps(
        timestamp: datetime,
        min_timestamp: Optional[datetime],
        max_timestamp: Optional[datetime],
    ) -> bool:
        """
        Check whether a timestamp falls within a given time range.

        :param datetime timestamp: The timestamp to check
        :param Optional[datetime] min_timestamp: The minimum timestamp (inclusive). If None, no minimum limit.
        :param Optional[datetime] max_timestamp: The maximum timestamp (inclusive). If None, no maximum limit.
        :return bool: True if the timestamp is within the range, False otherwise
        """
        if max_timestamp and timestamp > max_timestamp:
            return False
        if min_timestamp and timestamp < min_timestamp:
            return False
        return True

    def _filter_labelled_passages_by_timestamp(
        self,
        labelled_passages: list[LabelledPassage],
        min_timestamp: Optional[datetime] = None,
        max_timestamp: Optional[datetime] = None,
    ) -> list[LabelledPassage]:
        """Uses max- and min-timestampts to filter out the labelled passages"""
        filtered_passages = []
        for passage in labelled_passages:
            passage_copy = passage.model_copy(update={"spans": []})
            for span in passage.spans:
                span_copy = span.model_copy(update={"labellers": [], "timestamps": []})
                for labeller, timestamp in zip(span.labellers, span.timestamps):
                    if self._is_between_timestamps(
                        timestamp=timestamp,
                        min_timestamp=min_timestamp,
                        max_timestamp=max_timestamp,
                    ):
                        span_copy.labellers.append(labeller)
                        span_copy.timestamps.append(timestamp)

                if len(span_copy.labellers) > 0:
                    passage_copy.spans.append(span_copy)

            if len(passage_copy.spans) > 0:
                filtered_passages.append(passage_copy)

        return filtered_passages

    def pull_labelled_passages(
        self,
        concept: Concept,
        workspace: str = "knowledge-graph",
        min_timestamp: Optional[datetime] = None,
        max_timestamp: Optional[datetime] = None,
    ) -> list[LabelledPassage]:
        """
        Get the labelled passages from Argilla for a given concept.

        :param Concept concept: The concept to get the labelled passages for
        :param Optional[datetime] min_timestamp: Only get annotations made after this timestamp (inclusive), defaults to None
        :param Optional[datetime] max_timestamp: Only get annotations made before this timestamp (inclusive), defaults to None
        :raises ValueError: If no dataset matching the concept ID was found in Argilla
        :raises ValueError: If no datasets were found in Argilla, you may need to be granted access to the workspace(s)
        :return list[LabelledPassage]: A list of LabelledPassage objects
        """
        # First, see whether the dataset exists with the name we expect

        dataset = self.client.datasets(  # type: ignore
            self._concept_to_dataset_name(concept), workspace=workspace
        )

        labelled_passages = self.dataset_to_labelled_passages(dataset)  # type: ignore
        if min_timestamp or max_timestamp:
            labelled_passages = self._filter_labelled_passages_by_timestamp(
                labelled_passages, min_timestamp, max_timestamp
            )
        return labelled_passages

    def combine_datasets(self, *datasets: Dataset) -> Dataset:
        """
        Combine an arbitrary number of argilla datasets into one.

        :param FeedbackDataset *datasets: Unspecified number of datasets to combine, at
        least one.
        :return FeedbackDataset: The combined dataset
        """
        if not datasets:
            raise ValueError("At least one dataset must be provided")

        self._assert_datasets_of_the_same_type(*datasets)

        settings = datasets[0].settings

        combined_dataset = Dataset(
            name=f"combined-{'-'.join([dataset.name for dataset in datasets])}",
            settings=settings,
        ).create()

        for dataset in datasets:
            combined_dataset.records.log(list(dataset.records))

        return combined_dataset

    @staticmethod
    def _assert_datasets_of_the_same_type(*datasets: Dataset):
        """
        Assert that all datasets are of the same type.

        :param FeedbackDataset *datasets: The datasets to check
        :raises ValueError: If the datasets are not of the same type
        """
        fields = datasets[0].fields
        questions = datasets[0].questions

        for dataset in datasets[1:]:
            if dataset.fields != fields:
                raise ValueError("All datasets must have the same fields")
            if dataset.questions != questions:
                raise ValueError("All datasets must have the same questions")

    def delete_dataset(self, dataset_name: str, workspace: str = "knowledge-graph"):
        """
        Delete a dataset from Argilla.

        :param str dataset_name: The name of the dataset to delete
        """
        dataset = self.client.datasets(name=dataset_name, workspace=workspace)
        dataset.delete()  # type: ignore
