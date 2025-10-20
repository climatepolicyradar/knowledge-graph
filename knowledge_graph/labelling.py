import os
import uuid
from logging import getLogger
from typing import Optional, Sequence
from uuid import UUID

from argilla import (
    Argilla,
    Dataset,
    ResponseStatus,
    Settings,
    SpanQuestion,
    TaskDistribution,
    TextField,
    Workspace,
)
from dotenv import find_dotenv, load_dotenv

from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.span import Span
from knowledge_graph.wikibase import Concept

logger = getLogger(__name__)

load_dotenv(find_dotenv())


class ArgillaSession:
    """Session for interacting with Argilla"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        workspace: str = "knowledge-graph",
    ):
        """Initialize an Argilla session"""
        self.client = Argilla(
            api_key=api_key or os.getenv("ARGILLA_API_KEY"),
            api_url=api_url or os.getenv("ARGILLA_API_URL"),
        )
        self.default_workspace = workspace
        logger.debug(
            "Initialized Argilla session with default workspace: %s",
            self.default_workspace,
        )

    def __repr__(self) -> str:
        """Return a string representation of the Argilla session"""
        return f"<ArgillaSession: workspace={self.default_workspace}>"

    def get_workspace(self, name: Optional[str] = None) -> Workspace:
        """
        Get a workspace by name.

        Args:
            name: If a name is not provided, the session's default_workspace will be
            used. If the workspace is not found, a ValueError will be raised.

        Returns:
            Workspace object.
        """
        workspace_name = name or self.default_workspace
        logger.info("Fetching workspace: %s", workspace_name)
        workspace_object = self.client.workspaces(name=workspace_name)
        if not workspace_object:
            logger.warning("Workspace '%s' not found", workspace_name)
            raise ValueError(f"Workspace '{workspace_name}' not found")
        logger.info("Successfully retrieved workspace: %s", workspace_object.name)
        return workspace_object

    def get_dataset(
        self,
        wikibase_id: WikibaseID,
        workspace: Optional[str] = None,
    ) -> Dataset:
        """
        Get a dataset by its Wikibase ID.

        Args:
            wikibase_id: Wikibase ID of the dataset.
            workspace: Workspace name. Defaults to session's default_workspace.

        Returns:
            Dataset object.
        """
        logger.info("Fetching dataset '%s'", wikibase_id)
        workspace_object = self.get_workspace(workspace)
        dataset = self.client.datasets(
            name=str(wikibase_id), workspace=workspace_object
        )
        if not dataset:
            logger.warning(
                "Dataset '%s' not found in workspace '%s'",
                wikibase_id,
                workspace_object.name,
            )
            raise ValueError(
                f"Dataset '{wikibase_id}' not found in workspace '{workspace_object.name}'"
            )
        logger.debug("Successfully retrieved dataset: %s", wikibase_id)
        return dataset

    def get_all_datasets(self, workspace: Optional[str] = None) -> list[Dataset]:
        """
        Get all datasets in a workspace.

        Args:
            workspace: Workspace name. Defaults to session's default_workspace.

        Returns:
            List of Dataset objects.
        """
        workspace_object = self.get_workspace(workspace)
        datasets = workspace_object.datasets
        logger.info(
            "Found %d dataset(s) in workspace '%s'",
            len(datasets),
            workspace_object.name,
        )
        return datasets

    def create_dataset(
        self,
        concept: Concept,
        workspace: Optional[str] = None,
    ) -> Dataset:
        """
        Create a dataset for a concept in the given workspace.

        The dataset will be named after the concept's Wikibase ID.

        Args:
            concept: Concept to create a dataset for.
            workspace: Workspace name. Defaults to session's default_workspace.

        Returns:
            The created dataset.
        """

        logger.info("Creating dataset for concept: %s", concept)

        workspace_object = self.get_workspace(workspace)
        if not concept.wikibase_id:
            raise ValueError(
                f"Concept '{concept}' must have a Wikibase ID to create a dataset"
            )

        try:
            if dataset := self.get_dataset(concept.wikibase_id, workspace):
                logger.warning(
                    "Dataset '%s' already exists in workspace '%s'. "
                    "Returning existing dataset instead of creating a new one.",
                    dataset.name,
                    workspace_object.name,
                )
                return dataset
        except ValueError:
            logger.debug("Dataset for %s does not yet exist", concept)

        settings = Settings(
            guidelines="Highlight the entity if it is present in the text",
            fields=[TextField(name="text", title="Text", use_markdown=True)],
            questions=[
                SpanQuestion(
                    name="entities",
                    field="text",
                    labels={str(concept.wikibase_id): concept.preferred_label},
                    required=True,
                    allow_overlapping=False,
                )
            ],
            distribution=TaskDistribution(min_submitted=2),
        )
        dataset = Dataset(
            name=str(concept.wikibase_id),
            workspace=workspace_object,
            settings=settings,
            client=self.client,
        )
        created_dataset = dataset.create()
        logger.info(
            "Successfully created dataset %s for concept %s in workspace %s",
            created_dataset.name,
            concept,
            workspace_object.name,
        )
        return created_dataset

    def push_labelled_passages(
        self,
        dataset: Dataset,
        labelled_passages: list[LabelledPassage],
    ) -> Dataset:
        """
        Add labelled passages to an existing dataset in Argilla.

        Args:
            dataset: Dataset to add passages to.
            labelled_passages: List of LabelledPassage objects to add. Note that only
                the text and the metadata of each passage will be uploaded - spans
                attached to the labelled passages will be ignored.

        Returns:
            The updated dataset.
        """
        logger.info(
            "Pushing %d labelled passages to dataset: %s",
            len(labelled_passages),
            dataset.name,
        )
        records = []
        for passage in labelled_passages:
            records.append(
                {
                    "id": str(uuid.uuid4()),
                    "fields": {"text": passage.text},
                    "metadata": self._format_metadata(passage.metadata),
                }
            )
        logger.debug("Formatted %d records for Argilla ingestion", len(records))
        dataset.records.log(records)
        logger.info(
            "Successfully pushed %d labelled passages to dataset: %s",
            len(labelled_passages),
            dataset.name,
        )
        return dataset

    def pull_labelled_passages(
        self,
        dataset: Dataset,
        include_statuses: Optional[Sequence[ResponseStatus]] = None,
        limit: Optional[int] = None,
    ) -> list[LabelledPassage]:
        """
        Pull labelled passages from a dataset.

        Creates one LabelledPassage per response. For records with multiple responses
        (e.g., from different labellers), each response will produce a separate passage.
        This allows downstream code to explicitly decide how to merge or aggregate
        across labellers.

        Args:
            dataset: Dataset to pull passages from.
            include_statuses: Response statuses to include. Defaults to [ResponseStatus.submitted].
            limit: Max number of records to pull (note: may return more passages if
                records have multiple responses).

        Returns:
            List of LabelledPassage objects, one per response.
        """
        if include_statuses is None:
            include_statuses = [ResponseStatus.submitted]

        logger.info(
            "Pulling labelled passages from dataset: %s (limit: %s, statuses: %s)",
            dataset.name,
            limit or "none",
            ", ".join(str(s.value) for s in include_statuses),
        )

        passages = []
        for record in dataset.records(with_responses=True, limit=limit):
            text = record.fields.get("text", "")
            if not text:
                continue

            # create a LabelledPassage for each response in each record
            for response in record.responses:
                if response.status not in include_statuses:
                    continue

                user = self.client.users(id=UUID(response.user_id))
                labeller = user.username if user else str(response.user_id)

                spans = []
                if response.value:
                    for span_data in response.value:
                        span = Span(
                            text=text,
                            start_index=span_data["start"],
                            end_index=span_data["end"],
                            concept_id=span_data["label"],
                            labellers=[labeller],
                            timestamps=[record.updated_at or record.inserted_at],
                        )
                        spans.append(span)

                passages.append(
                    LabelledPassage(
                        text=text,
                        metadata=self._format_metadata(record.metadata),
                        spans=spans,
                    )
                )

        logger.info("Pulled %d labelled passages", len(passages))
        return passages

    def _format_metadata(self, metadata: dict) -> dict:
        """
        Format metadata for Argilla ingestion.

        Normalizes keys if needed (dot→hyphen, lowercase) and optionally
        surfaces whitelisted keys. Relies on Dataset.allow_extra_metadata=True
        to accept arbitrary keys.

        Args:
            metadata: Original metadata dict.

        Returns:
            Formatted metadata dict.
        """
        if not metadata:
            return {}

        formatted = {}
        for key, value in metadata.items():
            normalized_key = key.replace(".", "-").lower()
            formatted[normalized_key] = value

        return formatted
