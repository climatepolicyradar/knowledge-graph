import os
import uuid
from functools import lru_cache
from logging import getLogger
from typing import Optional, Sequence, Union
from uuid import UUID

from argilla import (
    Argilla,
    Dataset,
    ResponseStatus,
    Settings,
    SpanQuestion,
    TaskDistribution,
    TextField,
    User,
    Workspace,
)
from argilla._models import Role
from dotenv import find_dotenv, load_dotenv

from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import (
    LabelledPassage,
    consolidate_passages_by_text,
)
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
            raise ValueError(f"Workspace '{workspace_name}' not found")
        logger.info("Successfully retrieved workspace: %s", workspace_object.name)
        return workspace_object

    def create_workspace(self, name: str) -> Workspace:
        """Create a new workspace in Argilla, or return the existing workspace"""
        logger.info("Creating workspace: %s", name)
        try:
            workspace = Workspace(name=name)
            created_workspace: Workspace = workspace.create()  # type: ignore[assignment]
            logger.info("Successfully created workspace: %s", created_workspace.name)
            return created_workspace
        except ValueError as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "unique constraint" in error_msg:
                logger.warning(
                    "Workspace '%s' already exists, returning existing workspace", name
                )
                return self.get_workspace(name)
            else:
                raise ValueError(f"Failed to create workspace '{name}'") from e

    def get_dataset(
        self,
        wikibase_id: WikibaseID | str,
        workspace: Optional[str] = None,
    ) -> Dataset:
        """Get a dataset by its Wikibase ID (ie its name) in the given workspace"""
        logger.info("Fetching dataset '%s'", wikibase_id)
        workspace_object = self.get_workspace(workspace)
        dataset = self.client.datasets(
            name=str(wikibase_id), workspace=workspace_object
        )
        if not dataset:
            raise ValueError(
                f"Dataset '{wikibase_id}' not found in workspace '{workspace_object.name}'"
            )
        logger.debug("Successfully retrieved dataset: %s", wikibase_id)
        return dataset

    def get_all_datasets(self, workspace: Optional[str] = None) -> list[Dataset]:
        """Get all datasets in a workspace"""
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
        Create a new dataset for a concept in the given workspace

        If the dataset already exists, it will be returned without being re-created.
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

    @lru_cache(maxsize=64)
    def _get_user_by_id(self, user_id: Union[UUID, str]) -> User | None:
        """Get user object by ID"""
        return self.client.users(id=user_id)

    @lru_cache(maxsize=64)
    def _get_user_by_username(self, username: str) -> User | None:
        """Get user object by username"""
        return self.client.users(username=username)

    def get_user(
        self, username: Optional[str] = None, user_id: Union[UUID, str, None] = None
    ) -> User:
        """Get user object by username or ID"""
        if not (username or user_id):
            raise ValueError("One of 'username' or 'user_id' must be provided")
        if username and user_id:
            raise ValueError("Only one of 'username' or 'user_id' must be provided")

        if user_id is not None:
            user = self._get_user_by_id(user_id)
        else:
            assert username is not None
            user = self._get_user_by_username(username)
        if not user:
            raise ValueError(f"User '{username}' not found in Argilla")
        return user

    def create_user(
        self,
        username: str,
        password: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        role: Role | str = Role.annotator,
    ) -> User:
        """
        Create a new user in Argilla.

        Args:
            username: Username for the new user. Must be unique in Argilla.
            password: Password for the new user. If not provided, a random one will be generated.
            first_name: First name of the new user. Defaults to username if not provided.
            last_name: Last name of the new user.
            role: Role of the new user. Can be a Role enum or string. Options:
                - Role.annotator or "annotator" - Can annotate records and submit responses (default)
                - Role.admin or "admin" - Full administrative access
                - Role.owner or "owner" - Owner role

        If the user already exists, it will be returned without being re-created.
        """
        logger.info("Creating user: %s (role: %s)", username, role)

        try:
            user = User(
                username=username,
                password=password,
                first_name=first_name or username,
                last_name=last_name,
                role=Role(role),
            )
            created_user = user.create()
            logger.info("Successfully created user: %s", created_user.username)
            return created_user
        except ValueError as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "unique constraint" in error_msg:
                logger.warning(
                    "User '%s' already exists, retrieving existing user", username
                )
                return self.get_user(username=username)
            raise ValueError(f"Failed to create user '{username}'") from e

    def add_user_to_workspace(self, username: str, workspace: Optional[str] = None):
        """Add an existing user to a workspace"""
        workspace_name = workspace or self.default_workspace
        logger.info("Adding user '%s' to workspace '%s'", username, workspace_name)

        workspace_object = self.get_workspace(workspace_name)
        user_object = self.get_user(username)

        try:
            user_object.add_to_workspace(workspace_object)
            logger.info("Added user '%s' to workspace '%s'", username, workspace_name)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"Failed to add user '{username}' to workspace '{workspace_name}'"
            ) from e

    def remove_user_from_workspace(
        self, username: str, workspace: Optional[str] = None
    ):
        """Remove a user from a workspace"""
        workspace_name = workspace or self.default_workspace
        logger.info("Removing user '%s' from workspace '%s'", username, workspace_name)

        workspace_object = self.get_workspace(workspace_name)
        user_object = self.get_user(username)

        try:
            user_object.remove_from_workspace(workspace_object)
            logger.info(
                "Removed user '%s' from workspace '%s'", username, workspace_name
            )
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"Failed to remove user '{username}' from workspace '{workspace_name}'"
            ) from e

    def add_labelled_passages(
        self,
        labelled_passages: list[LabelledPassage],
        wikibase_id: WikibaseID | str,
        workspace: Optional[str] = None,
    ) -> Dataset:
        """
        Add labelled passages to an existing dataset in Argilla.

        Args:
            labelled_passages: List of LabelledPassage objects to add. Note that only
                the text and the metadata of each passage will be uploaded - spans
                attached to the labelled passages will be ignored.
            wikibase_id: Wikibase ID of the dataset to add passages to.
            workspace: Name of the workspace to add passages to. If not provided, the
                session's default_workspace will be used.

        Returns:
            The updated dataset.
        """
        dataset = self.get_dataset(wikibase_id, workspace)
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

    def get_labelled_passages(
        self,
        wikibase_id: WikibaseID | str,
        workspace: Optional[str] = None,
        include_statuses: Optional[Sequence[ResponseStatus]] = None,
        limit: Optional[int] = None,
        merge_responses: bool = True,
    ) -> list[LabelledPassage]:
        """
        Pull labelled passages from a dataset.

        Creates one LabelledPassage per response. For records with multiple responses
        (e.g., from different labellers), each response will produce a separate passage.
        This allows downstream code to explicitly decide how to merge or aggregate
        across labellers.

        Args:
            wikibase_id: Wikibase ID of the dataset to pull passages from.
            workspace: Name of the workspace to pull passages from. If not provided, the
                session's default_workspace will be used.
            include_statuses: Response statuses to include. Defaults to [ResponseStatus.submitted].
            limit: Max number of records to pull (note: may return more passages if
                records have multiple responses).
            merge_responses: Whether to merge responses from multiple labellers into
                single passages with combined spans. Defaults to True. If False, each
                response will be returned as a separate passage, which may be useful
                for tracking individual labeller contributions.

        Returns:
            List of LabelledPassage objects, one per response (or merged by text if merge_responses=True).
        """
        dataset = self.get_dataset(wikibase_id, workspace)

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

                user = self.get_user(user_id=response.user_id)
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
        if merge_responses:
            merged_passages = consolidate_passages_by_text(passages)
            logger.info(
                "Merged %d responses into %d LabelledPassages",
                len(passages),
                len(merged_passages),
            )
            return merged_passages
        else:
            return passages

    def _format_metadata(self, metadata: dict) -> dict:
        """Format metadata keys for Argilla by lowercasing and replacing dots with hyphens"""
        if not metadata:
            return {}

        formatted = {}
        for key, value in metadata.items():
            normalized_key = key.replace(".", "-").lower()
            formatted[normalized_key] = value

        return formatted
