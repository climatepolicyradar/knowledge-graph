import os
import tempfile
from contextlib import contextmanager
from functools import wraps
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import boto3
import wandb
from mypy_boto3_s3 import S3Client
from wandb.sdk.wandb_run import Run as WandbRun

from knowledge_graph.classifier import Classifier
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.version import Version

T = TypeVar("T")

logger = getLogger(__name__)


class ArtifactNotFoundError(Exception):
    """
    Raise this error when trying to fetch an artifact which does not exist.

    :param artifact_id: The artifact ID that does not exist
    :param artifact_type: The type of artifact that does not exist
    :param message: Optional custom error message. If provided, this will be used
        instead of the auto-generated message.
    """

    def __init__(
        self,
        artifact_id: str,
        artifact_type: str,
        *args: str,
        message: Optional[str] = None,
    ) -> None:
        super().__init__(*args)
        self.artifact_id = artifact_id
        self.artifact_type = artifact_type
        self.message = message

    def __str__(self) -> str:  # noqa: D105
        if self.message:
            return self.message
        return (
            f"{self.artifact_type.title()} artifact '{self.artifact_id}' not found. "
            f"Verify the artifact exists in W&B and check the artifact ID."
        )


def with_wandb_run(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that automatically manages W&B runs for session methods.

    This decorator ensures that the decorated methods have access to a W&B run,
    creating one automatically if needed. The job type is automatically inferred
    from the name of the decorated function.

    Usage:
        @with_wandb_run  # job_type will be "upload_classifier"
        def upload_classifier(self, classifier, ...):
            # This method will automatically have a run available
            pass
    """

    @wraps(func)
    def wrapper(self: "WeightsAndBiasesSession", *args, **kwargs) -> T:
        # Extract run from kwargs if present
        run = kwargs.pop("run", None)

        # Extract wikibase_id from args or kwargs
        wikibase_id = None
        if "wikibase_id" in kwargs:
            wikibase_id = kwargs["wikibase_id"]
        # Maybe the first positional argument is a wikibase_id?
        elif len(args) > 0 and isinstance(args[0], (str, WikibaseID)):
            wikibase_id = args[0]
        # Maybe the first positional argument is a classifier, which has a concept with a wikibase_id?
        elif (
            args
            and hasattr(args[0], "concept")
            and hasattr(args[0].concept, "wikibase_id")
        ):
            wikibase_id = args[0].concept.wikibase_id

        if not wikibase_id:
            raise ValueError(
                f"Couldn't determine a Wikibase ID for the {func.__name__} method. "
            )

        # Make sure that the found wikibase_id is actually a valid WikibaseID
        if isinstance(wikibase_id, str):
            wikibase_id = WikibaseID(wikibase_id)

        # Ensure a run exists, creating a temporary one if needed
        run_needs_to_be_finished = False
        if run is None:
            # Create a temporary run
            wandb.login(key=self.api._api_key)
            run = wandb.init(
                entity=self.entity,
                project=wikibase_id,
                job_type=func.__name__,
                config={},
            )
            run_needs_to_be_finished = True

        try:
            # Pass the found/created run to the decorated method
            kwargs["run"] = run
            return func(self, *args, **kwargs)
        finally:
            # If the run was created by the decorator, finish it
            if run_needs_to_be_finished:
                run.finish()

    return wrapper


class WeightsAndBiasesSession:
    """
    W&B session for managing classifiers, metadata and labelled passages.

    Classifier storage is backed by S3.

    Args:
        aws_env: AWS environment for S3 storage. If not provided, uses AWS_PROFILE
            environment variable.

    Raises:
        ValueError: If WANDB_API_KEY is not set or S3 client cannot be initialized.
    """

    def __init__(self, default_aws_env: Optional[AwsEnv] = None):
        """
        Initialize a WeightsAndBiasesSession.

        Args:
            default_aws_env: Default AWS environment for S3 storage operations.
                If not provided, uses AWS_PROFILE environment variable.
                Can be overridden per-operation for cross-environment tasks.

        Raises:
            ValueError: If WANDB_API_KEY is not set.
        """
        self.entity = os.getenv("WANDB_ENTITY", "climatepolicyradar")
        if not (api_key := os.getenv("WANDB_API_KEY")):
            raise ValueError(
                "Could not find a Weights & Biases API key. Please set the "
                "WANDB_API_KEY environment variable."
            )
        self.api: wandb.Api = wandb.Api(api_key=api_key)
        self.default_aws_env = default_aws_env or AwsEnv(
            os.getenv("AWS_PROFILE", "labs")
        )
        self._aws_sessions: dict[str, boto3.Session] = {}
        self._s3_clients: dict[str, S3Client] = {}

    def __repr__(self) -> str:
        """Return a string representation of the W&B session"""
        return (
            f"<WeightsAndBiasesSession: default_aws_env={self.default_aws_env.value}>"
        )

    def _get_s3_client(self, aws_env: Optional[AwsEnv] = None) -> S3Client:
        """Get S3 client for the specified AWS environment, with caching."""
        env = aws_env or self.default_aws_env
        env_name = env.value

        if env_name not in self._s3_clients:
            aws_region = os.getenv("AWS_REGION", "eu-west-1")
            session = boto3.Session(profile_name=env_name, region_name=aws_region)
            self._s3_clients[env_name] = session.client("s3")
            self._aws_sessions[env_name] = session

        return self._s3_clients[env_name]

    def _get_s3_bucket(self, aws_env: Optional[AwsEnv] = None) -> str:
        """Get S3 bucket name for the specified AWS environment."""
        env = aws_env or self.default_aws_env
        return f"cpr-{env.value}-models"

    def _project_artifact_id(
        self,
        wikibase_id: WikibaseID,
        classifier_id: ClassifierID,
        version: Optional[Version] = None,
    ) -> str:
        """Construct a W&B project artifact identifier (for training/storage)."""
        artifact_name = f"{self.entity}/{wikibase_id}/{classifier_id}"
        if version:
            artifact_name += f":{version}"
        return artifact_name

    def _registry_artifact_id(
        self,
        wikibase_id: WikibaseID,
        version: Optional[Version] = None,
    ) -> str:
        """Construct a W&B registry artifact identifier (for promotion/deployment)."""
        artifact_name = f"wandb-registry-model/{wikibase_id}"
        if version:
            artifact_name += f":{version}"
        return artifact_name

    @contextmanager
    def _get_artifact(
        self,
        artifact_id: str,
        artifact_type: str = "model",
        run: Optional[WandbRun] = None,
    ):
        """
        Context manager for working with W&B artifacts.

        Provides a unified way to get artifacts whether from a run or the API,
        with automatic saving.

        Args:
            artifact_id: The artifact ID to retrieve
            artifact_type: Type of artifact (default: "model")
            run: Optional W&B run for tracking (if None, uses API)

        Yields:
            W&B artifact object

        Raises:
            ArtifactNotFoundError: If artifact cannot be found
        """
        artifact = None
        try:
            if run is None:
                # If we're just doing read-only operations, use the W&B API
                artifact = self.api.artifact(artifact_id, type=artifact_type)
            else:
                # If we've got a run set up to track operations, use the run object to
                # get the artifact directly
                artifact = run.use_artifact(artifact_id, type=artifact_type)

            yield artifact

        except Exception as e:
            logger.error("Failed to get artifact %s: %s", artifact_id, e)
            raise ArtifactNotFoundError(
                artifact_id=artifact_id, artifact_type=artifact_type
            ) from e

    @with_wandb_run
    def upload_classifier(
        self,
        classifier: Classifier,
        wikibase_id: WikibaseID,
        version: Optional[Version] = None,
        classifiers_profiles: Optional[list[str]] = None,
        compute_environment: Optional[dict[str, Any]] = None,
        aws_env: Optional[AwsEnv] = None,
        run: Optional[WandbRun] = None,
    ) -> dict[str, Any]:
        """
        Upload a classifier to W&B artifacts with S3 reference.

        This method saves the classifier to S3 and creates a W&B artifact that
        references the S3 location. If no version is provided, a version is
        automatically determined and assigned.

        Args:
            classifier: The classifier to upload
            wikibase_id: Wikibase ID of the concept
            version: Version for the classifier (auto-determined if None)
            classifiers_profiles: Optional list of classifier profiles
            compute_environment: Optional compute environment requirements
            aws_env: AWS environment for S3 upload (uses default if None)
            run: Optional W&B run for tracking (if None, a run will be created automatically)

        Returns:
            Dictionary containing artifact information

        Raises:
            ValueError: If classifier concept is invalid
            ArtifactNotFoundError: If artifact operations fail
        """
        if not classifier.concept.wikibase_id:
            raise ValueError(
                "The supplied classifier must have a concept with wikibase_id"
            )

        # If no version is provided, determine the next version automatically
        if version is None:
            version = self.get_next_classifier_version(
                wikibase_id=wikibase_id, classifier_id=classifier.id
            )

        s3_key = f"{wikibase_id}/{classifier.id}/{version}/model.pickle"

        # Save classifier locally first
        with tempfile.NamedTemporaryFile(suffix=".pickle", delete=False) as temp_file:
            classifier.save(temp_file.name)
            temp_path = Path(temp_file.name)

        try:
            # Use specified AWS environment or fall back to default
            env = aws_env or self.default_aws_env
            s3_client = self._get_s3_client(env)
            s3_bucket = self._get_s3_bucket(env)

            # Upload the classifier to S3
            s3_client.upload_file(
                Filename=temp_path.name,
                Bucket=s3_bucket,
                Key=s3_key,
            )
            logger.info("Uploaded classifier to s3://%s/%s", s3_bucket, s3_key)

            metadata = {
                "aws_env": env.value,
                "classifier_name": classifier.name,
                "concept_id": classifier.concept.id,
                "concept_wikibase_revision": classifier.concept.wikibase_revision,
                "classifiers_profiles": classifiers_profiles,
                "compute_environment": compute_environment,
            }

            artifact = wandb.Artifact(
                name=self._project_artifact_id(wikibase_id, classifier.id, version),
                type="model",
                metadata=metadata,
            )

            # Add the S3 reference to the artifact
            s3_uri = f"s3://{s3_bucket}/{s3_key}"
            artifact.add_reference(uri=s3_uri, checksum=True)

            assert run is not None, (
                "The with_wandb_run decorator should ensure that a run is available, "
                "but none was found. Try supplying a run to upload_classifier "
                "explicitly, and raise an issue if the problem persists."
            )
            logged_artifact = run.log_artifact(artifact)
            logged_artifact = logged_artifact.wait()

            logger.info(f"Created W&B {artifact.type} artifact: {artifact.name}")

            return {
                "artifact_id": logged_artifact.id,
                "s3_uri": s3_uri,
                "metadata": metadata,
            }

        finally:
            # Make sure we clean up the temporary file, even if an error occurs
            if temp_path.exists():
                temp_path.unlink()

    def get_classifier(
        self,
        wikibase_id: WikibaseID,
        classifier_id: ClassifierID,
        version: Optional[Version | str] = None,
        model_to_cuda: bool = False,
        run: Optional[WandbRun] = None,
    ) -> Classifier:
        """
        Fetch a classifier from W&B artifacts and load it directly into memory.

        Args:
            wikibase_id: Wikibase ID of the concept
            classifier_id: Classifier ID to get
            version: Version of the classifier. If None or "latest", automatically determines the latest version
            model_to_cuda: Whether to move the model to CUDA
            run: Optional W&B run for lineage tracking (tracks artifact usage)

        Returns:
            Loaded classifier instance
        """
        if version is None or version == "latest":
            version = self.get_latest_classifier_version(
                wikibase_id=wikibase_id, classifier_id=classifier_id
            )
        elif isinstance(version, str):
            version = Version(version)

        artifact_id = self._project_artifact_id(wikibase_id, classifier_id, version)
        logger.info("Loading classifier into memory: %s", artifact_id)

        try:
            with self._get_artifact(artifact_id, run=run) as artifact:
                with tempfile.TemporaryDirectory() as temp_dir:
                    artifact_dir = artifact.download(root=temp_dir)
                    classifier = Classifier.load(
                        str(Path(artifact_dir) / "model.pickle"),
                        model_to_cuda=model_to_cuda,
                    )
                    logger.info(
                        "Successfully loaded %s from artifact %s",
                        str(classifier),
                        artifact_id,
                    )

                    return classifier

        except Exception as e:
            logger.error("Failed to load classifier %s: %s", artifact_id, e)
            raise ArtifactNotFoundError(
                artifact_id=artifact_id,
                artifact_type="model",
                message=(
                    f"Failed to load classifier artifact {artifact_id}. "
                    "Check that the artifact exists and is accessible in W&B."
                ),
            ) from e

    def update_classifier_metadata(
        self,
        wikibase_id: WikibaseID,
        classifier_id: ClassifierID,
        version: Version | str,
        metadata_updates: dict[str, Any],
        run: Optional[WandbRun] = None,
    ) -> None:
        """
        Update metadata for a classifier artifact.

        Args:
            classifier_id: Classifier ID to update
            version: Version string
            metadata_updates: Dictionary of metadata updates
            wikibase_id: Wikibase ID of the concept
            run: Optional W&B run for tracking (if None, uses API)
        """
        if isinstance(version, str):
            version = Version(version)

        artifact_id = self._project_artifact_id(wikibase_id, classifier_id, version)

        with self._get_artifact(artifact_id, run=run) as artifact:
            # Update metadata
            for key, value in metadata_updates.items():
                artifact.metadata[key] = value
            # Save the artifact after modifications
            artifact.save()

        logger.info("Updated metadata for %s", artifact_id)

    def get_classifier_metadata(
        self,
        wikibase_id: WikibaseID,
        classifier_id: ClassifierID,
        version: Version | str,
    ) -> dict[str, Any]:
        """
        Get metadata for a classifier artifact.

        Args:
            wikibase_id: Wikibase ID of the concept
            classifier_id: Classifier ID to get metadata for
            version: Version string

        Returns:
            Dictionary containing artifact metadata
        """
        if isinstance(version, str):
            version = Version(version)

        artifact_id = self._project_artifact_id(wikibase_id, classifier_id, version)

        with self._get_artifact(artifact_id) as artifact:
            return dict(artifact.metadata)

    @with_wandb_run
    def promote_classifier(
        self,
        wikibase_id: WikibaseID,
        classifier_id: ClassifierID,
        version: Version | str,
        aws_env: AwsEnv,
        classifiers_profiles: Optional[list[str]] = None,
        run: Optional[WandbRun] = None,
    ) -> None:
        """
        Promote a classifier to the registry for a specific environment.

        This method tags the classifier artifact with the environment and links it
        to the W&B model registry for easy discovery and deployment.

        Args:
            wikibase_id: Wikibase ID of the concept
            classifier_id: Classifier ID to promote
            version: Version string
            aws_env: AWS environment to promote the classifier TO
            classifiers_profiles: Optional list of classifier profiles
            run: Optional W&B run for tracking (automatically created if None)
        """
        if isinstance(version, str):
            version = Version(version)

        artifact_id = self._project_artifact_id(wikibase_id, classifier_id, version)

        with self._get_artifact(artifact_id, run=run) as artifact:
            # Add environment tag
            current_tags = set(artifact.tags or [])
            current_tags.add(aws_env.value)
            artifact.tags = list(current_tags)

            # Update classifiers profiles if provided
            if classifiers_profiles:
                artifact.metadata["classifiers_profiles"] = classifiers_profiles

            # Save the artifact after modifications
            artifact.save()

            # Link the artifact to the registry collection if we have a run
            if run is not None:
                registry_name = "model"
                target_path = f"wandb-registry-{registry_name}/{wikibase_id}"
                run.link_artifact(
                    artifact=artifact,
                    target_path=target_path,
                    aliases=None,
                )
                logger.info("Linked artifact to registry collection: %s", target_path)

        logger.info(
            "Promoted classifier %s for environment %s",
            artifact_id,
            aws_env.value,
        )

    def demote_classifier(
        self,
        wikibase_id: WikibaseID,
        version: Version | str,
        aws_env: AwsEnv,
        run: Optional[WandbRun] = None,
    ) -> None:
        """
        Demote a classifier by removing the specified environment tag.

        Works with registry artifacts, not project artifacts.

        Args:
            wikibase_id: Wikibase ID of the concept
            version: Version string
            aws_env: AWS environment to demote the classifier FROM
            run: Optional W&B run for tracking (if None, uses API)
        """
        if isinstance(version, str):
            version = Version(version)

        artifact_id = self._registry_artifact_id(wikibase_id, version)

        with self._get_artifact(artifact_id, run=run) as artifact:
            # Validate the environment tag exists
            if not artifact.tags or aws_env.value not in artifact.tags:
                raise ValueError(
                    f"Classifier {artifact_id} is not promoted in environment {aws_env.value}"
                )

            # Validate the model was trained for this environment
            if artifact.metadata.get("aws_env") != aws_env.value:
                raise ValueError(
                    f"Classifier {artifact_id} was not trained in environment {aws_env.value}"
                )

            # Remove environment tag
            current_tags = set(artifact.tags)
            current_tags.discard(aws_env.value)
            artifact.tags = list(current_tags)

            # Remove classifiers profiles
            artifact.metadata.pop("classifiers_profiles", None)

            # Save the artifact after modifications
            artifact.save()

        logger.info(
            "Demoted classifier %s from environment %s",
            artifact_id,
            aws_env.value,
        )

    def get_latest_classifier_version(
        self,
        wikibase_id: WikibaseID,
        classifier_id: ClassifierID,
        aws_env: Optional[AwsEnv] = None,
    ) -> Version:
        """
        Get the latest version of a classifier.

        Args:
            wikibase_id: Wikibase ID of the concept
            classifier_id: Classifier ID
            aws_env: Optional AWS environment to filter by. If provided, only returns
                    versions tagged with this environment.

        Returns:
            The latest Version of the classifier

        Raises:
            ArtifactNotFoundError: If no artifacts exist for the classifier
        """
        base_artifact_id = self._project_artifact_id(wikibase_id, classifier_id)
        artifact_type = "model"

        try:
            if aws_env is None:
                # Original behavior - get latest overall
                artifact = self.api.artifact(base_artifact_id, type=artifact_type)
                return Version(artifact.version)
            else:
                # Filter by AWS environment tag
                artifacts = self.api.artifacts(
                    type_name=artifact_type, name=base_artifact_id
                )
                env_tagged_artifacts = [
                    artifact
                    for artifact in artifacts
                    if artifact.tags and aws_env.value in artifact.tags
                ]

                if not env_tagged_artifacts:
                    raise ArtifactNotFoundError(
                        artifact_id=base_artifact_id,
                        artifact_type=artifact_type,
                        message=f"No artifacts found for classifier {classifier_id} in {wikibase_id} with environment tag {aws_env.value}",
                    )

                # Find the latest version among environment-tagged artifacts
                latest_artifact = max(
                    env_tagged_artifacts, key=lambda a: Version(a.version)
                )
                return Version(latest_artifact.version)

        except Exception as e:
            logger.error("Failed to get latest version for %s: %s", base_artifact_id, e)
            raise ArtifactNotFoundError(
                artifact_id=base_artifact_id,
                artifact_type=artifact_type,
                message=f"No artifacts found for classifier {classifier_id} in {wikibase_id}",
            ) from e

    def get_next_classifier_version(
        self,
        wikibase_id: WikibaseID,
        classifier_id: ClassifierID,
    ) -> Version:
        """
        Get the next version number for a classifier.

        This method gets the latest version and increments it. If no artifacts exist,
        it returns v0.

        Args:
            wikibase_id: Wikibase ID of the concept
            classifier_id: Classifier ID

        Returns:
            Next version (e.g., Version("v1"), Version("v2"))
        """
        try:
            latest_version = self.get_latest_classifier_version(
                wikibase_id=wikibase_id, classifier_id=classifier_id
            )
            next_version = latest_version.increment()
            logger.info(
                "Next version for %s/%s is %s",
                wikibase_id,
                classifier_id,
                next_version,
            )
        except ArtifactNotFoundError:
            # No artifacts exist yet, start with v0
            next_version = Version("v0")
            logger.info(
                "No existing artifacts found for %s/%s. Starting with %s",
                wikibase_id,
                classifier_id,
                next_version,
            )

        return next_version

    @with_wandb_run
    def upload_labelled_passages(
        self,
        labelled_passages: list[LabelledPassage],
        concept: Concept,
        classifier: Optional[Classifier] = None,
        run: Optional[WandbRun] = None,
    ) -> None:
        """
        Upload labelled passages as a W&B artifact.

        Args:
            labelled_passages: List of labelled passages to upload
            concept: Concept associated with the passages
            classifier: Optional classifier that generated the passages
            run: Optional W&B run for tracking (if None, creates temporary run)
        """
        if not concept.wikibase_id:
            raise ValueError(
                "Concept must have a wikibase_id to upload labelled passages"
            )

        artifact_name = (
            f"{classifier.id}-labelled-passages" if classifier else "labelled-passages"
        )
        artifact_metadata = {
            "concept_wikibase_revision": concept.wikibase_revision,
            "passage_count": len(labelled_passages),
        }

        if classifier:
            artifact_metadata["classifier_id"] = classifier.id

        labelled_passages_artifact = wandb.Artifact(
            name=artifact_name,
            type="labelled_passages",
            metadata=artifact_metadata,
        )

        with labelled_passages_artifact.new_file(
            "labelled_passages.jsonl", mode="w", encoding="utf-8"
        ) as f:
            data = "\n".join([entry.model_dump_json() for entry in labelled_passages])
            f.write(data)

        assert run is not None  # Decorator ensures run is not None
        run.log_artifact(
            labelled_passages_artifact, name=artifact_name, type="labelled_passages"
        )
        logger.info("Uploaded %d labelled passages", len(labelled_passages))

    def get_labelled_passages(
        self,
        run_identifier: str,
        artifact_type: str = "labelled_passages",
    ) -> list[LabelledPassage]:
        """
        Get labelled passages from a W&B run and load them into memory.

        Args:
            run_identifier: W&B run identifier
            artifact_type: Type of artifact to get

        Returns:
            List of labelled passages loaded into memory
        """
        run = self.api.run(run_identifier)

        # Find the labelled passages artifact
        artifacts = [
            artifact
            for artifact in run.logged_artifacts()
            if artifact.type == artifact_type
        ]

        if not artifacts:
            raise ArtifactNotFoundError(
                artifact_id=run_identifier,
                artifact_type=artifact_type,
                message=(
                    f"No {artifact_type} artifacts found in run {run_identifier}. "
                    "Check that the run contains the expected artifact type."
                ),
            )

        if len(artifacts) > 1:
            raise ArtifactNotFoundError(
                artifact_id=run_identifier,
                artifact_type=artifact_type,
                message=(
                    f"Multiple {artifact_type} artifacts found in run {run_identifier}. "
                    "Expected exactly one artifact of this type."
                ),
            )

        artifact = artifacts[0]

        # Use a temporary directory that will be cleaned up automatically
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = artifact.download(root=temp_dir)

            # Load labelled passages
            labelled_passages = []
            for json_file in Path(artifact_dir).glob("*.jsonl"):
                with open(json_file, "r", encoding="utf-8") as f:
                    file_labelled_passages = [
                        LabelledPassage.model_validate_json(line) for line in f
                    ]
                    labelled_passages.extend(file_labelled_passages)

        logger.info("Loaded %d labelled passages into memory", len(labelled_passages))
        return labelled_passages

    def add_classifiers_profiles(
        self,
        wikibase_id: WikibaseID,
        classifier_id: ClassifierID,
        version: Version | str,
        profiles_to_add: list[str],
        run: Optional[WandbRun] = None,
    ) -> None:
        """
        Add classifiers profiles to an existing classifier artifact.

        Args:
            wikibase_id: Wikibase ID of the concept
            classifier_id: Classifier ID
            version: Version string
            profiles_to_add: List of profiles to add
            run: Optional W&B run for tracking
        """
        if isinstance(version, str):
            version = Version(version)

        artifact_id = self._project_artifact_id(wikibase_id, classifier_id, version)

        with self._get_artifact(artifact_id, run=run) as artifact:
            current_profiles = set(artifact.metadata.get("classifiers_profiles", []))
            new_profiles = current_profiles | set(profiles_to_add)

            if len(new_profiles) > 1:
                raise ValueError(
                    f"Artifact can have maximum of one classifiers profile. "
                    f"Current: {current_profiles}, trying to add: {profiles_to_add}"
                )

            artifact.metadata["classifiers_profiles"] = list(new_profiles)
            artifact.save()

        logger.info("Added classifiers profiles %s to %s", profiles_to_add, artifact_id)

    def remove_classifiers_profiles(
        self,
        wikibase_id: WikibaseID,
        classifier_id: ClassifierID,
        version: Version | str,
        profiles_to_remove: list[str],
        run: Optional[WandbRun] = None,
    ) -> None:
        """
        Remove classifiers profiles from an existing classifier artifact.

        Args:
            wikibase_id: Wikibase ID of the concept
            classifier_id: Classifier ID
            version: Version string
            profiles_to_remove: List of profiles to remove
            run: Optional W&B run for tracking
        """
        if isinstance(version, str):
            version = Version(version)

        artifact_id = self._project_artifact_id(wikibase_id, classifier_id, version)

        with self._get_artifact(artifact_id, run=run) as artifact:
            current_profiles = set(artifact.metadata.get("classifiers_profiles", []))

            if new_profiles := current_profiles - set(profiles_to_remove):
                artifact.metadata["classifiers_profiles"] = list(new_profiles)
            else:
                artifact.metadata.pop("classifiers_profiles", None)

            artifact.save()

        logger.info(
            "Removed classifiers profiles %s from %s", profiles_to_remove, artifact_id
        )

    @contextmanager
    def run_context(self, wikibase_id: WikibaseID):
        """
        Context manager that yields an active W&B run.

        This allows classifiers and other components to use a session-managed
        W&B run instead of creating their own.

        Args:
            wikibase_id: Wikibase ID of the concept/project

        Yields:
            WandbRun: Active W&B run for the project

        Usage:
            with session.run_context(classifier.concept.wikibase_id) as run:
                classifier.fit(wandb_run=run)
        """
        wandb.login(key=self.api._api_key)
        run = wandb.init(entity=self.entity, project=wikibase_id, job_type="session")
        try:
            yield run
        finally:
            run.finish()
