import logging
import os
from datetime import datetime
from typing import Annotated, Literal

import wandb
from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field
from rich.logging import RichHandler
from starlette.requests import Request
from starlette.responses import JSONResponse

from knowledge_graph.config import WANDB_ENTITY


# Define response models for automatic schema generation
class Classifier(BaseModel):
    """Classifier model metadata and evaluation metrics."""

    id: str = Field(
        description=(
            "Classifier ID. Classifiers with the same behaviour will have "
            "identical IDs."
        )
    )
    concept_id: str = Field(
        description=(
            "Concept ID. A fingerprint of the contents of a concept - i.e. if a "
            "concept for a Wikibase ID changes, its ID will change."
        )
    )
    f1_score: float | None = Field(description="Passage-level F1 score")
    precision: float | None = Field(description="Passage-level precision")
    recall: float | None = Field(description="Passage-level recall")
    classifier_type: str | None = Field(
        description="Type of classifier (e.g., 'KeywordClassifier', 'LLMClassifier')"
    )
    model_name: str | None = Field(
        description="Name of the underlying model e.g. GPT-5, climateBERT."
    )
    validation_set_size: int | None = Field(
        description="Number of passages in the validation set"
    )
    version: str = Field(description="Artifact version string")
    created_at: datetime = Field(description="Artifact creation timestamp")


class ClassifiersResult(BaseModel):
    """All classifiers trained on a concept."""

    wikibase_id: str = Field(description="The Wikibase ID that was queried")
    classifiers: list[Classifier] = Field(description="The matching classifiers")
    total_found: int = Field(description="Number of classifiers found")


class ValidationPredictions(BaseModel):
    """Validation set predictions for a classifier."""

    classifier_id: str = Field(description="Classifier ID")
    columns: list[str] = Field(description="Column names")
    data: list[list] = Field(description="Row data")
    total_rows: int = Field(description="Total number of rows")


# Common error handler to reduce code duplication
async def handle_wandb_error(
    e: Exception, operation: str, ctx: Context | None = None
) -> str:
    """Centralized error handling for Weights & Biases operations"""
    error_msg = f"{operation}: {str(e)}"
    if ctx:
        await ctx.error(error_msg)
    else:
        print(error_msg)
    return error_msg


def _get_api():
    """Create a Weights & Biases API client authenticated from the environment."""
    return wandb.Api(api_key=os.environ["WANDB_API_KEY"])


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)

# Initialize FastMCP app
mcp = FastMCP("Climate Policy Radar Weights & Biases")


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint for load balancer monitoring."""
    _ = request  # Unused but required by the function signature
    return JSONResponse(
        {"status": "healthy", "service": "Climate Policy Radar Weights & Biases MCP"}
    )


@mcp.tool
async def get_classifiers(
    wikibase_id: Annotated[
        str,
        Field(description="Wikibase concept ID (e.g., 'Q69')", pattern=r"^Q\d+$"),
    ],
    sort_by: Annotated[
        Literal["latest", "precision", "recall", "f1_score"],
        Field(description="Sort classifiers by creation date or by a metric"),
    ] = "latest",
    ctx: Context | None = None,
) -> ClassifiersResult:
    """
    Get all classifiers that have been trained on a concept from Weights & Biases.

    Each concept has its own Weights & Biases project (named by the concept's
    Wikibase ID). This tool retrieves every model artifact logged across the
    project's runs, and returns each classifier's metadata together with its
    passage-level evaluation metrics (F1, precision, recall).

    Use sort_by to order the results by the latest creation date (default) or by
    a specific metric.
    """
    if ctx:
        await ctx.info(f"Retrieving classifiers for {wikibase_id} (sort_by: {sort_by})")

    project_path = f"{WANDB_ENTITY}/{wikibase_id}"

    try:
        api = _get_api()
        runs = api.runs(project_path)

        classifiers: list[Classifier] = []
        seen_artifact_ids: set[str] = set()

        for run in runs:
            try:
                for artifact in run.logged_artifacts():
                    if artifact.type != "model":
                        continue

                    # Skip if we've already processed this artifact
                    artifact_id = f"{artifact.name}:{artifact.version}"
                    if artifact_id in seen_artifact_ids:
                        continue
                    seen_artifact_ids.add(artifact_id)

                    metadata = artifact.metadata or {}

                    # Extract metrics from run summary
                    summary = run.summary
                    f1_score = summary.get("passage_level_f1")
                    precision = summary.get("passage_level_precision")
                    recall = summary.get("passage_level_recall")

                    # Get validation set size and model name from run config
                    run_config = run.config
                    validation_set_size = run_config.get(
                        "n_gold_standard_labelled_passages"
                    )

                    classifiers.append(
                        Classifier(
                            id=artifact.name,
                            concept_id=metadata.get("concept_id", wikibase_id),
                            f1_score=f1_score,
                            precision=precision,
                            recall=recall,
                            classifier_type=metadata.get("classifier_name"),
                            model_name=run_config.get("classifier_kwargs", {}).get(
                                "model_name", None
                            ),
                            validation_set_size=validation_set_size,
                            version=artifact.version,
                            created_at=artifact.created_at,
                        )
                    )
            except Exception as e:
                if ctx:
                    await ctx.warning(f"Failed to extract info from run {run.id}: {e}")
                continue

        # Sort classifiers based on the sort_by parameter
        if sort_by == "latest":
            classifiers.sort(key=lambda c: c.created_at, reverse=True)
        elif sort_by == "precision":
            classifiers.sort(key=lambda c: (c.precision is None, -(c.precision or 0)))
        elif sort_by == "recall":
            classifiers.sort(key=lambda c: (c.recall is None, -(c.recall or 0)))
        elif sort_by == "f1_score":
            classifiers.sort(key=lambda c: (c.f1_score is None, -(c.f1_score or 0)))

        return ClassifiersResult(
            wikibase_id=wikibase_id,
            classifiers=classifiers,
            total_found=len(classifiers),
        )
    except Exception as e:
        await handle_wandb_error(
            e, f"Failed to get classifiers for concept {wikibase_id}", ctx
        )
        return ClassifiersResult(wikibase_id=wikibase_id, classifiers=[], total_found=0)


@mcp.tool
async def get_classifier_validation_predictions(
    wikibase_id: Annotated[
        str,
        Field(description="Wikibase concept ID (e.g., 'Q69')", pattern=r"^Q\d+$"),
    ],
    classifier_id: Annotated[
        str,
        Field(description="The classifier ID (model artifact name)", min_length=1),
    ],
    ctx: Context | None = None,
) -> ValidationPredictions | None:
    """
    Get validation set predictions for a specific classifier.

    Finds the run that logged the given classifier (model artifact) within the
    concept's Weights & Biases project, then returns that run's logged
    validation-set-predictions table (columns and row data). Returns null if no
    predictions table is found for the classifier.

    Use get_classifiers first to discover the available classifier IDs for a
    concept.
    """
    if ctx:
        await ctx.info(
            f"Retrieving validation predictions for {classifier_id} ({wikibase_id})"
        )

    project_path = f"{WANDB_ENTITY}/{wikibase_id}"

    try:
        api = _get_api()
        runs = api.runs(project_path)

        for run in runs:
            try:
                has_classifier = any(
                    artifact.type == "model" and artifact.name == classifier_id
                    for artifact in run.logged_artifacts()
                )
                if not has_classifier:
                    continue

                for run_artifact in run.logged_artifacts():
                    if "validation_set_predictions" in run_artifact.name:
                        table = run_artifact.get("validation_set_predictions")
                        if table is not None:
                            return ValidationPredictions(
                                classifier_id=classifier_id,
                                columns=table.columns,
                                data=table.data,
                                total_rows=len(table.data),
                            )
            except Exception as e:
                if ctx:
                    await ctx.warning(
                        f"Failed to fetch predictions from run {run.id}: {e}"
                    )
                continue

        return None
    except Exception as e:
        await handle_wandb_error(
            e,
            f"Failed to get validation predictions for classifier {classifier_id}",
            ctx,
        )
        return None
