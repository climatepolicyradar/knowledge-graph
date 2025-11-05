import asyncio
import gc
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
import wandb
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from knowledge_graph.classifier.embedding import EmbeddingClassifier
from knowledge_graph.concept import Concept
from knowledge_graph.config import WANDB_ENTITY
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelling import ArgillaSession
from scripts.evaluate import evaluate_classifier
from scripts.get_concept import get_concept_async

console = Console()
app = typer.Typer()

# Use low batch size to be able to test big models below
BATCH_SIZE = 4

# from MTEB (https://huggingface.co/spaces/mteb/leaderboard)
# with filters under 500m parameters; sentence-transformers compatible
# Each model has recommended query and document prefixes for optimal performance
EMBEDDING_MODEL_CONFIG = [
    {
        "model": "BAAI/bge-small-en-v1.5",  # 33M; 512D; 512T
        "query_prefix": "",
        "document_prefix": "",
    },
    {
        "model": "ibm-granite/granite-embedding-107m-multilingual",  # 107M; 384D; 512T
        "query_prefix": "",
        "document_prefix": "",
    },
    {
        "model": "intfloat/multilingual-e5-small",  # 118M; 384D; 512T
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
    {
        "model": "BAAI/bge-base-en-v1.5",  # 109M; 768D; 512T
        "query_prefix": "",
        "document_prefix": "",
    },
    # failing with einops installation issue
    # {
    #     "model": "nomic-ai/nomic-embed-text-v1.5",  # 137M; 768D; 8192T
    #     "query_prefix": "search_query: ",
    #     "document_prefix": "search_document: ",
    # },
    # NOTE: below are too large to run many times on a machine with 16gb ram
    # {
    #     "model": "ibm-granite/granite-embedding-278m-multilingual", # 287M; 768D; 512T
    #     "query_prefix": "",
    #     "document_prefix": "",
    # },
    # {
    #     "model": "google/embeddinggemma-300m",  # ~300M; 768D; 2048T
    #     "query_prefix": "task: classification | query:",
    #     "document_prefix": "",
    # },
    # {
    #     "model": "Alibaba-NLP/gte-multilingual-base",  # ~300M; 768D; 8192T
    #     "query_prefix": "",
    #     "document_prefix": "",
    # },
    # needs extra xformers dependency
    # {
    #     "model": "Snowflake/snowflake-arctic-embed-m-v2.0",  # 305M; 768D; 8192T
    #     "query_prefix": "query: ",
    #     "document_prefix": "",
    # },
]

THRESHOLDS = [0.5, 0.6, 0.65, 0.75]


@dataclass
class FieldPreset:
    """Configuration for which concept fields to include in the embedding."""

    name: str
    include_description: bool
    include_definition: bool
    include_alternative_labels: bool
    include_negative_labels: bool
    use_markdown_headers: bool = True

    def to_dict(self) -> dict:
        """Convert preset to dictionary for hashing and config logging."""
        return {
            "name": self.name,
            "include_description": self.include_description,
            "include_definition": self.include_definition,
            "include_alternative_labels": self.include_alternative_labels,
            "include_negative_labels": self.include_negative_labels,
            "use_markdown_headers": self.use_markdown_headers,
        }


FIELD_PRESETS = [
    FieldPreset(
        name="title_alternative",
        include_description=False,
        include_definition=False,
        include_alternative_labels=True,
        include_negative_labels=False,
    ),
    FieldPreset(
        name="title_alternative_negative",
        include_description=False,
        include_definition=False,
        include_alternative_labels=True,
        include_negative_labels=True,
    ),
    FieldPreset(
        name="title_alternative_negative",
        include_description=False,
        include_definition=False,
        include_alternative_labels=True,
        include_negative_labels=True,
    ),
    FieldPreset(
        name="title_description_definition",
        include_description=True,
        include_definition=True,
        include_alternative_labels=False,
        include_negative_labels=False,
    ),
    FieldPreset(
        name="title_alternative_negative_description_definition",
        include_description=True,
        include_definition=True,
        include_alternative_labels=True,
        include_negative_labels=True,
    ),
    # NOTE: these are left commented as they are evaluated in the W&B workspace, but
    # perform poorly
    # FieldPreset(
    #     name="title_only",
    #     include_description=False,
    #     include_definition=False,
    #     include_alternative_labels=False,
    #     include_negative_labels=False,
    # ),
    # FieldPreset(
    #     name="title_description",
    #     include_description=True,
    #     include_definition=False,
    #     include_alternative_labels=False,
    #     include_negative_labels=False,
    # ),
    # FieldPreset(
    #     name="title_description_definition_no_headers",
    #     include_description=True,
    #     include_definition=True,
    #     include_alternative_labels=False,
    #     include_negative_labels=False,
    #     use_markdown_headers=False,
    # ),
    # FieldPreset(
    #     name="title_alternative_no_headers",
    #     include_description=False,
    #     include_definition=False,
    #     include_alternative_labels=True,
    #     include_negative_labels=False,
    #     use_markdown_headers=False,
    # ),
]


def generate_config_hash(
    model_name: str,
    threshold: float,
    field_preset: FieldPreset,
    query_prefix: str,
    document_prefix: str,
) -> str:
    """
    Generate a unique hash for a configuration.

    This hash is used to check if a configuration has already been evaluated
    and to ensure reproducibility across runs.
    """
    config_str = (
        f"{model_name}|{threshold}|{field_preset.name}|"
        f"{field_preset.include_description}|{field_preset.include_definition}|"
        f"{field_preset.include_alternative_labels}|{field_preset.include_negative_labels}|"
        f"{field_preset.use_markdown_headers}|"
        f"{query_prefix}|{document_prefix}"
    )
    # TODO: it could be better to use `Identifier.generate()` here, to take advantage
    # of the best practices developed there. This would cause checking against the
    # current runs based on their hash to break, so it's only worth doing should we
    # adapt this for a new benchmark or delete all the runs in the current benchmark.
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def check_run_exists_in_wandb(
    config_hash: str,
    concept_id: str,
    project_name: str = "benchmark-embeddingclassifier",
) -> bool:
    """
    Check for a completed run with the same configuration in W&B.

    Returns True only if at least one run with the same config_hash exists
    and has completed successfully (state == "finished").
    """
    try:
        api = wandb.Api()
        runs = api.runs(
            f"{WANDB_ENTITY}/{project_name}",
            filters={
                "config.config_hash": config_hash,
                "config.concept_id": concept_id,
            },
        )

        successful_runs = [run for run in runs if run.state == "finished"]
        return len(successful_runs) > 0
    except Exception as e:
        console.log(f"Warning: Could not check W&B for existing runs: {e}")
        return False


def load_concept_ids_from_file(file_path: Path) -> list[WikibaseID]:
    """
    Load concept IDs from a text file.

    Each line should contain one Wikibase ID (e.g. Q123).
    Lines starting with # are treated as comments and ignored.
    Empty lines are also ignored.

    :param file_path: Path to the text file containing concept IDs
    :return: List of concept ID strings
    """
    if not file_path.exists():
        console.log(f"Concepts file not found at: {file_path}")
        return []

    concepts = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue
            try:
                concept_id = WikibaseID(line)
                concepts.append(concept_id)
            except ValidationError:
                console.log(
                    f"Warning: Line {line_num} doesn't look like a Wikibase ID: {line}"
                )
                continue

    console.log(f"Loaded {len(concepts)} concept IDs from {file_path}")
    return concepts


async def get_concepts_with_validation_data(
    concept_ids: Optional[list[WikibaseID]] = None,
) -> list[Concept]:
    """
    Fetch concepts from Wikibase and labelled passages from Argilla.

    :param concept_ids: Optional list of specific concept IDs to fetch. If None, fetches all.
    :return: List of (wikibase_id, concept) tuples.
    """
    # Determine which concept IDs to process
    if concept_ids:
        console.log(f"Fetching {len(concept_ids)} specified concepts...")
        wikibase_ids = concept_ids
    else:
        console.log("Discovering all datasets in Argilla...")
        argilla = ArgillaSession()
        datasets = argilla.get_all_datasets()
        wikibase_ids = [WikibaseID(dataset.name) for dataset in datasets]
        console.log(f"Found {len(wikibase_ids)} datasets in Argilla")

    concepts = []
    for wikibase_id in wikibase_ids:
        try:
            console.log(f"Fetching concept {wikibase_id}...")
            concept = await get_concept_async(
                wikibase_id=wikibase_id,
                include_recursive_has_subconcept=True,
                include_labels_from_subconcepts=True,
            )

            if not concept.labelled_passages:
                console.log(
                    f"Skipping {wikibase_id}: no labelled passages found in Argilla"
                )
                continue

            concepts.append(concept)
            console.log(
                f"Found {wikibase_id} ({concept.preferred_label}) "
                f"with {len(concept.labelled_passages)} labelled passages"
            )
        except Exception as e:
            console.log(f"Error loading concept {wikibase_id}: {e}")
            continue

    console.log(f"Total concepts with validation data: {len(concepts)}")
    return concepts


class ConceptWrapper:
    """
    Wrapper around a Concept that overrides to_markdown with preset configuration.

    This wrapper delegates all attribute access to the underlying concept,
    but returns a custom markdown representation based on the field preset.
    """

    def __init__(self, concept: Concept, preset: FieldPreset):
        self._concept = concept
        self._preset = preset
        # Pre-generate the markdown text with the preset configuration
        self._markdown = concept.to_markdown(
            wikibase=None,
            include_description=preset.include_description,
            include_definition=preset.include_definition,
            include_alternative_labels=preset.include_alternative_labels,
            include_negative_labels=preset.include_negative_labels,
            include_concept_neighbourhood=False,  # Never include for embeddings
            include_example_passages=False,  # Never include for embeddings
            use_markdown_headers=preset.use_markdown_headers,
        )

    def to_markdown(self, wikibase=None):
        """Return the pre-generated markdown text."""
        _ = wikibase  # Parameter kept for API compatibility
        return self._markdown

    def __getattr__(self, name):
        """Delegate all other attribute access to the underlying concept."""
        return getattr(self._concept, name)


def create_custom_concept_for_preset(
    concept: Concept, preset: FieldPreset
) -> ConceptWrapper:
    """
    Create a concept wrapper with custom to_markdown behavior for the preset.

    Returns a wrapper that behaves like the original concept but with a customized
    to_markdown method based on the field preset configuration.
    """
    return ConceptWrapper(concept, preset)


@app.command()
def main(
    force_rerun: Annotated[
        bool,
        typer.Option(
            "--force-rerun",
            help="Re-run configurations that already exist in W&B",
        ),
    ] = False,
    concepts: Annotated[
        Optional[list[str]],
        typer.Option(
            "--concepts",
            help="List of wikibase IDs to evaluate (overrides concepts.txt file)",
        ),
    ] = None,
):
    """
    Run embedding classifier benchmark across all concepts with validation data.

    This script performs a grid search over:
    - Embedding models (each with model-specific prefixes)
    - Similarity thresholds
    - Concept field configurations

    By default, the script loads concept IDs from scripts/benchmarks/concepts.txt.
    If the file is empty or doesn't exist, all concepts with validation data will be evaluated.
    Use --concepts to override and specify concepts on the command line.

    Results are logged to W&B project 'benchmark-embeddingclassifier' for easy
    comparison and aggregation across concepts.

    Usage:
        - Edit the EMBEDDING_MODEL_CONFIG, THRESHOLDS or FIELD_PRESETS at the top of this script
        - Run the script (uv run python scripts/benchmarks/embeddingclassifier.py [OPTIONS]).
            This will skip over configs that have already been logged to W&B – these can be
            forced to rerun by deleting the runs in W&B, or using the --force-rerun option.
    """
    console.log("Starting embedding classifier benchmark")

    # Determine which concepts to evaluate
    # Priority: --concepts > default concepts.txt > all concepts
    concept_ids = None
    if concepts:
        console.log(f"Using {len(concepts)} concepts from command-line arguments")
        concept_ids = [WikibaseID(c) for c in concepts]
    else:
        concepts_file = Path(__file__).parent / "concepts.txt"
        if concepts_file.exists():
            concept_ids = load_concept_ids_from_file(concepts_file)
            if not concept_ids:
                console.log(
                    "concepts.txt is empty, will evaluate all concepts with validation data"
                )
        else:
            console.log(
                "No concepts.txt found, will evaluate all concepts with validation data"
            )

    all_concepts = asyncio.run(get_concepts_with_validation_data(concept_ids))

    if not all_concepts:
        console.log("No concepts found with validation data")
        raise typer.Exit(1)

    # Calculate total number of configurations
    total_configs = (
        len(all_concepts)
        * len(EMBEDDING_MODEL_CONFIG)
        * len(THRESHOLDS)
        * len(FIELD_PRESETS)
    )
    console.log(
        f"Grid search configuration:\n"
        f"   - {len(all_concepts)} concepts\n"
        f"   - {len(EMBEDDING_MODEL_CONFIG)} embedding models\n"
        f"   - {len(THRESHOLDS)} thresholds\n"
        f"   - {len(FIELD_PRESETS)} field presets\n"
        f"   = {total_configs} total configurations"
    )

    skipped = 0
    evaluated = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating configurations...", total=total_configs)

        for model_config in EMBEDDING_MODEL_CONFIG:
            model_name = model_config["model"]
            query_prefix = model_config["query_prefix"]
            document_prefix = model_config["document_prefix"]

            console.log(f"\n{'=' * 60}")
            console.log(f"Loading model: {model_name}")
            console.log(f"{'=' * 60}\n")

            for concept in all_concepts:
                for threshold in THRESHOLDS:
                    for preset in FIELD_PRESETS:
                        config_hash = generate_config_hash(
                            model_name,
                            threshold,
                            preset,
                            query_prefix,
                            document_prefix,
                        )

                        progress.update(
                            task,
                            description=f"[{concept.wikibase_id}] {model_name.split('/')[-1]} | {preset.name} @ {threshold}",
                        )

                        # Check if config already exists
                        if not force_rerun and check_run_exists_in_wandb(
                            config_hash, concept_id=concept.id
                        ):
                            console.log(
                                f"Skipping {concept.wikibase_id} | {model_name} | "
                                f"{preset.name} | {threshold} (already evaluated)"
                            )
                            skipped += 1
                            progress.advance(task)
                            continue

                        try:
                            # Initialize W&B run
                            run = wandb.init(
                                entity=WANDB_ENTITY,
                                project="benchmark-embeddingclassifier",
                                job_type="benchmark_evaluation",
                                config={
                                    "concept_id": str(concept.id),
                                    "concept_wikibase_id": str(concept.wikibase_id),
                                    "concept_preferred_label": concept.preferred_label,
                                    "embedding_model_name": model_name,
                                    "threshold": threshold,
                                    "field_preset": preset.to_dict(),
                                    "field_preset_name": preset.name,
                                    "query_prefix": query_prefix,
                                    "document_prefix": document_prefix,
                                    "config_hash": config_hash,
                                },
                                tags=[
                                    str(concept.wikibase_id),
                                    model_name,
                                    preset.name,
                                ],
                            )

                            # Create concept with custom field configuration
                            concept_for_eval = create_custom_concept_for_preset(
                                concept, preset
                            )

                            # Initialize embedding classifier
                            classifier = EmbeddingClassifier(
                                concept=concept_for_eval,  # type: ignore[arg-type]
                                embedding_model_name=model_name,
                                threshold=threshold,
                                document_prefix=document_prefix,
                                query_prefix=query_prefix,
                                # Avoids MPS out of memory errors
                                device="cpu",
                            )

                            # Evaluate classifier
                            start_time = time.time()
                            _, _ = evaluate_classifier(
                                classifier=classifier,
                                labelled_passages=concept.labelled_passages,
                                wandb_run=run,
                                batch_size=BATCH_SIZE,
                            )
                            evaluation_time = time.time() - start_time

                            # Clean up classifier and force garbage collection
                            del classifier
                            gc.collect()
                            if (
                                hasattr(torch, "mps")
                                and hasattr(torch, "backends")
                                and hasattr(torch.backends, "mps")
                                and torch.backends.mps.is_built()
                                and torch.backends.mps.is_available()
                            ):
                                torch.mps.empty_cache()

                            # Log evaluation time
                            run.log({"evaluation_time_seconds": evaluation_time})
                            run.summary["evaluation_time_seconds"] = evaluation_time

                            console.log(
                                f"Evaluated {concept.wikibase_id} | {model_name} | "
                                f"{preset.name} | {threshold} ({evaluation_time:.1f}s)"
                            )

                            evaluated += 1

                            # Finish the run
                            wandb.finish()

                        except Exception as e:
                            console.log(
                                f"Error evaluating {concept.wikibase_id} | {model_name} | "
                                f"{preset.name} | {threshold}: {e}"
                            )
                            failed += 1
                            wandb.finish(exit_code=1)

                        progress.advance(task)

    # Print summary
    console.log("\n" + "=" * 60)
    console.log("Benchmark Summary:")
    console.log(f"   Evaluated: {evaluated}")
    console.log(f"   Skipped (already exists): {skipped}")
    console.log(f"   Failed: {failed}")
    console.log(f"   Total configurations: {total_configs}")
    console.log("=" * 60)

    if evaluated > 0:
        console.log(
            f"\nResults available at: https://wandb.ai/{WANDB_ENTITY}/benchmark-embeddingclassifier"
        )


if __name__ == "__main__":
    app()
