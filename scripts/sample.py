"""
CLI wrapper for the sample operation.

The reusable, Prefect-free logic (`run_sampling`) lives in
`knowledge_graph.operations.sample` and is imported directly by the sampling flow. This
module only adds the Typer command used by `just sample`, which loads the dataset from
local disk and resolves the optional `--from-yaml-config` before calling `run_sampling`.
"""

from pathlib import Path
from typing import Annotated

import click
import pandas as pd
import typer

from knowledge_graph.config import processed_data_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.operations.sample import CORPUS_TYPES, run_sampling
from knowledge_graph.operations.train import resolve_config_inputs
from knowledge_graph.utils import get_logger, parse_kwargs_from_strings

app = typer.Typer()


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID | None,
        typer.Option(
            help="The Wikibase ID of the concept to sample passages for. Required unless --from-yaml-config is given.",
            parser=WikibaseID,
        ),
    ] = None,
    sample_size: int = typer.Option(130, help="The number of passages to sample"),
    min_negative_proportion: float = typer.Option(
        0.1, help="The minimum proportion of negative samples to take"
    ),
    dataset_name: str = typer.Option(
        "balanced",
        help="Dataset to use",
        click_type=click.Choice(["balanced", "combined"]),
    ),
    corpus_types_include: Annotated[
        list[str] | None,
        typer.Option(
            help="Corpus types to include. Can be specified multiple times. If not set, all types are included.",
            click_type=click.Choice(CORPUS_TYPES),
        ),
    ] = None,
    corpus_types_exclude: Annotated[
        list[str] | None,
        typer.Option(
            help="Corpus types to exclude. Can be specified multiple times.",
            click_type=click.Choice(CORPUS_TYPES),
        ),
    ] = None,
    max_size_to_sample_from: int = typer.Option(
        500_000,
        help="Maximum number of passages to load from the dataset before sampling",
    ),
    max_negative_proportion: float | None = typer.Option(
        None,
        help="Maximum proportion of the sample that can be negative. If not set, fills remaining sample_size after positives.",
    ),
    track_and_upload: bool = typer.Option(
        True,
        help="Whether to track the run and upload the labelled passages to W&B",
    ),
    concept_override: Annotated[
        list[str] | None,
        typer.Option(
            help="Concept property overrides in key=value format. Can be specified multiple times.",
        ),
    ] = None,
    from_yaml_config: Annotated[
        Path | None,
        typer.Option(
            help="Whether to use custom-classifier YAML config.",
        ),
    ] = None,
):
    logger = get_logger()

    wikibase_id, cfg = resolve_config_inputs(wikibase_id, from_yaml_config)
    if cfg is not None:
        concept_overrides = cfg.concept_overrides.as_overrides()
        dataset_name = cfg.sampling.dataset_name
        sample_size = cfg.sampling.sample_size
        min_negative_proportion = cfg.sampling.min_negative_proportion
        max_negative_proportion = cfg.sampling.max_negative_proportion
        corpus_types_include = cfg.sampling.corpus_types_include
        corpus_types_exclude = cfg.sampling.corpus_types_exclude
        max_size_to_sample_from = cfg.sampling.max_size_to_sample_from
    else:
        concept_overrides = parse_kwargs_from_strings(concept_override)

    if dataset_name == "balanced":
        dataset_filename = "sampled_dataset.feather"
    elif dataset_name == "combined":
        dataset_filename = "combined_dataset.feather"
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    dataset_path = processed_data_dir / dataset_filename

    try:
        dataset = pd.read_feather(dataset_path)
        logger.info(f"Loaded {len(dataset)} passages from {dataset_path}")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{dataset_path} not found. If you haven't already, you should run:\n"
            f"  just build-dataset"
        ) from e

    run_sampling(
        wikibase_id=wikibase_id,
        dataset=dataset,
        dataset_name=dataset_name,
        sample_size=sample_size,
        min_negative_proportion=min_negative_proportion,
        corpus_types_include=corpus_types_include,
        corpus_types_exclude=corpus_types_exclude,
        max_size_to_sample_from=max_size_to_sample_from,
        max_negative_proportion=max_negative_proportion,
        track_and_upload=track_and_upload,
        concept_overrides=concept_overrides,
    )


if __name__ == "__main__":
    app()
