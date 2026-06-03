import typer
from rich.console import Console

from knowledge_graph.config import processed_data_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.labelling import ArgillaSession
from knowledge_graph.utils import get_logger
from knowledge_graph.wikibase import WikibaseSession

app = typer.Typer()
console = Console()


def push_passages_to_argilla(
    labelled_passages: list[LabelledPassage],
    wikibase_id: WikibaseID,
    workspace_name: str = "knowledge-graph",
    limit: int | None = None,
    argilla_api_url: str | None = None,
    argilla_api_key: str | None = None,
    wikibase_username: str | None = None,
    wikibase_password: str | None = None,
    wikibase_url: str | None = None,
) -> None:
    logger = get_logger()

    if limit is not None:
        logger.info(f"Limiting number of labelled passages to {limit}")
        labelled_passages = labelled_passages[:limit]

    argilla = ArgillaSession(api_url=argilla_api_url, api_key=argilla_api_key)
    logger.info("✅ Connected to Argilla")

    wikibase = WikibaseSession(
        username=wikibase_username,
        password=wikibase_password,
        url=wikibase_url,
    )
    concept = wikibase.get_concept(wikibase_id)
    logger.info(f"✅ Loaded metadata for {concept}")

    dataset = argilla.create_dataset(concept, workspace=workspace_name)
    logger.info(f'✅ Created dataset "{dataset.name}" for {concept}')

    argilla.add_labelled_passages(
        labelled_passages=labelled_passages,
        wikibase_id=wikibase_id,
        workspace=workspace_name,
    )
    logger.info(f"✅ Pushed {len(labelled_passages)} passages to dataset")


@app.command()
def main(
    wikibase_id: WikibaseID = typer.Option(
        ...,
        help="The Wikibase ID of the concept to create a dataset for",
        parser=WikibaseID,
    ),
    workspace_name: str = typer.Option(
        "knowledge-graph",
        help="The name of the existing workspace in Argilla",
    ),
    limit: int | None = typer.Option(
        130, help="Limit the number of passages loaded to Argilla."
    ),
):
    """
    Push a new dataset for a concept to Argilla.

    Requires that the `sample` script has been run first.
    """
    sampled_passages_dir = processed_data_dir / "sampled_passages"
    sampled_passages_path = sampled_passages_dir / f"{wikibase_id}.jsonl"

    if not sampled_passages_path.exists():
        raise FileNotFoundError(
            f"Sampled passages not found for {wikibase_id}. Run the sample script (scripts/sample.py) first."
        )

    console.log(f"Loading sampled passages for {wikibase_id}")
    with open(sampled_passages_path, "r", encoding="utf-8") as f:
        labelled_passages = [LabelledPassage.model_validate_json(line) for line in f]
    n_annotations = sum([len(entry.spans) for entry in labelled_passages])
    console.log(
        f"Loaded {len(labelled_passages)} labelled passages "
        f"with {n_annotations} individual annotations"
    )

    push_passages_to_argilla(
        labelled_passages=labelled_passages,
        wikibase_id=wikibase_id,
        workspace_name=workspace_name,
        limit=limit,
    )


if __name__ == "__main__":
    app()
