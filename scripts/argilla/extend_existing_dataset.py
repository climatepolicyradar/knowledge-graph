import typer

from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.operations.extend_dataset import extend_dataset_locally

app = typer.Typer()


@app.command()
def main(
    wikibase_id: WikibaseID = typer.Option(
        ...,
        help="The Wikibase ID of the concept to add passages for",
        parser=WikibaseID,
    ),
    workspace_name: str = typer.Option(
        "knowledge-graph",
        help="The name of the workspace containing the existing dataset",
    ),
    limit: int | None = typer.Option(
        130, help="Limit the number of passages loaded to Argilla."
    ),
    require_full_limit: bool = typer.Option(
        False,
        help="Fail, rather than warn and add fewer, when fewer than --limit of the "
        "sampled passages are new.",
    ),
):
    """
    Extend an existing dataset for a concept to Argilla.

    Requires that the `sample` script has been run first, and that the dataset already
    exists. Deduplicates the input passages based on an exact text match against what's
    already in Argilla.
    """
    extend_dataset_locally(
        wikibase_id=wikibase_id,
        workspace=workspace_name,
        limit=limit,
        require_full_limit=require_full_limit,
    )


if __name__ == "__main__":
    app()
