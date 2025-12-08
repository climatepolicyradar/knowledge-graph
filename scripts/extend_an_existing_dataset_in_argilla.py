from typing import Annotated

import typer
from rich.console import Console

from knowledge_graph.config import processed_data_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.labelling import ArgillaSession
from knowledge_graph.wikibase import WikibaseSession

app = typer.Typer()
console = Console()


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ...,
            help="The Wikibase ID of the concept to add passages for",
            parser=WikibaseID,
        ),
    ],
    workspace_name: Annotated[
        str,
        typer.Option(
            ...,
            help="The name of the workspace containing the existing dataset",
        ),
    ],
):
    with console.status("Connecting to Argilla..."):
        argilla = ArgillaSession()
    console.log("✅ Connected to Argilla")

    sampled_passages_dir = processed_data_dir / "sampled_passages"
    sampled_passages_path = sampled_passages_dir / f"{wikibase_id}.jsonl"

    console.log(f"Loading sampled passages for {wikibase_id}")
    try:
        with open(sampled_passages_path, "r", encoding="utf-8") as f:
            labelled_passages = [
                LabelledPassage.model_validate_json(line) for line in f
            ]
        n_annotations = sum([len(entry.spans) for entry in labelled_passages])
        console.log(
            f"Loaded {len(labelled_passages)} labelled passages "
            f"with {n_annotations} individual annotations"
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"No sampled passages found for {wikibase_id}. Please run"
            f"  just sample {wikibase_id}"
        ) from e

    # Get concept metadata from wikibase
    wikibase = WikibaseSession()
    concept = wikibase.get_concept(wikibase_id)
    console.log(f"✅ Loaded metadata for {concept}")

    # Get existing dataset from Argilla
    with console.status(f"Looking for existing dataset for {concept}..."):
        dataset = argilla.get_dataset(wikibase_id, workspace=workspace_name)
    console.log(
        f"✅ Found existing dataset '{dataset.name}' with {len(list(dataset.records))} records"
    )

    # Deduplicate local dataset based on text in Argilla
    with console.status(
        "Deduplicating text in input labelled passages based on records in Argilla..."
    ):
        argilla_records = list(dataset.records)
        text_in_argilla: set[str] = set(
            [record.fields.get("text", "") for record in argilla_records]
        )

        lp_length_before = len(labelled_passages)
        labelled_passages = [
            lp for lp in labelled_passages if lp.text not in text_in_argilla
        ]

    console.print(
        f"{len(labelled_passages)}/{lp_length_before} input passages remaining after deduplication"
    )

    # Push labelled passages to the dataset
    with console.status(f"Adding {len(labelled_passages)} passages to dataset..."):
        argilla.add_labelled_passages(
            labelled_passages=labelled_passages,
            wikibase_id=wikibase_id,
            workspace=workspace_name,
        )

    console.log(
        f"✅ Successfully added {len(labelled_passages)} passages to dataset '{dataset.name}'"
    )
    console.log(f"Dataset now contains {len(list(dataset.records))} total records")


if __name__ == "__main__":
    app()
