import json
from pathlib import Path

import typer
from rich.console import Console
from typing_extensions import Annotated

from src.wikibase import WikibaseSession

console = Console(highlight=False)
wikibase = WikibaseSession()


def main(
    output_path: Annotated[
        Path,
        typer.Option(
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=False,
            resolve_path=True,
            help="Path to the output file",
        ),
    ],
):
    """
    Queries the wikibase instance for groups of items which could plausibly be merged

    Outputs a json file with the following structure:
    [
        [Q123, Q456, Q789],
        [Q234, Q567],
        ...
    ]

    The output file can be used as input to the merge_concepts.py script
    """
    assert output_path.suffix == ".json", "Output file must be a json file"
    console.print("Generating merge candidates...", style="blue")
    merge_candidates = []

    with open(output_path, "w") as f:
        json.dump(merge_candidates, f, indent=4)

    console.print(f"Merge candidates saved to {output_path}", style="bold green")


if __name__ == "__main__":
    typer.run(main)
