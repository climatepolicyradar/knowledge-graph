import json
from pathlib import Path

import typer
from rich.console import Console
from typing_extensions import Annotated

from src.wikibase import WikibaseSession

console = Console(highlight=False)
wikibase = WikibaseSession()


def main(
    input_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            resolve_path=True,
            help="Path to the input file",
        ),
    ],
):
    """
    Takes a file containing a list of concept ID groups, and merges each one into a single concept in the wikibase instance.

    You can generate the list of concept ID groups by running the generate_merge_candidates.py script.

    If generating the list of concept ID groups manually, they should be saved in a json file obeying the following structure:
    [
        [Q123, Q456, Q789],
        [Q234, Q567],
        ...
    ]

    """
    assert input_path.suffix == ".json", "Input file must be a json file"
    with open(input_path, "r") as f:
        merge_candidates = json.load(f)

    # make sure the data is a list of lists of strings
    assert isinstance(merge_candidates, list), "Data must be a list"
    for group in merge_candidates:
        assert isinstance(group, list), "Each group of merge candidates must be a list"
        for item in group:
            assert isinstance(item, str), "Each item in a group must be a string"

    console.print("Merging concepts...", style="blue")
    for group in merge_candidates:
        # remove any claims which link the items in the group to one another

        # then merge the items
        # https://www.wikidata.org/w/api.php?action=help&modules=wbmergeitems
        pass


if __name__ == "__main__":
    typer.run(main)
