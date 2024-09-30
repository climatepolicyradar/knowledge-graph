import json
from pathlib import Path

import typer
from rich.progress import Progress
from typing_extensions import Annotated

from src.wikibase import WikibaseSession

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

    The first item in each group will act as the target item, with all other items in the group being merged into it.
    """
    assert input_path.suffix == ".json", "Input file must be a json file"
    with open(input_path, "r", encoding="utf-8") as f:
        merge_candidates = json.load(f)

    # make sure the data is a list of lists of strings
    assert isinstance(merge_candidates, list), "Data must be a list"
    for group in merge_candidates:
        assert isinstance(group, list), "Each group of merge candidates must be a list"
        for item in group:
            assert isinstance(item, str), "Each item in a group must be a string"

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Merging concepts...", total=len(merge_candidates)
        )
        for group in merge_candidates:
            progress.advance(task)
            # get the claims for each item in the group
            for wikibase_id in group:
                response = wikibase.session.get(
                    url=wikibase.api_url,
                    params={
                        "action": "wbgetclaims",
                        "format": "json",
                        "entity": wikibase_id,
                    },
                ).json()
                statements = response.get("claims", {})
                flat_statements = [
                    statement
                    for statements_of_same_type in statements.values()
                    for statement in statements_of_same_type
                ]
                for statement in flat_statements:
                    item_id = (
                        statement.get("mainsnak", {})
                        .get("datavalue", {})
                        .get("value", {})
                        .get("id", None)
                    )
                    # if the statement refers to another item in the group, remove it
                    if item_id is not None and item_id in group:
                        # https://www.wikidata.org/w/api.php?action=help&modules=wbremoveclaims
                        remove_claim_response = wikibase.session.post(
                            url=wikibase.api_url,
                            data={
                                "action": "wbremoveclaims",
                                "claim": statement["id"],
                                "token": wikibase.csrf_token,
                                "bot": True,
                                "summary": "Removing claims between duplicate items",
                                "format": "json",
                            },
                        )
                        if remove_claim_response.status_code != 200:
                            progress.print(
                                f"Failed to remove claim {statement['id']} from item {wikibase_id}: {remove_claim_response.text}",
                                style="red",
                            )

                # then merge the items
                # https://www.wikidata.org/w/api.php?action=help&modules=wbmergeitems
                target_item, ids_to_merge = group[0], group[1:]
                for wikibase_id in ids_to_merge:
                    merge_response = wikibase.session.post(
                        url=wikibase.api_url,
                        data={
                            "action": "wbmergeitems",
                            "fromid": wikibase_id,
                            "toid": target_item,
                            "token": wikibase.csrf_token,
                            "bot": True,
                            "summary": "Merging duplicate items",
                            "format": "json",
                        },
                    )
                    if merge_response.status_code != 200:
                        progress.print(
                            f"Failed to merge item {wikibase_id} into item {target_item}: {merge_response.text}",
                            style="red",
                        )

        progress.update(task, completed=len(merge_candidates))
        progress.print("✅ Merging complete", style="green")


if __name__ == "__main__":
    typer.run(main)
