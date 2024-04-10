"""
Deletes all items, claims, and properties in a wikibase instance.

Note: This action is irreversible and should be used with caution. The script will
prompts the user for confirmation before performing the deletion.

Usage:
- Run `python scripts/nuke.py`
"""

import typer
from rich.console import Console
from rich.progress import Progress
from rich.prompt import Confirm

from src.wikibase import WikibaseSession

console = Console()

wikibase = WikibaseSession()


def main():
    if Confirm.ask(
        f"Are you sure that you want to delete all items and properties from [white]{wikibase.base_url}?[/]",
    ):
        if Confirm.ask(
            "[bold][red]REALLY sure?[/bold] This action is irreversible.[/]"
        ):
            console.print("[green]Okay, deleting all items and properties...[/]")
            all_items = wikibase.get_all_items()
            all_properties = wikibase.get_all_properties()

            with Progress() as progress:
                claim_removal_task = progress.add_task(
                    "[cyan]Deleting items...", total=len(all_items)
                )

                for item in all_items:
                    progress.update(
                        claim_removal_task,
                        advance=1,
                        description=f"Removing claims from {item['q_id']}",
                    )
                    for _, statements in wikibase.get_statements(item["q_id"]).items():
                        for statement in statements:
                            # https://www.wikidata.org/w/api.php?action=help&modules=wbremoveclaims
                            remove_claim_response = wikibase.session.post(
                                url=wikibase.api_url,
                                data={
                                    "action": "wbremoveclaims",
                                    "claim": statement["id"],
                                    "token": wikibase.csrf_token,
                                    "format": "json",
                                },
                            )
                            if remove_claim_response.status_code != 200:
                                console.print(
                                    f"[red]Failed to remove claim {statement['id']} from item {item['q_id']}[/]: {remove_claim_response.text}"
                                )

                item_delete_task = progress.add_task(
                    "[cyan]Deleting items...", total=len(all_items)
                )
                for item in all_items:
                    progress.update(
                        item_delete_task,
                        advance=1,
                        description=f"Deleting item {item['q_id']}",
                    )
                    # https://www.mediawiki.org/w/api.php?action=help&modules=delete
                    item_deletion_response = wikibase.session.post(
                        url=wikibase.api_url,
                        data={
                            "action": "delete",
                            "pageid": item["page_id"],
                            "token": wikibase.csrf_token,
                            "format": "json",
                            "reason": "Nuke script: delete all items",
                        },
                    )
                    if item_deletion_response.status_code != 200:
                        console.print(
                            f"[red]Failed to delete item {item['q_id']}[/]: {item_deletion_response.text}"
                        )

                property_deletion_task = progress.add_task(
                    "[cyan]Deleting properties...", total=len(all_properties)
                )
                for property in all_properties:
                    progress.update(
                        property_deletion_task,
                        advance=1,
                        description=f"Deleting property {property['p_id']}",
                    )
                    # https://www.mediawiki.org/w/api.php?action=help&modules=delete
                    property_deletion_response = wikibase.session.post(
                        url=wikibase.api_url,
                        data={
                            "action": "delete",
                            "pageid": property["page_id"],
                            "token": wikibase.csrf_token,
                            "format": "json",
                            "reason": "Nuke script: delete all properties",
                        },
                    )
                    if property_deletion_response.status_code != 200:
                        console.print(
                            f"[red]Failed to delete property {property['p_id']}[/]: {property_deletion_response.text}"
                        )
                progress.stop()

            # # now delete the properties


if __name__ == "__main__":
    typer.run(main)
