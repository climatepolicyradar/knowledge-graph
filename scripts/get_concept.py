"""
CLI wrapper for the get-concept operation.

The reusable logic (`get_concept_async`, `parse_wikibase_config`) lives in
`knowledge_graph.operations.get_concept` and is imported directly by
`knowledge_graph.classifier.autollm`. This module only adds the Typer command used by
`just get-concept`.
"""

import asyncio
from typing import Annotated, Optional

import typer
from rich.console import Console

from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.operations.get_concept import (
    get_concept_async,
    parse_wikibase_config,
)
from knowledge_graph.wikibase import WikibaseConfig

console = Console()
app = typer.Typer()


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ..., help="The Wikibase ID of the concept to fetch", parser=WikibaseID
        ),
    ],
    include_recursive_has_subconcept: bool = True,
    include_labels_from_subconcepts=True,
    wikibase_config: Annotated[
        Optional[WikibaseConfig],
        typer.Option(
            ...,
            parser=parse_wikibase_config,
            help=(
                "Optional override of env variables for wikibase (username, "
                "password, url)"
            ),
        ),
    ] = None,
):
    concept = asyncio.run(
        get_concept_async(
            wikibase_id=wikibase_id,
            include_recursive_has_subconcept=include_recursive_has_subconcept,
            include_labels_from_subconcepts=include_labels_from_subconcepts,
            wikibase_config=wikibase_config,
        )
    )

    return concept


if __name__ == "__main__":
    app()
