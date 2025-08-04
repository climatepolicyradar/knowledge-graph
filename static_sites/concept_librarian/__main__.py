import asyncio
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Set

import typer
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler

from src.concept import Concept, WikibaseID
from src.wikibase import WikibaseSession
from static_sites.concept_librarian.checks import (
    ConceptIssue,
    ConceptStoreIssue,
    MultiConceptIssue,
    RelationshipIssue,
    check_alternative_labels_for_pipes,
    check_description_and_definition_length,
    check_for_duplicate_preferred_labels,
    check_for_unconnected_concepts,
    ensure_positive_and_negative_labels_dont_overlap,
    validate_alternative_label_uniqueness,
    validate_circular_hierarchical_relationships,
    validate_concept_label_casing,
    validate_hierarchical_relationship_symmetry,
    validate_related_relationship_symmetry,
)
from static_sites.concept_librarian.template import (
    create_concept_page,
    create_index_page,
)

app = typer.Typer()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, highlighter=NullHighlighter())],
)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Get the directory where this file lives
current_dir = Path(__file__).parent.resolve()


def get_affected_concept_ids(issues: list[ConceptStoreIssue]) -> Set[WikibaseID]:
    """Get all concept IDs that are affected by any issues"""
    concept_ids = set()
    for issue in issues:
        if isinstance(issue, ConceptIssue):
            concept_ids.add(issue.concept.wikibase_id)
        elif isinstance(issue, RelationshipIssue):
            concept_ids.add(issue.from_concept.wikibase_id)
            concept_ids.add(issue.to_concept.wikibase_id)
        elif isinstance(issue, MultiConceptIssue):
            concept_ids.update(c.wikibase_id for c in issue.concepts)
    return concept_ids


def get_all_subconcepts(
    concept_id: WikibaseID,
    visited: set[WikibaseID],
    wikibase_id_to_concept: dict[WikibaseID, Concept],
) -> set[WikibaseID]:
    """
    Recursively retrieve all subconcept IDs for a given concept

    :param WikibaseID concept_id: The ID of the concept to get the subconcepts of
    :param set visited: A set of concept IDs that have already been visited (the
        function is recursive, so we use this to avoid infinite loops)
    :param dict[WikibaseID, Concept] wikibase_id_to_concept: A dictionary of concept
        IDs to concepts
    :return set[WikibaseID]: A set of all subconcept IDs, including nested subconcepts
    """
    if concept_id in visited:
        return set()

    visited.add(concept_id)
    result = set()

    if concept_id not in wikibase_id_to_concept:
        return result

    current_concept = wikibase_id_to_concept[concept_id]

    for subconcept_id in current_concept.has_subconcept:
        if subconcept_id in wikibase_id_to_concept:
            subconcept = wikibase_id_to_concept[subconcept_id]
            result.add(subconcept.wikibase_id)
            nested_subconcepts = get_all_subconcepts(
                subconcept_id, visited, wikibase_id_to_concept
            )
            result.update(nested_subconcepts)

    return result


@app.command()
def main():
    """CLI entry point to generate the Concept Librarian static site."""
    asyncio.run(async_main())


async def async_main():
    """Fetch concepts, run validation checks, and generate static HTML pages."""
    concepts: list[Concept] = []

    logging.info("Fetching all of our concepts from wikibase (async)...")
    async with WikibaseSession() as wikibase:
        concepts = await wikibase.get_concepts_async()
    logging.info("✅ Fetched %d concepts from wikibase", len(concepts))

    wikibase_id_to_recursive_subconcept_ids: dict[WikibaseID, set[WikibaseID]] = (
        defaultdict(set)
    )
    wikibase_id_to_concept = {c.wikibase_id: c for c in concepts}

    for concept in concepts:
        wikibase_id_to_recursive_subconcept_ids[concept.wikibase_id] = (
            get_all_subconcepts(
                concept.wikibase_id,
                visited=set(),
                wikibase_id_to_concept=wikibase_id_to_concept,
            )
        )

    wikibase_id_to_recursive_subconcept_ids = {
        k: list(v) for k, v in wikibase_id_to_recursive_subconcept_ids.items()
    }
    logging.info("✅ Mapped the list of subconcepts for all concepts")

    issues: list[ConceptStoreIssue] = []
    for check in [
        validate_related_relationship_symmetry,
        validate_hierarchical_relationship_symmetry,
        validate_alternative_label_uniqueness,
        ensure_positive_and_negative_labels_dont_overlap,
        check_description_and_definition_length,
        check_for_duplicate_preferred_labels,
        check_alternative_labels_for_pipes,
        validate_circular_hierarchical_relationships,
        check_for_unconnected_concepts,
        validate_concept_label_casing,
        # validate_concept_depth_and_descendant_balance,
    ]:
        issues.extend(check(concepts))
        logging.info('🔬 Ran "%s"', check.__name__)

    problematic_concepts = get_affected_concept_ids(issues)
    logging.info(
        "❗ Found %d issues in %d problematic concepts",
        len(issues),
        len(problematic_concepts),
    )

    # Delete and recreate the output directory
    output_dir = current_dir / "dist"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Generate and save the index page
    html_content = create_index_page(issues)
    output_path = output_dir / "index.html"
    output_path.write_text(html_content)
    logging.info("✅ Generated index page")

    # Generate and save individual concept pages
    for concept in concepts:
        subconcept_ids = wikibase_id_to_recursive_subconcept_ids[concept.wikibase_id]
        subconcepts = [
            wikibase_id_to_concept[wikibase_id]
            for wikibase_id in subconcept_ids
            if wikibase_id in wikibase_id_to_concept
        ]

        html_content = create_concept_page(
            concept=concept, subconcepts=subconcepts, all_issues=issues
        )
        output_path = output_dir / f"{concept.wikibase_id}.html"
        output_path.write_text(html_content)
    logging.info("✅ Generated %d concept pages", len(concepts))


if __name__ == "__main__":
    typer.run(main)
