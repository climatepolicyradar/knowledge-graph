import shutil
from collections import defaultdict
from pathlib import Path
from typing import Set

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.progress import track

from src.concept import Concept, WikibaseID
from src.wikibase import WikibaseSession
from static_sites.concept_librarian.checks import (
    ConceptIssue,
    ConceptStoreIssue,
    EmptyConcept,
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
console = Console()

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


@app.command()
def main():
    wikibase = WikibaseSession()
    concepts: list[Concept] = []
    concept_ids = wikibase.get_concept_ids()
    for wikibase_id in track(
        concept_ids,
        description="Fetching all concepts from wikibase",
        transient=True,
    ):
        try:
            concept = wikibase.get_concept(wikibase_id)
            concepts.append(concept)
        except ValidationError:
            concepts.append(EmptyConcept(wikibase_id=wikibase_id))
            console.log(f"Failed to fetch concept {wikibase_id}")

    console.log(f"Fetched {len(concepts)} concepts")

    # Get recursive subconcepts for each concept
    wikibase_id_to_subconcepts = defaultdict(set)
    # Create lookup dict for faster access
    wikibase_id_to_concept = {c.wikibase_id: c for c in concepts}

    for concept in track(
        concepts,
        description="Getting recursive subconcepts",
        transient=True,
    ):
        result = set()
        wikibase_ids_to_process = set(concept.has_subconcept)
        processed_wikibase_ids = set()

        while wikibase_ids_to_process:
            current_wikibase_id = wikibase_ids_to_process.pop()
            if current_wikibase_id in processed_wikibase_ids:
                continue

            processed_wikibase_ids.add(current_wikibase_id)
            if current_wikibase_id in wikibase_id_to_concept:
                current_concept = wikibase_id_to_concept[current_wikibase_id]
                result.add(current_concept)
                # Add this concept's subconcepts to be processed
                wikibase_ids_to_process.update(current_concept.has_subconcept)

        wikibase_id_to_subconcepts[concept.wikibase_id] = result
    console.log(
        f"Built a list of subconcepts for {len(wikibase_id_to_subconcepts)} concepts"
    )

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
    ]:
        issues.extend(check(concepts))
        console.log(f'Ran "{check.__name__}"')

    problematic_concepts = get_affected_concept_ids(issues)
    console.log(f"Found {len(issues)} issues in {len(problematic_concepts)} concepts")

    # Delete and recreate the output directory
    output_dir = current_dir / "dist"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Generate and save the index page
    html_content = create_index_page(issues)
    output_path = output_dir / "index.html"
    output_path.write_text(html_content)
    console.log("Generated index page")

    # Generate and save individual concept pages
    for concept in track(
        concepts, description="Generating concept pages", transient=True
    ):
        subconcepts = wikibase_id_to_subconcepts[concept.wikibase_id]
        html_content = create_concept_page(
            concept=concept, subconcepts=subconcepts, all_issues=issues
        )
        output_path = output_dir / f"{concept.wikibase_id}.html"
        output_path.write_text(html_content)
    console.log(f"Generated {len(concepts)} concept pages")


if __name__ == "__main__":
    typer.run(main)
