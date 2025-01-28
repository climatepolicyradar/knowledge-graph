import shutil

import typer
from rich.console import Console

from concept_librarian.checks import (
    ConceptStoreIssue,
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
from concept_librarian.template import create_concept_page, create_index_page
from scripts.config import data_dir, root_dir
from src.concept import Concept
from src.wikibase import WikibaseSession

app = typer.Typer()
console = Console()


@app.command()
def main():
    wikibase = WikibaseSession()
    console.log("Fetching all concepts from wikibase")
    concepts: list[Concept] = wikibase.get_concepts()
    console.log(f"Fetched {len(concepts)} concepts")

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
        console.log(f'Running "{check.__name__}"')
        issues.extend(check(concepts))

    problematic_concepts = set(
        [issue.fix_concept.wikibase_id for issue in issues if issue.fix_concept]
    )
    console.log(f"Found {len(issues)} issues in {len(problematic_concepts)} concepts")

    # Delete and recreate the output directory
    output_dir = data_dir / "build" / "concept_librarian"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Generate and save the index page
    html_content = create_index_page(issues)
    output_path = output_dir / "index.html"
    output_path.write_text(html_content)
    console.log(f"HTML report generated: '{output_path.relative_to(root_dir)}'")

    # Generate and save individual concept pages
    for concept in concepts:
        issues_for_concept = [
            issue
            for issue in issues
            if issue.fix_concept
            and issue.fix_concept.wikibase_id == concept.wikibase_id
        ]

        html_content = create_concept_page(issues_for_concept)
        output_path = output_dir / f"{concept.wikibase_id}.html"
        output_path.write_text(html_content)
        console.log(f"HTML report generated: '{output_path.relative_to(root_dir)}'")


if __name__ == "__main__":
    typer.run(main)
