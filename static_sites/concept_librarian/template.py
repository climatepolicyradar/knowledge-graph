from collections import Counter, defaultdict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.concept import Concept, WikibaseID
from static_sites.concept_librarian.checks import (
    ConceptIssue,
    ConceptStoreIssue,
    MultiConceptIssue,
    RelationshipIssue,
)

# Get the directory where this file lives and load the templates
current_dir = Path(__file__).parent.resolve()
env = Environment(loader=FileSystemLoader(current_dir / "templates"))

# Add type tests to Jinja environment
env.tests["concept_issue"] = lambda x: isinstance(x, ConceptIssue)
env.tests["relationship_issue"] = lambda x: isinstance(x, RelationshipIssue)
env.tests["multi_concept_issue"] = lambda x: isinstance(x, MultiConceptIssue)


def get_issues_for_concept(
    issues: list[ConceptStoreIssue], concept_id: WikibaseID
) -> list[ConceptStoreIssue]:
    """Get all issues that affect a given concept"""
    concept_issues = []
    for issue in issues:
        if isinstance(issue, ConceptIssue):
            if issue.concept.wikibase_id == concept_id:
                concept_issues.append(issue)
        elif isinstance(issue, RelationshipIssue):
            if (
                issue.from_concept.wikibase_id == concept_id
                or issue.to_concept.wikibase_id == concept_id
            ):
                concept_issues.append(issue)
        elif isinstance(issue, MultiConceptIssue):
            if any(c.wikibase_id == concept_id for c in issue.concepts):
                concept_issues.append(issue)
    return concept_issues


def create_index_page(issues: list[ConceptStoreIssue]) -> str:
    """Create an HTML report of all issues found using the Jinja template"""

    # Group issues by type
    issues_by_type = defaultdict(list)
    for issue in issues:
        issues_by_type[issue.issue_type].append(issue)

    # Count totals
    total_issues = len(issues)
    type_counts = Counter(issue.issue_type for issue in issues)

    return env.get_template("index.html").render(
        total_issues=total_issues,
        type_counts=type_counts,
        issues_by_type=issues_by_type,
    )


def create_concept_page(concept: Concept, all_issues: list[ConceptStoreIssue]) -> str:
    """Create an HTML page for a specific concept's issues"""
    concept_issues = get_issues_for_concept(all_issues, concept.wikibase_id)
    return env.get_template("concept.html").render(
        issues=concept_issues,
        concept=concept,
    )
