from collections import Counter, defaultdict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from static_sites.concept_librarian.checks import ConceptStoreIssue

# Get the directory where this file lives and load the templates
current_dir = Path(__file__).parent.resolve()
env = Environment(loader=FileSystemLoader(current_dir / "templates"))


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


def create_concept_page(issues: list[ConceptStoreIssue]) -> str:
    # make sure that all of the issues are for the same concept
    assert len(
        set(issue.fix_concept.wikibase_id for issue in issues if issue.fix_concept)
    ) in [0, 1]

    # Get the concept from the first issue that has a fix_concept, or None if no issues
    # have a fix_concept
    concept = next((issue.fix_concept for issue in issues if issue.fix_concept), None)

    return env.get_template("concept.html").render(issues=issues, concept=concept)
