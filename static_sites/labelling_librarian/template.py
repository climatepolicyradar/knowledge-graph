from functools import lru_cache
from pathlib import Path
from typing import Counter

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

from src.identifiers import WikibaseID
from src.wikibase import WikibaseSession
from static_sites.labelling_librarian.checks import (
    DatasetLevelIssue,
    LabellingIssue,
    PassageLevelIssue,
)

load_dotenv()

session = WikibaseSession()


# Get the directory where this file lives and load the templates
current_dir = Path(__file__).parent.resolve()
env = Environment(loader=FileSystemLoader(current_dir / "templates"))


@lru_cache(maxsize=1024)
def get_dataset_to_preferred_label_map(dataset_name: str) -> str:
    """Retrieves the name of the dataset to present in the UI"""
    concept = session.get_concept(WikibaseID(dataset_name))
    return concept.preferred_label


def create_index_page(issues: list[LabellingIssue]) -> str:
    """Create an HTML report of all issues found using the Jinja template"""
    passage_issues = dict(
        Counter(
            issue.dataset_name
            for issue in issues
            if isinstance(issue, PassageLevelIssue)
        )
    )

    dataset_issues = dict(
        Counter(
            issue.dataset_name
            for issue in issues
            if isinstance(issue, DatasetLevelIssue)
        )
    )

    dataset_info = {
        dataset_name: {
            "passage_issue_count": passage_issues.get(dataset_name, 0),
            "dataset_issue_count": dataset_issues.get(dataset_name, 0),
            "issue_types": list(
                set(
                    issue.type for issue in issues if issue.dataset_name == dataset_name
                )
            ),
            "preferred_label": get_dataset_to_preferred_label_map(dataset_name),
        }
        for dataset_name in {issues.dataset_name for issues in issues}
    }

    # Sorting, bringing those with the most issues to the top. Considering there are ~130 passages per dataset
    # discounting the passage-level issues as such.
    dataset_info = dict(
        sorted(
            dataset_info.items(),
            key=lambda item: item[1]["passage_issue_count"] / 130
            + item[1]["dataset_issue_count"],
            reverse=True,
        )
    )

    total_issue_counts = dict(Counter(issue.type for issue in issues))
    return env.get_template("index.html").render(
        dataset_info=dataset_info, total_issue_counts=total_issue_counts
    )


def create_dataset_page(dataset_name: str, issues: list[LabellingIssue]) -> str:
    """Create an HTML report of all issues found for a single dataset using the Jinja template"""
    assert all(issue.dataset_name == dataset_name for issue in issues)  # type: ignore

    dataset_issues: list[DatasetLevelIssue] = []
    passage_issues: list[PassageLevelIssue] = []
    for issue in issues:
        if isinstance(issue, DatasetLevelIssue):
            dataset_issues.append(issue)
        elif isinstance(issue, PassageLevelIssue):
            passage_issues.append(issue)

    passage_issue_counts = dict(Counter(issue.type for issue in passage_issues))

    return env.get_template("dataset.html").render(
        dataset_name=dataset_name,
        preferred_label=get_dataset_to_preferred_label_map(dataset_name),
        dataset_issues=dataset_issues,
        passage_issues=passage_issues,
        passage_issue_counts=passage_issue_counts,
    )
