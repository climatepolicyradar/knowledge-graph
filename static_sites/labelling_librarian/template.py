from pathlib import Path
from typing import Counter

from jinja2 import Environment, FileSystemLoader
from static_sites.labelling_librarian.checks import (
    PassageLevelIssue,
    DatasetLevelIssue,
    LabellingIssue,
)


# Get the directory where this file lives and load the templates
current_dir = Path(__file__).parent.resolve()
env = Environment(loader=FileSystemLoader(current_dir / "templates"))


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

    issue_counts = {
        dataset_name: {
            "passage": passage_issues.get(dataset_name, 0),
            "dataset": dataset_issues.get(dataset_name, 0),
        }
        for dataset_name in {issues.dataset_name for issues in issues}
    }

    issue_counts = dict(
        sorted(
            issue_counts.items(),
            key=lambda item: item[1]["passage"] / 130 + item[1]["dataset"],
            reverse=True,
        )
    )
    return env.get_template("index.html").render(issue_counts=issue_counts)


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

    return env.get_template("dataset.html").render(
        dataset_name=dataset_name,
        dataset_issues=dataset_issues,
        passage_issues=passage_issues,
    )
