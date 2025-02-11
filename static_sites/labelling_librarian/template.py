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
    dataset_issues = dict(Counter(issue.dataset_name for issue in issues))

    return env.get_template("index.html").render(dataset_issues=dataset_issues)


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
