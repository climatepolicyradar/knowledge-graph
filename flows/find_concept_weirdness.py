import time
from collections import Counter, defaultdict
from pathlib import Path
from string import punctuation

from prefect import flow, task
from pydantic import BaseModel

from src.concept import Concept
from src.wikibase import WikibaseSession


class ConceptStoreIssue(BaseModel):
    """Issue raised by concept store checks"""

    issue_type: str
    message: str
    metadata: dict


wikibase = WikibaseSession()


@flow(log_prints=True)
def validate_concept_store() -> list[ConceptStoreIssue]:
    print("Fetching all concepts from wikibase")
    concepts: list[Concept] = wikibase.get_concepts()
    print(f"Found {len(concepts)} concepts")

    issues = []
    issues.extend(validate_related_relationship_symmetry(concepts))
    issues.extend(validate_hierarchical_relationship_symmetry(concepts))
    issues.extend(validate_alternative_label_uniqueness(concepts))
    issues.extend(ensure_positive_and_negative_labels_dont_overlap(concepts))
    issues.extend(check_description_and_definition_length(concepts))
    issues.extend(check_for_duplicate_preferred_labels(concepts))

    librarian_output_dir = Path("../data/librarian")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    html_content = create_html_report(issues)
    (librarian_output_dir / f"librarian_report_{timestr}.html").write_text(html_content)
    print("HTML report generated: concept_store_issues.html")

    return issues


@task(log_prints=True)
def validate_related_relationship_symmetry(
    concepts: list[Concept],
) -> list[ConceptStoreIssue]:
    """Make sure related concepts relationships are symmetrical"""
    issues = []
    related_relationships = [
        (concept.wikibase_id, related_id)
        for concept in concepts
        for related_id in concept.related_concepts
    ]
    print(f"Found {len(related_relationships)} related concepts relationships")
    for concept_id, related_id in related_relationships:
        if (related_id, concept_id) not in related_relationships:
            issues.append(
                ConceptStoreIssue(
                    issue_type="asymmetric_related_relationship",
                    message=f"{concept_id} is related to {related_id}, but {related_id} is not related to {concept_id}",
                    metadata={"concept_id": concept_id, "related_id": related_id},
                )
            )
    return issues


@task(log_prints=True)
def validate_hierarchical_relationship_symmetry(
    concepts: list[Concept],
) -> list[ConceptStoreIssue]:
    """Make sure hierarchical subconcept relationships are symmetrical"""
    issues = []
    has_subconcept_relationships = [
        (concept.wikibase_id, subconcept_id)
        for concept in concepts
        for subconcept_id in concept.has_subconcept
    ]
    subconcept_of_relationships = [
        (concept.wikibase_id, parent_concept_id)
        for concept in concepts
        for parent_concept_id in concept.subconcept_of
    ]
    print(f"Found {len(has_subconcept_relationships)} subconcept relationships")
    print(f"Found {len(subconcept_of_relationships)} subconcept_of relationships")
    for concept_id, subconcept_id in has_subconcept_relationships:
        if (subconcept_id, concept_id) not in subconcept_of_relationships:
            issues.append(
                ConceptStoreIssue(
                    issue_type="asymmetric_subconcept_relationship",
                    message=f"{concept_id} has subconcept {subconcept_id}, but {subconcept_id} does not have parent concept {concept_id}",
                    metadata={"concept_id": concept_id, "subconcept_id": subconcept_id},
                )
            )
    for concept_id, parent_concept_id in subconcept_of_relationships:
        if (parent_concept_id, concept_id) not in has_subconcept_relationships:
            issues.append(
                ConceptStoreIssue(
                    issue_type="asymmetric_subconcept_relationship",
                    message=f"{concept_id} is subconcept of {parent_concept_id}, but {parent_concept_id} does not have subconcept {concept_id}",
                    metadata={
                        "concept_id": concept_id,
                        "parent_concept_id": parent_concept_id,
                    },
                )
            )
    return issues


@task(log_prints=True)
def validate_alternative_label_uniqueness(
    concepts: list[Concept],
) -> list[ConceptStoreIssue]:
    """Make sure alternative labels are unique"""
    issues = []
    for concept in concepts:
        duplicate_labels = [
            label
            for label in concept.alternative_labels
            if concept.alternative_labels.count(label) > 1
        ]
        if duplicate_labels:
            issues.append(
                ConceptStoreIssue(
                    issue_type="duplicate_alternative_labels",
                    message=f"{concept.wikibase_id} has duplicate alternative labels: {duplicate_labels}",
                    metadata={
                        "concept_id": concept.wikibase_id,
                        "duplicate_labels": duplicate_labels,
                    },
                )
            )
    return issues


@task(log_prints=True)
def ensure_positive_and_negative_labels_dont_overlap(
    concepts: list[Concept],
) -> list[ConceptStoreIssue]:
    """Make sure negative labels don't appear in positive labels"""
    issues = []
    for concept in concepts:
        overlapping_labels = set(concept.negative_labels) & set(concept.all_labels)
        if overlapping_labels:
            issues.append(
                ConceptStoreIssue(
                    issue_type="overlapping_labels",
                    message=f"{concept.wikibase_id} has negative labels which appear in its positive labels: {overlapping_labels}",
                    metadata={
                        "concept_id": concept.wikibase_id,
                        "overlapping_labels": list(overlapping_labels),
                    },
                )
            )
    return issues


@task(log_prints=True)
def check_description_and_definition_length(
    concepts: list[Concept],
) -> list[ConceptStoreIssue]:
    """Make sure descriptions and definitions are long enough"""
    issues = []
    minimum_length = 20
    for concept in concepts:
        if concept.description and len(concept.description) < minimum_length:
            issues.append(
                ConceptStoreIssue(
                    issue_type="short_description",
                    message=f"{concept.wikibase_id} has a short description",
                    metadata={
                        "concept_id": concept.wikibase_id,
                        "description": concept.description,
                    },
                )
            )
        if concept.definition and len(concept.definition) < minimum_length:
            issues.append(
                ConceptStoreIssue(
                    issue_type="short_definition",
                    message=f"{concept.wikibase_id} has a short definition",
                    metadata={
                        "concept_id": concept.wikibase_id,
                        "definition": concept.definition,
                    },
                )
            )
    return issues


@task(log_prints=True)
def check_for_duplicate_preferred_labels(
    concepts: list[Concept],
) -> list[ConceptStoreIssue]:
    """Make sure there are no duplicate concepts"""
    issues = []

    def clean(text: str) -> str:
        cleaned = text.lower().strip()
        cleaned = cleaned.translate(str.maketrans("", "", punctuation))
        return cleaned

    duplicate_dict = defaultdict(list)
    for concept in concepts:
        label = clean(concept.preferred_label)
        duplicate_dict[label].append(concept.wikibase_id)

    for label, ids in duplicate_dict.items():
        if len(ids) > 1:
            issues.append(
                ConceptStoreIssue(
                    issue_type="duplicate_preferred_labels",
                    message=f"Duplicate preferred labels: {label} -> {ids}",
                    metadata={"label": label, "concept_ids": ids},
                )
            )
    return issues


@task(log_prints=True)
def create_html_report(issues: list[ConceptStoreIssue]) -> str:
    """Create an HTML report of all issues found with tabs and shuffle functionality"""

    issues_by_type = defaultdict(list)
    for issue in issues:
        issues_by_type[issue.issue_type].append(issue)

    # Count totals
    total_issues = len(issues)
    type_counts = Counter(issue.issue_type for issue in issues)

    # Create HTML
    html = [
        "<html>",
        "<head>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 2em; }",
        ".tab { display: none; }",
        ".tab.active { display: block; }",
        ".tab-button { padding: 10px 20px; margin-right: 5px; cursor: pointer; }",
        ".tab-button.active { background: #007bff; color: white; }",
        ".issue { margin: 1em 0; padding: 0.5em; background: #f5f5f5; }",
        ".metadata { font-family: monospace; margin-left: 1em; }",
        ".shuffle-button { margin: 1em 0; padding: 10px 20px; background: #28a745; color: white; border: none; cursor: pointer; }",
        "</style>",
        "<script>",
        "function openTab(evt, tabName) {",
        "  const tabs = document.getElementsByClassName('tab');",
        "  for (let tab of tabs) { tab.classList.remove('active'); }",
        "  const buttons = document.getElementsByClassName('tab-button');",
        "  for (let button of buttons) { button.classList.remove('active'); }",
        "  document.getElementById(tabName).classList.add('active');",
        "  evt.currentTarget.classList.add('active');",
        "}",
        "function shuffleIssues(tabName) {",
        "  const tab = document.getElementById(tabName);",
        "  const issues = tab.getElementsByClassName('issue');",
        "  const issuesArr = Array.from(issues);",
        "  const parent = issues[0].parentNode;",
        "  for (let i = issuesArr.length - 1; i > 0; i--) {",
        "    const j = Math.floor(Math.random() * (i + 1));",
        "    [issuesArr[i], issuesArr[j]] = [issuesArr[j], issuesArr[i]];",
        "  }",
        "  issuesArr.forEach(issue => parent.appendChild(issue));",
        "}",
        "</script>",
        "</head>",
        "<body>",
        "<h1>Concept Store Librarian Report 📘</h1>",
        f"<p>Total issues found: {total_issues}</p>",
        "<div class='tab-buttons'>",
    ]

    # Add tab buttons
    for issue_type in issues_by_type.keys():
        html.append(
            f"<button class='tab-button' onclick=\"openTab(event, '{issue_type}')\">{issue_type} ({type_counts[issue_type]})</button>"
        )

    # Add tab content
    for issue_type, issue_list in issues_by_type.items():
        html.extend(
            [
                f"<div id='{issue_type}' class='tab'>",
                f"<h2>{issue_type}</h2>",
                f"<button class='shuffle-button' onclick=\"shuffleIssues('{issue_type}')\">Shuffle Issues</button>",
                "<div class='issues-container'>",
            ]
        )

        for issue in issue_list:
            html.extend(
                [
                    "<div class='issue'>",
                    f"<p>{issue.message}</p>",
                    "<pre class='metadata'>",
                    f"{issue.metadata}",
                    "</pre>",
                    "</div>",
                ]
            )

        html.extend(["</div>", "</div>"])

    # Add script to show first tab by default
    html.extend(
        [
            "<script>",
            "document.getElementsByClassName('tab-button')[0].click();",
            "</script>",
            "</body>",
            "</html>",
        ]
    )

    return "\n".join(html)


if __name__ == "__main__":
    issues = validate_concept_store()
    print(issues)
