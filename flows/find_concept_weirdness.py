from collections import defaultdict
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


if __name__ == "__main__":
    issues = validate_concept_store()
    print(issues)
