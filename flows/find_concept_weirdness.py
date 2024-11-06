from collections import defaultdict
from string import punctuation

from prefect import flow, task

from src.concept import Concept
from src.wikibase import WikibaseSession

wikibase = WikibaseSession()


@flow(log_prints=True)
def validate_concept_store():
    print("Fetching all concepts from wikibase")
    concepts: list[Concept] = wikibase.get_concepts()
    print(f"Found {len(concepts)} concepts")

    validate_related_relationship_symmetry(concepts)
    validate_hierarchical_relationship_symmetry(concepts)
    validate_alternative_label_uniqueness(concepts)
    ensure_positive_and_negative_labels_dont_overlap(concepts)
    check_description_and_definition_length(concepts)
    check_for_duplicate_preferred_labels(concepts)


@task(log_prints=True)
def validate_related_relationship_symmetry(concepts: list[Concept]):
    """Make sure related concepts relationships are symmetrical"""
    related_relationships = [
        (concept.wikibase_id, related_id)
        for concept in concepts
        for related_id in concept.related_concepts
    ]
    print(f"Found {len(related_relationships)} related concepts relationships")
    for concept_id, related_id in related_relationships:
        if (related_id, concept_id) not in related_relationships:
            print(
                f"{concept_id} is related to {related_id}, but "
                f"{related_id} is not related to {concept_id}"
            )


@task(log_prints=True)
def validate_hierarchical_relationship_symmetry(concepts: list[Concept]):
    """Make sure hierarchical subconcept relationships are symmetrical"""
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
            print(
                f"{concept_id} has subconcept {subconcept_id}, but "
                f"{subconcept_id} does not have parent concept {concept_id}"
            )
    for concept_id, parent_concept_id in subconcept_of_relationships:
        if (parent_concept_id, concept_id) not in has_subconcept_relationships:
            print(
                f"{concept_id} is subconcept of {parent_concept_id}, but "
                f"{parent_concept_id} does not have subconcept {concept_id}"
            )


@task(log_prints=True)
def validate_alternative_label_uniqueness(concepts: list[Concept]):
    """Make sure alternative labels are unique"""
    for concept in concepts:
        duplicate_labels = [
            label
            for label in concept.alternative_labels
            if concept.alternative_labels.count(label) > 1
        ]
        if duplicate_labels:
            print(
                f"{concept.wikibase_id} has duplicate alternative labels: {duplicate_labels}"
            )


@task(log_prints=True)
def ensure_positive_and_negative_labels_dont_overlap(concepts: list[Concept]):
    """Make sure negative labels don't appear in positive labels"""
    for concept in concepts:
        overlapping_labels = set(concept.negative_labels) & set(concept.all_labels)
        if overlapping_labels:
            print(
                f"{concept.wikibase_id} has negative labels which appear "
                f"in its positive labels: {overlapping_labels}"
            )


@task(log_prints=True)
def check_description_and_definition_length(concepts: list[Concept]):
    """Make sure descriptions and definitions are long enough"""
    minimum_length = 20
    for concept in concepts:
        if concept.description and len(concept.description) < minimum_length:
            print(f"{concept.wikibase_id} has a short description")
        if concept.definition and len(concept.definition) < minimum_length:
            print(f"{concept.wikibase_id} has a short definition")


@task(log_prints=True)
def check_for_duplicate_preferred_labels(concepts: list[Concept]):
    """Make sure there are no duplicate concepts"""

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
            print(f"Duplicate preferred labels: {label} -> {ids}")


if __name__ == "__main__":
    validate_concept_store()
