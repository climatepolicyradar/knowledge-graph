from collections import defaultdict
from string import punctuation
from typing import Optional

from pydantic import BaseModel, ValidationError

from src.concept import Concept, WikibaseID


class ConceptStoreIssue(BaseModel):
    """Issue raised by concept store checks"""

    issue_type: str
    message: str
    metadata: dict
    fix_concept: Optional[Concept] = None


def stringify_concept(
    wikibase_id: Optional[WikibaseID], all_concepts: list[Concept]
) -> str:
    # This is here as the `wikibase_id` property of `Concept` is Optional
    if wikibase_id is None:
        raise ValueError("wikibase_id cannot be None")

    single_concept_list = [c for c in all_concepts if c.wikibase_id == wikibase_id]

    if single_concept_list:
        concept = single_concept_list[0]
        return f"""<a href="{concept.wikibase_url}" target="_blank" class="concept-link">{concept.preferred_label} ({concept.wikibase_id})</a>"""
    else:
        return f"<a href='https://www.wikidata.org/wiki/{wikibase_id}' target='_blank' class='concept-link'>{wikibase_id}</a>"


def create_fix_button(concept: Concept) -> str:
    """Create a fix button that links to the concept's page"""
    return f'<a href="{concept.wikibase_url}" target="_blank" class="fix-button">Fix this</a>'


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
    for concept_id, related_id in related_relationships:
        if (related_id, concept_id) not in related_relationships:
            issues.append(
                ConceptStoreIssue(
                    issue_type="asymmetric_related_relationship",
                    message=f"{stringify_concept(concept_id, concepts)} is related to {stringify_concept(related_id, concepts)}, but {stringify_concept(related_id, concepts)} is not related to {stringify_concept(concept_id, concepts)}",
                    metadata={"concept_id": concept_id, "related_id": related_id},
                )
            )
    return issues


def validate_hierarchical_relationship_symmetry(
    concepts: list[Concept],
) -> list[ConceptStoreIssue]:
    """Make sure hierarchical subconcept relationships are symmetrical"""
    wikibase_id_to_concept = {concept.wikibase_id: concept for concept in concepts}
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
    for concept_id, subconcept_id in has_subconcept_relationships:
        if (subconcept_id, concept_id) not in subconcept_of_relationships:
            try:
                subconcept = wikibase_id_to_concept[subconcept_id]
            except (ValidationError, KeyError):
                subconcept = Concept(
                    wikibase_id=subconcept_id, preferred_label="unknown"
                )
            issues.append(
                ConceptStoreIssue(
                    issue_type="asymmetric_subconcept_relationship",
                    message=f"{stringify_concept(concept_id, concepts)} has subconcept {stringify_concept(subconcept_id, concepts)}, but {stringify_concept(subconcept_id, concepts)} does not have parent concept {stringify_concept(concept_id, concepts)}",
                    metadata={"concept_id": concept_id, "subconcept_id": subconcept_id},
                    fix_concept=subconcept,
                )
            )
    for concept_id, parent_concept_id in subconcept_of_relationships:
        if (parent_concept_id, concept_id) not in has_subconcept_relationships:
            try:
                parent_concept = wikibase_id_to_concept[parent_concept_id]
            except (ValidationError, KeyError):
                parent_concept = Concept(
                    wikibase_id=parent_concept_id, preferred_label="unknown"
                )
            issues.append(
                ConceptStoreIssue(
                    issue_type="asymmetric_subconcept_relationship",
                    message=f"{stringify_concept(concept_id, concepts)} is subconcept of {stringify_concept(parent_concept_id, concepts)}, but {stringify_concept(parent_concept_id, concepts)} does not have subconcept {stringify_concept(concept_id, concepts)}",
                    metadata={
                        "concept_id": concept_id,
                        "parent_concept_id": parent_concept_id,
                    },
                    fix_concept=parent_concept,
                )
            )
    return issues


def validate_circular_hierarchical_relationships(
    concepts: list[Concept],
) -> list[ConceptStoreIssue]:
    """Make sure hierarchical relationships are not circular"""

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

    has_subconcept_pairs_processed: list[set] = []

    for concept_id, subconcept_id in has_subconcept_relationships:
        if (subconcept_id, concept_id) in has_subconcept_relationships:
            if {concept_id, subconcept_id} not in has_subconcept_pairs_processed:
                has_subconcept_pairs_processed.append({concept_id, subconcept_id})
                issues.append(
                    ConceptStoreIssue(
                        issue_type="circular_subconcept_relationship",
                        message=f"{stringify_concept(concept_id, concepts)} has subconcept {stringify_concept(subconcept_id, concepts)}, and {stringify_concept(subconcept_id, concepts)} has subconcept {stringify_concept(concept_id, concepts)}",
                        metadata={
                            "concept_id": concept_id,
                            "subconcept_id": subconcept_id,
                        },
                    )
                )

    subconcept_of_pairs_processed: list[set] = []
    for concept_id, parent_concept_id in subconcept_of_relationships:
        if (parent_concept_id, concept_id) in subconcept_of_relationships:
            if {concept_id, parent_concept_id} not in subconcept_of_pairs_processed:
                subconcept_of_pairs_processed.append({concept_id, parent_concept_id})
                issues.append(
                    ConceptStoreIssue(
                        issue_type="circular_subconcept_relationship",
                        message=f"{stringify_concept(concept_id, concepts)} is subconcept of {stringify_concept(parent_concept_id, concepts)}, and {stringify_concept(parent_concept_id, concepts)} is a subconcept of {stringify_concept(concept_id, concepts)}",
                        metadata={
                            "concept_id": concept_id,
                            "parent_concept_id": parent_concept_id,
                        },
                    )
                )

    return issues


def check_for_unconnected_concepts(
    concepts: list[Concept],
) -> list[ConceptStoreIssue]:
    """Find all concepts that are not connected to any other concepts"""
    issues = []
    for concept in concepts:
        # TODO: a better way of doing this would be to query all properties with data type
        # "item" and check if each concept has a value for any of those properties
        if (
            not concept.related_concepts
            and not concept.has_subconcept
            and not concept.subconcept_of
        ):
            issues.append(
                ConceptStoreIssue(
                    issue_type="unconnected_concept",
                    message=f"{stringify_concept(concept.wikibase_id, concepts)} is not connected to any other concepts",
                    metadata={"concept_id": concept.wikibase_id},
                    fix_concept=concept,
                )
            )
    return issues


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
                    message=f"{stringify_concept(concept.wikibase_id, concepts)} has duplicate alternative labels: {duplicate_labels}",
                    metadata={
                        "concept_id": concept.wikibase_id,
                        "duplicate_labels": duplicate_labels,
                    },
                    fix_concept=concept,
                )
            )
    return issues


def check_alternative_labels_for_pipes(
    concepts: list[Concept],
) -> list[ConceptStoreIssue]:
    """Find all concepts that have alternative labels containing pipes (|)"""

    issues = []
    for concept in concepts:
        if alt_labels_containing_pipes := [
            label for label in concept.alternative_labels if "|" in label
        ]:
            issues.append(
                ConceptStoreIssue(
                    issue_type="alternative_label_contains_pipe",
                    message=f"{stringify_concept(concept.wikibase_id, concepts)} has alternative labels containing pipes",
                    metadata={
                        "concept_id": concept.wikibase_id,
                        "aliases_with_pipes": alt_labels_containing_pipes,
                    },
                    fix_concept=concept,
                )
            )

    return issues


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
                    message=f"{stringify_concept(concept.wikibase_id, concepts)} has negative labels which appear in its positive labels: {overlapping_labels}",
                    metadata={
                        "concept_id": concept.wikibase_id,
                        "overlapping_labels": list(overlapping_labels),
                    },
                )
            )
    return issues


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
                    message=f"{stringify_concept(concept.wikibase_id, concepts)} has a short description",
                    metadata={
                        "concept_id": concept.wikibase_id,
                        "description": concept.description,
                    },
                    fix_concept=concept,
                )
            )
        if concept.definition and len(concept.definition) < minimum_length:
            issues.append(
                ConceptStoreIssue(
                    issue_type="short_definition",
                    message=f"{stringify_concept(concept.wikibase_id, concepts)} has a short definition",
                    metadata={
                        "concept_id": concept.wikibase_id,
                        "definition": concept.definition,
                    },
                    fix_concept=concept,
                )
            )
    return issues


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
            duplicate_concepts = [
                concept for concept in concepts if concept.wikibase_id in ids
            ]
            duplicate_concepts_string = ", ".join(
                [
                    stringify_concept(concept.wikibase_id, concepts)
                    for concept in duplicate_concepts
                ]
            )
            issues.append(
                ConceptStoreIssue(
                    issue_type="duplicate_preferred_labels",
                    message=f"{len(ids)} concepts have the same label '{label}': {duplicate_concepts_string}",
                    metadata={"label": label, "concept_ids": ids},
                )
            )
    return issues


def validate_concept_label_casing(
    concepts: list[Concept],
):
    """Find concepts with labels that are not UPPERCASED or lowercased"""
    issues = []
    for concept in concepts:
        if concept.preferred_label and not (
            concept.preferred_label.isupper() or concept.preferred_label.islower()
        ):
            issues.append(
                ConceptStoreIssue(
                    issue_type="label_mixed_casing",
                    message=f"{stringify_concept(concept.wikibase_id, concepts)} has a label that uses mixed casing.",
                    metadata={"concept_id": concept.wikibase_id},
                    fix_concept=concept,
                )
            )
    return issues
