from collections import defaultdict
from string import punctuation
from typing import Optional

from pydantic import BaseModel

from src.concept import Concept, WikibaseID


class ConceptStoreIssue(BaseModel):
    """Issue raised by concept store checks"""

    issue_type: str
    message: str


class ConceptIssue(ConceptStoreIssue):
    """Issue raised by concept store checks"""

    concept: Concept


class RelationshipIssue(ConceptStoreIssue):
    """Issue raised by concept store checks"""

    from_concept: Concept
    to_concept: Concept


class MultiConceptIssue(ConceptStoreIssue):
    """Issue raised by concept store checks involving multiple concepts as a group"""

    concepts: list[Concept]


class EmptyConcept(Concept):
    """A concept which comes from Wikibase but is missing key data"""

    wikibase_id: WikibaseID
    preferred_label: Optional[str] = None


def format_concept_link(concept: Concept) -> str:
    """Format a concept as an HTML link"""
    style = "text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 underline underline-offset-4"
    display_text = (
        concept.wikibase_id if isinstance(concept, EmptyConcept) else str(concept)
    )
    return f"<a href='{concept.wikibase_url}' target='_blank' class='{style}'>{display_text}</a>"


def validate_related_relationship_symmetry(
    concepts: list[Concept],
) -> list[RelationshipIssue]:
    """Make sure related concepts relationships are symmetrical"""
    issues = []
    concepts_by_id = {concept.wikibase_id: concept for concept in concepts}
    related_relationships = [
        (concept.wikibase_id, related_id)
        for concept in concepts
        for related_id in concept.related_concepts
    ]
    for concept_id, related_id in related_relationships:
        if (related_id, concept_id) not in related_relationships:
            from_concept = concepts_by_id[concept_id]
            try:
                to_concept = concepts_by_id[related_id]
            except KeyError:
                to_concept = EmptyConcept(wikibase_id=related_id)
            issues.append(
                RelationshipIssue(
                    issue_type="asymmetric_related_relationship",
                    message=f"{format_concept_link(from_concept)} is related to {format_concept_link(to_concept)}, but {format_concept_link(to_concept)} is not related to {format_concept_link(from_concept)}",
                    from_concept=from_concept,
                    to_concept=to_concept,
                )
            )
    return issues


def validate_hierarchical_relationship_symmetry(
    concepts: list[Concept],
) -> list[RelationshipIssue]:
    """Make sure hierarchical subconcept relationships are symmetrical"""
    concepts_by_id = {concept.wikibase_id: concept for concept in concepts}
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
            from_concept = concepts_by_id[concept_id]
            try:
                to_concept = concepts_by_id[subconcept_id]
            except KeyError:
                to_concept = EmptyConcept(wikibase_id=subconcept_id)
            issues.append(
                RelationshipIssue(
                    issue_type="asymmetric_subconcept_relationship",
                    message=f"{format_concept_link(from_concept)} has subconcept {format_concept_link(to_concept)}, but {format_concept_link(to_concept)} does not have parent concept {format_concept_link(from_concept)}",
                    from_concept=from_concept,
                    to_concept=to_concept,
                )
            )
    for concept_id, parent_concept_id in subconcept_of_relationships:
        if (parent_concept_id, concept_id) not in has_subconcept_relationships:
            from_concept = concepts_by_id[concept_id]
            try:
                to_concept = concepts_by_id[parent_concept_id]
            except KeyError:
                to_concept = EmptyConcept(wikibase_id=parent_concept_id)
            issues.append(
                RelationshipIssue(
                    issue_type="asymmetric_subconcept_relationship",
                    message=f"{format_concept_link(from_concept)} is subconcept of {format_concept_link(to_concept)}, but {format_concept_link(to_concept)} does not have subconcept {format_concept_link(from_concept)}",
                    from_concept=from_concept,
                    to_concept=to_concept,
                )
            )
    return issues


def validate_circular_hierarchical_relationships(
    concepts: list[Concept],
) -> list[RelationshipIssue]:
    """Make sure hierarchical relationships are not circular"""
    concepts_by_id = {concept.wikibase_id: concept for concept in concepts}
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
                from_concept = concepts_by_id[concept_id]
                to_concept = concepts_by_id[subconcept_id]
                issues.append(
                    RelationshipIssue(
                        issue_type="circular_subconcept_relationship",
                        message=f"{format_concept_link(from_concept)} has subconcept {format_concept_link(to_concept)}, and {format_concept_link(to_concept)} has subconcept {format_concept_link(from_concept)}",
                        from_concept=from_concept,
                        to_concept=to_concept,
                    )
                )

    subconcept_of_pairs_processed: list[set] = []
    for concept_id, parent_concept_id in subconcept_of_relationships:
        if (parent_concept_id, concept_id) in subconcept_of_relationships:
            if {concept_id, parent_concept_id} not in subconcept_of_pairs_processed:
                subconcept_of_pairs_processed.append({concept_id, parent_concept_id})
                from_concept = concepts_by_id[concept_id]
                to_concept = concepts_by_id[parent_concept_id]
                issues.append(
                    RelationshipIssue(
                        issue_type="circular_subconcept_relationship",
                        message=f"{format_concept_link(from_concept)} is subconcept of {format_concept_link(to_concept)}, and {format_concept_link(to_concept)} is a subconcept of {format_concept_link(from_concept)}",
                        from_concept=from_concept,
                        to_concept=to_concept,
                    )
                )

    return issues


def check_for_unconnected_concepts(
    concepts: list[Concept],
) -> list[ConceptIssue]:
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
                ConceptIssue(
                    issue_type="unconnected_concept",
                    message=f"{format_concept_link(concept)} is not connected to any other concepts",
                    concept=concept,
                )
            )
    return issues


def validate_alternative_label_uniqueness(
    concepts: list[Concept],
) -> list[ConceptIssue]:
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
                ConceptIssue(
                    issue_type="duplicate_alternative_labels",
                    message=f"{format_concept_link(concept)} has duplicate alternative labels: {duplicate_labels}",
                    concept=concept,
                )
            )
    return issues


def check_alternative_labels_for_pipes(
    concepts: list[Concept],
) -> list[ConceptIssue]:
    """Find all concepts that have alternative labels containing pipes (|)"""

    issues = []
    for concept in concepts:
        if any("|" in label for label in concept.alternative_labels):
            issues.append(
                ConceptIssue(
                    issue_type="alternative_label_contains_pipe",
                    message=f"{format_concept_link(concept)} has alternative labels containing pipes",
                    concept=concept,
                )
            )

    return issues


def ensure_positive_and_negative_labels_dont_overlap(
    concepts: list[Concept],
) -> list[ConceptIssue]:
    """Make sure negative labels don't appear in positive labels"""
    issues = []
    for concept in concepts:
        overlapping_labels = set(concept.negative_labels) & set(concept.all_labels)
        if overlapping_labels:
            issues.append(
                ConceptIssue(
                    issue_type="overlapping_labels",
                    message=f"{format_concept_link(concept)} has negative labels which appear in its positive labels: {overlapping_labels}",
                    concept=concept,
                )
            )
    return issues


def check_description_and_definition_length(
    concepts: list[Concept],
) -> list[ConceptIssue]:
    """Make sure descriptions and definitions are long enough"""
    issues = []
    minimum_length = 20
    for concept in concepts:
        if concept.description and len(concept.description) < minimum_length:
            issues.append(
                ConceptIssue(
                    issue_type="short_description",
                    message=f"{format_concept_link(concept)} has a short description",
                    concept=concept,
                )
            )
        if concept.definition and len(concept.definition) < minimum_length:
            issues.append(
                ConceptIssue(
                    issue_type="short_definition",
                    message=f"{format_concept_link(concept)} has a short definition",
                    concept=concept,
                )
            )
    return issues


def check_for_duplicate_preferred_labels(
    concepts: list[Concept],
) -> list[MultiConceptIssue]:
    """Make sure there are no duplicate concepts"""
    issues = []

    def clean(text: str) -> str:
        cleaned = text.lower().strip()
        cleaned = cleaned.translate(str.maketrans("", "", punctuation))
        return cleaned

    duplicate_dict = defaultdict(list)
    for concept in concepts:
        label = clean(concept.preferred_label)
        duplicate_dict[label].append(concept)

    for label, duplicate_concepts in duplicate_dict.items():
        if len(duplicate_concepts) > 1:
            duplicate_concepts_string = ", ".join(
                format_concept_link(concept) for concept in duplicate_concepts
            )
            issues.append(
                MultiConceptIssue(
                    issue_type="duplicate_preferred_labels",
                    message=f"{len(duplicate_concepts)} concepts have the same label '{label}': {duplicate_concepts_string}",
                    concepts=duplicate_concepts,
                )
            )
    return issues


def validate_concept_label_casing(
    concepts: list[Concept],
) -> list[ConceptIssue]:
    """Find concepts with labels that are not UPPERCASED or lowercased"""
    issues = []
    for concept in concepts:
        if concept.preferred_label and not (
            concept.preferred_label.isupper() or concept.preferred_label.islower()
        ):
            issues.append(
                ConceptIssue(
                    issue_type="label_mixed_casing",
                    message=f"{format_concept_link(concept)} has a label that uses mixed casing.",
                    concept=concept,
                )
            )
    return issues
