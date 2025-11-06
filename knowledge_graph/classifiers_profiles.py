from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING, List, Set

from pydantic import BaseModel, Field

from flows.result import Err, Error, Result
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.wikibase import StatementRank

if TYPE_CHECKING:
    # only import this circular dependency if we're running in a type-checking
    # environment, eg for pyright
    pass


class Profile(str, Enum):
    """Accepted classifiers profile values"""

    PRIMARY = "primary"
    EXPERIMENTAL = "experimental"
    RETIRED = "retired"

    def __str__(self):
        """Return a string representation"""
        return self.value

    @classmethod
    def generate(cls, rank: StatementRank):
        """Generate a Profile from a StatementRank"""
        if rank == StatementRank.PREFERRED:
            return cls.PRIMARY
        elif rank == StatementRank.NORMAL:
            return cls.EXPERIMENTAL
        elif rank == StatementRank.DEPRECATED:
            return cls.RETIRED
        else:
            raise ValueError(f"Unknown StatementRank: {rank}")


class ClassifiersProfileMapping(BaseModel):
    """Base class for a classifier profile"""

    wikibase_id: WikibaseID = Field(description="Wikibase ID (e.g. Q100)")
    classifier_id: ClassifierID = Field(description="Canonical Classifier ID")
    classifiers_profile: Profile = Field(
        description=("The classifiers profile for specified classifier ID"),
    )


def validate_mappings_multiplicity(
    profiles: List[ClassifiersProfileMapping],
    profile_validation: Profile,
    max_count: int,
) -> List[Result[WikibaseID, Error]]:
    """Ensure no concept is present in more than the specified number of classifiers"""
    errors = []
    counts = Counter(
        profile.wikibase_id
        for profile in profiles
        if profile.classifiers_profile == profile_validation
    )

    for wikibase_id, count in counts.items():
        if count > max_count:
            errors.append(
                Err(
                    Error(
                        msg="Validation Error: classifier multiplicity",
                        metadata={
                            "wikibase_id": {wikibase_id},
                            "profile": {str(profile_validation)},
                            "count": {count},
                        },
                    )
                )
            )

    return errors


def validate_unique_classifier_ids(
    profiles: List[ClassifiersProfileMapping],
) -> List[Result[WikibaseID, Error]]:
    """Ensure no classifier_id has more than 1 profile"""
    classifier_to_wikibase = {}
    errors = []

    for profile in profiles:
        if profile.classifier_id in classifier_to_wikibase:
            # Duplicate classifier_id found
            errors.append(
                Err(
                    Error(
                        msg="Validation Error: classifiers ID has multiple wikibase IDs",
                        metadata={
                            "wikibase_id": {
                                profile.wikibase_id,
                                classifier_to_wikibase[profile.classifier_id],
                            },
                            "classifier_id": {profile.classifier_id},
                        },
                    )
                )
            )

        else:
            classifier_to_wikibase[profile.classifier_id] = profile.wikibase_id

    return errors


def get_valid_wikibase_ids(
    profiles: List[ClassifiersProfileMapping], invalid_wikibase_ids: Set[WikibaseID]
) -> List[ClassifiersProfileMapping]:
    """Return only wikibase ids that have passed validation"""
    return [
        profile
        for profile in profiles
        if profile.wikibase_id not in invalid_wikibase_ids
    ]


def validate_classifiers_profiles_mappings(
    profiles: List[ClassifiersProfileMapping],
) -> tuple[List[ClassifiersProfileMapping], List[Result[WikibaseID, Error]]]:
    """Perform validation on the list of ClassifiersProfileMapping objects"""
    errors: list[Result[WikibaseID, Error]] = []

    errors.extend(validate_mappings_multiplicity(profiles, Profile.RETIRED, 3))
    errors.extend(validate_mappings_multiplicity(profiles, Profile.PRIMARY, 1))
    errors.extend(validate_mappings_multiplicity(profiles, Profile.EXPERIMENTAL, 1))
    errors.extend(validate_unique_classifier_ids(profiles))

    failures = [r._error for r in errors if isinstance(r, Err)]
    invalid_wikibase_ids = {
        wikibase_id
        for error in failures
        for wikibase_id in (error.metadata or {}).get("wikibase_id", set())
    }

    # Identify all profiles with invalid wikibase IDs
    valid_profiles = get_valid_wikibase_ids(profiles, invalid_wikibase_ids)

    return valid_profiles, errors
