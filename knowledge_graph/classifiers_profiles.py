from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING, Iterable

from pydantic import BaseModel, Field

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

    wikibase_id: WikibaseID = Field(description="Wikibase ID")
    classifier_id: ClassifierID = Field(description="Classifier ID")
    classifiers_profile: Profile = Field(
        description=("The classifiers profile for specified classifier ID"),
    )


# TODO: update naming similar to mapping
# TODO: update class structure
class ClassifiersProfiles(list[ClassifiersProfileMapping]):
    """Class for managing and validating a list of ClassifiersProfileMapping objects"""

    def validate(self):
        """Perform validation on the list of ClassifiersProfileMapping objects"""
        errors = []
        invalid_wikibase_ids = set()

        errors.extend(
            self._validate_mappings_multiplicity(
                Profile.RETIRED, 3, invalid_wikibase_ids
            )
        )
        errors.extend(
            self._validate_mappings_multiplicity(
                Profile.PRIMARY, 1, invalid_wikibase_ids
            )
        )
        errors.extend(
            self._validate_mappings_multiplicity(
                Profile.EXPERIMENTAL, 1, invalid_wikibase_ids
            )
        )
        errors.extend(self._validate_unique_classifier_ids(invalid_wikibase_ids))

        # Remove all profiles with invalid wikibase IDs
        if invalid_wikibase_ids:
            self._remove_invalid_wikibase_ids(invalid_wikibase_ids)

        # Raise an exception if there are validation errors
        if errors:
            raise ValueError("\n".join(errors))

    def _validate_mappings_multiplicity(
        self, profile_validation, max_count, invalid_wikibase_ids: set
    ) -> list[str]:
        """Ensure no concept is present in more than the specified number of classifiers"""
        counts = Counter(
            profile.wikibase_id
            for profile in self
            if profile.classifiers_profile == profile_validation
        )
        errors = [
            f"Validation error: Wikibase ID '{wikibase_id}' has {count} {str(profile_validation)} profiles (maximum allowed is {max_count})."
            for wikibase_id, count in counts.items()
            if count > max_count
        ]
        invalid_wikibase_ids.update(
            wikibase_id for wikibase_id, count in counts.items() if count > max_count
        )
        return errors

    def _validate_unique_classifier_ids(self, invalid_wikibase_ids: set) -> list[str]:
        """Ensure no classifier_id has more than 1 profile"""
        classifier_to_wikibase = {}
        errors = []

        for profile in self:
            if profile.classifier_id in classifier_to_wikibase:
                # Duplicate classifier_id found
                errors.append(
                    f"Validation error: Classifier ID '{profile.classifier_id}' is associated with multiple wikibase IDs "
                    f"('{classifier_to_wikibase[profile.classifier_id]}' and '{profile.wikibase_id}')."
                )
                # Add both wikibase_ids to invalid_wikibase_ids
                invalid_wikibase_ids.add(profile.wikibase_id)
            else:
                classifier_to_wikibase[profile.classifier_id] = profile.wikibase_id

        return errors

    def _remove_invalid_wikibase_ids(self, invalid_wikibase_ids: set):
        """Remove wikibase IDs with invalid profiles"""
        self[:] = [
            profile
            for profile in self
            if profile.wikibase_id not in invalid_wikibase_ids
        ]

    def append(self, item: ClassifiersProfileMapping):
        """Override append to validate the item before adding it"""
        if not isinstance(item, ClassifiersProfileMapping):
            raise TypeError(
                "Only ClassifiersProfileMapping objects can be added to ClassifiersProfiles."
            )
        super().append(item)
        self.validate()

    def extend(self, items: Iterable[ClassifiersProfileMapping]):
        """Override extend to validate the items before adding them"""
        if not all(isinstance(item, ClassifiersProfileMapping) for item in items):
            raise TypeError(
                "Only ClassifiersProfileMapping objects can be added to ClassifiersProfiles."
            )
        super().extend(items)
        self.validate()
