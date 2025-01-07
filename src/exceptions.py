from datetime import datetime
from typing import Optional


class ConceptNotFoundError(Exception):
    """Exception raised when a requested concept cannot be found."""

    def __init__(self, wikibase_id: str):
        self.concept_id = wikibase_id
        self.message = f"Concept not found: {wikibase_id}"
        super().__init__(self.message)


class RevisionNotFoundError(Exception):
    """Exception raised when a requested revision cannot be found."""

    def __init__(self, wikibase_id: str, timestamp: Optional[datetime] = None):
        self.concept_id = wikibase_id
        self.message = f"No revision found for ID: {wikibase_id}" + (
            f" at timestamp: {timestamp}" if timestamp else ""
        )

        super().__init__(self.message)
