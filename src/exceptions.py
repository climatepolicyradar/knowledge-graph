from datetime import datetime
from typing import Optional

from src.concept import WikibaseID


class ConceptNotFoundError(Exception):
    """Exception raised when a requested concept cannot be found."""

    def __init__(self, wikibase_id: WikibaseID):
        self.concept_id = wikibase_id
        self.message = f"Concept not found: {wikibase_id}"
        super().__init__(self.message)


class RevisionNotFoundError(Exception):
    """Exception raised when a requested revision cannot be found."""

    def __init__(self, wikibase_id: WikibaseID, timestamp: Optional[datetime] = None):
        self.concept_id = wikibase_id
        self.message = f"No revision found for ID: {wikibase_id}" + (
            f" at timestamp: {timestamp}" if timestamp else ""
        )

        super().__init__(self.message)


class ConceptCountUpdateError(Exception):
    """Exception raised when concept count updates fail"""

    def __init__(self, document_id: str, status_code: int):
        super().__init__(
            f"Failed to update concept counts for document {document_id}. "
            f"Received status code: {status_code}"
        )
        self.document_id = document_id
        self.status_code = status_code


class QueryError(Exception):
    """Exception raised when concept count updates fail"""

    def __init__(self, status_code: int):
        super().__init__(f"Failed to query Vespa. Received status code: {status_code}")
        self.status_code = status_code


class PartialUpdateError(Exception):
    """Exception raised when partial updates of concepts fail."""

    def __init__(self, id: str, status_code: int):
        super().__init__(
            f"Failed to update `{id}`. " f"Received status code: {status_code}"
        )
        self.id = id
        self.status_code = status_code
