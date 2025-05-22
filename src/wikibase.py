import json
import os
from datetime import datetime, timezone
from logging import getLogger
from typing import Optional

import dotenv
import httpx
from httpx import HTTPError
from pydantic import ValidationError

from src.concept import Concept
from src.exceptions import ConceptNotFoundError, RevisionNotFoundError
from src.identifiers import WikibaseID

logger = getLogger(__name__)

dotenv.load_dotenv()


class WikibaseSession:
    """A session for interacting with Wikibase"""

    session = httpx.Client()

    has_subconcept_property_id = os.getenv("WIKIBASE_HAS_SUBCONCEPT_PROPERTY_ID", "P1")
    subconcept_of_property_id = os.getenv("WIKIBASE_SUBCONCEPT_OF_PROPERTY_ID", "P2")
    related_concept_property_id = os.getenv(
        "WIKIBASE_RELATED_CONCEPT_PROPERTY_ID", "P3"
    )
    negative_labels_property_id = os.getenv(
        "WIKIBASE_NEGATIVE_LABELS_PROPERTY_ID", "P9"
    )
    definition_property_id = os.getenv("WIKIBASE_DEFINITION_PROPERTY_ID", "P7")

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        url: Optional[str] = None,
    ):
        """Log in to Wikibase and get a CSRF token"""
        self.username = username or os.getenv("WIKIBASE_USERNAME")
        self.password = password or os.getenv("WIKIBASE_PASSWORD")
        self.base_url = url or os.getenv("WIKIBASE_URL")
        self.api_url = f"{self.base_url}/w/api.php"

        if not self.username or not self.password or not self.base_url:
            raise ValueError(
                "username, password and url must be set, either as arguments or "
                "the environment variables: WIKIBASE_USERNAME, WIKIBASE_PASSWORD, "
                "and WIKIBASE_URL"
            )
        self._login()

        # Get all redirects, so that any fetched references to concepts can be updated
        # to use the correct wikibase ID
        self.redirects = self._get_all_redirects()

    def __repr__(self) -> str:
        """Return a string representation of the Wikibase session"""
        return f"<WikibaseSession: {self.username} at {self.api_url}>"

    def _login(self):
        """Log in to Wikibase and get a CSRF token"""
        login_token_response = self.session.get(
            url=self.api_url,
            params={
                "action": "query",
                "meta": "tokens",
                "type": "login",
                "format": "json",
            },
        ).json()
        login_token = login_token_response["query"]["tokens"]["logintoken"]

        self.session.post(
            url=self.api_url,
            data={
                "action": "login",
                "lgname": self.username,
                "lgpassword": self.password,
                "lgtoken": login_token,
                "format": "json",
            },
        )

        csrf_token_response = self.session.get(
            url=self.api_url,
            params={"action": "query", "meta": "tokens", "format": "json"},
        )
        csrf_token = csrf_token_response.json()["query"]["tokens"]["csrftoken"]

        logger.debug(f"Got session CSRF token: {csrf_token}")

        self.csrf_token = csrf_token
        logger.debug("Session headers updated")

    def _resolve_redirect(self, wikibase_id: WikibaseID) -> WikibaseID:
        """
        Check if a Wikibase ID is a redirect and return its target ID if it is.

        :param WikibaseID wikibase_id: The Wikibase ID to check
        :return WikibaseID: The resolved Wikibase ID (either the same ID or its redirect target)
        """
        return self.redirects.get(wikibase_id, wikibase_id)

    def get_concept(
        self,
        wikibase_id: WikibaseID,
        timestamp: Optional[datetime] = None,
        include_labels_from_subconcepts: bool = False,
        include_recursive_subconcept_of: bool = False,
        include_recursive_has_subconcept: bool = False,
    ) -> Concept:
        """
        Get a concept from Wikibase by its Wikibase ID

        :param WikibaseID wikibase_id: The Wikibase ID of the concept
        :param Optional[datetime] timestamp: The timestamp to fetch the concept at.
            If not provided, the latest version of the concept will be fetched.
        :param bool include_labels_from_subconcepts: Whether to include the labels
            from subconcepts in the concept
        :param bool include_recursive_subconcept_of: Whether to include the concept's
            complete ancestry of all parent concepts, recursively up the hierarchy
        :param bool include_recursive_has_subconcept: Whether to include the concept's
            complete subconcepts, recursively down the hierarchy
        :return Concept: The concept with the given Wikibase ID
        """
        # Get the base concept first
        concept = self.get_concepts(wikibase_ids=[wikibase_id], timestamp=timestamp)[0]

        # Get recursive relationships if needed
        recursive_subconcept_ids = []
        if include_recursive_has_subconcept or include_labels_from_subconcepts:
            recursive_subconcept_ids = self.get_recursive_has_subconcept_relationships(
                wikibase_id
            )
            if include_recursive_has_subconcept:
                concept.recursive_has_subconcept = recursive_subconcept_ids

        if include_recursive_subconcept_of:
            concept.recursive_subconcept_of = (
                self.get_recursive_subconcept_of_relationships(wikibase_id)
            )

        # Get labels from subconcepts if needed
        if include_labels_from_subconcepts and recursive_subconcept_ids:
            subconcepts = self.get_concepts(
                wikibase_ids=recursive_subconcept_ids, timestamp=timestamp
            )

            # Collect all labels
            all_positive_labels = set(concept.all_labels)
            all_negative_labels = set(concept.negative_labels)
            for subconcept in subconcepts:
                all_positive_labels.update(subconcept.all_labels)
                all_negative_labels.update(subconcept.negative_labels)

            concept.alternative_labels = list(all_positive_labels)
            concept.negative_labels = list(all_negative_labels)

        return concept

    def _get_recursive_relationships(
        self,
        wikibase_id: WikibaseID,
        property_id: str,
        max_depth: int = 50,
        current_depth: int = 0,
        visited: Optional[set[WikibaseID]] = None,
    ) -> list[WikibaseID]:
        """
        Helper method to fetch recursive relationships by following a property.

        Uses a 'visited' set to avoid cycles and redundant API calls.

        :param WikibaseID wikibase_id: The Wikibase ID to start from
        :param str property_id: The property ID to traverse
        :param int max_depth: The maximum number of hops to traverse across the
            hierarchy, defaults to 50
        :param int current_depth: Internal parameter to track recursion depth
        :param set[WikibaseID] visited: Set of already visited IDs to avoid cycles
        :return list[WikibaseID]: List of related concept IDs
        """
        if visited is None:
            visited = set()

        if current_depth >= max_depth or wikibase_id in visited:
            return []

        visited.add(wikibase_id)

        valid_property_ids = [
            self.subconcept_of_property_id,
            self.has_subconcept_property_id,
        ]
        if property_id not in valid_property_ids:
            raise ValueError(
                f"Invalid property ID: {property_id}. "
                f"Valid property IDs are: {valid_property_ids}"
            )

        # Resolve any redirects first
        wikibase_id = self._resolve_redirect(wikibase_id)

        response = self.session.get(
            url=self.api_url,
            params={
                "action": "wbgetentities",
                "format": "json",
                "ids": wikibase_id,
                "props": "claims",  # Only fetch claims to reduce response size
            },
        ).json()

        entity = response["entities"][str(wikibase_id)]
        hierarchically_related_concepts = []

        if "claims" in entity:
            related_ids = []
            for claim in entity["claims"].values():
                for statement in claim:
                    if (
                        statement["mainsnak"]["snaktype"] == "value"
                        and statement["mainsnak"]["property"] == property_id
                    ):
                        related_id = self._resolve_redirect(
                            statement["mainsnak"]["datavalue"]["value"]["id"]
                        )
                        if related_id not in visited:
                            related_ids.append(related_id)

            # Batch fetch the next level of relationships
            if related_ids:
                hierarchically_related_concepts.extend(related_ids)
                for related_id in related_ids:
                    hierarchically_related_concepts.extend(
                        self._get_recursive_relationships(
                            related_id,
                            property_id,
                            max_depth=max_depth,
                            current_depth=current_depth + 1,
                            visited=visited,
                        )
                    )

        return list(set(hierarchically_related_concepts))

    def get_recursive_subconcept_of_relationships(
        self, wikibase_id: WikibaseID
    ) -> list[WikibaseID]:
        """Fetch the complete parentage of a concept, recursively up the hierarchy"""
        return self._get_recursive_relationships(
            wikibase_id, self.subconcept_of_property_id
        )

    def get_recursive_has_subconcept_relationships(
        self, wikibase_id: WikibaseID
    ) -> list[WikibaseID]:
        """Fetch the complete subconcepts of a concept, recursively down the hierarchy"""
        return self._get_recursive_relationships(
            wikibase_id, self.has_subconcept_property_id
        )

    def _get_pages(self, extra_params: dict) -> list[dict]:
        """
        Helper method to get pages from Wikibase with pagination.

        :param dict extra_params: The parameters to pass to the API
        :return list[dict]: List of page data from all batches
        """
        PAGE_REQUEST_SIZE = 500
        MAX_PAGE_REQUESTS = 2000  # Suitable up to 1M pages (500*2000)

        base_params = {
            "action": "query",
            "format": "json",
            "list": "allpages",  # See https://www.mediawiki.org/wiki/API:Allpages
            "apnamespace": 120,
            "aplimit": PAGE_REQUEST_SIZE,
        }
        # Update with any additional params supplied by the user
        base_params.update(extra_params)

        pages = []
        for i in range(MAX_PAGE_REQUESTS):
            response = self.session.get(
                url=self.api_url,
                params=base_params,
            ).json()

            if "error" in response:
                raise HTTPError(response["error"])
            if "warnings" in response:
                logger.warning(response["warnings"])

            batch_pages = response["query"]["allpages"]
            pages.extend(batch_pages)

            if continue_params := response.get("continue"):
                base_params.update(continue_params)
                logger.info(f"Retrieved {len(pages)} pages after iteration {i}")
            else:
                break

        return pages

    def get_all_concept_ids(self) -> list[WikibaseID]:
        """
        Get a complete list of all concept ids in the Wikibase instance.

        :return list[WikibaseID]: The concept ids, e.g ["Q123", "Q456"]
        """
        pages = self._get_pages(extra_params={"apfilterredir": "nonredirects"})
        return [page["title"].replace("Item:", "") for page in pages]

    def _get_all_redirects(self, batch_size: int = 50) -> dict[WikibaseID, WikibaseID]:
        """
        Get all redirects from Wikibase.

        :return dict[WikibaseID, WikibaseID]: The redirects, e.g
            {"Q123": "Q456", "Q456": "Q789"}
        """
        pages = self._get_pages(extra_params={"apfilterredir": "redirects"})
        redirects = {}

        # For each redirect, we need to find the wikibase ids of the source and target.
        # We process the pages in batches of 50
        for batch in [
            pages[i : i + batch_size] for i in range(0, len(pages), batch_size)
        ]:
            ids_to_fetch = [page["title"].replace("Item:", "") for page in batch]
            response = self.session.get(
                url=self.api_url,
                params={
                    "action": "wbgetentities",
                    "format": "json",
                    "ids": "|".join(ids_to_fetch),
                    "props": "info",
                },
            ).json()

            for wikibase_id, entity in response.get("entities", {}).items():
                if "redirects" in entity:
                    redirects[WikibaseID(wikibase_id)] = WikibaseID(
                        entity["redirects"]["to"]
                    )

        return redirects

    def get_concepts(
        self,
        limit: Optional[int] = None,
        wikibase_ids: Optional[list[WikibaseID]] = None,
        timestamp: Optional[datetime] = None,
    ) -> list[Concept]:
        """
        Get concepts from Wikibase, optionally specified by their Wikibase IDs.

        Fetches concepts in batches for better performance.

        :param Optional[int] limit: The maximum number of concepts to fetch
        :param list[WikibaseID] wikibase_ids: The Wikibase IDs of the concepts
        :param Optional[datetime] timestamp: The timestamp to fetch concepts at.
            If not provided, the latest version will be fetched.
        :return list[Concept]: The concepts, optionally with the given Wikibase IDs
        :raises ConceptNotFoundError: If a concept doesn't exist
        :raises RevisionNotFoundError: If a concept exists but no revision is found at the specified timestamp
        """
        if not wikibase_ids:
            wikibase_ids = self.get_all_concept_ids()

        if limit:
            wikibase_ids = wikibase_ids[:limit]

        if timestamp:
            if timestamp.tzinfo is None:
                timestamp = timestamp.astimezone(timezone.utc)
            if timestamp > datetime.now(timezone.utc):
                raise ValueError(
                    "Can't fetch concepts from the future... "
                    "The value of timestamp must be in the past"
                )
        else:
            timestamp = datetime.now(timezone.utc)

        # Process in batches of 50 for better performance
        BATCH_SIZE = 50
        concepts = []

        for i in range(0, len(wikibase_ids), BATCH_SIZE):
            batch_ids = wikibase_ids[i : i + BATCH_SIZE]
            # First resolve any redirects
            batch_ids = [self._resolve_redirect(wid) for wid in batch_ids]

            # Then get the basic entity data
            entity_response = self.session.get(
                url=self.api_url,
                params={
                    "action": "wbgetentities",
                    "format": "json",
                    "ids": "|".join(batch_ids),
                    "props": "info",
                },
            ).json()

            if "error" in entity_response:
                logger.warning(f"Error fetching batch: {entity_response['error']}")
                continue

            # Then get the full concept data using revisions API
            for wikibase_id in batch_ids:
                try:
                    # Get the pageid for this wikibase ID
                    page_id = str(
                        entity_response.get("entities", {})
                        .get(str(wikibase_id), {})
                        .get("pageid")
                    )
                    if not page_id:
                        raise ConceptNotFoundError(wikibase_id)

                    # Get the revision at the specified timestamp
                    revisions_response = self.session.get(
                        url=self.api_url,
                        params={
                            "action": "query",
                            "format": "json",
                            "pageids": page_id,
                            "prop": "revisions",
                            "rvdir": "older",
                            "rvlimit": 1,
                            "rvprop": "content",
                            "rvslots": "main",
                            "rvstart": timestamp.isoformat(),
                        },
                    ).json()

                    pages = revisions_response.get("query", {}).get("pages", {})
                    if not pages:
                        raise ConceptNotFoundError(wikibase_id)

                    page = next(iter(pages.values()))
                    revisions = page.get("revisions", [])
                    if not revisions:
                        raise RevisionNotFoundError(wikibase_id, timestamp)

                    entity = json.loads(
                        revisions[0].get("slots", {}).get("main", {}).get("*", "{}")
                    )

                    if not entity:
                        raise ConceptNotFoundError(wikibase_id)

                    preferred_label = (
                        entity.get("labels", {})
                        .get("en", {})
                        .get("value", f"concept {wikibase_id}")
                    )
                    alternative_labels = []
                    if isinstance(entity.get("aliases"), dict):
                        alternative_labels = [
                            alias.get("value")
                            for alias in entity.get("aliases", {}).get("en", [])
                            if alias.get("language") == "en"
                        ]

                    description = (
                        entity.get("descriptions", {}).get("en", {}).get("value", "")
                        if isinstance(entity.get("descriptions"), dict)
                        else ""
                    )

                    concept = Concept(
                        preferred_label=preferred_label,
                        alternative_labels=alternative_labels,
                        description=description,
                        wikibase_id=wikibase_id,
                    )

                    if "claims" in entity and entity["claims"] != []:
                        for claim in entity["claims"].values():
                            for statement in claim:
                                if statement["mainsnak"]["snaktype"] == "value":
                                    property_id = statement["mainsnak"]["property"]
                                    value = statement["mainsnak"]["datavalue"]["value"]
                                    if property_id == self.subconcept_of_property_id:
                                        concept.subconcept_of = (
                                            concept.subconcept_of
                                            + [self._resolve_redirect(value["id"])]
                                        )
                                    elif property_id == self.has_subconcept_property_id:
                                        concept.has_subconcept = (
                                            concept.has_subconcept
                                            + [self._resolve_redirect(value["id"])]
                                        )
                                    elif (
                                        property_id == self.related_concept_property_id
                                    ):
                                        concept.related_concepts = (
                                            concept.related_concepts
                                            + [self._resolve_redirect(value["id"])]
                                        )
                                    elif (
                                        property_id == self.negative_labels_property_id
                                    ):
                                        concept.negative_labels = (
                                            concept.negative_labels + [value]
                                        )
                                    elif property_id == self.definition_property_id:
                                        concept.definition = value

                    concepts.append(concept)
                except (KeyError, json.JSONDecodeError) as e:
                    logger.warning(
                        f"Failed to parse concept with Wikibase ID: {wikibase_id} with error: {e}"
                    )
                except (ConceptNotFoundError, RevisionNotFoundError) as e:
                    logger.warning(str(e))
                except ValidationError as e:
                    logger.warning(
                        f"Failed to validate concept with Wikibase ID: {wikibase_id} with error: {e}"
                    )

        if not concepts and wikibase_ids:
            raise ConceptNotFoundError(wikibase_ids[0])

        return concepts

    def get_statements(self, wikibase_id: WikibaseID) -> list[dict]:
        """
        Get all statements for a Wikibase item

        :param str wikibase_id: The Wikibase ID of the item
        :return list[dict]: A list of all statements for the item
        """
        response = self.session.get(
            url=self.api_url,
            params={
                "action": "wbgetclaims",
                "format": "json",
                "entity": wikibase_id,
            },
        ).json()

        statements = response["claims"]
        return statements

    def add_statement(
        self, subject_id: str, predicate_id: str, value: str, summary: Optional[str]
    ) -> dict:
        """
        Add a statement to a Wikibase item

        :param str subject_id: The Wikibase ID of the subject entity
        :param str predicate_id: The Wikibase ID of the predicate property
        :param str value: Should take the form
            {"entity-type": "item", "id": object_id} or
            {"datatype": "string", "value": "string"}
        :param Optional[str] summary: A summary message for the edit
        :return dict: The response from the Wikibase API
        """
        data = {
            "action": "wbcreateclaim",
            "format": "json",
            "entity": subject_id,
            "property": predicate_id,
            "snaktype": "value",
            "value": json.dumps(value),
            "token": self.csrf_token,
            "bot": True,
        }

        if summary:
            data["summary"] = summary

        create_claim_response = self.session.post(url=self.api_url, data=data).json()
        return create_claim_response

    def add_alternative_labels(
        self,
        wikibase_id: WikibaseID,
        alternative_labels: list[str],
        language: str = "en",
    ):
        """Add a list of alternative labels to a Wikibase item, preserving existing ones"""
        existing_alternative_labels = self.get_concept(wikibase_id).alternative_labels

        all_alternative_labels = list(
            set(existing_alternative_labels + alternative_labels)
        )

        response = self.session.post(
            url=self.api_url,
            data={
                "action": "wbeditentity",
                "format": "json",
                "id": wikibase_id,
                "token": self.csrf_token,
                "data": json.dumps(
                    {
                        "aliases": {
                            language: [
                                {"language": language, "value": alias}
                                for alias in all_alternative_labels
                            ]
                        }
                    }
                ),
            },
        )
        response_data = response.json()
        if "error" in response_data:
            raise Exception(
                f"Error adding alternative labels: {response_data['error']}"
            )
        return response_data
