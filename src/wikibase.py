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
        self.redirects = self.get_all_redirects()

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
        include_recursive_parent_concepts: bool = False,
    ) -> Concept:
        """
        Get a concept from Wikibase by its Wikibase ID

        :param WikibaseID wikibase_id: The Wikibase ID of the concept
        :param Optional[datetime] timestamp: The timestamp to fetch the concept at.
            If not provided, the latest version of the concept will be fetched.
        :param bool include_labels_from_subconcepts: Whether to include the labels
            from subconcepts in the concept
        :param bool include_recursive_parent_concepts: Whether to include the concept's
            complete ancestry of all parent concepts, recursively up the hierarchy
        :return Concept: The concept with the given Wikibase ID
        """
        # Resolve any redirects first
        wikibase_id = self._resolve_redirect(wikibase_id)

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

        # First get the pageid for this wikibase ID
        page_id_response = self.session.get(
            url=self.api_url,
            params={
                "action": "wbgetentities",
                "format": "json",
                "ids": wikibase_id,
                "props": "info",
            },
        ).json()

        page_id = str(
            page_id_response.get("entities", {}).get(wikibase_id, {}).get("pageid")
        )
        if not page_id:
            raise ConceptNotFoundError(wikibase_id)

        redirects = (
            page_id_response.get("entities", {})
            .get(wikibase_id, {})
            .get("redirects", [])
        )
        if redirects:
            logger.warning(
                f"Made a request to {self.base_url} for {wikibase_id} but was "
                f"redirected to {redirects.get('to')}"
            )
            wikibase_id = redirects.get("to")

        # Use the pageid to get the latest revision before the supplied timestamp
        # https://www.mediawiki.org/wiki/API:Revisions
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

        # Extract useful content from the revision response
        pages = revisions_response.get("query", {}).get("pages", {})
        if not pages:
            raise ConceptNotFoundError(wikibase_id)

        page = next(iter(pages.values()))
        revisions = page.get("revisions", [])
        if not revisions:
            raise RevisionNotFoundError(wikibase_id, timestamp)

        entity = json.loads(revisions[0].get("slots", {}).get("main", {}).get("*"))
        preferred_label = entity.get("labels", {}).get("en", {}).get("value", "")

        if isinstance(entity["aliases"], dict):
            alternative_labels = [
                alias.get("value")
                for alias in entity.get("aliases", {}).get("en", [])
                if alias.get("language") == "en"
            ]
        else:
            alternative_labels = []

        description = (
            entity.get("descriptions", {}).get("en", {}).get("value", "")
            if isinstance(entity["descriptions"], dict)
            else ""
        )

        preferred_label = entity.get("labels", {}).get("en", {}).get("value", "")

        if isinstance(entity["aliases"], dict):
            alternative_labels = [
                alias.get("value")
                for alias in entity.get("aliases", {}).get("en", [])
                if alias.get("language") == "en"
            ]
        else:
            alternative_labels = []

        description = (
            entity.get("descriptions", {}).get("en", {}).get("value", "")
            if isinstance(entity["descriptions"], dict)
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
                            concept.subconcept_of = concept.subconcept_of + [
                                self._resolve_redirect(value["id"])
                            ]
                        elif property_id == self.has_subconcept_property_id:
                            concept.has_subconcept = concept.has_subconcept + [
                                self._resolve_redirect(value["id"])
                            ]
                        elif property_id == self.related_concept_property_id:
                            concept.related_concepts = concept.related_concepts + [
                                self._resolve_redirect(value["id"])
                            ]
                        elif property_id == self.negative_labels_property_id:
                            concept.negative_labels = concept.negative_labels + [value]
                        elif property_id == self.definition_property_id:
                            concept.definition = value

        if include_labels_from_subconcepts:
            subconcepts = self.get_subconcepts(wikibase_id, recursive=True)

            # fetch all of the labels and negative_labels for all of the subconcepts
            # and the concept itself
            all_positive_labels = set(concept.all_labels)
            all_negative_labels = set(concept.negative_labels)
            for subconcept in subconcepts:
                all_positive_labels.update(subconcept.all_labels)
                all_negative_labels.update(subconcept.negative_labels)

            concept.alternative_labels = list(all_positive_labels)
            concept.negative_labels = list(all_negative_labels)

        if include_recursive_parent_concepts:
            concept.recursive_parent_concepts = self.get_recursive_parent_concepts(
                wikibase_id
            )

        return concept

    def get_recursive_parent_concepts(
        self, wikibase_id: WikibaseID
    ) -> list[WikibaseID]:
        """Fetch the complete ancestry of a concept, recursively up the hierarchy"""
        # Resolve any redirects first
        wikibase_id = self._resolve_redirect(wikibase_id)

        response = self.session.get(
            url=self.api_url,
            params={
                "action": "wbgetentities",
                "format": "json",
                "ids": wikibase_id,
                "props": "claims",
            },
        ).json()

        entity = response["entities"][wikibase_id]
        recursive_parent_concepts = []
        if "claims" in entity:
            for claim in entity["claims"].values():
                for statement in claim:
                    if statement["mainsnak"]["snaktype"] == "value":
                        property_id = statement["mainsnak"]["property"]
                        value = statement["mainsnak"]["datavalue"]["value"]
                        if property_id == self.subconcept_of_property_id:
                            resolved_id = self._resolve_redirect(value["id"])
                            recursive_parent_concepts.append(resolved_id)
                            recursive_parent_concepts.extend(
                                self.get_recursive_parent_concepts(resolved_id)
                            )

        # Duplicates can occur in the concept's list of recursive parents, so they need
        # to be removed before returning
        unique_recursive_parent_concepts = list(set(recursive_parent_concepts))
        return unique_recursive_parent_concepts

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

    def get_concept_ids(self) -> list[WikibaseID]:
        """
        Get concept ids from Wikibase.

        :return list[WikibaseID]: The concept ids, e.g ["Q123", "Q456"]
        """
        pages = self._get_pages(extra_params={"apfilterredir": "nonredirects"})
        return [page["title"].replace("Item:", "") for page in pages]

    def get_all_redirects(self, batch_size: int = 50) -> dict[WikibaseID, WikibaseID]:
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
                    redirects[wikibase_id] = entity["redirects"]["to"]

        return redirects

    def get_concepts(
        self,
        limit: Optional[int] = None,
        wikibase_ids: Optional[list[WikibaseID]] = None,
    ) -> list[Concept]:
        """
        Get concepts from Wikibase, optionally specified by their Wikibase IDs

        :param Optional[int] limit: The maximum number of concepts to fetch
        :param list[WikibaseID] wikibase_ids: The Wikibase IDs of the concepts
        :return list[Concept]: The concepts, optionally with the given Wikibase IDs
        """
        if not wikibase_ids:
            wikibase_ids = self.get_concept_ids()

        concepts = []
        for wikibase_id in wikibase_ids[:limit]:
            try:
                concept = self.get_concept(wikibase_id)
                concepts.append(concept)
            except ValidationError as e:
                logger.warning(
                    f"Failed to fetch concept with Wikibase ID: {wikibase_id} with error: {e}"
                )

        return concepts

    def get_subconcepts(
        self, wikibase_id: WikibaseID, recursive: bool = True
    ) -> list[Concept]:
        """
        Get all subconcepts of a concept

        :param str wikibase_id: The Wikibase ID of the concept
        :param bool recursive: Whether to get subconcepts recursively
        :return list[Concept]: A list of all subconcepts of the concept
        """
        response = self.session.get(
            url=self.api_url,
            params={
                "action": "wbgetentities",
                "format": "json",
                "ids": wikibase_id,
                "props": "claims",
            },
        ).json()

        entity = response["entities"][wikibase_id]
        subconcepts = []
        if "claims" in entity:
            for claim in entity["claims"].values():
                for statement in claim:
                    if statement["mainsnak"]["snaktype"] == "value":
                        property_id = statement["mainsnak"]["property"]
                        value = statement["mainsnak"]["datavalue"]["value"]
                        if property_id == self.has_subconcept_property_id:
                            subconcepts.append(self.get_concept(value["id"]))
                            if recursive:
                                subconcepts.extend(self.get_subconcepts(value["id"]))

        return subconcepts

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

    def get_alternative_labels(
        self, wikibase_id: WikibaseID, language: str = "en"
    ) -> list[str]:
        """
        Get all alternative labels for a Wikibase item in a specific language

        :param WikibaseID wikibase_id: The Wikibase ID of the item
        :param str language: The language code for the alternative labels
        :return list[str]: A list of alternative label strings
        """
        response = self.session.get(
            url=self.api_url,
            params={
                "action": "wbgetentities",
                "format": "json",
                "ids": wikibase_id,
                "props": "aliases",
            },
        ).json()

        aliases = (
            response.get("entities", {})
            .get(wikibase_id, {})
            .get("aliases", {})
            .get(language, [])
        )
        if aliases:
            return [alias["value"] for alias in aliases]
        return []

    def add_alternative_labels(
        self,
        wikibase_id: WikibaseID,
        alternative_labels: list[str],
        language: str = "en",
    ):
        """Add a list of alternative labels to a Wikibase item, preserving existing ones"""
        existing_alternative_labels = self.get_alternative_labels(wikibase_id, language)

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
