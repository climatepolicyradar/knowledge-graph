import json
import os
from datetime import datetime, timezone
from logging import getLogger
from typing import Dict, Optional

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

    has_subconcept_property_id = os.getenv("WIKIBASE_HAS_SUBCONCEPT_PROPERTY_ID")
    subconcept_of_property_id = os.getenv("WIKIBASE_SUBCONCEPT_OF_PROPERTY_ID")
    related_concept_property_id = os.getenv("WIKIBASE_RELATED_CONCEPT_PROPERTY_ID")
    negative_labels_property_id = os.getenv("WIKIBASE_NEGATIVE_LABELS_PROPERTY_ID")
    definition_property_id = os.getenv("WIKIBASE_DEFINITION_PROPERTY_ID")

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

    def get_all_properties(self) -> list[Dict[str, str]]:
        """
        Get all property IDs from the Wikibase instance

        :return list[Dict[str, str]]: A list of all property IDs (and their
        corresponding page_ids) in the Wikibase instance
        """
        all_properties_response = self.session.get(
            url=self.api_url,
            params={
                "action": "query",
                "format": "json",
                "list": "allpages",
                "apnamespace": "122",
                "aplimit": "max",
            },
        ).json()
        all_properties = [
            {"p_id": page["title"].replace("Property:", ""), "page_id": page["pageid"]}
            for page in all_properties_response["query"]["allpages"]
        ]
        sorted_properties = sorted(all_properties, key=lambda x: int(x["p_id"][1:]))
        return sorted_properties

    def get_concept(
        self,
        wikibase_id: WikibaseID,
        timestamp: Optional[datetime] = None,
        include_labels_from_subconcepts: bool = False,
    ) -> Concept:
        """
        Get a concept from Wikibase by its Wikibase ID

        :param WikibaseID wikibase_id: The Wikibase ID of the concept
        :param Optional[datetime] timestamp: The timestamp to fetch the concept at.
            If not provided, the latest version of the concept will be fetched.
        :param bool include_labels_from_subconcepts: Whether to include the labels
            from subconcepts in the concept
        :return Concept: The concept with the given Wikibase ID
        """

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
                                value["id"]
                            ]
                        elif property_id == self.has_subconcept_property_id:
                            concept.has_subconcept = concept.has_subconcept + [
                                value["id"]
                            ]
                        elif property_id == self.related_concept_property_id:
                            concept.related_concepts = concept.related_concepts + [
                                value["id"]
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

        return concept

    def get_concept_ids(self) -> list[WikibaseID]:
        """
        Get concept ids from Wikibase.

        :return list[WikibaseID]: The concept ids, e.g ["Q123", "Q456"]
        """
        PAGE_REQUEST_SIZE = 500
        # An extra precaution against infinite loops, suitable up to 1M concepts (500*2000)
        MAX_PAGE_REQUESTS = 2000

        params = {
            "action": "query",
            "format": "json",
            "list": "allpages",  # See https://www.mediawiki.org/wiki/API:Allpages
            "apnamespace": 120,
            "aplimit": PAGE_REQUEST_SIZE,
            "apfilterredir": "nonredirects",  # Only fetch non-redirect pages
        }

        wikibase_ids = []
        for i in range(MAX_PAGE_REQUESTS):
            # Get and process a page into wikibase_ids
            response = self.session.get(
                url=self.api_url,
                params=params,
            ).json()
            wikibase_ids.extend(
                [
                    page["title"].replace("Item:", "")
                    for page in response["query"]["allpages"]
                ]
            )

            #  Handle errors / warnings, see:
            # https://www.mediawiki.org/wiki/API:Continue#Example_3:_Python_code_for_iterating_through_all_results
            if "error" in response:
                raise HTTPError(response["error"])
            if "warnings" in response:
                logger.warning(response["warnings"])

            # Handle pagination if there are more pages
            if continue_params := response.get("continue"):
                params.update(continue_params)
                logger.info(
                    f"Retrieved {len(wikibase_ids)} ids after page iteration {i}"
                )
            else:
                break

        return wikibase_ids

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
