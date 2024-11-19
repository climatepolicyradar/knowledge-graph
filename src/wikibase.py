import json
import os
from logging import getLogger
from typing import Dict, Optional

import dotenv
import httpx

from src.concept import Concept, WikibaseID

logger = getLogger(__name__)

dotenv.load_dotenv()


class WikibaseSession:
    """A session for interacting with Wikibase"""

    session = httpx.Client()
    username = os.getenv("WIKIBASE_USERNAME")
    password = os.getenv("WIKIBASE_PASSWORD")
    base_url = os.getenv("WIKIBASE_URL")
    api_url = f"{base_url}/w/api.php"

    has_subconcept_property_id = os.getenv("WIKIBASE_HAS_SUBCONCEPT_PROPERTY_ID")
    subconcept_of_property_id = os.getenv("WIKIBASE_SUBCONCEPT_OF_PROPERTY_ID")
    related_concept_property_id = os.getenv("WIKIBASE_RELATED_CONCEPT_PROPERTY_ID")
    negative_labels_property_id = os.getenv("WIKIBASE_NEGATIVE_LABELS_PROPERTY_ID")
    definition_property_id = os.getenv("WIKIBASE_DEFINITION_PROPERTY_ID")

    def __init__(self):
        """Log in to Wikibase and get a CSRF token"""
        if not self.username or not self.password:
            raise ValueError(
                "WIKIBASE_USERNAME and WIKIBASE_PASSWORD environment variables must be set"
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

    def get_concept(self, wikibase_id: WikibaseID) -> Concept:
        """
        Get a concept from Wikibase by its Wikibase ID

        :param WikibaseID wikibase_id: The Wikibase ID of the concept
        :return Concept: The concept with the given Wikibase ID
        """
        response = self.session.get(
            url=self.api_url,
            params={
                "action": "wbgetentities",
                "format": "json",
                "ids": wikibase_id,
            },
        ).json()

        entity = response["entities"][wikibase_id]

        concept = Concept(
            preferred_label=entity.get("labels", {}).get("en", {}).get("value", ""),
            alternative_labels=[
                alias.get("value") for alias in entity.get("aliases", {}).get("en", [])
            ],
            description=entity.get("descriptions", {}).get("en", {}).get("value", ""),
            wikibase_id=wikibase_id,
        )

        if "claims" in entity:
            for claim in entity["claims"].values():
                for statement in claim:
                    if statement["mainsnak"]["snaktype"] == "value":
                        property_id = statement["mainsnak"]["property"]
                        value = statement["mainsnak"]["datavalue"]["value"]
                        if property_id == self.subconcept_of_property_id:
                            concept.subconcept_of.append(value["id"])
                        elif property_id == self.has_subconcept_property_id:
                            concept.has_subconcept.append(value["id"])
                        elif property_id == self.related_concept_property_id:
                            concept.related_concepts.append(value["id"])
                        elif property_id == self.negative_labels_property_id:
                            concept.negative_labels.append(value)
                        elif property_id == self.definition_property_id:
                            concept.definition = value

        return concept

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
            # NOTE: Because this call has a max `aplimit` of 5000, this implementation will
            # work up to a limit of 5000 item pages in the concept store. Beyond that, we'll
            # need to start paginating over the results
            response = self.session.get(
                url=self.api_url,
                params={
                    "action": "query",
                    "format": "json",
                    "list": "allpages",
                    "apnamespace": 120,
                    "aplimit": limit or "max",
                    "apfilterredir": "nonredirects",  # Only fetch non-redirect pages
                },
            ).json()
            wikibase_ids = [
                page["title"].replace("Item:", "")
                for page in response["query"]["allpages"]
            ]

        concepts = []
        for wikibase_id in wikibase_ids:
            concept = self.get_concept(wikibase_id)
            concepts.append(concept)

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
