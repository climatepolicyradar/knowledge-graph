import json
import os
from functools import wraps
from logging import getLogger
from typing import List, Optional, Union

import dotenv
import httpx

from src.concept import Concept

logger = getLogger(__name__)

dotenv.load_dotenv()


def update_progress_bar(func):
    """
    Update a progress bar

    Decorator to update a progress bar after a function call, if `progress_bar` is
    passed as a keyword argument
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if progress_bar := kwargs.get("progress_bar"):
            progress_bar.update(1)
        return func(*args, **kwargs)

    return wrapper


class WikibaseSession:
    """A session for interacting with Wikibase"""

    session = httpx.Client()
    username = os.getenv("WIKIBASE_USERNAME")
    password = os.getenv("WIKIBASE_PASSWORD")
    api_url = f"{os.getenv('WIKIBASE_URL')}/w/api.php"

    has_subconcept_property_id = os.getenv("WIKIBASE_HAS_SUBCONCEPT_PROPERTY_ID")
    subconcept_of_property_id = os.getenv("WIKIBASE_SUBCONCEPT_OF_PROPERTY_ID")
    related_concept_property_id = os.getenv("WIKIBASE_RELATED_CONCEPT_PROPERTY_ID")

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

    @update_progress_bar
    def create_concept(
        self, concept: Concept, subconcept_of: Optional[str] = None, **kwargs
    ) -> Concept:
        """
        Create a concept in Wikibase, with all of its subconcepts and relationships

        :param Concept concept: The concept to create
        :param Optional[str] subconcept_of: The Wikibase ID of the parent concept
        :return Concept: The original concept with a newly populated wikibase_id property
        """
        new_item_response = self.session.post(
            url=self.api_url,
            data={
                "action": "wbeditentity",
                "format": "json",
                "new": "item",
                "token": self.csrf_token,
                "data": json.dumps(
                    {
                        "labels": {
                            "en": {"language": "en", "value": concept.preferred_label}
                        },
                        "aliases": {
                            "en": [
                                {"language": "en", "value": alias}
                                for alias in concept.alternative_labels
                            ]
                        },
                    }
                ),
            },
        ).json()

        concept.wikibase_id = new_item_response["entity"]["id"]
        logger.debug(
            f"Created concept {concept.preferred_label} with ID {concept.wikibase_id}"
        )

        if subconcept_of:
            self.session.post(
                url=self.api_url,
                data={
                    "action": "wbcreateclaim",
                    "format": "json",
                    "entity": concept.wikibase_id,
                    "property": self.subconcept_of_property_id,
                    "snaktype": "value",
                    "value": json.dumps({"entity-type": "item", "id": subconcept_of}),
                    "token": self.csrf_token,
                },
            ).json()
            logger.debug(
                f"Created 'subconcept of' relationship between {concept.preferred_label} and {subconcept_of}"
            )

            self.session.post(
                url=self.api_url,
                data={
                    "action": "wbcreateclaim",
                    "format": "json",
                    "entity": subconcept_of,
                    "property": self.has_subconcept_property_id,
                    "snaktype": "value",
                    "value": json.dumps(
                        {"entity-type": "item", "id": concept.wikibase_id}
                    ),
                    "token": self.csrf_token,
                },
            ).json()

            logger.debug(
                f"Created 'has subconcept' relationship between {subconcept_of} and {concept.preferred_label}"
            )

        if concept.related_concepts:
            for related_concept in concept.related_concepts:
                self.session.post(
                    url=self.api_url,
                    data={
                        "action": "wbcreateclaim",
                        "format": "json",
                        "entity": concept.wikibase_id,
                        "property": self.related_concept_property_id,
                        "snaktype": "value",
                        "value": json.dumps(
                            {"entity-type": "item", "id": related_concept.wikibase_id}
                        ),
                        "token": self.csrf_token,
                    },
                ).json()

                self.session.post(
                    url=self.api_url,
                    data={
                        "action": "wbcreateclaim",
                        "format": "json",
                        "entity": related_concept.wikibase_id,
                        "property": self.related_concept_property_id,
                        "snaktype": "value",
                        "value": json.dumps(
                            {"entity-type": "item", "id": concept.wikibase_id}
                        ),
                        "token": self.csrf_token,
                    },
                ).json()

                logger.debug(
                    f"Created 'related concept' relationship between {concept.preferred_label} and {related_concept.preferred_label}"
                )

        for subconcept in concept.subconcepts:
            self.create_concept(subconcept, subconcept_of=concept.wikibase_id, **kwargs)

        return concept

    def get_all_item_ids(self) -> List[str]:
        """
        Get all item IDs from the Wikibase instance

        NOTE: Because this call has a max `aplimit` of 5000, this implementation will
        work up to a limit of 5000 item pages in the concept store. Beyond that, we'll
        need to start paginating over the results

        :return List[str]: A list of all item IDs in the Wikibase instance
        """
        all_pages_response = self.session.get(
            url=self.api_url,
            params={
                "action": "query",
                "format": "json",
                "list": "allpages",
                "apnamespace": "120",
                "aplimit": "max",
            },
        ).json()
        all_item_ids = [
            page["title"].replace("Item:", "")
            for page in all_pages_response["query"]["allpages"]
        ]
        return all_item_ids

    def get_concepts(self, wikibase_ids: Union[str, List[str]]) -> List[Concept]:
        """
        Get concepts from Wikibase by their Wikibase IDs

        :param Union[str  |  List[str]] wikibase_ids: A Wikibase ID or a list of Wikibase IDs
        :return Concept: The concepts with the given Wikibase IDs
        """
        if isinstance(wikibase_ids, str):
            wikibase_ids = [wikibase_ids]

        concepts = []
        for wikibase_id in wikibase_ids:
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
                preferred_label=entity["labels"]["en"]["value"],
                alternative_labels=[
                    alias["value"] for alias in entity["aliases"]["en"]
                ],
                wikibase_id=wikibase_id,
            )

            concepts.append(concept)

        return concepts

    def list_concepts(self) -> List[Concept]:
        """
        List all concepts in Wikibase

        :return List[Concept]: A list of all concepts in the Wikibase instance
        """
        response = self.session.get(
            url=self.api_url,
            params={
                "action": "query",
                "format": "json",
                "list": "allpages",
                "aplimit": 500,
            },
        ).json()

        concepts = []
        for page in response["query"]["allpages"]:
            wikibase_id = page["title"]
            concepts.append(self.get_concepts(wikibase_id)[0])

        return concepts

    def add_statement(
        self, subject_id: str, predicate_id: str, object_id: str, summary: Optional[str]
    ) -> dict:
        """
        Add a statement to a Wikibase item

        :param str subject_id: The Wikibase ID of the subject entity
        :param str predicate_id: The Wikibase ID of the predicate property
        :param str object_id: The Wikibase ID of the object entity
        :param Optional[str] summary: A summary message for the edit
        :return dict: The response from the Wikibase API
        """
        data = {
            "action": "wbcreateclaim",
            "format": "json",
            "entity": subject_id,
            "property": predicate_id,
            "snaktype": "value",
            "value": json.dumps({"entity-type": "item", "id": object_id}),
            "token": self.csrf_token,
            "bot": True,
        }

        if summary:
            data["summary"] = summary

        create_claim_response = self.session.post(url=self.api_url, data=data).json()
        return create_claim_response

    def remove_statement(
        self, subject_id: str, predicate_id: str, object_id: str, summary: Optional[str]
    ) -> None:
        """
        Remove a statement from a Wikibase entity

        :param str subject_id: The Wikibase ID of the subject entity
        :param str predicate_id: The Wikibase ID of the predicate property
        :param str object_id: The Wikibase ID of the object entity
        :param Optional[str] summary: A summary message for the edit
        """
        raise NotImplementedError
