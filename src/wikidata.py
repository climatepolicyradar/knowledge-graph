import logging

import httpx
from pydantic import ValidationError

from src.concept import Concept
from src.exceptions import ConceptNotFoundError
from src.identifiers import WikibaseID

logger = logging.getLogger(__name__)


class WikidataSession:
    """A session for interacting with Wikidata's API"""

    WIKIDATA_API_BASE_URL = "https://www.wikidata.org/w/api.php"
    WIKIDATA_SPARQL_BASE_URL = "https://query.wikidata.org/sparql"

    def __init__(self):
        self.session = httpx.Client(headers={"Accept": "application/json"})

    def __repr__(self) -> str:
        """Return a string representation of the WikidataSession"""
        return f"<WikidataSession: {self.WIKIDATA_API_BASE_URL}>"

    def get_property_values(
        self, property_id: str, value_id: WikibaseID, inverse: bool = False
    ) -> list[WikibaseID]:
        """
        Get all entities related to a given entity through a specific property.

        Args:
            property_id (str): The Wikidata property ID (e.g., 'P31' for instance-of)
            value_id (WikibaseID): The Wikidata ID to search for
            inverse (bool): If True, search for entities that are the object of the property
                          If False, search for entities that are the subject of the property
                          Example: For P31 (instance-of):
                          - inverse=False: "What are instances of X?"
                          - inverse=True: "What is X an instance of?"

        Returns:
            list[WikibaseID]: A list of Wikidata IDs related through the property
        """
        # Put together a SPARQL query based on the direction of the relationship
        if inverse:
            query = f"""
            SELECT ?item ?itemLabel WHERE {{
              wd:{value_id} wdt:{property_id} ?item.
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            """
        else:
            query = f"""
            SELECT ?item ?itemLabel WHERE {{
              ?item wdt:{property_id} wd:{value_id}.
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            """

        try:
            response = self.session.get(
                self.WIKIDATA_SPARQL_BASE_URL,
                params={"query": query, "format": "json"},
            )
            response.raise_for_status()

            results = response.json()["results"]["bindings"]
            qids: list[WikibaseID] = []
            for result in results:
                item_id = result["item"]["value"].split("/")[-1]
                qids.append(WikibaseID(item_id))

            logger.debug(
                f"Found {len(qids)} entities related to {value_id} through property {property_id}"
                f" (inverse={inverse})"
            )
            return qids

        except httpx.HTTPError as e:
            logger.error(f"HTTP error during SPARQL query: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected response format: {e}")
            raise ValueError(
                f"Unexpected response format from Wikidata SPARQL endpoint: {e}"
            )
        except Exception as e:
            logger.error(f"Error during SPARQL query: {e}")
            raise

    def get_instances_of(self, wikidata_id: WikibaseID) -> list[WikibaseID]:
        """
        Get all instances of a given Wikidata entity

        This is a convenience wrapper around get_property_values for the common
        case of finding instances of a class using the P31 property.
        """
        return self.get_property_values("P31", wikidata_id)

    def get_parent_entities(self, wikidata_id: WikibaseID) -> list[WikibaseID]:
        """
        Get all parent entities of a given Wikidata entity

        This is a convenience wrapper around get_property_values for finding parent
        items of a given Wikidata entity using the P31 property.
        """
        return self.get_property_values("P31", wikidata_id, inverse=True)

    def get_concept(self, entity_id: WikibaseID) -> Concept:
        """Get a concept from Wikidata by its ID"""
        response = self.session.get(
            self.WIKIDATA_API_BASE_URL,
            params={"action": "wbgetentities", "ids": entity_id, "format": "json"},
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ConceptNotFoundError(wikibase_id=entity_id) from e

        entity_data = response.json().get("entities", {}).get(entity_id, {})

        preferred_label = entity_data.get("labels", {}).get("en", {}).get("value")
        if preferred_label is None:
            all_labels = entity_data.get("labels", {})
            if all_labels:
                preferred_label = next(iter(all_labels.values()), {}).get("value")

        description = entity_data.get("descriptions", {}).get("en", {}).get("value")

        non_english_preferred_labels = [
            label.get("value")
            for label in entity_data.get("labels", {}).values()
            if label.get("language") != "en"
        ]
        all_language_aliases = [
            alias.get("value")
            for group_of_aliases_by_language in entity_data.get("aliases", {}).values()
            for alias in group_of_aliases_by_language
        ]

        return Concept(
            wikibase_id=entity_id,
            preferred_label=preferred_label,
            description=description,
            alternative_labels=list(
                set(non_english_preferred_labels + all_language_aliases)
            ),
        )

    def get_concepts(self, wikibase_ids: list[WikibaseID]) -> list[Concept]:
        """Get multiple concepts from Wikidata"""
        concepts = []
        for wikibase_id in wikibase_ids:
            try:
                concept = self.get_concept(wikibase_id)
                concepts.append(concept)
            except (ConceptNotFoundError, ValidationError) as e:
                logger.warning(f"Failed to fetch concept {wikibase_id}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error fetching concept {wikibase_id}: {e}")

        return concepts
