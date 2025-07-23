import logging
from typing import Optional

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

    INSTANCE_OF_PROPERTY = "P31"

    def __init__(self):
        """Initialize a WikidataSession with appropriate timeouts and headers"""
        self.session = httpx.Client(
            headers={"Accept": "application/json"},
            timeout=300.0,
        )

    def __repr__(self) -> str:
        """Return a string representation of the WikidataSession"""
        return f"<WikidataSession: {self.WIKIDATA_API_BASE_URL}>"

    def get_property_values(
        self,
        property_id: str,
        entity_id: WikibaseID,
        inverse: bool = False,
        limit: Optional[int] = None,
        page_size: int = 1000,
        offset: int = 0,
    ) -> list[WikibaseID]:
        """
        Get all entities related to a given entity through a specific property.

        :param str property_id: The Wikidata property ID (e.g., 'P31' for "instance of")
        :param WikibaseID entity_id: The Wikidata ID to search for
        :param bool inverse: If True, search for entities that are the object of the property.
            If False, search for entities that are the subject of the property.
            Example: For P31 (instance of):
            - inverse=False: "What are instances of X?"
            - inverse=True: "What is X an instance of?"
        :param Optional[int] limit: Maximum number of results to return in total. If
            None, all results will be returned.
        :param int page_size: Maximum number of results to return per query to the
            wikidata SPARQL endpoint
        :param int offset: Offset for pagination
        :return list[WikibaseID]: A list of Wikidata IDs related through the property
        """
        # Adjust page_size if it would exceed the total limit
        current_page_size = page_size
        if limit is not None:
            remaining = limit - offset
            if remaining <= 0:
                return []
            current_page_size = min(page_size, remaining)

        # Determine the subject-predicate-object pattern based on the direction specified
        # by the user
        triple_pattern = (
            f"wd:{entity_id} wdt:{property_id} ?item"
            if inverse
            else f"?item wdt:{property_id} wd:{entity_id}"
        )

        # Construct the full SPARQL query
        query = f"""
        SELECT ?item ?itemLabel WHERE {{
          {triple_pattern}.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {current_page_size}
        OFFSET {offset}
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
                f"Found {len(qids)} entities related to {entity_id} through property {property_id} "
                f"(inverse={inverse}, offset={offset}, page_size={page_size})"
            )

            # Check whether we've got a full page of results and if we're still under the
            # user-specified limit. If so, fetch the next page of results
            got_a_full_page_of_results = len(qids) == current_page_size
            still_under_the_limit = limit is None or (offset + len(qids)) < limit

            if got_a_full_page_of_results and still_under_the_limit:
                next_page = self.get_property_values(
                    property_id,
                    entity_id,
                    inverse,
                    limit,
                    page_size,
                    offset + len(qids),
                )
                qids.extend(next_page)

            return qids

        except httpx.HTTPError as e:
            logger.error(
                f"HTTP error during SPARQL query for {entity_id} with property {property_id}: {e}"
            )
            raise
        except KeyError as e:
            logger.error(
                f"Unexpected response format for {entity_id} with property {property_id}: {e}"
            )
            raise ValueError(
                f"Unexpected response format from Wikidata SPARQL endpoint: {e}"
            )
        except Exception as e:
            logger.error(
                f"Error during SPARQL query for {entity_id} with property {property_id}: {e}"
            )
            raise

    def get_instances_of(
        self, entity_id: WikibaseID, limit: Optional[int] = None
    ) -> list[WikibaseID]:
        """
        Get all instances of an entity, based on the "instance of" property

        :param WikibaseID entity_id: The Wikidata ID to find instances of
        :param Optional[int] limit: Maximum number of results to return
        :return list[WikibaseID]: A list of Wikidata IDs that are instances of the entity
        """
        return self.get_property_values(
            self.INSTANCE_OF_PROPERTY, entity_id, limit=limit
        )

    def get_instance_of_values(
        self, entity_id: WikibaseID, limit: Optional[int] = None
    ) -> list[WikibaseID]:
        """
        Get all entities that this entity is an instance of.

        For example, if entity "London (Q84)" is an instance of "city (Q515)" (among
        other things), then get_instance_of_values("Q84") would return ["Q515", ...].

        :param WikibaseID entity_id: The Wikidata ID to find instance-of values for
        :param Optional[int] limit: Maximum number of results to return
        :return list[WikibaseID]: A list of Wikidata IDs that this entity is an instance of
        """
        return self.get_property_values(
            self.INSTANCE_OF_PROPERTY, entity_id, inverse=True, limit=limit
        )

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
        try:
            entity_data = response.json()["entities"][str(entity_id)]
        except KeyError as e:
            raise ValueError(f"Could not parse data from {response.url}") from e

        preferred_label = entity_data.get("labels", {}).get("en", {}).get("value")
        if preferred_label is None:
            if all_labels := entity_data.get("labels", {}):
                preferred_label = next(iter(all_labels.values()), {}).get("value")

        description = entity_data.get("descriptions", {}).get("en", {}).get("value")

        aliases = [
            alias.get("value") for alias in entity_data.get("aliases", {}).get("en", [])
        ]

        concept = Concept(
            wikibase_id=entity_id,
            preferred_label=preferred_label,
            description=description,
            alternative_labels=aliases,
        )
        return concept

    def get_concepts(
        self, entity_ids: list[WikibaseID], limit: Optional[int] = None
    ) -> list[Concept]:
        """
        Get multiple concepts from Wikidata.

        :param list[WikibaseID] entity_ids: The Wikidata IDs to fetch concepts for
        :param Optional[int] limit: Maximum number of concepts to return
        :return list[Concept]: A list of concepts
        """
        concepts = []
        for entity_id in entity_ids[:limit]:
            try:
                concept = self.get_concept(entity_id)
                concepts.append(concept)
            except (ConceptNotFoundError, ValidationError) as e:
                logger.warning(f"Failed to fetch concept {entity_id}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error fetching concept {entity_id}: {e}")

        return concepts
