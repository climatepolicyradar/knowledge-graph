import asyncio
import logging
from typing import Optional

import httpx
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from knowledge_graph.async_bridge import async_to_sync
from knowledge_graph.concept import Concept
from knowledge_graph.exceptions import ConceptNotFoundError
from knowledge_graph.identifiers import WikibaseID

logger = logging.getLogger(__name__)


MAX_RETRIES = 5
RETRY_INITIAL_WAIT = 1.0
RETRY_MAX_WAIT = 60.0


class WikidataSession:
    """Async-first session for interacting with Wikidata's API"""

    # API endpoints
    base_url = "https://www.wikidata.org"
    api_url = base_url + "/w/api.php"
    sparql_url = "https://query.wikidata.org/sparql"

    # Magic numbers
    DEFAULT_TIMEOUT = 30
    DEFAULT_BATCH_SIZE = 50
    MAX_CONCURRENT_REQUESTS = 3  # Limit concurrent requests to avoid rate limiting
    REQUEST_DELAY_SECONDS = 0.5  # Small delay between requests to be gentle on server
    SPARQL_PAGE_SIZE = 1000  # Default page size for SPARQL queries

    INSTANCE_OF_PROPERTY = "P31"

    def __init__(self):
        """Initialize a WikidataSession with appropriate timeouts and headers"""
        # These will be initialized on first use
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    def __repr__(self) -> str:
        """Return a string representation of the WikidataSession"""
        return f"<WikidataSession: {self.api_url}>"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get the HTTP client, initializing session on first use"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"Accept": "application/json"},
                timeout=self.DEFAULT_TIMEOUT,
            )

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)

        return self._client

    async def close(self):
        """Close the async client"""
        if self._client:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential_jitter(initial=RETRY_INITIAL_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
    )
    async def _get_property_values_async(
        self,
        property_id: str,
        entity_id: WikibaseID,
        inverse: bool = False,
        limit: Optional[int] = None,
        page_size: int = SPARQL_PAGE_SIZE,
        offset: int = 0,
    ) -> list[WikibaseID]:
        """
        Async version: Get all entities related to a given entity through a specific property.

        :param str property_id: The Wikidata property ID (e.g., 'P31' for "instance of")
        :param WikibaseID entity_id: The Wikidata ID to search for
        :param bool inverse: If True, search for entities that are the object of the property.
            If False, search for entities that are the subject of the property.
        :param Optional[int] limit: Maximum number of results to return in total.
        :param int page_size: Maximum number of results to return per query
        :param int offset: Offset for pagination
        :return list[WikibaseID]: A list of Wikidata IDs related through the property
        """
        client = await self._get_client()

        # Use semaphore to limit concurrent requests
        assert self._semaphore is not None, "Semaphore should be initialized"
        async with self._semaphore:
            # Adjust page_size if it would exceed the total limit
            current_page_size = page_size
            if limit is not None:
                remaining = limit - offset
                if remaining <= 0:
                    return []
                current_page_size = min(page_size, remaining)

            # Determine the subject-predicate-object pattern based on the direction specified
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
                response = await client.get(
                    self.sparql_url,
                    params={"query": query, "format": "json"},
                )
                response.raise_for_status()

                results = response.json()["results"]["bindings"]
                qids: list[WikibaseID] = []
                for result in results:
                    item_id = result["item"]["value"].split("/")[-1]
                    qids.append(WikibaseID(item_id))

                logger.debug(
                    "Found %d entities related to %s through property %s (inverse=%s, offset=%d, page_size=%d)",
                    len(qids),
                    entity_id,
                    property_id,
                    inverse,
                    offset,
                    page_size,
                )

                # Check whether we've got a full page of results and if we're still under the
                # user-specified limit. If so, fetch the next page of results
                got_a_full_page_of_results = len(qids) == current_page_size
                still_under_the_limit = limit is None or (offset + len(qids)) < limit

                if got_a_full_page_of_results and still_under_the_limit:
                    next_page = await self._get_property_values_async(
                        property_id,
                        entity_id,
                        inverse,
                        limit,
                        page_size,
                        offset + len(qids),
                    )
                    qids.extend(next_page)

                # Small delay to be gentle on the server
                await asyncio.sleep(self.REQUEST_DELAY_SECONDS)

                return qids

            except httpx.HTTPError as e:
                logger.error(
                    "HTTP error during SPARQL query for %s with property %s: %s",
                    entity_id,
                    property_id,
                    e,
                )
                raise
            except KeyError as e:
                logger.error(
                    "Unexpected response format for %s with property %s: %s",
                    entity_id,
                    property_id,
                    e,
                )
                raise ValueError(
                    f"Unexpected response format from Wikidata SPARQL endpoint: {e}"
                ) from e

    @async_to_sync
    async def get_property_values(
        self,
        property_id: str,
        entity_id: WikibaseID,
        inverse: bool = False,
        limit: Optional[int] = None,
        page_size: int = SPARQL_PAGE_SIZE,
        offset: int = 0,
    ) -> list[WikibaseID]:
        """
        Get all entities related to a given entity through a specific property.

        :param str property_id: The Wikidata property ID (e.g., 'P31' for "instance of")
        :param WikibaseID entity_id: The Wikidata ID to search for
        :param bool inverse: If True, search for entities that are the object of the property.
            If False, search for entities that are the subject of the property.
        :param Optional[int] limit: Maximum number of results to return in total.
        :param int page_size: Maximum number of results to return per query
        :param int offset: Offset for pagination
        :return list[WikibaseID]: A list of Wikidata IDs related through the property
        """
        return await self._get_property_values_async(
            property_id, entity_id, inverse, limit, page_size, offset
        )

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

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential_jitter(initial=RETRY_INITIAL_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
    )
    async def _get_concept_async(self, entity_id: WikibaseID) -> Concept:
        """Async version: Get a concept from Wikidata by its ID"""
        client = await self._get_client()

        # Use semaphore to limit concurrent requests
        assert self._semaphore is not None, "Semaphore should be initialized"
        async with self._semaphore:
            try:
                response = await client.get(
                    self.api_url,
                    params={
                        "action": "wbgetentities",
                        "ids": entity_id,
                        "format": "json",
                    },
                )
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    raise ConceptNotFoundError(wikibase_id=entity_id) from e

                try:
                    entity_data = response.json()["entities"][str(entity_id)]
                except KeyError as e:
                    raise ValueError(f"Could not parse data from {response.url}") from e

                preferred_label = (
                    entity_data.get("labels", {}).get("en", {}).get("value")
                )
                if preferred_label is None:
                    if all_labels := entity_data.get("labels", {}):
                        preferred_label = next(iter(all_labels.values()), {}).get(
                            "value"
                        )

                description = (
                    entity_data.get("descriptions", {}).get("en", {}).get("value")
                )

                # Collect aliases from English and multilingual sources
                aliases_data = entity_data.get("aliases", {})
                aliases = list(
                    set(
                        alias.get("value")
                        for alias in (
                            aliases_data.get("en", [])  # english
                            + aliases_data.get("mul", [])  # multilingual
                        )
                    )
                )

                concept = Concept(
                    wikibase_id=entity_id,
                    preferred_label=preferred_label or "Unknown",  # type: ignore[arg-type]
                    description=description,
                    alternative_labels=aliases,
                )

                # Small delay to be gentle on the server
                await asyncio.sleep(self.REQUEST_DELAY_SECONDS)

                return concept

            except httpx.HTTPError as e:
                logger.error("HTTP error fetching concept %s: %s", entity_id, e)
                raise
            except KeyError as e:
                logger.error(
                    "Unexpected response format for concept %s: %s", entity_id, e
                )
                raise ValueError("Could not parse data from Wikidata API") from e

    @async_to_sync
    async def get_concept(self, entity_id: WikibaseID) -> Concept:
        """Get a concept from Wikidata by its ID"""
        return await self._get_concept_async(entity_id)

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential_jitter(initial=RETRY_INITIAL_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
    )
    async def _get_concepts_async(
        self, entity_ids: list[WikibaseID], limit: Optional[int] = None
    ) -> list[Concept]:
        """
        Async version: Get multiple concepts from Wikidata with concurrent fetching.

        :param list[WikibaseID] entity_ids: The Wikidata IDs to fetch concepts for
        :param Optional[int] limit: Maximum number of concepts to return
        :return list[Concept]: A list of concepts
        """
        if limit:
            entity_ids = entity_ids[:limit]

        if not entity_ids:
            return []

        logger.info(
            "Starting to fetch %d concepts with rate limiting (max %d concurrent)",
            len(entity_ids),
            self.MAX_CONCURRENT_REQUESTS,
        )

        # Create tasks for concurrent concept fetching
        tasks = [self._get_concept_async(entity_id) for entity_id in entity_ids]

        # Execute all concept fetches concurrently
        concepts = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results and exceptions
        successful_concepts = []
        failed_concepts = 0
        for i, concept in enumerate(concepts):
            if isinstance(concept, Concept):
                successful_concepts.append(concept)
            elif isinstance(concept, Exception):
                entity_id = entity_ids[i]
                if isinstance(concept, (ConceptNotFoundError, ValidationError)):
                    logger.warning("Failed to fetch concept %s: %s", entity_id, concept)
                else:
                    logger.error(
                        "Unexpected error fetching concept %s: %s", entity_id, concept
                    )
                failed_concepts += 1

        logger.info(
            "Concept fetching complete: %d successful, %d failed out of %d total",
            len(successful_concepts),
            failed_concepts,
            len(entity_ids),
        )

        return successful_concepts

    @async_to_sync
    async def get_concepts(
        self, entity_ids: list[WikibaseID], limit: Optional[int] = None
    ) -> list[Concept]:
        """
        Get multiple concepts from Wikidata.

        :param list[WikibaseID] entity_ids: The Wikidata IDs to fetch concepts for
        :param Optional[int] limit: Maximum number of concepts to return
        :return list[Concept]: A list of concepts
        """
        return await self._get_concepts_async(entity_ids, limit)
