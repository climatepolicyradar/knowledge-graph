import asyncio
import functools
import json
import os
from datetime import datetime, timezone
from logging import getLogger
from typing import Any, Callable, Coroutine, Optional, TypeVar, cast

import dotenv
import html2text
import httpx
from httpx import HTTPError, HTTPStatusError, RequestError
from more_itertools import chunked
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from knowledge_graph.concept import Concept
from knowledge_graph.exceptions import ConceptNotFoundError, RevisionNotFoundError
from knowledge_graph.identifiers import WikibaseID

logger = getLogger(__name__)
dotenv.load_dotenv()

T = TypeVar("T")
MAX_RETRIES = 5
RETRY_INITIAL_WAIT = 1.0
RETRY_MAX_WAIT = 60.0


def async_to_sync(
    async_func: Callable[..., Coroutine[None, None, T]],
) -> Callable[..., T]:
    """
    Decorator that converts async methods to synchronous interface

    This decorator wraps async methods to provide a synchronous interface by
    automatically managing the event loop. It creates a new event loop if none exists,
    or raises an error if called from within an existing async context to prevent
    deadlocks.

    The decorator preserves the original function's return type, so type checkers should
    understand that sync wrappers return the actual objects, not coroutines.

    Args:
        async_func: An async function that returns a Coroutine[Any, Any, T]

    Returns:
        A synchronous function that returns T (the unwrapped result type)

    Example:
        @async_to_sync
        async def get_data(self) -> MyData:
            return await self.get_data_async()

        # Type checker knows this returns MyData, not Awaitable[MyData]
        data = session.get_data()
    """

    @functools.wraps(async_func)
    def wrapper(self, *args, **kwargs) -> T:
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                raise RuntimeError(
                    f"Cannot call sync version of {async_func.__name__} from async context. "
                    f"Use {async_func.__name__}_async directly."
                )
        except RuntimeError as e:
            if "Cannot call sync version" in str(e):
                raise
        return asyncio.run(async_func(self, *args, **kwargs))

    return cast(Callable[..., T], wrapper)


class WikibaseSession:
    """Async-first session for interacting with Wikibase, with sync proxy methods"""

    # Magic numbers
    DEFAULT_TIMEOUT = 30
    DEFAULT_BATCH_SIZE = 50
    PAGE_REQUEST_SIZE = 500
    MAX_PAGE_REQUESTS = 2000  # Suitable up to 1M pages (500*2000)
    MAX_CONCURRENT_REQUESTS = 3  # Limit concurrent requests to avoid rate limiting
    REQUEST_DELAY_SECONDS = 0.5  # Small delay between requests to be gentle on server
    ITEM_NAMESPACE = 120
    ITEM_PREFIX = "Item:"
    HELP_NAMESPACE = 12

    # Property IDs
    has_subconcept_property_id = os.getenv("WIKIBASE_HAS_SUBCONCEPT_PROPERTY_ID", "P1")
    subconcept_of_property_id = os.getenv("WIKIBASE_SUBCONCEPT_OF_PROPERTY_ID", "P2")
    related_concept_property_id = os.getenv(
        "WIKIBASE_RELATED_CONCEPT_PROPERTY_ID", "P3"
    )
    negative_labels_property_id = os.getenv(
        "WIKIBASE_NEGATIVE_LABELS_PROPERTY_ID", "P9"
    )
    definition_property_id = os.getenv("WIKIBASE_DEFINITION_PROPERTY_ID", "P7")
    negative_concept_property_id = os.getenv(
        "WIKIBASE_NEGATIVE_CONCEPT_PROPERTY_ID", "P11"
    )

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        url: Optional[str] = None,
    ):
        """Initialize session - login happens on first use"""
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

        # These will be initialized on first use
        self._client: Optional[httpx.AsyncClient] = None
        self._csrf_token: Optional[str] = None
        self._redirects: Optional[dict[WikibaseID, WikibaseID]] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    def __repr__(self) -> str:
        """Return a string representation of the Wikibase session"""
        return f"<WikibaseSession: {self.username} at {self.api_url}>"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get the HTTP client, initializing session on first use"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT)
            await self._login()
            self._redirects = await self._get_all_redirects()

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)

        return self._client

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential_jitter(initial=RETRY_INITIAL_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type(HTTPStatusError),
    )
    async def _login(self):
        """Log in to Wikibase and get a CSRF token"""
        if not self._client:
            raise RuntimeError("Client not initialized")

        # Get login token
        login_token_response = await self._client.get(
            url=self.api_url,
            params={
                "action": "query",
                "meta": "tokens",
                "type": "login",
                "format": "json",
            },
        )
        login_token_response.raise_for_status()
        login_token_data = login_token_response.json()
        login_token = login_token_data["query"]["tokens"]["logintoken"]

        # Login
        login_response = await self._client.post(
            url=self.api_url,
            data={
                "action": "login",
                "lgname": self.username,
                "lgpassword": self.password,
                "lgtoken": login_token,
                "format": "json",
            },
        )
        login_response.raise_for_status()

        # Get CSRF token
        csrf_token_response = await self._client.get(
            url=self.api_url,
            params={
                "action": "query",
                "meta": "tokens",
                "format": "json",
            },
        )
        csrf_token_response.raise_for_status()
        csrf_token_data = csrf_token_response.json()
        self._csrf_token = csrf_token_data["query"]["tokens"]["csrftoken"]
        logger.debug("Got session CSRF token: %s", self._csrf_token)

    def _resolve_redirect(self, wikibase_id: WikibaseID) -> WikibaseID:
        """Check if a Wikibase ID is a redirect and return its target ID if it is."""
        if self._redirects is None:
            raise RuntimeError(
                "Session not initialized - call a method that initializes the session first"
            )
        return self._redirects.get(wikibase_id, wikibase_id)

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential_jitter(initial=RETRY_INITIAL_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type(HTTPStatusError),
    )
    async def _get_all_redirects(
        self, batch_size: Optional[int] = None
    ) -> dict[WikibaseID, WikibaseID]:
        """Get all redirects from Wikibase."""
        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE

        client = await self._get_client()
        pages = await self._get_pages(extra_params={"apfilterredir": "redirects"})
        redirects = {}

        # Process redirects in batches
        for batch in chunked(pages, batch_size):
            ids_to_fetch = [
                page["title"].removeprefix(self.ITEM_PREFIX) for page in batch
            ]

            response = await client.get(
                url=self.api_url,
                params={
                    "action": "wbgetentities",
                    "format": "json",
                    "ids": "|".join(ids_to_fetch),
                    "props": "info",
                },
                timeout=self.DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

            for wikibase_id, entity in data.get("entities", {}).items():
                if "redirects" in entity:
                    redirects[WikibaseID(wikibase_id)] = WikibaseID(
                        entity["redirects"]["to"]
                    )

        return redirects

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential_jitter(initial=RETRY_INITIAL_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type(HTTPStatusError),
    )
    async def _get_pages(self, extra_params: dict[str, str]) -> list[dict]:
        """Helper method to get pages from Wikibase with pagination."""
        client = await self._get_client()

        base_params = {
            "action": "query",
            "format": "json",
            "list": "allpages",  # See https://www.mediawiki.org/wiki/API:Allpages
            "apnamespace": self.ITEM_NAMESPACE,
            "aplimit": self.PAGE_REQUEST_SIZE,
        }
        base_params.update(extra_params)

        pages = []
        for i in range(self.MAX_PAGE_REQUESTS):
            response = await client.get(url=self.api_url, params=base_params)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                raise HTTPError(data["error"])
            if "warnings" in data:
                logger.warning(data["warnings"])

            batch_pages = data["query"]["allpages"]
            pages.extend(batch_pages)

            if continue_params := data.get("continue"):
                base_params.update(continue_params)
                logger.info("Retrieved %s pages after iteration %s", len(pages), i)
            else:
                break

        return pages

    async def get_all_concept_ids_async(self) -> list[WikibaseID]:
        """Get a complete list of all concept IDs in the Wikibase instance."""
        pages = await self._get_pages(extra_params={"apfilterredir": "nonredirects"})
        return [page["title"].replace(self.ITEM_PREFIX, "") for page in pages]

    @async_to_sync
    async def get_all_concept_ids(self) -> list[WikibaseID]:
        """Sync wrapper for get_all_concept_ids_async"""
        return await self.get_all_concept_ids_async()

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential_jitter(initial=RETRY_INITIAL_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type(HTTPStatusError),
    )
    async def _fetch_concept_async(
        self,
        wikibase_id: WikibaseID,
        timestamp: datetime,
        entity_info: dict[str, Any],
    ) -> Optional[Concept]:
        """Async helper to fetch a single concept."""
        client = await self._get_client()

        # Use semaphore to limit concurrent requests
        assert self._semaphore is not None, "Semaphore should be initialized"
        async with self._semaphore:
            try:
                page_id = str(entity_info.get("pageid"))
                if not page_id:
                    raise ConceptNotFoundError(wikibase_id)

                # Get the revision at the specified timestamp
                revisions_response = await client.get(
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
                )

                try:
                    revisions_response.raise_for_status()
                except HTTPStatusError as e:
                    if e.response.status_code == 404:
                        # 404 is not transient, just return None
                        logger.warning("Concept %s not found (404)", wikibase_id)
                        return None
                    # Re-raise for retry logic to handle
                    raise

                try:
                    revisions_data = revisions_response.json()
                except json.JSONDecodeError:
                    logger.warning(
                        "❌ Invalid JSON response for concept %s: %s",
                        wikibase_id,
                        revisions_response.text[:200],
                    )
                    return None

                pages = revisions_data.get("query", {}).get("pages", {})
                if not pages:
                    raise ConceptNotFoundError(wikibase_id)

                page = next(iter(pages.values()))
                revisions = page.get("revisions", [])
                if not revisions:
                    raise RevisionNotFoundError(wikibase_id, timestamp)

                # Get the revision content, handling empty content
                content = revisions[0].get("slots", {}).get("main", {}).get("*", "{}")
                if not content or content.strip() == "":
                    content = "{}"

                entity = json.loads(content)

                if not entity:
                    raise ConceptNotFoundError(wikibase_id)

                # Parse concept data
                concept = self._parse_wikibase_entity(wikibase_id, entity)

                concept = await self._incorporate_negative_concepts(concept)

                # Small delay to be gentle on the server
                await asyncio.sleep(self.REQUEST_DELAY_SECONDS)

                return concept

            except (KeyError, json.JSONDecodeError) as e:
                logger.warning("❌ Failed to parse concept %s: %s", wikibase_id, e)
            except (ConceptNotFoundError, RevisionNotFoundError) as e:
                logger.warning("❌ %s", str(e))
            except ValidationError as e:
                logger.warning("❌ Failed to validate concept %s: %s", wikibase_id, e)
            return None

    def _parse_wikibase_entity(
        self, wikibase_id: WikibaseID, entity: dict[str, Any]
    ) -> Concept:
        """Parse a Wikibase entity JSON into a Concept object."""
        # Extract basic concept information
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
            wikibase_id=WikibaseID(wikibase_id),
        )

        # Parse claims/properties
        if "claims" in entity and entity["claims"]:
            self._parse_concept_claims(concept, entity["claims"])

        return concept

    def _parse_concept_claims(self, concept: Concept, claims: dict[str, list]) -> None:
        """Parse Wikibase claims and populate concept properties."""
        for claim in claims.values():
            for statement in claim:
                if statement["mainsnak"]["snaktype"] == "value":
                    property_id = statement["mainsnak"]["property"]
                    value = statement["mainsnak"]["datavalue"]["value"]

                    if property_id == self.subconcept_of_property_id:
                        concept.subconcept_of.append(
                            self._resolve_redirect(value["id"])
                        )
                    elif property_id == self.has_subconcept_property_id:
                        concept.has_subconcept.append(
                            self._resolve_redirect(value["id"])
                        )
                    elif property_id == self.related_concept_property_id:
                        concept.related_concepts.append(
                            self._resolve_redirect(value["id"])
                        )
                    elif property_id == self.negative_labels_property_id:
                        concept.negative_labels.append(value)
                    elif property_id == self.definition_property_id:
                        concept.definition = value
                    elif property_id == self.negative_concept_property_id:
                        concept.negative_concepts.append(
                            self._resolve_redirect(value["id"])
                        )

    async def _incorporate_negative_concepts(self, concept: Concept) -> Concept:
        """
        Add positive labels from negative concepts

        For the given concept, fetch all negative concepts and add their positive
        labels to the concept's negative labels, returning a new concept with the
        additional negative labels.
        """
        # If there are no negative concepts, return the concept unchanged
        if not concept.negative_concepts:
            return concept

        try:
            # Fetch all negative concepts
            negative_concepts = await self.get_concepts_async(
                wikibase_ids=concept.negative_concepts
            )

            # Combine existing negative labels with new negative labels
            existing_negative_labels = set(concept.negative_labels)
            new_negative_labels = set(
                label
                for negative_concept in negative_concepts
                for label in negative_concept.all_labels
            )
            combined_negative_labels = list(
                existing_negative_labels | new_negative_labels
            )

            updated_concept = concept.model_copy(
                update={"negative_labels": combined_negative_labels},
                deep=True,
            )
            return updated_concept

        except (ConceptNotFoundError, HTTPStatusError, ValidationError) as e:
            logger.warning(
                "Failed to merge negative concept labels for %s: %s",
                concept.wikibase_id,
                e,
            )
            return concept

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential_jitter(initial=RETRY_INITIAL_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type(
            (HTTPStatusError, RequestError, json.JSONDecodeError)
        ),
    )
    async def get_concepts_async(
        self,
        limit: Optional[int] = None,
        wikibase_ids: Optional[list[WikibaseID]] = None,
        timestamp: Optional[datetime] = None,
    ) -> list[Concept]:
        """
        Async method to get concepts from Wikibase with concurrent fetching.

        This is the core async implementation that provides high performance
        through concurrent HTTP requests.
        """
        client = await self._get_client()

        if not wikibase_ids:
            logger.info("Fetching all concept IDs from Wikibase...")
            wikibase_ids = await self.get_all_concept_ids_async()
            logger.info("Found %d total concept IDs", len(wikibase_ids))

        if limit:
            wikibase_ids = wikibase_ids[:limit]
            logger.info("Limited to %d concepts", limit)

        logger.info(
            "Starting to fetch %d concepts with rate limiting (max %d concurrent)",
            len(wikibase_ids),
            self.MAX_CONCURRENT_REQUESTS,
        )

        # Handle timestamp
        if timestamp:
            if timestamp.tzinfo is None:
                timestamp = timestamp.astimezone(timezone.utc)
            if timestamp > datetime.now(timezone.utc):
                raise ValueError("Timestamp cannot be in the future")
        else:
            timestamp = datetime.now(timezone.utc)

        # Resolve redirects
        wikibase_ids = [self._resolve_redirect(wid) for wid in wikibase_ids]

        # Process in batches for entity info, then fetch concepts concurrently
        all_concepts = []
        failed_concepts = 0
        batch_size = self.DEFAULT_BATCH_SIZE
        total_batches = (len(wikibase_ids) + batch_size - 1) // batch_size

        for batch_num, batch_ids in enumerate(chunked(wikibase_ids, batch_size), 1):
            logger.info(
                "Processing batch %d/%d (%d concepts): %s",
                batch_num,
                total_batches,
                len(batch_ids),
                ", ".join(batch_ids[:5]) + ("..." if len(batch_ids) > 5 else ""),
            )

            # Get entity info for the batch
            entity_response = await client.get(
                url=self.api_url,
                params={
                    "action": "wbgetentities",
                    "format": "json",
                    "ids": "|".join(batch_ids),
                    "props": "info",
                },
            )
            entity_response.raise_for_status()
            entity_data = entity_response.json()

            if "error" in entity_data:
                logger.warning(
                    "Error fetching batch %d: %s", batch_num, entity_data["error"]
                )
                failed_concepts += len(batch_ids)
                continue

            # Create tasks for concurrent concept fetching
            tasks = []
            valid_concepts_in_batch = 0
            for wikibase_id in batch_ids:
                if entity_info := entity_data.get("entities", {}).get(
                    str(wikibase_id), {}
                ):
                    task = self._fetch_concept_async(
                        wikibase_id, timestamp, entity_info
                    )
                    tasks.append(task)
                    valid_concepts_in_batch += 1
                else:
                    # No entity info for this concept - skip it
                    continue

            # Execute all concept fetches concurrently
            if tasks:
                batch_concepts = await asyncio.gather(*tasks, return_exceptions=True)

                # Filter out None results and exceptions
                batch_successful = 0
                batch_failed = 0
                for concept in batch_concepts:
                    if isinstance(concept, Concept):
                        all_concepts.append(concept)
                        batch_successful += 1
                    elif isinstance(concept, Exception):
                        logger.warning("Exception in batch concept fetch: %s", concept)
                        batch_failed += 1
                    elif concept is None:
                        batch_failed += 1

                logger.info(
                    "Batch %d complete: %d successful, %d failed",
                    batch_num,
                    batch_successful,
                    batch_failed,
                )
                failed_concepts += batch_failed
            else:
                logger.warning("No valid concepts found in batch %d", batch_num)

        logger.info(
            "Concept fetching complete: %d successful, %d failed out of %d total",
            len(all_concepts),
            failed_concepts,
            len(wikibase_ids),
        )

        if not all_concepts and wikibase_ids:
            raise ConceptNotFoundError(wikibase_ids[0])

        return all_concepts

    @async_to_sync
    async def get_concepts(
        self,
        limit: Optional[int] = None,
        wikibase_ids: Optional[list[WikibaseID]] = None,
        timestamp: Optional[datetime] = None,
    ) -> list[Concept]:
        """
        Sync wrapper for get_concepts_async.

        This provides the familiar sync interface while leveraging
        async performance internally.
        """
        return await self.get_concepts_async(limit, wikibase_ids, timestamp)

    async def get_concept_async(
        self,
        wikibase_id: WikibaseID,
        timestamp: Optional[datetime] = None,
        include_labels_from_subconcepts: bool = False,
        include_recursive_subconcept_of: bool = False,
        include_recursive_has_subconcept: bool = False,
    ) -> Concept:
        """Async version of get_concept"""
        # Get the base concept first
        concepts = await self.get_concepts_async(
            wikibase_ids=[wikibase_id], timestamp=timestamp
        )
        if not concepts:
            raise ConceptNotFoundError(wikibase_id)
        concept = concepts[0]

        # Handle recursive relationships concurrently
        tasks = []
        if include_recursive_has_subconcept or include_labels_from_subconcepts:
            tasks.append(
                self.get_recursive_has_subconcept_relationships_async(wikibase_id)
            )
        if include_recursive_subconcept_of:
            tasks.append(
                self.get_recursive_subconcept_of_relationships_async(wikibase_id)
            )

        if tasks:
            results = await asyncio.gather(*tasks)
            result_index = 0

            # Handle subconcept relationships
            if include_recursive_has_subconcept or include_labels_from_subconcepts:
                recursive_subconcept_ids = results[result_index]
                result_index += 1

                if include_recursive_has_subconcept:
                    concept.recursive_has_subconcept = recursive_subconcept_ids

                # Get labels from subconcepts if needed
                if include_labels_from_subconcepts and recursive_subconcept_ids:
                    subconcepts = await self.get_concepts_async(
                        wikibase_ids=recursive_subconcept_ids, timestamp=timestamp
                    )

                    all_positive_labels = set(concept.all_labels)
                    all_negative_labels = set(concept.negative_labels)
                    for subconcept in subconcepts:
                        all_positive_labels.update(subconcept.all_labels)
                        all_negative_labels.update(subconcept.negative_labels)

                    concept.alternative_labels = list(all_positive_labels)
                    concept.negative_labels = list(all_negative_labels)

            # Handle parent concept relationships
            if include_recursive_subconcept_of:
                concept.recursive_subconcept_of = results[result_index]

        return concept

    @async_to_sync
    async def get_concept(
        self,
        wikibase_id: WikibaseID,
        timestamp: Optional[datetime] = None,
        include_labels_from_subconcepts: bool = False,
        include_recursive_subconcept_of: bool = False,
        include_recursive_has_subconcept: bool = False,
    ) -> Concept:
        """Sync wrapper for get_concept_async"""
        return await self.get_concept_async(
            wikibase_id,
            timestamp,
            include_labels_from_subconcepts,
            include_recursive_subconcept_of,
            include_recursive_has_subconcept,
        )

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential_jitter(initial=RETRY_INITIAL_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type(
            (HTTPStatusError, RequestError, json.JSONDecodeError)
        ),
    )
    async def get_recursive_has_subconcept_relationships_async(
        self, wikibase_id: WikibaseID
    ) -> list[WikibaseID]:
        """Async version of recursive subconcept fetching"""
        return await self._get_recursive_relationships_async(
            wikibase_id, self.has_subconcept_property_id
        )

    @async_to_sync
    async def get_recursive_has_subconcept_relationships(
        self, wikibase_id: WikibaseID
    ) -> list[WikibaseID]:
        """Sync wrapper for get_recursive_has_subconcept_relationships_async"""
        return await self.get_recursive_has_subconcept_relationships_async(wikibase_id)

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential_jitter(initial=RETRY_INITIAL_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type(
            (HTTPStatusError, RequestError, json.JSONDecodeError)
        ),
    )
    async def get_recursive_subconcept_of_relationships_async(
        self, wikibase_id: WikibaseID
    ) -> list[WikibaseID]:
        """Async version of recursive parent concept fetching"""
        return await self._get_recursive_relationships_async(
            wikibase_id, self.subconcept_of_property_id
        )

    @async_to_sync
    async def get_recursive_subconcept_of_relationships(
        self, wikibase_id: WikibaseID
    ) -> list[WikibaseID]:
        """Sync wrapper for get_recursive_subconcept_of_relationships_async"""
        return await self.get_recursive_subconcept_of_relationships_async(wikibase_id)

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential_jitter(initial=RETRY_INITIAL_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type(
            (HTTPStatusError, RequestError, json.JSONDecodeError)
        ),
    )
    async def _get_recursive_relationships_async(
        self,
        wikibase_id: WikibaseID,
        property_id: str,
        max_depth: int = 50,
        current_depth: int = 0,
        visited: Optional[set[WikibaseID]] = None,
    ) -> list[WikibaseID]:
        """Async version of recursive relationship fetching with concurrency"""
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
            raise ValueError(f"Invalid property ID: {property_id}")

        client = await self._get_client()
        wikibase_id = self._resolve_redirect(wikibase_id)

        response = await client.get(
            url=self.api_url,
            params={
                "action": "wbgetentities",
                "format": "json",
                "ids": wikibase_id,
                "props": "claims",
            },
        )
        response.raise_for_status()

        try:
            data = response.json()
        except json.JSONDecodeError:
            logger.warning(
                "❌ Invalid JSON response for concept %s: %s",
                wikibase_id,
                response.text,
            )
            raise

        entity = data["entities"][str(wikibase_id)]
        related_concepts = []

        if "claims" in entity:
            direct_related_ids = []
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
                            direct_related_ids.append(related_id)

            # Fetch relationships concurrently
            if direct_related_ids:
                related_concepts.extend(direct_related_ids)

                # Create concurrent tasks for the next level
                if tasks := [
                    self._get_recursive_relationships_async(
                        related_id,
                        property_id,
                        max_depth,
                        current_depth + 1,
                        visited.copy(),
                    )
                    for related_id in direct_related_ids
                ]:
                    results = await asyncio.gather(*tasks)
                    for result in results:
                        related_concepts.extend(result)

        return list(set(related_concepts))

    async def close(self):
        """Close the async client"""
        if self._client:
            await self._client.aclose()

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
        retry=retry_if_exception_type(
            (HTTPStatusError, RequestError, json.JSONDecodeError)
        ),
    )
    async def add_alternative_labels_async(
        self,
        wikibase_id: WikibaseID,
        alternative_labels: list[str],
        language: str = "en",
    ):
        """Add a list of alternative labels to a Wikibase item, preserving existing ones"""
        existing_alternative_labels = (
            await self.get_concept_async(wikibase_id)
        ).alternative_labels

        all_alternative_labels = list(
            set(existing_alternative_labels + alternative_labels)
        )

        client = await self._get_client()
        response = await client.post(
            url=self.api_url,
            data={
                "action": "wbeditentity",
                "format": "json",
                "id": wikibase_id,
                "token": self._csrf_token,
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
        response.raise_for_status()
        response_data = response.json()
        if "error" in response_data:
            raise HTTPError(
                f"Error adding alternative labels: {response_data['error']}"
            )
        return response_data

    @async_to_sync
    async def add_alternative_labels(
        self,
        wikibase_id: WikibaseID,
        alternative_labels: list[str],
        language: str = "en",
    ):
        """Sync wrapper for add_alternative_labels_async"""
        return await self.add_alternative_labels_async(
            wikibase_id, alternative_labels, language
        )

    async def search_help_pages_async(self, search_term: str) -> list[str]:
        """Async method to search for help pages in Wikibase."""
        client = await self._get_client()
        response = await client.get(
            url=self.api_url,
            params={
                "action": "query",
                "list": "search",
                "srsearch": search_term,
                "srnamespace": self.HELP_NAMESPACE,
                "format": "json",
            },
        )
        response.raise_for_status()
        data = response.json()
        return [result["title"] for result in data.get("query", {}).get("search", [])]

    @async_to_sync
    async def search_help_pages(self, search_term: str) -> list[str]:
        """Sync wrapper for search_help_pages_async"""
        return await self.search_help_pages_async(search_term)

    async def get_help_page_content_async(self, page_title: str) -> str:
        """Async method to get help page content as markdown."""
        client = await self._get_client()
        response = await client.get(
            url=self.api_url,
            params={
                "action": "parse",
                "page": page_title,
                "format": "json",
            },
        )
        response.raise_for_status()
        data = response.json()
        html_content = data.get("parse", {}).get("text", {}).get("*", "")
        if not html_content:
            return ""
        markdown_content = html2text.html2text(html_content)
        return markdown_content

    @async_to_sync
    async def get_help_page_content(self, page_title: str) -> str:
        """Sync wrapper for get_help_page_content_async"""
        return await self.get_help_page_content_async(page_title)
