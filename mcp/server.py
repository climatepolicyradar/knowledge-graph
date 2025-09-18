import logging
from typing import Annotated

from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field
from rich.logging import RichHandler
from starlette.requests import Request
from starlette.responses import JSONResponse

from knowledge_graph.concept import Concept
from knowledge_graph.wikibase import WikibaseID, WikibaseSession


# Define response models for automatic schema generation
class ConceptSearchResult(BaseModel):
    """Search results from the concept store."""

    concepts: list[Concept] = Field(description="List of matching concepts")
    total_found: int = Field(description="Total number of concepts found")
    query: str = Field(description="The search query that was used")


class ConceptIdsResult(BaseModel):
    """All concept IDs in the knowledge graph."""

    concept_ids: list[WikibaseID] = Field(
        description="Complete list of concept identifiers"
    )
    total_count: int = Field(description="Total number of concepts available")


class HelpPagesResult(BaseModel):
    """Help documentation search results."""

    page_titles: list[str] = Field(description="Titles of matching help pages")
    search_terms: str = Field(description="The search terms that were used")
    total_found: int = Field(description="Number of help pages found")


class HelpPagesListResult(BaseModel):
    """List of all help documentation pages available in the concept store."""

    page_titles: list[str] = Field(description="Titles of all help pages")
    total_count: int = Field(description="Total number of help pages available")


class MultipleConceptsResult(BaseModel):
    """Results from retrieving multiple concepts by ID."""

    concepts: list[Concept] = Field(description="The retrieved concepts")
    requested_ids: list[str] = Field(description="Wikibase IDs that were requested")
    found_count: int = Field(description="Number of concepts successfully retrieved")


class HelpPageContent(BaseModel):
    """Content of a help documentation page."""

    title: str = Field(description="Title of the help page")
    content: str = Field(description="Full markdown content")
    character_count: int = Field(description="Length of content in characters")


# Common error handler to reduce code duplication
async def handle_wikibase_error(
    e: Exception, operation: str, ctx: Context = None
) -> str:
    """Centralized error handling for Wikibase operations"""
    error_msg = f"{operation}: {str(e)}"
    if ctx:
        await ctx.error(error_msg)
    else:
        print(error_msg)
    return error_msg


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)

# Initialize FastMCP app
mcp = FastMCP("Climate Policy Radar Concept Store")


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint for load balancer monitoring."""
    _ = request  # Unused but required by the function signature
    return JSONResponse(
        {"status": "healthy", "service": "Climate Policy Radar Concept Store MCP"}
    )


@mcp.tool
async def search_concepts(
    query: Annotated[
        str, Field(description="Search keywords, phrases, or Wikibase ID", min_length=1)
    ],
    limit: Annotated[
        int, Field(description="Maximum results to return", ge=1, le=100)
    ] = 10,
    ctx: Context = None,
) -> ConceptSearchResult:
    """
    Search the concept store by keywords, phrases, or Wikibase IDs.

    Wikibase will match your search terms to terms found in the preferred labels,
    alternative labels, and descriptions of the concepts. The search index and matching
    algorithm are optimised for precision over recall, and the way that terms are
    matched is strict.

    As a result, longer, compound queries (ie those which reference multiple topics, eg
    "drainage infrastructure urban planning") rarely deliver good results, and are not
    recommended. If searching for multiple topics, it's better to search for each topic
    separately and then combine the results.

    The concept store contains many concepts, each with many labels, so even if you
    don't find a result based on a single term, you might be able to find it with one
    or two successive searches. You should try both technical and non-technical terms.
    """

    if ctx:
        await ctx.info(f"Searching for concepts: '{query}' (limit: {limit})")

    try:
        wikibase = WikibaseSession()
        concepts = await wikibase.search_concepts_async(search_term=query, limit=limit)

        return ConceptSearchResult(
            concepts=concepts, total_found=len(concepts), query=query
        )
    except Exception as e:
        await handle_wikibase_error(e, f"Search failed for query '{query}'", ctx)
        return ConceptSearchResult(concepts=[], total_found=0, query=query)


@mcp.tool
async def list_concept_ids(ctx: Context = None) -> ConceptIdsResult:
    """Get a list of all concept IDs available in the knowledge graph."""

    if ctx:
        await ctx.info("Retrieving all concept IDs...")

    try:
        wikibase = WikibaseSession()
        concept_ids = await wikibase.get_all_concept_ids_async()

        return ConceptIdsResult(concept_ids=concept_ids, total_count=len(concept_ids))
    except Exception as e:
        await handle_wikibase_error(e, "Failed to retrieve concept IDs", ctx)
        return ConceptIdsResult(concept_ids=[], total_count=0)


@mcp.tool
async def get_multiple_concepts(
    wikibase_ids: Annotated[
        list[str],
        Field(description="Wikibase IDs to retrieve", min_length=1, max_length=50),
    ],
    ctx: Context = None,
) -> MultipleConceptsResult:
    """
    Retrieve multiple concepts by their Wikibase IDs.

    This tool is useful when you have retrieved a list of concept IDs (eg a list of
    related concepts), and you want to get more detailed information about them.
    """

    if ctx:
        await ctx.info(f"Retrieving {len(wikibase_ids)} concepts")

    try:
        # Validate and convert IDs
        valid_ids = [WikibaseID(wid) for wid in wikibase_ids]

        if not valid_ids and ctx:
            await ctx.warning("No valid Wikibase IDs provided")

        wikibase = WikibaseSession()
        concepts = await wikibase.get_concepts_async(wikibase_ids=valid_ids)

        return MultipleConceptsResult(
            concepts=concepts, requested_ids=wikibase_ids, found_count=len(concepts)
        )
    except Exception as e:
        await handle_wikibase_error(e, "Failed to retrieve concepts", ctx)
        return MultipleConceptsResult(
            concepts=[], requested_ids=wikibase_ids, found_count=0
        )


@mcp.tool
async def get_concept(
    wikibase_id: Annotated[
        str, Field(description="Wikibase concept ID (e.g., 'Q69')", pattern=r"^Q\d+$")
    ],
    include_labels_from_subconcepts: Annotated[
        bool, Field(description="Include child concept labels")
    ] = False,
    include_recursive_subconcept_of: Annotated[
        bool,
        Field(
            description=(
                "Include the concept's complete ancestor hierarchy, ie its parent "
                "concepts, and its parent concepts' parent concepts, etc."
            )
        ),
    ] = False,
    include_recursive_has_subconcept: Annotated[
        bool,
        Field(
            description=(
                "Include the concept's complete descendant hierarchy, ie its "
                "subconcepts, and its subconcepts' subconcepts' subconcepts, etc"
            )
        ),
    ] = False,
    ctx: Context = None,
) -> Concept:
    """
    Get information about a single concept

    If specified, the returned concept can include extra detail about the hierarchy in
    which it sits, beyond its immediate neighbours.
    """

    if ctx:
        await ctx.info(f"Retrieving concept {wikibase_id}")

    try:
        wikibase = WikibaseSession()
        concept = await wikibase.get_concept_async(
            wikibase_id=WikibaseID(wikibase_id),
            include_labels_from_subconcepts=include_labels_from_subconcepts,
            include_recursive_subconcept_of=include_recursive_subconcept_of,
            include_recursive_has_subconcept=include_recursive_has_subconcept,
        )
        return concept
    except Exception as e:
        await handle_wikibase_error(e, f"Failed to retrieve concept {wikibase_id}", ctx)
        return Concept(
            preferred_label="Error: Concept not found", wikibase_id=WikibaseID("Q0")
        )


@mcp.tool
async def search_help_pages(
    search_terms: Annotated[
        str,
        Field(
            description="Search terms to match in help page titles and content",
            min_length=1,
        ),
    ],
    ctx: Context = None,
) -> HelpPagesResult:
    """
    Search for help documentation pages by topic.

    Help pages exist for a variety of topics, including:
    - Heuristics on how to structure a concept, where to draw boundaries between
      concepts, and how concepts can be connected to one another
    - A style guide for concepts' labels, descriptions, definitions, negative labels,
      etc in the concept store
    - What the concept store is for, ie how the data is used by downstream classifiers
      and other tools/services

    Wikibase will match your search terms to terms found in the titles and content of
    the help pages. The search index and matching algorithm are optimised for precision
    over recall, and the way that terms are matched is strict.

    As a result, longer, compound queries rarely deliver good results, and are not
    recommended. If searching for multiple topics, it's better to search for each topic
    separately and then combine the results.
    """

    if ctx:
        await ctx.info(f"Searching help documentation for: '{search_terms}'")

    try:
        wikibase = WikibaseSession()
        page_titles = await wikibase.search_help_pages_async(search_term=search_terms)

        return HelpPagesResult(
            page_titles=page_titles,
            search_terms=search_terms,
            total_found=len(page_titles),
        )
    except Exception as e:
        await handle_wikibase_error(
            e, f"Help page search failed for query '{search_terms}'", ctx
        )
        return HelpPagesResult(page_titles=[], search_terms=search_terms, total_found=0)


@mcp.tool
async def get_help_page(
    page_title: Annotated[
        str,
        Field(
            description="Exact help page title including 'Help:' prefix", min_length=1
        ),
    ],
    ctx: Context = None,
) -> HelpPageContent:
    """Retrieve the full content of a help documentation page."""

    if ctx:
        await ctx.info(f"Retrieving help page: '{page_title}'")

    try:
        wikibase = WikibaseSession()
        content = await wikibase.get_help_page_async(page_title=page_title)

        if not content:
            content = f"Help page '{page_title}' not found. Use search_help_pages to discover available documentation."

        return HelpPageContent(
            title=page_title, content=content, character_count=len(content)
        )
    except Exception as e:
        await handle_wikibase_error(
            e, f"Failed to retrieve help page '{page_title}'", ctx
        )
        error_content = f"Error: {str(e)}"
        return HelpPageContent(
            title=page_title, content=error_content, character_count=len(error_content)
        )


@mcp.tool
async def list_help_pages(ctx: Context = None) -> HelpPagesListResult:
    """Get a list of all help documentation pages available in the concept store."""

    if ctx:
        await ctx.info("Retrieving all help documentation pages...")

    try:
        wikibase = WikibaseSession()
        page_titles = await wikibase.get_all_help_pages_async()

        return HelpPagesListResult(
            page_titles=page_titles,
            total_count=len(page_titles),
        )
    except Exception as e:
        await handle_wikibase_error(e, "Failed to retrieve help pages", ctx)
        return HelpPagesListResult(page_titles=[], total_count=0)
