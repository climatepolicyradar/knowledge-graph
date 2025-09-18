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
    query_used: str = Field(description="The search query that was used")
    total_found: int = Field(description="Number of help pages found")


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
    """Search the concept store by keywords, phrases, or Wikibase IDs."""

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
async def get_all_concept_ids(ctx: Context = None) -> ConceptIdsResult:
    """Get all concept IDs available in the knowledge graph."""

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
    """Retrieve multiple concepts by their Wikibase IDs."""

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
        bool, Field(description="Include full parent hierarchy")
    ] = False,
    include_recursive_has_subconcept: Annotated[
        bool, Field(description="Include full child hierarchy")
    ] = False,
    ctx: Context = None,
) -> Concept:
    """Get information about a single concept with optional hierarchical data."""

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
    query: Annotated[
        str, Field(description="Search terms for help documentation", min_length=1)
    ],
    ctx: Context = None,
) -> HelpPagesResult:
    """Search for help documentation pages by topic."""

    if ctx:
        await ctx.info(f"Searching help documentation for: '{query}'")

    try:
        wikibase = WikibaseSession()
        page_titles = await wikibase.search_help_pages_async(search_term=query)

        return HelpPagesResult(
            page_titles=page_titles, query_used=query, total_found=len(page_titles)
        )
    except Exception as e:
        await handle_wikibase_error(
            e, f"Help page search failed for query '{query}'", ctx
        )
        return HelpPagesResult(page_titles=[], query_used=query, total_found=0)


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
