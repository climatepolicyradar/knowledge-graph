# Concept store MCP server

This MCP server is a wrapper around the `WikibaseSession` class, providing MCP access to our concept store. The MCP endpoints allow LLMs to make use of the concept store as a tool in their reasoning process.

## Local Development

The `.env` file in the root of your repo should be populated with the following variables:

```
WIKIBASE_USERNAME=
WIKIBASE_PASSWORD=
WIKIBASE_URL=
```

### Running the server locally

To run the server locally, you can run:

```bash
just serve-mcp
```

### Running the server through Docker

To build the Docker image:

```bash
docker build -t wikibase-mcp-server -f mcp/Dockerfile .
```

To run the Docker container with an environment file (e.g., `.env`):

```bash
docker run -p 8000:8000 --env-file .env wikibase-mcp-server
```

### Using the local MCP server in Cursor

First, get the MCP URL by running `pulumi stack output mcp_url --stack labs` from the `mcp/infra` directory.

To use the MCP server in eg Cursor, you need to add it to the application's `mcp.json`.

```json
{
  "mcpServers": {
    "ClimatePolicyRadarConceptStore": {
      "url": THE_MCP_URL,
      "transport": "streamable-http",
      "headers": {
        "Accept": "application/json, text/event-stream"
      }
    }
  }
}
```

## Deploying the MCP server to AWS

The MCP server is deployed to AWS using Pulumi. See the [infra README](./infra/README.md) for more details.
