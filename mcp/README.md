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
docker run -p 80:80 --env-file .env wikibase-mcp-server
```

### Using the local MCP server in Cursor

To use the MCP server in eg Cursor, you need to add it to the `mcp.json`.

```json
{
  "mcpServers": {
    "ClimatePolicyRadarConceptStore": {
      "url": "http://localhost/mcp"
    }
  }
}
```
