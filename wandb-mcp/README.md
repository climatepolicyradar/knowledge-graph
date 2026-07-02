# Weights & Biases MCP server

This MCP server exposes our Weights & Biases classifier training data as tools an LLM can call. It provides read-only access to the classifiers trained on each concept and their validation-set predictions, so LLMs can reason about classifier performance.

Each concept has its own Weights & Biases project (named by the concept's Wikibase ID, e.g. `Q69`), under the `climatepolicyradar` entity.

## Tools

- `get_classifiers` — get all classifiers trained on a concept, with their passage-level metrics (F1, precision, recall), model type/name, validation-set size, and version. Sortable by `latest`, `precision`, `recall`, or `f1_score`.
- `get_classifier_validation_predictions` — get the validation-set predictions table (columns + rows) logged for a specific classifier.

## Local Development

The `.env` file in the root of your repo should be populated with:

```
WANDB_API_KEY=
```

### Running the server locally

```bash
just serve-wandb-mcp
```

This serves on port 8001 locally so it can run alongside the concept store MCP (`just serve-mcp`, port 8000).

### Running the server through Docker

To build the Docker image:

```bash
docker build -t wandb-mcp-server -f wandb-mcp/Dockerfile .
```

To run the Docker container with an environment file (e.g., `.env`):

```bash
docker run -p 8000:8000 --env-file .env wandb-mcp-server
```

## Using the MCP server

The server has been deployed on AWS - take a look at the [infra README](./infra/README.md) for more details.

You can fetch the URL of the running MCP server by running `pulumi stack output mcp_url --stack labs` from the `wandb-mcp/infra` directory. You can use this in various tools, a couple of which are described below.

### Using the remote MCP server in Cursor

To use the MCP server in eg Cursor, you need to add it to the application's `mcp.json`.

```json
{
  "mcpServers": {
    "ClimatePolicyRadarWeightsAndBiases": {
      "url": THE_MCP_URL,
      "transport": "streamable-http",
      "headers": {
        "Accept": "application/json, text/event-stream"
      }
    }
  }
}
```

For more information, see [the cursor documentation](https://cursor.com/docs/context/mcp).

### Using the MCP server in Claude Desktop

Grab the MCP URL, and follow the instructions on adding a custom remote MCP connection in [Claude's documentation](https://docs.claude.com/en/docs/claude-code/mcp).
