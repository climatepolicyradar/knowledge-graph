FROM python:3.13-slim-bookworm

WORKDIR /app

# Install build essentials
RUN apt-get update && apt-get install -y build-essential

# Install uv and dependencies
RUN pip install uv
COPY pyproject.toml uv.lock README.md ./ 
RUN uv sync --extra mcp --locked

# Copy the rest of the application code
COPY . .

# Expose the FastMCP server on port 8000
EXPOSE 8000

# Run the FastMCP server
CMD ["uv", "run", "fastmcp", "run", "mcp/server.py:mcp", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
