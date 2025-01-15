# Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy project files
COPY pyproject.toml poetry.lock ./
COPY predictions_api/README.md ./README.md

# Configure poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Copy all source code
COPY src ./src
COPY predictions_api ./predictions_api
COPY scripts ./scripts
COPY tests ./tests

# Install dependencies and the local package
RUN poetry install --with predictions_api

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uvicorn for running the application
RUN pip install uvicorn

# Copy installed packages and source code from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /app/src ./src
COPY --from=builder /app/predictions_api ./predictions_api
COPY --from=builder /app/scripts ./scripts

# Add the app directory to Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run the application directly with uvicorn
CMD ["uvicorn", "predictions_api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
