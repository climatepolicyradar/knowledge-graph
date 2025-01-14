FROM python:3.10-slim

WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy project files
COPY README.md pyproject.toml poetry.lock ./

# Configure poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --with predictions_api --no-root

# Copy application code
COPY predictions_api ./predictions_api
COPY scripts ./scripts
COPY src ./src

# Add the app directory to Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run the application
CMD ["poetry", "run", "uvicorn", "predictions_api.app.main:app", "--host", "0.0.0.0", "--port", "8000"] 
