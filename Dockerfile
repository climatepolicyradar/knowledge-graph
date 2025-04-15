FROM prefecthq/prefect:2.20.7-python3.10

RUN pip install --upgrade pip
RUN pip install "poetry==1.8.3"

WORKDIR /opt/prefect/knowledge-graph

# Set up environment
COPY ./poetry.lock ./pyproject.toml ./
RUN poetry config virtualenvs.create false
RUN poetry install --no-root --no-interaction --with transformers --without dev

# Set up package
COPY ./flows ./flows/
COPY ./src ./src/
COPY ./scripts ./scripts
COPY ./static_sites ./static_sites/
RUN poetry install --no-interaction --only-root
