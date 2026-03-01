# Builder stage with full image, as we need compilation software
FROM python:3.14-bookworm@sha256:6cc50e409fb008d64c8c119c33488cf0e2288fb657a9ceeae237d381f842713e AS builder
COPY --from=ghcr.io/astral-sh/uv@sha256:f64ad69940b634e75d2e4d799eb5238066c5eeda49f76e782d4873c3d014ea33 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_SYSTEM_PYTHON=1

# Install the AWS CLI at v2 to assist with training classifiers within
# the Docker container.
RUN uv pip install awscliv2==2.3.1

WORKDIR /app

# Use bind mounts to install dependencies without copying pyproject.toml into the layer.
# This allows the dependency layer to remain cached even when the version field changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv pip install -r pyproject.toml --extra transformers_gpu --extra coiled --link-mode=copy

# Runtime stage with slim image
FROM python:3.14-slim-bookworm@sha256:5404df00cf00e6e7273375f415651837b4d192ac6859c44d3b740888ac798c99

WORKDIR /app

# Install git, git-lfs and github cli for git operations within knowledge graph
# Required in runtime image as it does not copy from builder image
RUN apt-get update \
    && apt-get install -y git git-lfs wget \
    && mkdir -p -m 755 /etc/apt/keyrings \
    && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    && cat $out > /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && mkdir -p -m 755 /etc/apt/sources.list.d \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*
RUN git lfs install

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
# Copy uv from builder stage
COPY --from=builder /bin/uv /bin/uvx /bin/

# Copy the project into the image
COPY pyproject.toml README.md ./
COPY knowledge_graph ./knowledge_graph/
COPY flows ./flows/
COPY scripts ./scripts/
COPY static_sites ./static_sites/
COPY vibe-checker ./vibe-checker/

# Install the project
RUN uv pip install --system -e .

# Set PYTHONPATH to ensure modules can be found for distributed tasks
# This is a workaround for when running on coiled when functions are serialised 
# and it seems the cwd context can be lost
ENV PYTHONPATH="/app:/app/knowledge_graph:/app/flows:/app/scripts"

ENV PREFECT_LOGGING_LEVEL=DEBUG
# Setting PYTHONUNBUFFERED to a non-empty value different from 0 ensures that the python output i.e. the stdout and
ENV PYTHONUNBUFFERED=1
# Example of a segmentation fault on Linux with and without enabling the fault handler:
#
# $ python -c "import ctypes; ctypes.string_at(0)"
# Segmentation fault

# $ python -q -X faulthandler
# >>> import ctypes
# >>> ctypes.string_at(0)
# Fatal Python error: Segmentation fault

# Current thread 0x00007fb899f39700 (most recent call first):
#   File "/home/python/cpython/Lib/ctypes/__init__.py", line 486 in string_at
#   File "<stdin>", line 1 in <module>
# Segmentation fault
ENV PYTHONFAULTHANDLER=1
