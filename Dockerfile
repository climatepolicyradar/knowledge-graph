# Builder stage with full image, as we need compilation software
FROM python:3.13-bookworm@sha256:8238a23384ea691ebe26e31745b7e3b4a65be1a17204b71c490846ed6209eb4c AS builder
COPY --from=ghcr.io/astral-sh/uv@sha256:f64ad69940b634e75d2e4d799eb5238066c5eeda49f76e782d4873c3d014ea33 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_SYSTEM_PYTHON=1

# Install the AWS CLI at v2 to assist with training classifiers within
# the Docker container.
RUN uv pip install awscliv2==2.3.1

WORKDIR /app

# This allows the dependencies of the project (which do not change
# often) to be cached separately from the project itself (which
# changes very frequently).
COPY pyproject.toml README.md ./
RUN uv pip install -r pyproject.toml --extra transformers_gpu --extra coiled

# Runtime stage with slim image
FROM python:3.13-slim-bookworm@sha256:4e368beeae9bf2720518bf8f45a5ecb701adadfcbb01142561fd7a4c99989a86

WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --from=builder /app/pyproject.toml ./
COPY --from=builder /app/README.md ./

# Copy the project into the image
COPY knowledge_graph ./knowledge_graph/
COPY flows ./flows/
COPY scripts ./scripts/
COPY static_sites ./static_sites/

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
