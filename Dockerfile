# Builder stage with full image, as we need compilation software
FROM python:3.13-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_SYSTEM_PYTHON=1

# Install the aws cli at v2 to assist with training classifiers within the docker container.
RUN pip3 install awscliv2

WORKDIR /app

# This allows the dependencies of the project (which do not change
# often) to be cached separately from the project itself (which
# changes very frequently).
COPY pyproject.toml README.md ./
RUN uv pip install -r pyproject.toml --extra transformers --extra coiled

# Copy the project into the image
COPY src ./src/
COPY flows ./flows/
COPY scripts ./scripts/

# Install the project
RUN uv pip install -e .

# Runtime stage with slim image
FROM python:3.13-slim-bookworm

WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code to runtime stage
COPY --from=builder /app/src ./src/
COPY --from=builder /app/flows ./flows/
COPY --from=builder /app/scripts ./scripts/

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
