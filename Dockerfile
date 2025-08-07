FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

ENV UV_COMPILE_BYTECODE=1
ENV UV_SYSTEM_PYTHON=1

# This allows the dependencies of the project (which do not change
# often) to be cached separately from the project itself (which
# changes very frequently).
COPY pyproject.toml .
RUN uv pip install -r pyproject.toml --extra transformers
COPY . .
RUN uv pip install -e .

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
