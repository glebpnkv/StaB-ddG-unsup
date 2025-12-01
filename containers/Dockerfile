ARG PYTHON_VERSION=3.10.16
FROM python:${PYTHON_VERSION}-slim

# Set the working directory in the container
WORKDIR /app

# System build deps
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    bash \
  && rm -rf /var/lib/apt/lists/*

# Install uv and sync project deps into a venv at /app/.venv
COPY pyproject.toml .
RUN python -m pip install -U pip && \
    python -m pip install --user uv
ENV PATH="/root/.local/bin:${PATH}"
RUN uv sync
ENV PATH="/app/.venv/bin:${PATH}"

# Ensure pip is available in the venv (for Vertex components which sometimes invoke it)
RUN python -m ensurepip || true && python -m pip install -U pip

# Removing build deps to slim the final image
RUN apt-get purge -y --auto-remove build-essential python3-dev git libffi-dev || true && \
    rm -rf /var/lib/apt/lists/*

# Copy app code
COPY . .

# Install the local package so `import stabddg` works
RUN python -m pip install -e .

# Adding execution permissions to the scripts
RUN chmod +x /app/scripts/intact_pretrain.sh