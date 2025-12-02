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
    wget \
    tar \
    unzip \
  && rm -rf /var/lib/apt/lists/*

# Copy app code
COPY . .
# Install uv and sync project deps into a venv at /app/.venv
RUN python -m pip install -U pip && \
    python -m pip install --user uv
ENV PATH="/root/.local/bin:${PATH}"
RUN uv sync --extra gcp --extra pytorch_cpu
ENV PATH="/app/.venv/bin:${PATH}"

# Ensure pip is available in the venv (for Vertex components which sometimes invoke it)
RUN python -m ensurepip || true && python -m pip install -U pip

# Removing build deps to slim the final image
RUN apt-get purge -y --auto-remove build-essential python3-dev git libffi-dev || true && \
    rm -rf /var/lib/apt/lists/*

# Adding execution permissions to the scripts
RUN chmod +x /app/scripts/intact_pretrain.sh
RUN chmod +x /app/scripts/download_stabddg_data.sh