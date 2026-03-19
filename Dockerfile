# =============================================================================
# FRAGMENTOMICS-TOOLS CONTAINER
# =============================================================================
#
# Container for fragmentomics-tools with Cython extensions.
#
# Build:
#   make docker-build
# Or manually (requires JFrog credentials for fragments-h5):
#   docker build --build-arg JFROG_CONDA_CHANNEL=https://USER:PASS@karius.jfrog.io/artifactory/api/conda/karius-conda \
#     -t fragmentomics-tools:1.0.0 .
#
# =============================================================================

FROM mambaorg/micromamba:1.5-jammy

LABEL maintainer="Karius Analytics"
LABEL description="Fragmentomics tools and datastructures"

# Switch to root for system package installation
USER root

# Install system dependencies (build-essential needed for Cython compilation)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        pigz \
        gzip \
        bzip2 \
        procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to micromamba user
USER $MAMBA_USER

# Copy environment file
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

# JFrog conda channel (for internal packages like fragments-h5)
# Pass via: --build-arg JFROG_CONDA_CHANNEL=https://USER:PASS@karius.jfrog.io/...
ARG JFROG_CONDA_CHANNEL=""

# Create the conda environment (add JFrog channel if provided)
RUN if [ -n "$JFROG_CONDA_CHANNEL" ]; then \
        micromamba install -y -n base -f /tmp/environment.yml -c "$JFROG_CONDA_CHANNEL" && \
        micromamba clean --all --yes; \
    else \
        micromamba install -y -n base -f /tmp/environment.yml && \
        micromamba clean --all --yes; \
    fi

# Activate the environment by default
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Copy package source
COPY --chown=$MAMBA_USER:$MAMBA_USER . /tmp/fragmentomics_tools

# Build Cython extensions and install package
RUN cd /tmp/fragmentomics_tools && \
    pip install --no-deps . && \
    cd / && \
    rm -rf /tmp/fragmentomics_tools

# Verify installation
RUN python --version && \
    python -c "import fragmentomics_tools; print('fragmentomics-tools OK')"

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["/bin/bash"]
