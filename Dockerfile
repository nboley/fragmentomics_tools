# =============================================================================
# FRAGMENTOMICS-TOOLS CONTAINER
# =============================================================================
#
# Container for fragmentomics-tools with Cython extensions.
#
# Build (no JFrog needed — fragments-h5 and datamanifest install from GitHub):
#   docker build -t karius-fragmentomics-tools:latest .
#
# =============================================================================

FROM mambaorg/micromamba:1.5-jammy

LABEL maintainer="Karius Analytics"
LABEL description="Fragmentomics tools and datastructures"

# Switch to root for system package installation
USER root

# Install system dependencies (build-essential for Cython, git for pip GitHub installs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
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

# Create the conda environment (fragments-h5 and datamanifest install via pip from GitHub)
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

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
    python -c "import fragmentomics_tools; print('fragmentomics-tools OK')" && \
    python -c "from fragments_h5 import FragmentsH5; print('fragments-h5 OK')" && \
    python -c "from datamanifest import DataManifest; print('datamanifest OK')"

ENV PATH="/opt/conda/bin:${PATH}"

# Nextflow hardcodes /usr/local/bin/aws in its Batch command wrapper
USER root
RUN ln -sf /opt/conda/bin/aws /usr/local/bin/aws
USER $MAMBA_USER

WORKDIR /work
CMD ["/bin/bash"]
