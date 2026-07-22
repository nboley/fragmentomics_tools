# Makefile for building and publishing conda packages and Docker images
# fragmentomics-tools

PACKAGE_NAME := fragmentomics-tools
VERSION := $(shell grep '^\s*version:' recipe/recipe.yaml | head -1 | sed -E 's/.*version:\s*"?([0-9.]+)"?.*/\1/')
ECR_REGISTRY := 573640641260.dkr.ecr.us-east-1.amazonaws.com
IMAGE_NAME := karius-$(PACKAGE_NAME)
IMAGE_TAG := $(ECR_REGISTRY)/$(IMAGE_NAME):$(VERSION)
IMAGE_LATEST := $(ECR_REGISTRY)/$(IMAGE_NAME):latest

.PHONY: all login conda-login docker-login tag conda conda-build conda-publish docker docker-build docker-push clean help

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  login         Verify credentials for conda and docker"
	@echo "  conda-login   Verify JFrog credentials for conda publishing"
	@echo "  docker-login  Verify AWS authentication for ECR"
	@echo "  conda-build   Build conda package with rattler-build"
	@echo "  conda-publish Publish conda package to JFrog Artifactory"
	@echo "  conda         Build and publish conda package"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-push   Push Docker image to ECR"
	@echo "  docker        Build and push Docker image"
	@echo "  tag           Create and push git tag v$$VERSION"
	@echo "  all           Build/upload conda, tag repo, build/push docker"
	@echo "  clean         Remove build artifacts"
	@echo ""
	@echo "Configuration:"
	@echo "  PACKAGE_NAME=$(PACKAGE_NAME)"
	@echo "  VERSION=$(VERSION)"
	@echo "  IMAGE_TAG=$(IMAGE_TAG)"

# Main target: login, tag, build conda, build docker, clean
all: login tag conda docker clean
	@echo ""
	@echo "========================================"
	@echo "Release $(VERSION) complete!"
	@echo "  ✓ Git tagged: v$(VERSION)"
	@echo "  ✓ Conda package built and published"
	@echo "  ✓ Docker pushed: $(IMAGE_TAG)"
	@echo "  ✓ Build artifacts cleaned"
	@echo "========================================"

# Verify credentials before building
login: conda-login docker-login
	@echo ""
	@echo "✓ All credentials verified successfully!"
	@echo ""

# Check for JFrog credentials (in pip.conf or environment)
conda-login:
	@echo "Checking for JFrog credentials..."
	@HAS_ENV_CREDS=0; \
	HAS_PIP_CREDS=0; \
	if [ -n "$$JFROG_URL" ] && { [ -n "$$JFROG_USER" ] || [ -n "$$JFROG_ACCESS_TOKEN" ]; }; then \
		HAS_ENV_CREDS=1; \
	fi; \
	if [ -f ~/.config/pip/pip.conf ] && grep -q "index-url.*jfrog" ~/.config/pip/pip.conf 2>/dev/null; then \
		HAS_PIP_CREDS=1; \
	fi; \
	if [ -f ~/.pip/pip.conf ] && grep -q "index-url.*jfrog\|extra-index-url.*jfrog" ~/.pip/pip.conf 2>/dev/null; then \
		HAS_PIP_CREDS=1; \
	fi; \
	if [ $$HAS_ENV_CREDS -eq 0 ] && [ $$HAS_PIP_CREDS -eq 0 ]; then \
		echo "❌ Error: JFrog credentials not found"; \
		echo "Please set JFROG_URL, JFROG_USER/JFROG_PASSWORD or JFROG_ACCESS_TOKEN"; \
		echo "Or configure ~/.config/pip/pip.conf or ~/.pip/pip.conf with JFrog index-url"; \
		exit 1; \
	fi; \
	echo "✓ JFrog credentials found"

# Check for AWS CLI (needed for ECR)
docker-login:
	@echo "Checking for AWS CLI..."
	@if ! command -v aws >/dev/null 2>&1; then \
		echo "❌ Error: aws CLI not found"; \
		echo "Please install AWS CLI and configure with: aws configure"; \
		exit 1; \
	fi
	@echo "✓ AWS CLI found"
	@echo "Logging in to ECR..."
	@aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $(ECR_REGISTRY)

# Create and push git tag
tag:
	@echo "Creating git tag v$(VERSION)..."
	@if git rev-parse "v$(VERSION)" >/dev/null 2>&1; then \
		echo "❌ Error: Tag v$(VERSION) already exists"; \
		echo "Please bump the version in recipe/recipe.yaml and pyproject.toml first"; \
		exit 1; \
	fi
	@git tag -a "v$(VERSION)" -m "Release version $(VERSION)"
	@git push origin "v$(VERSION)"
	@echo "✓ Tagged and pushed v$(VERSION)"

# Build and publish conda package
conda: conda-build conda-publish

# Build conda package with rattler-build
conda-build:
	@echo "Building conda package..."
	@rattler-build build --recipe recipe/recipe.yaml --channel conda-forge --channel bioconda --no-test; \
	BUILD_EXIT=$$?; \
	if [ $$BUILD_EXIT -ne 0 ] && { [ ! -d output ] || [ -z "$$(find output -name '*.conda' 2>/dev/null)" ]; }; then \
		echo "❌ Error: Conda build failed (exit code $$BUILD_EXIT)"; \
		exit $$BUILD_EXIT; \
	elif [ $$BUILD_EXIT -ne 0 ]; then \
		echo "⚠️  Warning: Build succeeded but cleanup failed (exit code $$BUILD_EXIT) - this is a known rattler-build issue"; \
	fi
	@echo "✓ Conda package built"

# Publish conda package to JFrog
conda-publish:
	@echo "Publishing conda package to JFrog..."
	bash scripts/publish_conda_package.sh
	@echo "✓ Conda package published"

# Build and push Docker image
docker: docker-build docker-push

# Build Docker image
docker-build:
	@echo "Building Docker image $(IMAGE_NAME):$(VERSION)..."
	@# Construct JFrog conda channel URL from environment vars or pip.conf
	@JFROG_CHANNEL=""; \
	if [ -n "$$JFROG_URL" ] && [ -n "$$JFROG_USER" ] && [ -n "$$JFROG_PASSWORD" ]; then \
		JFROG_CHANNEL="https://$$JFROG_USER:$$JFROG_PASSWORD@$$JFROG_URL/artifactory/api/conda/karius-conda"; \
	elif [ -n "$$JFROG_URL" ] && [ -n "$$JFROG_ACCESS_TOKEN" ]; then \
		JFROG_CHANNEL="https://token:$$JFROG_ACCESS_TOKEN@$$JFROG_URL/artifactory/api/conda/karius-conda"; \
	else \
		PIP_URL=$$(pip config get global.extra-index-url 2>/dev/null || true); \
		if echo "$$PIP_URL" | grep -q "jfrog"; then \
			USER_PASS=$$(echo "$$PIP_URL" | sed -n 's|https://\([^@]*\)@.*|\1|p'); \
			HOST=$$(echo "$$PIP_URL" | sed -n 's|https://[^@]*@\([^/]*\)/.*|\1|p'); \
			if [ -n "$$USER_PASS" ] && [ -n "$$HOST" ]; then \
				JFROG_CHANNEL="https://$$USER_PASS@$$HOST/artifactory/api/conda/karius-conda"; \
			fi; \
		fi; \
	fi; \
	if [ -n "$$JFROG_CHANNEL" ]; then \
		echo "  Using JFrog conda channel for internal packages"; \
		docker build --build-arg JFROG_CONDA_CHANNEL="$$JFROG_CHANNEL" \
			-t $(IMAGE_TAG) -t $(IMAGE_LATEST) .; \
	else \
		echo "  Warning: No JFrog credentials found. Internal packages may fail to install."; \
		docker build -t $(IMAGE_TAG) -t $(IMAGE_LATEST) .; \
	fi
	@echo "✓ Docker image built: $(IMAGE_TAG)"

# Push Docker image to ECR
docker-push:
	@echo "Pushing Docker image to ECR..."
	docker push $(IMAGE_TAG)
	docker push $(IMAGE_LATEST)
	@echo "✓ Docker image pushed: $(IMAGE_TAG)"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/ output/ work/ .pytest_cache/
	rm -f fragmentomics_tools/sequence.c fragmentomics_tools/*.so
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Clean complete"
