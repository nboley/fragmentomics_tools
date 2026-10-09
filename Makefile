# Makefile for building and publishing conda packages and Docker images
# fragmentomics-tools

PACKAGE_NAME := fragmentomics-tools
VERSION ?= $(shell grep '^version' pyproject.toml | head -1 | sed -E 's/.*"([^"]+)".*/\1/')
ECR_REGISTRY := 573640641260.dkr.ecr.us-east-1.amazonaws.com
IMAGE_NAME := karius-$(PACKAGE_NAME)
IMAGE_TAG := $(ECR_REGISTRY)/$(IMAGE_NAME):$(VERSION)
IMAGE_LATEST := $(ECR_REGISTRY)/$(IMAGE_NAME):latest

.PHONY: all login conda-login docker-login tag conda conda-build conda-publish docker docker-build docker-push clean help test test-realdata

# Pin the interpreter so `make test` uses the correct conda env. Override with
# `make test PYTHON=/path/to/other/python`.
PYTHON ?= /home/nathanboley/miniconda3/envs/biomarker_env/bin/python

# Pinning the interpreter is not sufficient on its own: `test_formats.py`
# drives the `bedToBigBed`, `tabix` and `bedtools` BINARIES, which live in the
# same env's bin/ and are not found via the interpreter. Without them on PATH
# about 40 test_formats tests fail and none of it is a regression. Putting
# $(PYTHON)'s own bin/ first makes `make test` self-contained rather than
# PATH-dependent.
export PATH := $(dir $(PYTHON)):$(PATH)

# Wall-clock bound on a test run. This is a hang detector, not a perf budget:
# the library suite finishes in well under a minute. Override for slow hosts
# or a bigger suite: make test TEST_TIMEOUT=7200
TEST_TIMEOUT ?= 3600

# --doctest-modules runs docstring examples as tests. Turning it on found
# three LIVE defects that the test suite and a five-reviewer static pass had
# all missed: numpy.product (removed in numpy 2.0), a crash on strandless
# Regions, and a missing Fragment import breaking three public entry points.
# Keep it on.
#
# The two ignored packages import optional dependencies (datamanifest, fbio)
# that are absent from the test environment, so their modules cannot even be
# collected. That is an environment gap, not a code defect.
# `tests/` (background_model) IS collected, alongside `test/` (library). It was
# omitted until 2026-10-07, which meant 677 tests never ran under the default
# target -- including a module whose only coverage had just been added. A suite
# that does not run by default is close to no suite.
# The two suites are easy to confuse: `test/` is the library, `tests/` is
# background_model. Both are here on purpose.
# `--ignore=tests/conftest.py` is NOT about skipping tests. It exists so that
# a `tests/conftest.py` CAN exist at all.
#
# --doctest-modules makes pytest COLLECT conftest.py files as modules to scan
# for doctests. With no __init__.py in these directories, prepend import mode
# imports every one under the bare name `conftest`, so the second collection
# fails with "import file mismatch" against test/fragment_array/conftest.py.
# An agent hit this, diagnosed it correctly and then deleted its conftest,
# which is the wrong lever.
#
# --ignore stops the COLLECTION only; pytest still loads the file as a plugin,
# so fixtures, markers and hooks all work. Verified by a pytest_report_header
# hook firing under this exact invocation.
#
# Rejected alternatives: importmode=importlib and adding __init__.py both fix
# the collision but break `import cut_site_oracle` in
# tests/test_cut_site_simulator.py, which relies on prepend mode putting
# tests/ on sys.path. Moving test/fragment_array/conftest.py to the root
# breaks its DATA_DIR, which is built from __file__.
PYTEST_ARGS ?= test/ tests/ fragmentomics_tools/ -q --doctest-modules \
	--ignore=fragmentomics_tools/bias_correction \
	--ignore=fragmentomics_tools/public_data_resources \
	--ignore=tests/conftest.py

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
	@echo "  test          Run the test suite under a $(TEST_TIMEOUT)s timeout"
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
	@# VERSION is read from the WORKING TREE. If pyproject.toml is uncommitted,
	@# the tag would point at HEAD, which declares a different version.
	@if ! git diff --quiet HEAD -- pyproject.toml; then \
		echo "❌ Error: pyproject.toml has uncommitted changes; refusing to tag."; \
		echo "  Working tree declares $(VERSION), but the tag would point at HEAD,"; \
		echo "  which declares something else. Commit the version bump first."; \
		exit 1; \
	fi
	@echo "Creating git tag v$(VERSION)..."
	@if git rev-parse "v$(VERSION)" >/dev/null 2>&1; then \
		echo "❌ Error: Tag v$(VERSION) already exists"; \
		echo "Please bump the version in pyproject.toml first"; \
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
	@rattler-build build --recipe recipe/recipe.yaml --channel conda-forge --channel bioconda --no-test --variant pkg_version=$(VERSION); \
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

# Run the tests under a wall-clock bound.
#
# Always run the suite this way rather than invoking pytest bare. This suite
# can wedge rather than fail -- a fork deadlock in parallel_apply once ran for
# 12 hours before anyone looked, producing no output at all, because a bare
# `pytest -q | tail` never reaches EOF when the process never exits.
# SIGKILL rather than SIGTERM: a process stuck in an uninterruptible futex
# wait will not act on a catchable signal.

# The interpreter is PINNED, not taken from PATH. A bare `python` resolves to
# whatever the caller's shell has, and in an agent/MCP shell that was
# /opt/conda/envs/claude-mcp/bin/python -- no pybedtools, no torch. The run then
# died with 12 COLLECTION ERRORS that look exactly like broken tests. That cost
# two people time independently on 2026-10-07, each initially reading it as repo
# breakage rather than a wrong interpreter.
# Override for a different env: make test PYTHON=/path/to/python
PYTHON ?= /home/nathanboley/miniconda3/envs/biomarker_env/bin/python

# The interpreter's own bin/ goes on PATH too, and pinning PYTHON alone is NOT
# enough. Several tests reach BINARIES that live beside it, not python modules:
# `bedtools` (via pybedtools -- CLAUDE.md flags this specifically), plus `bgzip`
# and `tabix`. Pinning only the interpreter took the suite from 2 failed to
# **54 failed**, every one of them "intersectBed does not appear to be installed
# or on the path". The failures look like broken interval logic, not a missing
# binary, which is what makes it worth spelling out here.
PYTHON_BIN := $(dir $(PYTHON))

test:
	@if [ ! -x "$(PYTHON)" ]; then \
		echo "❌ interpreter not found: $(PYTHON)"; \
		echo "   This target pins the interpreter on purpose -- a bare 'python'"; \
		echo "   picks up whatever is on PATH and fails as collection errors."; \
		echo "   Override with: make test PYTHON=/path/to/python"; \
		exit 2; \
	fi
	@PATH="$(PYTHON_BIN):$$PATH" \
	timeout --signal=KILL $(TEST_TIMEOUT) $(PYTHON) -m pytest $(PYTEST_ARGS); \
	rc=$$?; \
	if [ $$rc -eq 137 ]; then \
		echo ""; \
		echo "❌ Suite KILLED after $(TEST_TIMEOUT)s. It HUNG -- it did not fail."; \
		echo "   A hang is a finding, not a flake. To find out where:"; \
		echo "     py-spy dump --pid <the pytest pid>"; \
		echo "   Check child processes too; a stuck fork child shows 0s CPU."; \
	fi; \
	exit $$rc

# Real-data orientation check. An ordinary `make test` SKIPS it when the EFS
# h5 inputs are absent, so an off-EFS checkout still has a usable suite. This
# target passes --realdata, which turns absent inputs into a failure.
#
# That distinction is the whole point. On the version_2 branch a committed
# interval manifest went unread by any test for a whole phase, and two real
# regressions lived in the repo as a result: a 592-interval `merge` movement
# recorded as "deliberate", and cross-process non-determinism no in-process
# test could observe. A regression net that can silently skip is how both
# survived.
#
# Only the orientation test is wired here. The interval and bedtools
# equivalence suites live on version_2 and do not exist on this branch.
test-realdata:
	@timeout --signal=KILL $(TEST_TIMEOUT) $(PYTHON) -m pytest \
		test/test_orientation_real_data.py -v --realdata; \
	rc=$$?; \
	if [ $$rc -eq 137 ]; then \
		echo ""; \
		echo "❌ KILLED after $(TEST_TIMEOUT)s. It HUNG -- it did not fail."; \
		exit 137; \
	fi; \
	exit $$rc

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/ output/ work/ .pytest_cache/
	rm -f fragmentomics_tools/sequence.c fragmentomics_tools/*.so
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Clean complete"
