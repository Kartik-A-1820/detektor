.DEFAULT_GOAL := help
PY ?= python
WEIGHTS ?= runs/chimera/chimera_best.pt

.PHONY: help install install-dev lint fix test test-fast bench bench-quick bench-all serve serve-ui docker docker-run clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	$(PY) -m pip install -r requirements.txt

install-dev: ## Install runtime + development dependencies
	$(PY) -m pip install -r requirements-dev.txt

lint: ## Static checks (ruff)
	ruff check .

fix: ## Apply safe lint fixes
	ruff check . --fix

test: ## Run the full test suite
	$(PY) -m pytest

test-fast: ## Run tests, stopping at the first failure
	$(PY) -m pytest -x -q

bench-quick: ## 1-minute benchmark smoke run (smallest profile)
	$(PY) -m benchmarks run --suites fast --profiles firefly --img-sizes 320 --quick --tag quick

bench: ## Core benchmarks for the default profiles (≈5–10 min on CPU)
	$(PY) -m benchmarks run --suites fast --tag core

bench-all: ## Every suite for every profile, incl. API load + synthetic end-to-end (≈30+ min on CPU)
	$(PY) -m benchmarks run --suites all --profiles all --tag full

serve: ## Serve WEIGHTS over HTTP on :8000
	$(PY) serve.py --weights $(WEIGHTS)

serve-ui: ## Serve WEIGHTS with the web console at /ui
	$(PY) serve.py --weights $(WEIGHTS) --ui

docker: ## Build the CPU image
	docker build -t detektor:latest .

docker-run: ## Run the CPU image (expects ./artifacts/model.pt)
	docker run --rm -p 8000:8000 -v $(PWD)/artifacts:/artifacts:ro detektor:latest

clean: ## Remove caches and local benchmark output
	rm -rf .pytest_cache .ruff_cache runs/benchmarks bench-out
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
