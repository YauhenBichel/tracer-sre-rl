.PHONY: install test demo train train-real crawl update-data export lint format typecheck security check clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

install-dev:  ## Install with test dependencies
	pip install -r requirements-dev.txt

test:  ## Run all tests
	python -m pytest tests/ -v

demo:  ## Run baseline comparison (random vs heuristic vs oracle)
	python demo.py --quiet

train:  ## Train using config/training.yaml (edit to change data/agent/episodes)
	python run_training.py

train-real:  ## Train on real Aiops-Dataset faults (241 incidents + builtins)
	python run_training.py

train-all:  ## Crawl + train on everything
	python run_crawler.py --db-path data/incidents.db
	python run_training.py

crawl:  ## Crawl public incident APIs into data/incidents.db
	python run_crawler.py --db-path data/incidents.db

update-data:  ## Re-crawl incidents and refresh dataset
	python run_crawler.py --db-path data/incidents.db
	@echo ""
	@echo "Dataset updated. Edit config/training.yaml to include:"
	@echo "  incident_db: data/incidents.db"
	@echo "Then: make train"

export:  ## Export training trajectories as JSONL
	python run_training.py --export training_data/trajectories.jsonl

lint:  ## Run ruff linter
	ruff check src/ tests/

lint-fix:  ## Auto-fix lint issues
	ruff check --fix src/ tests/

format:  ## Check code style formatting
	ruff format --check src/ tests/

format-fix:  ## Auto-format code style
	ruff format src/ tests/

typecheck:  ## Run mypy static type analysis
	mypy src/

security:  ## Run bandit security scan
	bandit -r src/ -c pyproject.toml

check:  ## Run all checks (lint + format + typecheck + security)
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/
	bandit -r src/ -c pyproject.toml

clean:  ## Remove generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -f data/incidents.db
	rm -rf training_data/ scenarios/generated/ .pytest_cache/
