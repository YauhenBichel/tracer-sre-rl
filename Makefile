.PHONY: install test check-reward train crawl export finetune clean help

help:  ## Show all available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

quickstart:  ## Set up the project and verify everything works
	@echo "============================================================"
	@echo "Step 1: Installing dependencies"
	@echo "============================================================"
	pip install -r requirements.txt -q
	@echo ""
	@echo "============================================================"
	@echo "Step 2: Running tests"
	@echo "============================================================"
	python -m pytest tests/ -q
	@echo ""
	@echo "============================================================"
	@echo "Step 3: Validating reward signal"
	@echo "============================================================"
	python -m app.main check-reward --quiet
	@echo ""
	@echo "============================================================"
	@echo "Done! Next steps:"
	@echo "  make train     # Run training episodes"
	@echo "  make export    # Save trajectories for LLM fine-tuning"
	@echo "  make help      # See all commands"
	@echo "============================================================"

install:  ## Install Python dependencies
	pip install -r requirements.txt

install-dev:  ## Install with test and lint tools
	pip install -r requirements-dev.txt

test:  ## Run all tests
	python -m pytest tests/ -v

check-reward:  ## Verify the reward function scores agents correctly (random < heuristic < oracle)
	python -m app.main check-reward --quiet

train:  ## Run training episodes and show accuracy per failure type
	python -m app.main train

crawl:  ## Fetch real incidents from GCP, Cloudflare, and GitHub APIs
	python -m app.main crawl --db-path data/incidents.db

export:  ## Save training trajectories to a file for LLM fine-tuning
	python -m app.main export

finetune:  ## Run LLM fine-tuning on saved trajectories (shows what would happen without GPU)
	python -m app.main finetune

clean:  ## Delete generated files (database, trajectories, caches)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -f data/incidents.db
	rm -rf training_data/ scenarios/generated/ .pytest_cache/
