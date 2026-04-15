# Makefile for verb-gloss-wsd

UV=uv
FIND=find
MAKE=make
SRC=aidu.ai.llm
APP=app/

.PHONY: help install clean wipe serve run smoke test curl web.build

help:    ## Show this help
	@grep -h "##" $(MAKEFILE_LIST) | grep -v grep | sed -e "s/\$$//" -e "s/##//"

install: ## Install dependencies and set up environment
	@echo "Installing dependencies"
	@$(UV) sync

	@echo "Upgrading pip"
	@$(UV) run python -m ensurepip --upgrade

clean:  ## Clean temporary and cache files
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .venv
	$(FIND) . -type f -name '*~' -delete
	$(FIND) . -type f -name '*.pyc' -delete
	$(FIND) . -type d -name '__pycache__' -delete

wipe:   ## Delete all uv-related files for a fresh start
wipe: clean
	@echo "Removing uv.lock"
	rm -f uv.lock

serve:	## Run the web server for the application
	$(UV) run python -m serve.app

run:    ## Run the analysis application (default)
	$(UV) run python -m $(SRC).main

smoke:  ## Run a quick smoke on each src file
	@echo "smoke test"

	@echo "smoke test for client. Press Enter to continue..."
	@read dummy

	$(UV) run python -m $(SRC).client

	@echo "smoke test for request. Press Enter to continue..."
	@read dummy

	$(UV) run python -m $(SRC).requester

	
test:   ## Run all tests
	@echo "Running tests..."
	$(UV) run pytest

curl:	## Runs curl tests against the server
	@echo "Running curl tests..."
	test/curl_tests.sh

web.build:	## Build the web frontend
	cd web && $(Make) build
