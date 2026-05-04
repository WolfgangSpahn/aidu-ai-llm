# Makefile for verb-gloss-wsd

UV=uv
FIND=find
MAKE=make
SRC=aidu.ai.llm
APP=app/

.PHONY: help install clean wipe serve run smoke test curl web.build

help:                                     ## Show this help
	@grep -h "##" $(MAKEFILE_LIST) | grep -v grep | sed -e "s/\$$//" -e "s/##//"

# install targets

install:                                  ## Install python dependencies and set up environment
	@echo "Installing dependencies"
	@$(UV) sync

	@echo "Upgrading pip"
	@$(UV) run python -m ensurepip --upgrade

# Cleanup targets

clean.server:                             ## Clean temporary and cache files
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .venv
	$(FIND) . -type f -name '*~' -delete
	$(FIND) . -type f -name '*.pyc' -delete
	$(FIND) . -type d -name '__pycache__' -delete

wipe:                                     ## Delete all uv-related files for a fresh start
wipe: clean
	@echo "Removing uv.lock"
	rm -f uv.lock


# Application targets

serve:	                                  ## Run the web server for the application
	$(UV) run python -m serve.app

run:                                      ## Run the analysis application (default)
	@echo "Running the application"
	@echo "no main yet"

# Smoke test targets

smoke.client:                             ## Run a quick smoke test for the client
	$(UV) run python -m $(SRC).client

smoke.client.llm:						  ## Run a quick smoke test for the LLM client	
	$(UV) run python -m $(SRC).clients.llm

smoke.client.sympy:					      ## Run a quick smoke test for the SymPy client	
	$(UV) run python -m $(SRC).clients.sympy

smoke.requester:                          ## Run a quick smoke test for the requester
	$(UV) run python -m $(SRC).requester

smoke.actor:                              ## Run a quick smoke test for the actor
	$(UV) run python -m $(SRC).actor

smoke.actors.mathTutor:                   ## Run a quick smoke test for the math tutor actor
	$(UV) run python -m $(SRC).actors.mathTutor

smoke:									  ## Run all smoke tests
	$(MAKE) smoke.client
	$(MAKE) smoke.client.llm
	$(MAKE) smoke.client.sympy
	$(MAKE) smoke.requester
	$(MAKE) smoke.actor
	$(MAKE) smoke.actors.mathTutor

# Testing targets
	
test:                                     ## Run all tests
	@echo "Running tests..."
	$(UV) run pytest

curl:	                                  ## Runs curl tests against the server
	@echo "Running curl tests..."
	test/curl_tests.sh

# Web frontend targets

web.clean:                                ## Clean up the web frontend
	cd web && $(MAKE) clean
web.install:	                          ## Install web frontend dependencies
	cd web && $(MAKE) install
web.build:                                ## Build the web frontend
	cd web && $(MAKE) build


clean: clean.server web.clean
	@echo "Cleaned server and web frontend"