# Makefile for verb-gloss-wsd

UV=uv
FIND=find
MAKE=make
LLM_SRC=aidu.ai.llm
SYM_SRC=aidu.ai.symbolic
APP=app/

.PHONY: help install clean wipe serve run smoke test curl web.build

help:                                     ## Show this help
	@grep -h "##" $(MAKEFILE_LIST) | grep -v grep | sed -e "s/\$$//" -e "s/##//"

# install targets

server.install:                                  ## Install python dependencies and set up environment
	@echo "Installing dependencies"
	@$(UV) sync

	@echo "Upgrading pip"
	@$(UV) run python -m ensurepip --upgrade

# Cleanup targets

server.clean:                             ## Clean temporary and cache files
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .venv
	$(FIND) . -type f -name '*~' -delete
	$(FIND) . -type f -name '*.pyc' -delete
	$(FIND) . -type d -name '__pycache__' -delete

wipe:                                     ## Delete all uv-related files for a fresh start
wipe: server.clean
	@echo "Removing uv.lock"
	rm -f uv.lock


# Application targets

server.run:	                               ## Run the web server for the application
	$(UV) run python -m serve.app

app.run:                                   ## Run the analysis application (default)
	@echo "Running the application"
	@echo "no main yet"

# Smoke test targets

smoke.client:                             ## Run a quick smoke test for the client
	$(UV) run python -m $(LLM_SRC).client

smoke.clients.openai:						  ## Run a quick smoke test for the LLM client	
	$(UV) run python -m $(LLM_SRC).clients.openai

smoke.clients.google:						  ## Run a quick smoke test for the Google Gemini client	
	$(UV) run python -m $(LLM_SRC).clients.google

smoke.clients.sympy:					      ## Run a quick smoke test for the SymPy client	
	$(UV) run python -m $(LLM_SRC).clients.sympy

smoke.requester:                          ## Run a quick smoke test for the requester
	$(UV) run python -m $(LLM_SRC).requester

smoke.actor:                              ## Run a quick smoke test for the actor
	$(UV) run python -m $(LLM_SRC).actor

smoke.agents.mathTutor:                   ## Run a quick smoke test for the math tutor agent
	$(UV) run python -m $(LLM_SRC).agents.mathTutor

smoke.solver.mathSolver:				   ## Run a quick smoke test for the math solver
	$(UV) run python -m $(LLM_SRC).solver.MathSolver

smoke.plugin:								   ## Run a quick smoke test for the plugin
	$(UV) run python -m $(LLM_SRC).plugin

smoke.engines.symbolicSolver:						   ## Run a quick smoke test for the symbolic solver
	$(UV) run python -m $(SYM_SRC).engines.SymbolicSolver

smoke:									  ## Run all smoke tests
	$(MAKE) smoke.clients.openai
	$(MAKE) smoke.clients.google
	$(MAKE) smoke.clients.sympy
	$(MAKE) smoke.requester
	$(MAKE) smoke.agents.mathTutor
	$(MAKE) smoke.solver.mathSolver
	$(MAKE) smoke.plugin
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

jupyter:        ## Start a jupyter notebook server
	@if [ ! -d ".venv" ]; then uv venv; fi
	uv pip install jupyter
	uv run jupyter lab

clean: server.clean web.clean
	@echo "Cleaned server and web frontend"

install: server.install web.install
	@echo "Installed server and web frontend"

serve: server.run
	@echo "Running the application"