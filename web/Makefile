.PHONY: help install build clean

help:    ## Show this help
	@grep -h "##" $(MAKEFILE_LIST) | grep -v grep | sed -e "s/\$$//" -e "s/##//"



install: ## Install frontend dependencies (Vite)
	@echo "Installing frontend dependencies"
	npm install

build: ## Build frontend for production
	@echo "Building frontend"
	npm run build

clean: ## Clean node_modules and lock files
	@echo "Cleaning dependencies"
	rm -rf node_modules package-lock.json
