.PHONY: help install build clean smoke

help:    ## Show this help
	@grep -h "##" $(MAKEFILE_LIST) | grep -v grep | sed -e "s/\$$//" -e "s/##//"



install: ## Install frontend dependencies (Vite)
	@echo "Installing frontend dependencies"
	npm install

build: build.solidjs ## Build frontend for production (alias for build.solidjs)

build.raw: ## Build raw TypeScript variant (main.ts + index.html)
	@echo "Building raw TypeScript variant"
	npm run build -- --config vite.config.ts

build.solidjs: ## Build SolidJS variant (main_solidjs.tsx + index.solidjs.html)
	@echo "Building SolidJS variant"
	npm run build -- --config vite.config.solidjs.ts

clean: ## Clean node_modules and lock files
	@echo "Cleaning dependencies"
	rm -rf node_modules package-lock.json
