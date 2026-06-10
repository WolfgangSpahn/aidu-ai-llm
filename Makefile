PACKAGE=aidu-ai-llm

EXAMPLE= aidu.ai.llm.demo.app

SMOKE_MODULES=\
	aidu.ai.llm.client \
	aidu.ai.llm.clients.openai \
	aidu.ai.llm.clients.google \
	aidu.ai.llm.clients.sympy \
	aidu.ai.llm.requester \
	aidu.ai.llm.plugin

include ../aidu-dev-tools/python-package.mk


web.install:                             ## Install web dependencies
	cd web && npm install

web.build:                               ## Build frontend and copy into package
	cd web && $(MAKE) build

	rm -rf src/aidu/ai/llm/demo/web
	cp -r web/dist src/aidu/ai/llm/demo/web

web.clean:                               ## Clean frontend build artifacts
	cd web && $(MAKE) clean

	rm -rf src/aidu/ai/llm/demo/web