# aidu-ai-llm

Small Python LLM playground from PHBern to help people to understand how to build AI-Tutors

with:
- reusable LLM utilities under `src/aidu/ai/llm`
- a FastAPI chat backend in `serve/app.py`
- a minimal Vite frontend in `web/`

## Prerequisites

The following tools are required to build and run the project:

- **make** — for running common development and build tasks  
- **Python (>=3.11)** — backend runtime  
- **uv (latest)** — Python package manager and environment runner  
- **Node.js (>=18, includes npm)** — required to build the frontend (TypeScript via Vite)

### 💡 Notes

* `uv` replaces `pip`, `venv`, etc. → state of the art: no manual environment activation needed
* `npm` is included with Node.js, so you don’t need to install it separately

## Quick Start

Copy .env_example to .env and add your OpenAI Token there.

When you have ensured the prerequisites, this should work

```bash
make install
make serve
```

Open: http://localhost:8000

## Project Layout

```text
serve/
  app.py                 FastAPI app (chat/session endpoints + static web mount)

src/aidu/ai/llm/
  client.py              OpenAI client wrapper
  requester.py           Base class interacting with ai client
  builder.py             Prompt template builder
  safeformat.py          Safe string formatting helpers
  tool_registry.py       Tool registration/execution helpers
  spec.py                Plugin spec definitions
  evaluator.py           Evaluator overloads Requestor

test/
  curl_tests.sh          Curl-based API smoke checks

web/
  index.html             Chat page shell
  src/main.ts            Browser chat logic
  HELP.md                License/help notes
```

## Useful Commands

```bash
make smoke              # runs smoke test on each individual file
make serve              # run FastAPI server
make curl               # run curl tests against above server

cd web && make install  # install frontend deps
cd web && make build    # build frontend
```

## Notes

- Requires `OPENAI_API_KEY` in your environment (or `.env`).
- The frontend talks to the backend session endpoints under `/sessions/...`.

