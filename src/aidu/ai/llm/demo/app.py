# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
Small FastAPI server exposing a stateful chat interface.
Mirrors run_smoke_test_chat: one system prompt, per-session message history,
single POST endpoint for user turns.

Run:
     uv run python -m serve.app
     # or
     uvicorn serve.app:app --reload

 OpenAPI docs available at http://localhost:8000/docs
"""

import logging
import uuid
import uvicorn
from pathlib import Path as FsPath
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rich.logging import RichHandler
from rich.console import Console


from aidu.ai.llm.clients.openai import OpenAIClient
from aidu.ai.llm.agents.mathTutor import MathTutor
from aidu.ai.llm.evaluators.uncertainty import UncertaintyEvaluator
from aidu.ai.llm.client import Context, Trace, State

load_dotenv()

# ---------------------------------------------------------------------------
# Rich Logging Setup
# ---------------------------------------------------------------------------

console = Console()

# Configure logging with Rich handler - apply to root logger so uvicorn uses it too
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=True, show_level=True)])

# Get the root logger and apply Rich handler to it for uvicorn
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if isinstance(handler, RichHandler):
        break
else:
    root_logger.addHandler(RichHandler(console=console, rich_tracebacks=True, show_time=True, show_level=True))

# Suppress verbose third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.INFO)  # Keep INFO to see startup messages
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # Suppress HTTP access logs

logger = logging.getLogger(__name__)
logger.info("✓ Logging system initialized - AIDU LLM Chat API Starting")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Smoke Chat API",
    description="Stateful chat backed by gpt-4o-mini, one session per conversation thread.",
    version="0.1.0",
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("✓ CORS enabled for all origins")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_ACTOR = "MathTutor"
WEB_DIST_DIR = FsPath("web/dist")

# Registry for actor classes. Add new actors here as they are introduced.
ACTOR_REGISTRY = {
    "MathTutor": MathTutor,
}

# ---------------------------------------------------------------------------
# State  (in-memory; replace with a real store for production)
# ---------------------------------------------------------------------------

# session_id -> Context object (contains trace + state)
_sessions: dict[str, Context] = {}


def _get_or_raise(session_id: str) -> Context:
    """
    Retrieve the Context for a session_id or raise 404 if not found.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return _sessions[session_id]


def _make_client() -> OpenAIClient:
    """
    Create and return an LLMClient instance with the configured model and API key.
    """
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("✗ OPENAI_API_KEY not configured")
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")
    logger.debug("OpenAI client created")
    return OpenAIClient("gpt-4o-mini", config={"enforce_json": False}, api_key=api_key)


def _make_actor(actor_name: str = DEFAULT_ACTOR):
    """
    Create and return an actor instance with an LLMClient.
    """
    actor_cls = ACTOR_REGISTRY.get(actor_name)
    if actor_cls is None:
        available = ", ".join(sorted(ACTOR_REGISTRY.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Unknown actor '{actor_name}'. Available actors: {available}",
        )

    client = _make_client()
    actor = actor_cls(client)
    logger.debug(f"{actor_name} actor instantiated")
    return actor


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    context: dict


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[dict]


class EvaluateRequest(BaseModel):
    message: str
    context: str = ""
    correct_answer: str = ""


class EvaluateResponse(BaseModel):
    session_id: str
    distribution: list[float]


def _resolve_frontend_index() -> FsPath | None:
    """Return the built frontend entrypoint, supporting both raw and SolidJS builds."""
    primary = WEB_DIST_DIR / "index.html"
    if primary.exists():
        return primary

    solid = WEB_DIST_DIR / "index.solidjs.html"
    if solid.exists():
        return solid

    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post(
    "/sessions",
    response_model=SessionResponse,
    summary="Create a new chat session",
    status_code=201,
)
def create_session() -> SessionResponse:
    """
    Open a fresh conversation thread.  Returns the ``session_id`` to use in
    subsequent ``/sessions/{session_id}/chat`` calls.
    """
    session_id = str(uuid.uuid4())
    context = Context(
        trace=Trace(messages=[{"role": "system", "content": SYSTEM_PROMPT}]),
        state=State(data={}),
    )
    _sessions[session_id] = context
    logger.info(f"✓ Session created: {session_id}")
    return SessionResponse(session_id=session_id)


@app.post(
    "/sessions/{session_id}/{actor_id}/chat",
    response_model=ChatResponse,
    summary="Send a message and get the assistant reply",
)
def chat(
    session_id: Annotated[str, Path(description="Session ID returned by POST /sessions")],
    actor_id: Annotated[str, Path(description="Actor ID (e.g. MathTutor)")],
    body: ChatRequest,
) -> ChatResponse:
    """
    Append the user message to the session history, call the MathTutor actor,
    append the assistant reply, and return it. Supports math problem solving
    and student progress tracking through function calls.
    """
    try:
        msg_preview = body.message[:60] + ("..." if len(body.message) > 60 else "")
        logger.info(f"→ Session {session_id} | Message: {msg_preview}")

        context = _get_or_raise(session_id)
        user_message = {"role": "user", "content": body.message}
        logger.debug(f"  History: {len(context.trace.messages)} messages")

        actor = _make_actor(actor_id)
        logger.debug(f"Calling {actor_id}.chat_turn()...")
        reply, context = actor.chat_turn(context=context, user_message=user_message)
        logger.debug(f" {actor_id}.chat_turn() completed and updated context")

        # Persist updated context
        _sessions[session_id] = context

        reply_preview = reply[:60] + ("..." if len(reply) > 60 else "")
        logger.info(f"✓ Reply (first 60 chars): {reply_preview}")
        logger.info(f"✓ Full reply ({len(reply)} chars): {reply}")
        logger.info(
            f"✓ Updated context: {len(context.trace.messages)} messages in context.trace.messages, context.state keys: {list(context.state.data.keys())}, context.control.data: {context.control.data}, context.control.duration: {context.control.duration:.2f}s"
        )
        return ChatResponse(session_id=session_id, reply=reply, context=context.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/sessions/{session_id}/history",
    response_model=HistoryResponse,
    summary="Retrieve full message history for a session",
)
def get_history(
    session_id: Annotated[str, Path(description="Session ID")],
) -> HistoryResponse:
    """Return the full conversation history (including the system prompt)."""
    context = _get_or_raise(session_id)
    return HistoryResponse(session_id=session_id, messages=context.trace.messages)


@app.delete(
    "/sessions/{session_id}",
    status_code=204,
    summary="Delete a session and its history",
)
def delete_session(
    session_id: Annotated[str, Path(description="Session ID")],
) -> None:
    """Remove a session, freeing its stored message history."""
    _get_or_raise(session_id)
    del _sessions[session_id]
    logger.info(f"⊘ Session deleted: {session_id}")


@app.delete(
    "/sessions",
    status_code=204,
    summary="Clear all sessions",
)
def clear_all_sessions() -> None:
    """Clear all sessions. Useful for development/testing."""
    global _sessions
    count = len(_sessions)
    _sessions.clear()
    logger.info(f"✓ Cleared all {count} sessions")


@app.post(
    "/sessions/{session_id}/evaluate",
    response_model=EvaluateResponse,
    summary="Evaluate student confidence calibration",
)
def evaluate(
    session_id: Annotated[str, Path(description="Session ID")],
    body: EvaluateRequest,
) -> EvaluateResponse:
    """
    Evaluate a student's confidence calibration for a given message and context.
    Returns a probability distribution across the Likert scale.
    """
    try:
        _get_or_raise(session_id)

        client = _make_client()
        evaluator = UncertaintyEvaluator(client)

        distribution = evaluator.evaluate(eval_params={"text": body.message, "context": body.context, "correct_answer": body.correct_answer})

        if distribution is None:
            raise HTTPException(status_code=500, detail="Evaluation failed")

        logger.info(f"✓ Evaluation complete: {[f'{v:.2f}' for v in distribution]}")
        return EvaluateResponse(session_id=session_id, distribution=distribution)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error in evaluate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", include_in_schema=False)
def frontend_index() -> FileResponse:
    """Serve whichever frontend variant was built most recently."""
    index_file = _resolve_frontend_index()
    if index_file is None:
        raise HTTPException(
            status_code=404,
            detail="Frontend build not found. Build web assets first (e.g., make web.build).",
        )
    return FileResponse(index_file)


app.mount("/", StaticFiles(directory="web/dist", html=True))


def main():
    uvicorn.run(
        "aidu.ai.llm.demo.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
