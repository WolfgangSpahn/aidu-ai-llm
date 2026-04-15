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
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aidu.ai.llm.client import LLMClient, clean_message

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(funcName)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Smoke Chat API",
    description="Stateful chat backed by gpt-4o-mini, one session per conversation thread.",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant."

# ---------------------------------------------------------------------------
# State  (in-memory; replace with a real store for production)
# ---------------------------------------------------------------------------

# session_id -> list of message dicts
_sessions: dict[str, list[dict]] = {}


def _get_or_raise(session_id: str) -> list[dict]:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return _sessions[session_id]


def _make_client() -> LLMClient:
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")
    return LLMClient(api_key)


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


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[dict]


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
    _sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    logger.info("New session: %s", session_id)
    return SessionResponse(session_id=session_id)


@app.post(
    "/sessions/{session_id}/chat",
    response_model=ChatResponse,
    summary="Send a message and get the assistant reply",
)
def chat(
    session_id: Annotated[str, Path(description="Session ID returned by POST /sessions")],
    body: ChatRequest,
) -> ChatResponse:
    """
    Append the user message to the session history, call the LLM, append
    the assistant reply, and return it.
    """
    messages = _get_or_raise(session_id)

    messages.append({"role": "user", "content": body.message})

    client = _make_client()
    msg = client.chat(model=MODEL, messages=messages)

    reply = msg.get("content", "")
    messages.append(clean_message(msg))

    logger.info("session=%s  user=%r  reply=%r", session_id, body.message[:60], reply[:60])
    return ChatResponse(session_id=session_id, reply=reply)


@app.get(
    "/sessions/{session_id}/history",
    response_model=HistoryResponse,
    summary="Retrieve full message history for a session",
)
def get_history(
    session_id: Annotated[str, Path(description="Session ID")],
) -> HistoryResponse:
    """Return the full conversation history (including the system prompt)."""
    messages = _get_or_raise(session_id)
    return HistoryResponse(session_id=session_id, messages=messages)


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
    logger.info("Deleted session: %s", session_id)

app.mount("/", StaticFiles(directory="web/dist", html=True))

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve.app:app", host="0.0.0.0", port=8000, reload=True)
