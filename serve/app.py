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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rich.logging import RichHandler
from rich.console import Console

from aidu.ai.llm.client import LLMClient
from aidu.ai.llm.actors import MathTutor
from aidu.ai.llm.evaluators.uncertainty import UncertaintyEvaluator

load_dotenv()

# ---------------------------------------------------------------------------
# Rich Logging Setup
# ---------------------------------------------------------------------------

console = Console()

# Configure logging with Rich handler - apply to root logger so uvicorn uses it too
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=True, show_level=True)]
)

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

# ---------------------------------------------------------------------------
# State  (in-memory; replace with a real store for production)
# ---------------------------------------------------------------------------

# session_id -> list of message dicts
_sessions: dict[str, list[dict]] = {}
# session_id -> actor state dict
_session_states: dict[str, dict] = {}


def _clean_message_for_storage(msg: dict) -> dict:
    """
    Clean message for safe storage in message history.
    Removes internal fields and ensures no conflicting fields (tool_calls vs function_call).
    """
    cleaned = {}
    
    # Keep essential fields
    if "role" in msg:
        cleaned["role"] = msg["role"]
    if "content" in msg and msg["content"]:
        cleaned["content"] = msg["content"]
    
    # Keep function_call but NOT tool_calls (they conflict in the API)
    if "function_call" in msg and msg["function_call"]:
        cleaned["function_call"] = msg["function_call"]
    
    # Remove internal fields
    # (skip _fc_message, tool_calls, etc.)
    
    return cleaned


def _get_or_raise(session_id: str) -> list[dict]:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return _sessions[session_id]


def _get_state_or_create(session_id: str) -> dict:
    if session_id not in _session_states:
        _session_states[session_id] = {}
    return _session_states[session_id]


def _make_client() -> LLMClient:
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("✗ OPENAI_API_KEY not configured")
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")
    logger.debug("OpenAI client created")
    return LLMClient(api_key)


def _make_actor() -> MathTutor:
    client = _make_client()
    logger.debug("MathTutor actor instantiated")
    return MathTutor(client, prompt_template=SYSTEM_PROMPT)


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


class EvaluateRequest(BaseModel):
    message: str
    context: str = ""
    correct_answer: str = ""


class EvaluateResponse(BaseModel):
    session_id: str
    distribution: list[float]


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
    _session_states[session_id] = {}
    logger.info(f"✓ Session created: {session_id}")
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
    Append the user message to the session history, call the MathTutor actor,
    append the assistant reply, and return it. Supports math problem solving
    and student progress tracking through function calls.
    """
    try:
        msg_preview = body.message[:60] + ("..." if len(body.message) > 60 else "")
        logger.info(f"→ Session {session_id} | Message: {msg_preview}")
        
        messages = _get_or_raise(session_id)
        state = _get_state_or_create(session_id)

        messages.append({"role": "user", "content": body.message})
        logger.debug(f"  History: {len(messages)} messages")

        actor = _make_actor()
        logger.debug("Calling MathTutor.run()...")
        msg, state = actor.run(messages=messages, model=MODEL, state=state)
        logger.debug(f"  Response role: {msg.get('role', 'unknown')} | Content: {len(msg.get('content', ''))} chars")

        # Extract reply from content, function call result, or state update
        reply = msg.get("content", "")
        if not reply and msg.get("_fc_message"):
            reply = msg.get("_fc_message")
            logger.info(f"  Function result: {reply[:60]}...")
        if not reply and msg.get("function_call"):
            fc = msg.get("function_call")
            reply = f"Executing {fc['name']}..."
            logger.info(f"  Function call: {fc['name']}")
        
        # Store cleaned message (removes tool_calls/function_call conflicts)
        messages.append(_clean_message_for_storage(msg))
        _session_states[session_id] = state

        reply_preview = reply[:60] + ("..." if len(reply) > 60 else "")
        logger.info(f"✓ Reply (first 60 chars): {reply_preview}")
        logger.info(f"✓ Full reply ({len(reply)} chars): {reply}")
        return ChatResponse(session_id=session_id, reply=reply)
    
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
    if session_id in _session_states:
        del _session_states[session_id]
    logger.info(f"⊘ Session deleted: {session_id}")


@app.delete(
    "/sessions",
    status_code=204,
    summary="Clear all sessions",
)
def clear_all_sessions() -> None:
    """Clear all sessions. Useful for development/testing."""
    global _sessions, _session_states
    count = len(_sessions)
    _sessions.clear()
    _session_states.clear()
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
        
        distribution = evaluator.evaluate(
            eval_params={
                "text": body.message,
                "context": body.context,
                "correct_answer": body.correct_answer
            }
        )
        
        if distribution is None:
            raise HTTPException(status_code=500, detail="Evaluation failed")
        
        logger.info(f"✓ Evaluation complete: {[f'{v:.2f}' for v in distribution]}")
        return EvaluateResponse(session_id=session_id, distribution=distribution)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error in evaluate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/", StaticFiles(directory="web/dist", html=True))

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve.app:app", host="0.0.0.0", port=8000, reload=True)
