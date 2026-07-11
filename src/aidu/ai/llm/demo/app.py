# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
Small FastAPI server exposing a stateful chat interface.
The demo owns HTTP/session concerns; assistants own prompts and turn handling.

Run:
     uv run python -m serve.app
     # or
     uvicorn serve.app:app --reload

 OpenAPI docs available at http://localhost:8000/docs
"""

import logging
import json
import uuid
import uvicorn
from pathlib import Path as FsPath
import time
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rich.logging import RichHandler
from rich.console import Console


from aidu.ai.llm.clients.openai import OpenAIClient
from aidu.ai.llm.assistants.mathAssistent_ass import MathAssistent
from aidu.ai.llm.evaluators.uncertainty import UncertaintyEvaluator
from aidu.ai.llm.agent import EndAgent
from aidu.ai.llm.agent_runner import run_agent_artifact_chat_turn, run_agent_chat_turn
from aidu.ai.core.artifacts import AppletArtifact
from aidu.ai.core.context import Context
from aidu.ai.agents.chem_applet_tutor import AppletRuleResponder, ChemLlmTutor, ChemLlmUserInput
from aidu.ai.agents.math_tutor import MathTutor
from aidu.ai.agents.symbolic_solver import SymbolicSolver

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
    title="AIDU LLM Demo API",
    description="Stateful chat API backed by registered AIDU assistants.",
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
DEFAULT_RUNNABLE_KIND = "assistant"
DEFAULT_RUNNABLE_ID = "MathAssistent"
WEB_DIR = FsPath(__file__).parent / "web"
ASSETS_DIR = WEB_DIR / "assets"

logger.info("ASSETS_DIR=%s", ASSETS_DIR)
logger.info("ASSETS exists=%s", ASSETS_DIR.exists())

INDEX_FILE = "index.html"
logger.info("WEB_DIR = %s", WEB_DIR)
logger.info("exists = %s", WEB_DIR.exists())


# Registry for assistant classes. Add new assistants here as they are introduced.
ASSISTANT_REGISTRY = {
    "MathAssistent": MathAssistent,
}

AGENT_REGISTRY = {
    "AppletRuleResponder": AppletRuleResponder,
    "ChemLlmTutor": ChemLlmTutor,
    "MathTutor": MathTutor,
}

RUNNABLES = [
    {
        "kind": "assistant",
        "id": "MathAssistent",
        "label": "Math Assistant",
        "input_types": ["text"],
    },
    {
        "kind": "agent",
        "id": "MathTutor",
        "label": "Math Tutor Agent",
        "input_types": ["text"],
    },
    {
        "kind": "agent",
        "id": "ChemLlmTutor",
        "label": "Chem Applet Tutor",
        "input_types": ["text"],
    },
    {
        "kind": "agent",
        "id": "AppletRuleResponder",
        "label": "Applet Rule Responder",
        "input_types": ["applet"],
    },
]

AGENT_PROMPT_ARGS = {
    "ChemLlmTutor": {
        **ChemLlmTutor.default_args,
        "tutor_name": "",
        "level": "beginner",
        "history": "",
        "student_progress": "",
        "student_belief": "",
        "domain_id": "chemistry-basics",
        "domain_label": "Chemistry Basics",
        "domain_description": "Explore atoms, particles, valence electrons, and bonding with an interactive chemistry applet.",
        "learning_targets": "- Use particle counts to reason about atoms and ions\n- Connect valence electrons to bonding and reactivity\n- Explain observations from the current applet state",
        "applet_id": "chem-applet",
        "applet_name": "Chemistry Applet",
        "applet_description": "Interactive chemistry simulation connected to the tutoring conversation.",
        "applet_remote_control": "{}",
        "applet_info_store_schema": "{}",
        "applet_state": "{}",
    },
    "MathTutor": {
        "tutor_name": "",
        "focus_area": "general math",
        "level": "beginner",
        "history": "",
        "student_progress": "",
        "student_beliefs": "",
    },
}

AGENT_WORKFLOW_COMPANIONS = {
    "AppletRuleResponder": lambda: [EndAgent()],
    "ChemLlmTutor": lambda: [ChemLlmUserInput(), AppletRuleResponder(), EndAgent()],
    "MathTutor": lambda: [SymbolicSolver(), EndAgent()],
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
    return OpenAIClient(MODEL, config={"enforce_json": False}, api_key=api_key)


def _make_assistant(assistant_id: str = DEFAULT_RUNNABLE_ID):
    """
    Create and return an assistant instance with an LLMClient.
    """
    assistant_cls = ASSISTANT_REGISTRY.get(assistant_id)
    if assistant_cls is None:
        available = ", ".join(sorted(ASSISTANT_REGISTRY.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Unknown assistant '{assistant_id}'. Available assistants: {available}",
        )

    client = _make_client()
    assistant = assistant_cls(client)
    logger.debug(f"{assistant_id} assistant instantiated")
    return assistant


def _make_agent_workflow(agent_id: str):
    """
    Create and return an agent workflow rooted at ``agent_id``.
    """
    if agent_id == "AppletRuleResponder":
        raise HTTPException(status_code=400, detail=f"Agent '{agent_id}' does not accept text input.")

    agent_cls = AGENT_REGISTRY.get(agent_id)
    if agent_cls is None:
        available = ", ".join(sorted(AGENT_REGISTRY.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent '{agent_id}'. Available agents: {available}",
        )

    client = _make_client()
    starting_agent = agent_cls(client=client)
    companions_factory = AGENT_WORKFLOW_COMPANIONS.get(agent_id, lambda: [EndAgent()])
    agents = [starting_agent, *companions_factory()]
    logger.debug(f"{agent_id} agent workflow instantiated")
    return starting_agent, agents


def _make_agent_applet_workflow(agent_id: str):
    """
    Create an applet-input workflow for an agent runnable.
    """
    if agent_id != "AppletRuleResponder":
        raise HTTPException(status_code=400, detail=f"Agent '{agent_id}' does not accept applet input.")

    starting_agent = AppletRuleResponder()
    agents = [starting_agent, EndAgent()]
    logger.debug(f"{agent_id} applet workflow instantiated")
    return starting_agent, agents


def _runnable_key(kind: str, runnable_id: str) -> str:
    return f"{kind}:{runnable_id}"


def _prepare_context_for_assistant(context: Context, assistant) -> Context:
    """
    Bind an empty session to an assistant.

    The selected assistant supplies the system prompt. Once a session has chat
    history, it remains bound to that assistant so the system prompt and trace
    stay coherent.
    """
    assistant_id = assistant.id
    active_assistant_id = context.state.data.get("assistant_id")
    active_runnable = context.state.data.get("runnable")
    runnable_key = _runnable_key("assistant", assistant_id)

    if active_runnable and active_runnable != runnable_key:
        raise HTTPException(
            status_code=409,
            detail=f"Session is already bound to runnable '{active_runnable}'. Start a new session to use '{runnable_key}'.",
        )

    if active_assistant_id and active_assistant_id != assistant_id:
        raise HTTPException(
            status_code=409,
            detail=f"Session is already bound to assistant '{active_assistant_id}'. Start a new session to use '{assistant_id}'.",
        )

    if not context.trace.messages:
        context.trace.messages = assistant.build_system_prompt()
        context.state.data["assistant_id"] = assistant_id
        context.state.data["runnable"] = runnable_key

    return context


def _prepare_context_for_agent(context: Context, agent_id: str) -> Context:
    """Bind an empty session to an agent workflow."""
    runnable_key = _runnable_key("agent", agent_id)
    active_runnable = context.state.data.get("runnable")

    if active_runnable and active_runnable != runnable_key:
        raise HTTPException(
            status_code=409,
            detail=f"Session is already bound to runnable '{active_runnable}'. Start a new session to use '{runnable_key}'.",
        )

    if not context.trace.messages:
        context.state.data["agent_id"] = agent_id
        context.state.data["runnable"] = runnable_key
        prompt_args = AGENT_PROMPT_ARGS.get(agent_id)
        if prompt_args:
            context.state.data.setdefault(agent_id, dict(prompt_args))

    return context


def _store_plain_chat_turn(context: Context, user_message: dict, reply: str) -> Context:
    """
    Store a web-visible user/assistant turn for agent workflows.

    Assistants use ``LLMAssistant.chat_turn`` which already persists the turn.
    Agent workflows are artifact-oriented, so the web adapter writes the trace.
    """
    response_timestamp = time.time()
    stored_user = {
        "role": "user",
        "content": user_message.get("content", ""),
        "duration": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "timestamp": response_timestamp - context.control.duration,
    }
    context.trace.messages.append(stored_user)
    context.trace.messages.append(
        {
            "role": "assistant",
            "content": reply,
            "duration": context.control.duration,
            "timestamp": response_timestamp,
            **context.control.data.get("usage", {}),
        }
    )
    return context


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SessionResponse(BaseModel):
    session_id: str


class RunnableSelection(BaseModel):
    kind: Literal["assistant", "agent"] = DEFAULT_RUNNABLE_KIND
    id: str = DEFAULT_RUNNABLE_ID


class ChatRequest(BaseModel):
    message: str = ""
    input_type: Literal["text", "applet"] = "text"
    applet: dict[str, Any] | None = None
    runnable: RunnableSelection | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    context: dict


class RunnableDescriptor(BaseModel):
    kind: Literal["assistant", "agent"]
    id: str
    label: str
    input_types: list[Literal["text", "applet"]]


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


@app.get(
    "/runnables",
    response_model=list[RunnableDescriptor],
    summary="List assistants and agents available in the demo",
)
def list_runnables() -> list[RunnableDescriptor]:
    """Return chat-capable demo runnables."""
    return [RunnableDescriptor(**item) for item in RUNNABLES]


@app.post(
    "/sessions",
    response_model=SessionResponse,
    summary="Create a new chat session",
    status_code=201,
)
def create_session() -> SessionResponse:
    """
    Open a fresh conversation thread.  Returns the ``session_id`` to use in
    subsequent chat calls.
    """
    session_id = str(uuid.uuid4())
    context = Context()
    _sessions[session_id] = context
    logger.info(f"✓ Session created: {session_id}")
    return SessionResponse(session_id=session_id)


def _run_chat(
    session_id: str,
    body: ChatRequest,
    runnable: RunnableSelection,
) -> ChatResponse:
    """
    Send a user message through the selected runnable and return its reply.
    """
    try:
        if body.input_type == "applet":
            if body.applet is None:
                raise HTTPException(status_code=400, detail="Applet input requires an 'applet' object.")
            info_store = body.applet.get("infoStore")
            if not isinstance(info_store, dict):
                raise HTTPException(status_code=400, detail="Applet input requires an 'infoStore' object.")
            for key in ("action", "followup"):
                if not isinstance(info_store.get(key), str) or not info_store[key].strip():
                    raise HTTPException(status_code=400, detail=f"Applet input requires a non-empty 'infoStore.{key}' string.")
            body.message = body.message or json.dumps(body.applet, ensure_ascii=False, indent=2)
        elif not body.message.strip():
            raise HTTPException(status_code=400, detail="Text input requires a non-empty message.")

        msg_preview = body.message[:60] + ("..." if len(body.message) > 60 else "")
        logger.info(f"→ Session {session_id} | {runnable.kind}:{runnable.id} | {body.input_type}: {msg_preview}")

        context = _get_or_raise(session_id)
        user_message = {"role": "user", "content": body.message}
        logger.debug(f"  History: {len(context.trace.messages)} messages")

        if runnable.kind == "assistant":
            if body.input_type != "text":
                raise HTTPException(status_code=400, detail=f"Runnable '{runnable.id}' does not accept {body.input_type} input.")
            assistant = _make_assistant(runnable.id)
            context = _prepare_context_for_assistant(context, assistant)
            logger.debug(f"Calling {runnable.id}.chat_turn()...")
            reply, context = assistant.chat_turn(context=context, user_message=user_message)
            logger.debug(f" {runnable.id}.chat_turn() completed and updated context")
        elif runnable.kind == "agent":
            context = _prepare_context_for_agent(context, runnable.id)
            if body.input_type == "applet":
                starting_agent, agents = _make_agent_applet_workflow(runnable.id)
                logger.debug(f"Calling {runnable.id} applet workflow...")
                reply, context = run_agent_artifact_chat_turn(
                    starting_agent=starting_agent,
                    artifact=AppletArtifact(producer="user", step=context.step, content=body.applet or {}),
                    context=context,
                    agents=agents,
                    max_hops=0,
                )
            else:
                starting_agent, agents = _make_agent_workflow(runnable.id)
                logger.debug(f"Calling {runnable.id} text workflow...")
                reply, context = run_agent_chat_turn(
                    starting_agent=starting_agent,
                    user_text=body.message,
                    context=context,
                    agents=agents,
                    prompt_params=AGENT_PROMPT_ARGS.get(runnable.id),
                    max_hops=0,
                )
            context = _store_plain_chat_turn(context, user_message, reply)
            logger.debug(f" {runnable.id} agent workflow completed and updated context")
        else:
            raise HTTPException(status_code=400, detail=f"Unknown runnable kind '{runnable.kind}'.")

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


@app.post(
    "/sessions/{session_id}/chat",
    response_model=ChatResponse,
    summary="Send a message and get a runnable reply",
)
def chat(
    session_id: Annotated[str, Path(description="Session ID returned by POST /sessions")],
    body: ChatRequest,
) -> ChatResponse:
    runnable = body.runnable or RunnableSelection()
    return _run_chat(session_id=session_id, body=body, runnable=runnable)


@app.post(
    "/sessions/{session_id}/{assistant_id}/chat",
    response_model=ChatResponse,
    summary="Compatibility route for assistant chat",
)
def chat_assistant_compat(
    session_id: Annotated[str, Path(description="Session ID returned by POST /sessions")],
    assistant_id: Annotated[str, Path(description="Assistant ID (e.g. MathAssistent)")],
    body: ChatRequest,
) -> ChatResponse:
    return _run_chat(session_id=session_id, body=body, runnable=RunnableSelection(kind="assistant", id=assistant_id))


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


app.mount(
    "/assets",
    StaticFiles(directory=WEB_DIR / "assets"),
    name="assets",
)


@app.get("/", include_in_schema=False)
def frontend_index():

    response = FileResponse(WEB_DIR / INDEX_FILE)

    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response


def main():
    logger.warning("⚠️  Start app with http://127.0.0.1:8000 or http://localhost:8000")
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
