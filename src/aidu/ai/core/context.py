# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
# src/aidu/ai/core/context.py

from __future__ import annotations

import copy
import logging
from rich.console import Console, Group
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text
from rich import box

from collections.abc import Iterator
from typing import Any, overload

from pydantic import BaseModel, ConfigDict, Field, RootModel

from .artifacts import Artifact, TextArtifact
from .applet_info import AppletInfo
from .belief import StudentBelief
from .knowledge_progress import StudentKnowledgeProgress
from .supervisor import SupervisorState

logger = logging.getLogger(__name__)

# role and content are harmonized across providers; all other fields are provider-specific

class Message(BaseModel):
    """
    Target message type.

    Only these keys are valid:

        role
        content
        actor
        kind

    Any additional key is rejected during validation.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    role: str | None = None
    content: str | dict[str, Any] | list[Any] | None = None
    actor: str | None = None
    kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    def keys(self):
        return self.to_dict().keys()

    def items(self):
        return self.to_dict().items()

    def values(self):
        return self.to_dict().values()

    @staticmethod
    def clean_dialog_record(
        message: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Normalize one persisted record for LLM dialog history."""
        role = message.get("role")
        if role not in {"user", "assistant"}:
            return None

        applet_info = AppletInfo.from_message(message)
        if applet_info:
            applet_state_text = applet_info.to_text()
            student_text = str(message.get("content") or "").strip()
            content = (
                f"{applet_state_text}\nStudent said: {student_text}"
                if student_text
                and not student_text.lower().startswith("applet event:")
                else applet_state_text
            )
        else:
            content = str(message.get("content") or "").strip()

        if not content:
            return None

        cleaned: dict[str, Any] = {"role": role, "content": content}
        applet_input = message.get("applet_input")
        if message.get("kind") == "applet" and isinstance(applet_input, dict):
            cleaned["kind"] = "applet"
            cleaned["applet_input"] = applet_input
        return cleaned


class Messages(RootModel[list[dict[str, Any]]]):
    """Conversation records plus access to the latest persisted learner state.

    The root list preserves the API wire format. Each record may contain the
    core message fields and backend-owned per-turn state snapshots.
    """

    root: list[dict[str, Any]] = Field(default_factory=list)

    def latest_knowledge_progress(self) -> StudentKnowledgeProgress:
        """Return the newest persisted knowledge snapshot, or an empty state."""
        for message in reversed(self.root):
            state = message.get("backend_knowledge_progress_state")
            if state is not None:
                return StudentKnowledgeProgress.model_validate(state).clamped()
        return StudentKnowledgeProgress(root={})

    def latest_belief(self) -> StudentBelief:
        """Return the newest persisted student-belief snapshot, or its default."""
        for message in reversed(self.root):
            state = message.get("backend_belief_state")
            if state is not None:
                return StudentBelief.model_validate(state)
        return StudentBelief()

    def before_last(self) -> "Messages":
        """Return all records preceding the current turn."""
        return Messages(root=self.root[:-1])

    def recent(self, limit: int) -> "Messages":
        """Return at most the newest ``limit`` conversation records."""
        return Messages(root=self.root[-limit:])

    def cleaned_dialog(self, limit: int = 10) -> "Messages":
        """Return recent user and assistant records normalized for an LLM."""
        return Messages(
            root=[
                cleaned
                for message in self.recent(limit)
                if (cleaned := Message.clean_dialog_record(message))
            ]
        )

    def dialog_history(self, limit: int = 10) -> str:
        """Render recent normalized records as tutor prompt history."""
        cleaned = self.cleaned_dialog(limit)
        if not cleaned:
            return " - No previous dialog turns are available."
        return "\n".join(
            f" - {message['role']}: {message['content']}"
            for message in cleaned
        )

    def applet_state_before_last_tutor_message(self) -> dict[str, Any]:
        """Return the applet snapshot visible when the last tutor turn began."""
        last_tutor_index = next(
            (
                index
                for index in range(len(self.root) - 1, -1, -1)
                if self.root[index].get("role") == "assistant"
            ),
            None,
        )
        if last_tutor_index is None:
            return {}

        for message in reversed(self.root[:last_tutor_index]):
            applet_info = AppletInfo.from_message(message)
            if applet_info:
                return applet_info.to_state()
        return {}

    def append(self, message: Message | dict[str, Any]) -> None:
        """Append one core or backend-enriched message record."""
        if isinstance(message, Message):
            self.root.append(message.to_dict())
            return
        self.root.append(message)

    def __len__(self) -> int:
        return len(self.root)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.root)

    def __bool__(self) -> bool:
        return bool(self.root)

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, Any]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> dict[str, Any] | list[dict[str, Any]]:
        return self.root[index]

    def __setitem__(
        self, index: int, message: Message | dict[str, Any]
    ) -> None:
        self.root[index] = message.to_dict() if isinstance(message, Message) else message

    def __add__(
        self, other: list[Message | dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return self.root + [
            message.to_dict() if isinstance(message, Message) else message
            for message in other
        ]

    def __reversed__(self) -> Iterator[dict[str, Any]]:
        return reversed(self.root)


# class Message(BaseModel):

#     sender: str | None = None
#     recipient: str | None = None
# type: str = "message"

#     role: str | None = None
#     content: Any = None


class Trace(BaseModel):
    """Trace of messages exchanged so far in the conversation."""

    model_config = ConfigDict(validate_assignment=True)

    messages: Messages = Field(
        default_factory=Messages,
        description="List of messages exchanged in the conversation so far.",
    )

    def __init__(self, messages: Messages | list[dict[str, Any]] | None = None):
        super().__init__(messages=messages or Messages())

    def __str__(self):
        trace_messages_str = "\n - ".join(f"{msg}" for msg in self.messages[1:])
        return f"trace.messages[0].content: \n{self.messages[0]['content']}" f"\ntrace.messages:\n - {trace_messages_str} \n" #, len={len(self.messages)}"

    def pretty(self):
        """Return a Group of Rich Panels, one per message.

        Each panel's title is the message `role`. The panel body focuses on
        `function_call` if present, otherwise `content`. Any other keys are
        shown in the panel subtitle (footer).
        """
        panels: list[Panel] = []
        for msg in self.messages:
            role = msg.get("role", "message")

            # Prefer function_call over content for the main display
            main_renderable = None
            if "function_call" in msg and msg["function_call"] is not None:
                main_renderable = Pretty(msg["function_call"])
            else:
                content = msg.get("content")
                if isinstance(content, str):
                    main_renderable = Text(content)
                else:
                    main_renderable = Pretty(content)

            # Footer: show any remaining fields (excluding role, content, function_call)
            footer_kv = {k: v for k, v in msg.items() if k not in ("role", "content", "function_call")}
            subtitle = None
            if footer_kv:
                try:
                    subtitle = ", ".join(f"{k}={v}" for k, v in footer_kv.items())
                except Exception:
                    subtitle = str(footer_kv)

            panels.append(
                Panel(
                    main_renderable,
                    title=str(role),
                    subtitle=subtitle or "",
                    border_style="cyan",
                    box=box.ROUNDED,
                    padding=(1, 1),
                    expand=True,
                )
            )

        # Wrap all message panels in an outer Trace panel
        return Panel(
            Group(*panels),
            title="Trace",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 0),
            expand=True,
        )


class State(BaseModel):
    """Mutable state that can be updated and shared across turns."""

    data: dict[str, Any] = Field(
        default_factory=dict,
    )

    def __init__(self, data=None):
        super().__init__(data=data or {})

    def __str__(self):
        return str(self.data)

    def pretty(self) -> Panel:
        """Return a Rich Panel renderable for the state."""
        return Panel(Pretty(self.data), title="State", border_style="magenta", expand=True)


class Control(BaseModel):
    """Control information for execution and flow decisions."""

    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs representing control information for execution and flow decisions.",
    )
    duration: float = Field(
        default=0.0,
        description="Duration of the last chat request in seconds.",
    )

    def __init__(self, data=None, duration=0.0):
        super().__init__(data=data or {}, duration=duration)

    def __str__(self):
        control_display = {**self.data, "duration": f"{self.duration:.2f} s"}
        return str(control_display)

    def pretty(self) -> Panel:
        """Return a Rich Panel renderable for control information (includes duration)."""
        control_display = {**self.data, "duration": f"{self.duration:.2f} s"}
        return Panel(Pretty(control_display), title="Control", border_style="yellow", expand=True)


class Context(BaseModel):
    """
        Typed runtime context carrying history, mutable state, and control data.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
    )

    step: int = 0
    on_air: bool = Field(
        default=True,
        description="Whether ask-capable LLM clients may contact an external provider.",
    )

    trace: Trace = Field(
        default_factory=Trace,
        description="Former messages in the conversation.",
    )
    state: State = Field(
        default_factory=State,
        description="Mutable application state shared across turns.",
    )
    control: Control = Field(
        default_factory=Control,
        description="Control information for execution and flow decisions.",
    )

    artifacts: dict[str, Artifact] = Field(default_factory=dict)

    def for_assessor(self) -> "Context":
        """Copy this turn context without its response-stream callback."""
        assessor_context = copy.deepcopy(self)
        assessor_context.control.data.pop("stream_callback", None)
        return assessor_context

    def __str__(self):
        artifacts_str = ", ".join(f"{k}: {v}" for k, v in self.artifacts.items())
        return f"Context(step={self.step}, on_air={self.on_air}, trace={self.trace}, state={self.state}, control={self.control}, artifacts={artifacts_str})"


    def create_agent_states(self, agents):
        """
            Ensure that the context has state entries for all agents, initializing with their default state if not already present.
            This should be called at the start of a conversation or when new agents are introduced, to ensure that all agents have
            a place to store their state in the context. It does not overwrite existing state entries, allowing for persistence across turns.
        """
        for agent in agents:
            self.state.data.setdefault(
                agent.__class__.__name__,
                getattr(agent, "default_state", {}).copy(),
            )
    def check_agents_have_state(self, agents):
        """
            Check that the context has state entries for all agents. Raises ValueError if any agent is missing.
        """
        for agent in agents:
            if agent.__class__.__name__ not in self.state.data:
                raise ValueError(f"Agent '{agent.__class__.__name__}' does not have a state in the context. Please call 'create_agent_states' first.")


    def create_messages_trace(self, last_message_only: bool = False):
        """
            Create message traces from artifacts of type TextArtifact.
        """
        system_message = self.get_system_message()
        if system_message is None:
            self.trace.messages = [None]
        else:
            self.trace.messages = [system_message]

        text_artifacts = [artifact for artifact in self.artifacts.values() if isinstance(artifact, TextArtifact)]

        if text_artifacts:
            if last_message_only:
                self.trace.messages.append({"role": "assistant", 
                                            "content": text_artifacts[-1].content})
            else:
                for i, artifact in enumerate(self.artifacts.values()):
                    if isinstance(artifact, TextArtifact):
                        self.trace.messages.append(
                            {
                                "role": "assistant" if i % 2 == 0 else "user", # TODO: improve role assignment logic
                                "content": artifact.content,
                            }
                        )

    def get_system_message(self) -> Message | None:
        """
            Convenience method to get the initial system message from the trace, if present.
        """
        if self.trace.messages and self.trace.messages[0].get("role") == "system":
            return self.trace.messages[0]
        logger.error("No system message found in trace; We return None")
        return None

    def pretty(self, console: Console):
        """Pretty-print the context using Rich panels for a boxed view."""

        console.print(self.trace.pretty())
        console.print(self.state.pretty())
        console.print(self.control.pretty())

        # Artifacts: render a panel containing a mapping of id -> artifact.pretty() output
        # For compactness, display artifacts as a dict of id -> model_dump() inside a panel
        artifacts_dump = {k: v.model_dump() for k, v in self.artifacts.items()}
        console.print(Panel.fit(Pretty(artifacts_dump), title="Artifacts", border_style="green"))


class ActivityContext(BaseModel):
    """Complete learner state passed from one ordered activity to the next.

    Activity ``n`` is derived only from the context of activity ``n - 1``.
    Entry-test processing replaces knowledge for tested targets and carries
    belief and supervisor state forward unchanged.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    knowledge: StudentKnowledgeProgress
    belief: StudentBelief
    supervisor: SupervisorState

    @classmethod
    def neutral(cls) -> "ActivityContext":
        """Create the explicit state preceding the first ordered activity."""
        return cls(
            knowledge=StudentKnowledgeProgress(root={}),
            belief=StudentBelief(),
            supervisor=SupervisorState.prior(),
        )


# -------------------------------------------------------------------
# Smoke Test
# -------------------------------------------------------------------


def _smoke_test():

    from rich.console import Console
    from rich.panel import Panel
    from rich.pretty import Pretty

    from aidu.ai.core.artifacts import (
        TextArtifact,
        SymbolicArtifact,
        EvidenceArtifact,
        BeliefArtifact,
    )

    console = Console()

    console.rule("[bold blue]Context Smoke Test")

    context = Context()

    # -----------------------------------------------------------------
    # Trace
    # -----------------------------------------------------------------

    context.trace.messages.append(
        {
            "role": "user",
            "content": "Why is NaCl stable?",
        }
    )

    context.trace.messages.append(
        {
            "role": "assistant",
            "content": "Let's investigate the octet rule.",
        }
    )

    # -----------------------------------------------------------------
    # State
    # -----------------------------------------------------------------

    context.state.data["student"] = "alice"
    context.state.data["topic"] = "chemical bonding"

    # -----------------------------------------------------------------
    # Control
    # -----------------------------------------------------------------

    context.control.data["current_agent"] = "ChemistryTutor"
    context.control.duration = 0.42

    # -----------------------------------------------------------------
    # Artifacts
    # -----------------------------------------------------------------

    text = TextArtifact(
        id="text_1",
        content="Why is NaCl stable?",
    )

    symbolic = SymbolicArtifact(
        id="symbolic_1",
        content={
            "formula": "NaCl",
            "type": "ionic_compound",
        },
    )

    evidence = EvidenceArtifact(
        id="evidence_1",
        content={
            "concept": "octet_rule",
            "signal": "partial_understanding",
            "strength": 0.7,
        },
    )

    belief = BeliefArtifact(
        id="belief_1",
        content={
            "octet_rule": [
                0.05,
                0.10,
                0.20,
                0.50,
                0.15,
            ]
        },
    )

    context.artifacts[text.id] = text
    context.artifacts[symbolic.id] = symbolic
    context.artifacts[evidence.id] = evidence
    context.artifacts[belief.id] = belief

    # -----------------------------------------------------------------
    # Display
    # -----------------------------------------------------------------

    console.print(
        Panel.fit(
            Pretty(context),
            title="Context",
            border_style="green",
        )
    )

    console.print(
        Panel.fit(
            Pretty(context.model_dump()),
            title="Serialized",
            border_style="cyan",
        )
    )

    # -----------------------------------------------------------------
    # Assertions
    # -----------------------------------------------------------------

    assert len(context.trace.messages) == 2

    assert "text_1" in context.artifacts
    assert "symbolic_1" in context.artifacts
    assert "evidence_1" in context.artifacts
    assert "belief_1" in context.artifacts

    assert isinstance(
        context.artifacts["text_1"],
        TextArtifact,
    )

    assert isinstance(
        context.artifacts["symbolic_1"],
        SymbolicArtifact,
    )

    assert isinstance(
        context.artifacts["evidence_1"],
        EvidenceArtifact,
    )

    assert isinstance(
        context.artifacts["belief_1"],
        BeliefArtifact,
    )

    console.print()
    console.print("[bold green]✓ Context smoke test passed[/bold green]")


if __name__ == "__main__":
    _smoke_test()
