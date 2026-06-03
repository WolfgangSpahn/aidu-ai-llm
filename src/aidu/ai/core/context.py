# src/aidu/ai/core/context.py

from __future__ import annotations

from rich.console import Console, Group
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text
from rich import box

from pydantic import BaseModel, Field
from typing import Any

from .artifacts import Artifact

# role and content are harmonized across providers; all other fields are provider-specific
Message = dict[str, Any]

# class Message(BaseModel):

#     sender: str | None = None
#     recipient: str | None = None
# type: str = "message"

#     role: str | None = None
#     content: Any = None


class Trace(BaseModel):
    """Trace of messages exchanged so far in the conversation."""

    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of messages exchanged in the conversation so far.",
    )

    def __init__(self, messages=None):
        super().__init__(messages=messages or [])

    def __str__(self):
        return "messages:" + "len=" + str(len(self.messages))

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
    """Typed runtime context carrying history, mutable state, and control data."""

    step: int = 0

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

    def __str__(self):
        return f"Context(step={self.step}, trace={self.trace}, state={self.state}, control={self.control}, artifacts={list(self.artifacts.keys())})"

    def pretty(self, console: Console):
        """Pretty-print the context using Rich panels for a boxed view."""

        console.print(self.trace.pretty())
        console.print(self.state.pretty())
        console.print(self.control.pretty())

        # Artifacts: render a panel containing a mapping of id -> artifact.pretty() output
        # For compactness, display artifacts as a dict of id -> model_dump() inside a panel
        artifacts_dump = {k: v.model_dump() for k, v in self.artifacts.items()}
        console.print(Panel.fit(Pretty(artifacts_dump), title="Artifacts", border_style="green"))


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
