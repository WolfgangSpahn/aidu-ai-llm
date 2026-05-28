# src/aidu/ai/core/context.py

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any

from .artifacts import Artifact

# role and content are harmonized across providers; all other fields are provider-specific
Message = dict[str, Any]

# class Message(BaseModel):

#     sender: str | None = None
#     recipient: str | None = None
#     type: str = "message"

#     role: str | None = None
#     content: Any = None

class Trace(BaseModel):
    """Trace of messages exchanged so far in the conversation."""

    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of messages exchanged in the conversation so far.",
    )

class State(BaseModel):
    """Mutable state that can be updated and shared across turns."""

    data: dict[str, Any] = Field(
        default_factory=dict,
   )

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

class Context(BaseModel):
    """Typed runtime context carrying history, mutable state, and control data."""

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
 
    artifacts: dict[str, Artifact] = Field(
        default_factory=dict
    )


# -------------------------------------------------------------------
# Smoke Test
# -------------------------------------------------------------------

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
    console.print(
        "[bold green]✓ Context smoke test passed[/bold green]"
    )


if __name__ == "__main__":

    _smoke_test()
