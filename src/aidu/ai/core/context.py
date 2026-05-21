# src/aidu/ai/core/context.py

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any

# role and content are harmonized across providers; all other fields are provider-specific
Message = dict[str, Any]

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
 
