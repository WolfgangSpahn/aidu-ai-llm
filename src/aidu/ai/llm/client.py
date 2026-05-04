# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
    Base client interface and shared context models.

    This module includes:
    - Client: Abstract base class for chat-capable clients
    - Context / Trace / State / Control: Typed runtime context models
    - clean_message: Utility for recursive cleaning of message objects
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

def clean_message(obj):
    """
    Recursively remove:
    - None
    - empty lists []
    - empty dicts {}
    """
    if isinstance(obj, dict):
        cleaned = {
            k: clean_message(v)
            for k, v in obj.items()
            if v is not None
        }

        # remove empty containers
        return {
            k: v
            for k, v in cleaned.items()
            if v not in ({}, [])
        }

    elif isinstance(obj, list):
        cleaned = [
            clean_message(v)
            for v in obj
            if v is not None
        ]

        return [v for v in cleaned if v not in ({}, [])]

    return obj


# define Message type as dict for now, can be extended to a Pydantic model if needed
Message = dict[str, Any]

class Trace(BaseModel):
    """Trace of messages exchanged so far in the conversation."""

    messages: list[Message] = Field(
        default_factory=list,
        description="List of messages exchanged in the conversation so far.",
    )

class State(BaseModel):
    """Mutable state that can be updated and shared across turns."""

    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs representing mutable state for the conversation.",
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

class Client(ABC):
    """Base interface for chat-capable clients."""

    def __init__(self, model, config):
        self.model = model
        self.config = config

    @abstractmethod
    def chat(
        self,
        message: Message,
        context: Context
    ):
        """Run a chat completion and return a normalized message dict."""
        raise NotImplementedError