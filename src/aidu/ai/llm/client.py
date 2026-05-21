# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
Base client interface and utility helpers.

Entity types (Context, Trace, State, Control, Message, ChatConfig) live in
aidu.ai.core and are re-exported here for backward compatibility.
"""

from abc import ABC, abstractmethod

# --- re-exports from core (kept for backward compatibility) ---
from aidu.ai.core.context import Context, Trace, State, Control, Message  # noqa: F401
from aidu.ai.core.config import ChatConfig  # noqa: F401


def clean_message(obj):
    """Recursively remove None values and empty containers from a message dict."""
    if isinstance(obj, dict):
        cleaned = {k: clean_message(v) for k, v in obj.items() if v is not None}
        return {k: v for k, v in cleaned.items() if v not in ({}, [])}
    if isinstance(obj, list):
        cleaned = [clean_message(v) for v in obj if v is not None]
        return [v for v in cleaned if v not in ({}, [])]
    return obj


class Client(ABC):
    """Base interface for chat-capable clients."""

    def __init__(self, model: str, config: dict):
        self.model = model
        self.config = config

    @abstractmethod
    def chat(
        self,
        message: Message,
        context: Context,
        config: ChatConfig | None = None,
    ) -> Message:
        """Run a chat completion and return a normalized message dict."""
        raise NotImplementedError