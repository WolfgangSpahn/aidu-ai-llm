# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
Base client interface and utility helpers.

Entity types (Context, Trace, State, Control, Message, AskConfig) live in
aidu.ai.core and are re-exported here for backward compatibility.
"""

from abc import ABC, abstractmethod

# --- re-exports from core (kept for backward compatibility) ---
from aidu.ai.core.context import Context, Trace, State, Control, Message  # noqa: F401
from aidu.ai.core.config import AskConfig  # noqa: F401

OFF_AIR_MESSAGE = "Sorry, I can not answer, as I have **no token budget** for anonymous, but you can use the rest."


def clean_message(obj):
    """Recursively remove None values and empty containers from a message dict."""
    if isinstance(obj, dict):
        cleaned = {k: clean_message(v) for k, v in obj.items() if v is not None}
        return {k: v for k, v in cleaned.items() if v not in ({}, [])}
    if isinstance(obj, list):
        cleaned = [clean_message(v) for v in obj if v is not None]
        return [v for v in cleaned if v not in ({}, [])]
    return obj


def off_air_response(model: str | None = None) -> dict:
    """Return a normalized assistant message without contacting a provider."""
    return {
        "role": "assistant",
        "content": OFF_AIR_MESSAGE,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "model": model,
    }


def maybe_off_air_response(context: Context, model: str | None = None) -> dict | None:
    """Return an off-air response when the runtime context disables provider I/O."""
    return None if context.on_air else off_air_response(model)


class Client(ABC):
    """Base interface for ask-capable clients."""

    def __init__(self, model: str, config: dict):
        self.model = model
        self.config = config

    @abstractmethod
    def ask(
        self,
        message: Message,
        context: Context,
        config: AskConfig | None = None,
    ) -> Message:
        """Run a completion request and return a normalized message dict."""
        raise NotImplementedError
