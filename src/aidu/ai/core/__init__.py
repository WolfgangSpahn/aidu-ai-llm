# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
# src/aidu/ai/core/__init__.py

from aidu.ai.core.context import Context, Trace, State, Control, Message
from aidu.ai.core.config import AskConfig
from aidu.ai.core.protocols import ClientProtocol, ChatAgentProtocol
from aidu.ai.core.hookspecs import HookSpecs, hookimpl, hookspec

__all__ = [
    "Context",
    "Trace",
    "State",
    "Control",
    "Message",
    "AskConfig",
    "ClientProtocol",
    "ChatAgentProtocol",
    "HookSpecs",
    "hookimpl",
    "hookspec",
]
