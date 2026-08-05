# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
# src/aidu/ai/core/__init__.py

from aidu.ai.core.context import ActivityContext, Context, Trace, State, Control, Message, Messages, PersistedTurn
from aidu.ai.core.entry_test import populate_activity_context, process_entry_test
from aidu.ai.core.config import AskConfig
from aidu.ai.core.protocols import ClientProtocol, ChatAgentProtocol
from aidu.ai.core.hookspecs import HookSpecs, hookimpl, hookspec
from aidu.ai.core.knowledge_progress import (
    EvidenceKnowledgeProgress,
    StudentKnowledgeProgress,
)
from aidu.ai.core.supervisor import (
    SUPERVISOR_DIMENSIONS,
    SupervisorResult,
    SupervisorState,
)

__all__ = [
    "Context",
    "ActivityContext",
    "populate_activity_context",
    "process_entry_test",
    "Trace",
    "State",
    "Control",
    "Message",
    "Messages",
    "PersistedTurn",
    "AskConfig",
    "ClientProtocol",
    "ChatAgentProtocol",
    "HookSpecs",
    "hookimpl",
    "hookspec",
    "EvidenceKnowledgeProgress",
    "StudentKnowledgeProgress",
    "SUPERVISOR_DIMENSIONS",
    "SupervisorResult",
    "SupervisorState",
]
