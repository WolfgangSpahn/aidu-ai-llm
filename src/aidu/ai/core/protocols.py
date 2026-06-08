# src/aidu/ai/core/protocols.py

from __future__ import annotations

from typing import Protocol, Any

from aidu.ai.core.context import Context
from aidu.ai.core.config import AskConfig
from aidu.ai.core.agent_result import AgentResult


class ClientProtocol(Protocol):
    model: str

    def chat(
        self,
        message: dict[str, Any],
        context: Context,
        config: AskConfig | None = None,
    ) -> dict[str, Any]:
        """
        Execute a chat completion against the provider.
        """
        ...


class ChatAgentProtocol(Protocol):
    client: ClientProtocol

    def build_system_prompt(
        self,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Build initial system messages.
        """
        ...

    def chat(
        self,
        message: dict[str, Any],
        context: Context,
        chat_params: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Context]:
        """
        Process a conversational turn.
        """
        ...


class CognitiveAgentProtocol(Protocol):
    def run(
        self,
        context: Context,
    ) -> AgentResult: ...
