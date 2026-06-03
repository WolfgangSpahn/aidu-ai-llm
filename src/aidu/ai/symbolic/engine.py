# engine.py

from abc import ABC, abstractmethod

from aidu.ai.core.context import (
    Message,
)


class Engine(ABC):
    role: str = "engine"

    def build_system_prompt(self, **kwargs) -> list[Message]:
        """
        Build initial system messages.
        """
        return [{"role": "system", "content": "Symbolic math engine"}]

    @abstractmethod
    def ask(
        self,
        message,
        context,
        config=None,
    ):
        raise NotImplementedError
