# engine.py

from abc import ABC, abstractmethod

from aidu.ai.core.context import (
    Context,
    Message,
)

from aidu.ai.core.config import AskConfig


class Engine(ABC):

    @abstractmethod
    def ask(
        self,
        message,
        context,
        config=None,
    ):
        raise NotImplementedError