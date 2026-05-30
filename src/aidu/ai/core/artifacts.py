# src/aidu/ai/core/artifacts.py

from pydantic import BaseModel, Field
from typing import Any
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text
from rich.console import Group
from rich import box


class Artifact(BaseModel):

    id: str
    type: str

    content: Any = None
    def pretty(self) -> Panel:
        """Return a Rich Panel renderable for this artifact.

        Preserve newlines when content is a string.
        """
        header = Text(f"id: {self.id}    type: {self.type}")
        if isinstance(self.content, str):
            body = Text(self.content)
        else:
            body = Pretty(self.content)

        return Panel(
            Group(header, body),
            title=f"Artifact {self.id}",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 1),
            expand=True,
        )

class TextArtifact(Artifact):

    type: str = "text"

    content: str


class SymbolicArtifact(Artifact):

    type: str = "symbolic"

    content: Any


class EvidenceArtifact(Artifact):

    type: str = "evidence"

    content: dict[str, Any]


class BeliefArtifact(Artifact):

    type: str = "belief"

    content: dict[str, Any]

class ErrorArtifact(Artifact):

    type: str = "error"

    content: Any