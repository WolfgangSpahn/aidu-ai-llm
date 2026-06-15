# src/aidu/ai/core/artifacts.py
import logging
from pydantic import BaseModel, Field
from typing import Any, Literal, Annotated
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text
from rich.console import Group
from rich import box
from uuid import uuid4

logger = logging.getLogger(__name__)


class Artifact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    producer: str
    type: str
    step: int
    content: Any = None

    def pretty(self) -> Panel:
        """Return a Rich Panel renderable for this artifact.

        Preserve newlines when content is a string.
        """
        header = Text(f"id: {self.id}    type: {self.type}    producer: {self.producer}")
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
    type: Literal["text"] = "text"
    content: str


class SymbolicArtifact(Artifact):
    type: Literal["symbolic"] = "symbolic"
    content: Any


class EvidenceArtifact(Artifact):
    type: Literal["evidence"] = "evidence"
    content: dict[str, Any]


class BeliefArtifact(Artifact):
    type: Literal["belief"] = "belief"
    content: dict[str, Any]


class ErrorArtifact(Artifact):
    type: Literal["error"] = "error"
    content: Any

class EndArtifact(Artifact):
    type: Literal["text"] = "text"
    content: str

ArtifactType = Annotated[
    TextArtifact | SymbolicArtifact | EvidenceArtifact | BeliefArtifact | ErrorArtifact,
    Field(discriminator="type"),
]


def create_artifact(artifact_type: str, id: str, producer: str, step: int, content: Any) -> Artifact:
    """Factory function to create an artifact based on the type."""
    # Validate content matches expected types before creating the artifact to
    # provide clearer error messages than Pydantic's ValidationError.
    if artifact_type == "text":
        if not isinstance(content, str):
            raise TypeError(f"text artifact requires 'content' of type str, got {type(content).__name__}")
        return TextArtifact(id=id, producer=producer, step=step, content=content)
    elif artifact_type == "symbolic":
        # Symbolic artifacts can be any serializable structure, but must not be None.
        if content is None:
            raise TypeError("symbolic artifact requires non-None 'content'")
        return SymbolicArtifact(id=id, producer=producer, step=step, content=content)
    elif artifact_type == "evidence":
        if not isinstance(content, dict):
            raise TypeError(f"evidence artifact requires 'content' of type dict, got {type(content).__name__}")
        return EvidenceArtifact(id=id, producer=producer, step=step, content=content)
    elif artifact_type == "belief":
        if not isinstance(content, dict):
            raise TypeError(f"belief artifact requires 'content' of type dict, got {type(content).__name__}")
        return BeliefArtifact(id=id, producer=producer, step=step, content=content)
    elif artifact_type == "error":
        # Error artifacts may contain any content (exception info, message, etc.)
        return ErrorArtifact(id=id, producer=producer, step=step, content=content)
    else:
        raise ValueError(f"Unknown artifact type: {artifact_type}")
