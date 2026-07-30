# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
# src/aidu/ai/core/artifacts.py
import json
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

    def to_text(self) -> str:
        """Return the canonical text representation of structured content."""
        return json.dumps(self.content, ensure_ascii=False, sort_keys=True)

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

    def to_text(self) -> str:
        """Return text content unchanged."""
        return self.content


class SymbolicArtifact(Artifact):
    type: Literal["symbolic"] = "symbolic"
    content: Any


class AppletArtifact(Artifact):
    type: Literal["applet"] = "applet"
    content: dict[str, Any]

    def to_text(self) -> str:
        """Serialize the structured applet event deterministically."""
        return json.dumps(self.content, ensure_ascii=False, sort_keys=True)

    def is_outbound_command(self) -> bool:
        """Return whether this artifact contains a command addressed to an applet."""
        return bool(self.content.get("applet") and self.content.get("command"))


class JsonArtifact(Artifact):
    type: Literal["json"] = "json"
    content: dict[str, Any]


class ActivityEventArtifact(Artifact):
    """Structured lifecycle event emitted for the current learning activity."""

    type: Literal["activity_event"] = "activity_event"
    content: dict[str, Any]


class EvidenceArtifact(Artifact):
    type: Literal["evidence"] = "evidence"
    content: dict[str, Any]


class BeliefArtifact(Artifact):
    type: Literal["belief"] = "belief"
    content: dict[str, Any]


class ErrorArtifact(Artifact):
    type: Literal["error"] = "error"
    content: Any

class EndArtifact(TextArtifact):
    """Final text artifact emitted when a workflow ends."""

ArtifactType = Annotated[
    TextArtifact
    | SymbolicArtifact
    | AppletArtifact
    | ActivityEventArtifact
    | EvidenceArtifact
    | BeliefArtifact
    | ErrorArtifact,
    Field(discriminator="type"),
]


def latest_display_artifact(artifacts: list[Artifact]) -> TextArtifact | None:
    """Return the most recent text artifact suitable for display to the user."""
    return next(
        (
            artifact
            for artifact in reversed(artifacts)
            if isinstance(artifact, TextArtifact)
        ),
        None,
    )


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
    elif artifact_type == "applet":
        if not isinstance(content, dict):
            raise TypeError(f"applet artifact requires 'content' of type dict, got {type(content).__name__}")
        return AppletArtifact(id=id, producer=producer, step=step, content=content)
    elif artifact_type == "activity_event":
        if not isinstance(content, dict):
            raise TypeError(f"activity_event artifact requires 'content' of type dict, got {type(content).__name__}")
        return ActivityEventArtifact(id=id, producer=producer, step=step, content=content)
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
