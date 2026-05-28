# src/aidu/ai/core/artifacts.py

from pydantic import BaseModel, Field
from typing import Any


class Artifact(BaseModel):

    id: str
    type: str

    content: Any = None

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