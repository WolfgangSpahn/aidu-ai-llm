# src/aidu/ai/core/recommendation.py

from pydantic import BaseModel, Field
from typing import Any


class Recommendation(BaseModel):
    target: Any

    continuations: list[Any] = Field(default_factory=list)

    utility: float = 0.0

    rationale: str = ""

    metadata: dict[str, Any] = Field(default_factory=dict)
