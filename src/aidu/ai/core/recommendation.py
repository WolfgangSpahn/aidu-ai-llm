# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
# src/aidu/ai/core/recommendation.py

from pydantic import BaseModel, Field
from typing import Any


class Recommendation(BaseModel):
    target: Any

    continuations: list[Any] = Field(default_factory=list)

    utility: float = 0.0

    rationale: str = ""

    metadata: dict[str, Any] = Field(default_factory=dict)
