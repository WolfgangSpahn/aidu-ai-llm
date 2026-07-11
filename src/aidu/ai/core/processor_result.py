# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
# src/aidu/ai/core/agent_result.py

from pydantic import BaseModel, Field

from aidu.ai.core.artifacts import Artifact
from aidu.ai.core.recommendation import Recommendation


class AgentResult(BaseModel):
    artifacts: list[Artifact] = Field(default_factory=list)

    recommendations: list[Recommendation] = Field(default_factory=list)

    def __str__(self) -> str:
        return f"AgentResult(artifacts={self.artifacts}, recommendations={self.recommendations})"
