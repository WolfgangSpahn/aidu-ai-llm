# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
"""Canonical state produced when supervising an AI tutor response."""

from pydantic import BaseModel, Field


class SupervisorResult(BaseModel):
    """Fit score and rationale retained for a later human reviewer."""

    fit: float = Field(ge=0.0, le=1.0)
    reason: str


class SupervisorState(BaseModel):
    """Complete pedagogical and factual assessment of one tutor response."""

    factual_fit: SupervisorResult
    goal_alignment: SupervisorResult
    knowledge_alignment: SupervisorResult
    belief_alignment: SupervisorResult
    scaffolding_fit: SupervisorResult

    @classmethod
    def prior(cls) -> "SupervisorState":
        """Return the explicit neutral state before any tutor response exists."""
        return cls.model_validate({
            dimension: {
                "fit": 0.5,
                "reason": "No tutor response has been assessed yet.",
            }
            for dimension in cls.model_fields
        })


SUPERVISOR_DIMENSIONS = frozenset(SupervisorState.model_fields)
