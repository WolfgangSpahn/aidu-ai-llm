# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
"""Canonical state produced when supervising an AI tutor response."""

from typing import Literal

from pydantic import BaseModel, Field


InterventionName = Literal[
    "CONTINUE", "PROBE", "RECALL", "ELICIT_EXPLANATION", "CHALLENGE",
    "CONTRAST", "HINT", "DIRECT_ATTENTION", "EXPLAIN", "MODEL_EXAMPLE",
    "TRANSFER", "META_REFLECT", "AFFECT_REGULATE", "ADJUST_DIFFICULTY",
]


class InterventionLabel(BaseModel):
    """Dominant pedagogical intention attributed to a tutor response."""

    intervention: InterventionName
    reason: str


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
    intervention: InterventionLabel | None = None
    assessed_tutor_turn_index: int | None = Field(default=None, ge=0)
    outcome_student_turn_index: int | None = Field(default=None, ge=0)
    outcome_evidence_available: bool = True

    @classmethod
    def prior(cls) -> "SupervisorState":
        """Return the explicit neutral state before any tutor response exists."""
        return cls.model_validate({
            dimension: {
                "fit": 0.5,
                "reason": "No tutor response has been assessed yet.",
            }
            for dimension in (
                "factual_fit", "goal_alignment", "knowledge_alignment",
                "belief_alignment", "scaffolding_fit",
            )
        })


SUPERVISOR_DIMENSIONS = frozenset({
    "factual_fit",
    "goal_alignment",
    "knowledge_alignment",
    "belief_alignment",
    "scaffolding_fit",
})
