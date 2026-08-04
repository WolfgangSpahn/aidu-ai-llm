"""Canonical teacher-target accumulated evidence models."""

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from aidu.support.scoring import KnowledgeEvidenceState


class EvidenceKnowledgeProgress(BaseModel):
    """Strict persisted form of one target's accumulated evidence."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    mastery: float = Field(ge=0.0, le=1.0)
    positive_evidence: float = Field(ge=0.0)
    negative_evidence: float = Field(ge=0.0)
    entry_prior: float = Field(ge=0.0, le=1.0)
    entry_weight: float = Field(ge=0.0)
    source_count: int = Field(ge=0)
    turn_assessment_count: int = Field(ge=0)
    last_updated_turn: int | None = Field(ge=0)
    evidence_fingerprints: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_mastery_matches_evidence(self) -> "EvidenceKnowledgeProgress":
        expected = self.as_evidence_state().mastery
        if abs(self.mastery - expected) > 1e-9:
            raise ValueError(
                "mastery must equal positive_evidence divided by total evidence"
            )
        return self

    @classmethod
    def from_evidence_state(
        cls,
        state: KnowledgeEvidenceState,
    ) -> "EvidenceKnowledgeProgress":
        return cls(
            mastery=state.mastery,
            positive_evidence=state.positive_evidence,
            negative_evidence=state.negative_evidence,
            entry_prior=state.entry_prior,
            entry_weight=state.entry_weight,
            source_count=state.source_count,
            turn_assessment_count=state.turn_assessment_count,
            last_updated_turn=state.last_updated_turn,
            evidence_fingerprints=list(state.evidence_fingerprints),
        )

    def as_evidence_state(self) -> KnowledgeEvidenceState:
        return KnowledgeEvidenceState(
            positive_evidence=self.positive_evidence,
            negative_evidence=self.negative_evidence,
            entry_prior=self.entry_prior,
            entry_weight=self.entry_weight,
            source_count=self.source_count,
            turn_assessment_count=self.turn_assessment_count,
            last_updated_turn=self.last_updated_turn,
            evidence_fingerprints=tuple(self.evidence_fingerprints),
        )

    @property
    def evidence_weight(self) -> float:
        """Return the authoritative total accumulated evidence."""
        return self.positive_evidence + self.negative_evidence

    def clamped(self) -> "EvidenceKnowledgeProgress":
        """Return the already validated canonical record."""
        return self.model_copy(deep=True)


class StudentKnowledgeProgress(
    RootModel[dict[str, EvidenceKnowledgeProgress]]
):
    """Knowledge progress indexed by teacher-defined target ID."""

    def clamped(self) -> "StudentKnowledgeProgress":
        """Return the serializable per-turn snapshot with numeric values clamped."""
        return StudentKnowledgeProgress(
            root={
                target_id: progress.clamped()
                for target_id, progress in self.root.items()
            }
        )

    def to_tutor_text(self) -> str:
        """Describe target estimates and their evidence strength for planning."""
        if not self.root:
            return " - We have not started yet."
        lines = [
            f" - Active learning targets: {len(self.root)}.",
            " - Interpret 0.50 as neutral/unknown, below 0.50 as evidence against mastery, and above 0.50 as evidence toward mastery.",
            " - Entry-test-only estimates are tentative, low-weight priors. Do not treat a high entry prior as demonstrated mastery; confirm it through an accessible explanation or applet investigation.",
            " - Targets with no evidence are unknown, not mastered and not failed.",
        ]
        for target_id, progress in self.root.items():
            if progress.entry_weight == 0 and progress.turn_assessment_count == 0:
                evidence = "no evidence"
            elif progress.turn_assessment_count == 0:
                question_count = max(1, round(progress.entry_weight / 0.75))
                evidence = (
                    "tentative entry-test prior from "
                    f"{question_count} question{'s' if question_count != 1 else ''}; "
                    "not yet confirmed in dialog"
                )
            else:
                evidence = (
                    f"updated from {progress.turn_assessment_count} learner-turn "
                    f"assessment{'s' if progress.turn_assessment_count != 1 else ''}; "
                    f"total evidence weight {progress.evidence_weight:.2f}"
                )
            lines.append(
                f"   - {target_id}: {progress.mastery:.0%} ({evidence})"
            )
        return "\n".join(lines)
