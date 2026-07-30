"""Canonical teacher-target knowledge progress models."""

from pydantic import BaseModel, RootModel


class EvidenceKnowledgeProgress(BaseModel):
    """Mastery and accumulated evidence for one teacher-defined target."""

    mastery: float
    positive_evidence: float
    negative_evidence: float

    def clamped(self) -> "EvidenceKnowledgeProgress":
        """Return a copy with probabilities and evidence inside valid ranges."""
        return EvidenceKnowledgeProgress(
            mastery=max(0.0, min(1.0, self.mastery)),
            positive_evidence=max(0.0, self.positive_evidence),
            negative_evidence=max(0.0, self.negative_evidence),
        )


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
        """Summarize the available teacher-defined targets for tutor prompts."""
        if not self.root:
            return " - We have not started yet."
        return (
            f" - Learning targets loaded: {len(self.root)}. "
            "Each domain target progress starts at 0.0."
        )
