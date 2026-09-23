"""Store and present a learner's progress for each learning target.

This module is the boundary between the scoring code and the data saved by the
AI tutoring application.  The scoring code uses
:class:`aidu.support.scoring.KnowledgeEvidenceState`; the Pydantic models here
validate that state and turn it into a JSON-serializable form.

There are two levels of data:

* :class:`EvidenceKnowledgeProgress` describes evidence for one learning
  target, for example ``"solve-linear-equations"``.
* :class:`StudentKnowledgeProgress` maps every target ID to its evidence.

When extending this module, keep ``mastery`` derived from the evidence rather
than updating it independently.  The validator below enforces this invariant.
Changes to the actual scoring formula belong in ``aidu.support.scoring``.
"""

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from aidu.support.scoring import KnowledgeEvidenceState


class EvidenceKnowledgeProgress(BaseModel):
    """Validated, persistent progress for one learning target.

    Think of positive and negative evidence as weighted observations, not as
    numbers of correct and incorrect answers.  A strong independent explanation
    can therefore contribute more weight than a guess or a heavily hinted
    response.  ``mastery`` is calculated by the scoring model from these
    weights; callers must not invent a separate value for it.

    Attributes:
        mastery: Current estimate between 0.0 (no mastery) and 1.0 (mastery).
            It must agree with the evidence fields.
        positive_evidence: Total weight supporting mastery of this target.
        negative_evidence: Total weight suggesting the target is not mastered.
        entry_prior: Initial mastery estimate, between 0.0 and 1.0.
        entry_weight: Evidence weight assigned to the entry-test estimate.
            Zero means that no entry-test evidence was used.
        source_count: Number of evidence contributions with non-zero weight.
        turn_assessment_count: Number of learner-turn assessments, including
            assessments whose applied weight was zero.
        last_updated_turn: Zero-based index of the most recently assessed turn,
            or ``None`` if dialog has not updated this target yet.
        evidence_fingerprints: Identifiers used elsewhere to recognize evidence
            that has already been processed and avoid counting it twice.

    Pydantic rejects unknown fields and validates assignments after the object
    has been created.  This strictness is intentional: it catches incompatible
    saved data early instead of silently losing information.
    """

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
        """Reject records whose stored mastery disagrees with their evidence.

        Returns:
            The validated model.  Pydantic requires an ``after`` validator to
            return the model being validated.

        Raises:
            ValueError: If ``mastery`` differs from the value calculated by
                :class:`KnowledgeEvidenceState` by more than floating-point
                rounding tolerance.
        """
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
        """Create the persistent Pydantic model from the scoring model.

        Args:
            state: Authoritative evidence state produced by the scoring code.

        Returns:
            An equivalent validated record suitable for JSON serialization.

        The tuple of fingerprints in the immutable scoring model becomes a
        list because lists have a natural JSON representation.
        """
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
        """Convert this persistent record back to the scoring representation.

        Returns:
            An immutable :class:`KnowledgeEvidenceState` containing the same
            evidence.  Its ``mastery`` property is the authoritative calculated
            value used by this module's validator.
        """
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
        """Return the total weight of positive and negative observations.

        A larger value generally means that the mastery estimate is supported
        by more evidence.  It does not say whether that evidence is positive or
        negative; inspect ``mastery`` or the individual totals for direction.
        """
        return self.positive_evidence + self.negative_evidence

    def clamped(self) -> "EvidenceKnowledgeProgress":
        """Return an independent copy of this already validated record.

        The method name is retained for compatibility with callers that expect
        a safe snapshot.  No numeric adjustment is needed because Pydantic has
        already enforced all bounds.  ``deep=True`` also copies mutable values
        such as ``evidence_fingerprints``.
        """
        return self.model_copy(deep=True)


class StudentKnowledgeProgress(
    RootModel[dict[str, EvidenceKnowledgeProgress]]
):
    """All known progress for one student, indexed by learning-target ID.

    This is a Pydantic ``RootModel``, so its serialized JSON is a plain mapping
    rather than an object with an extra ``root`` key.  For example::

        progress.root["solve-linear-equations"].mastery

    Target IDs are defined by the teacher or activity configuration.  An absent
    target means "not tracked"; a present target with no evidence means
    "unknown", not "failed".
    """

    def clamped(self) -> "StudentKnowledgeProgress":
        """Return a deep, serializable snapshot of every target's progress.

        Returns:
            A new :class:`StudentKnowledgeProgress`.  Mutating its fingerprint
            lists will not modify the original object.

        Numeric values are already within their declared bounds; the historical
        method name is kept because other parts of the application call it.
        """
        return StudentKnowledgeProgress(
            root={
                target_id: progress.clamped()
                for target_id, progress in self.root.items()
            }
        )

    def mean_mastery_percent(self) -> int:
        """Return rounded mean target mastery as a learner-facing percentage.

        Targets with neither entry-test nor dialog evidence are excluded:
        their neutral 50% mastery is unknown, not demonstrated progress. If no
        target has evidence, the activity starts at 0%.
        """
        assessed = [
            item for item in self.root.values()
            if item.entry_weight > 0 or item.evidence_weight > 0
        ]
        if not assessed:
            return 0
        mean_mastery = sum(item.mastery for item in assessed) / len(assessed)
        return round(mean_mastery * 100)

    def to_tutor_text(self) -> str:
        """Format progress as guidance that can be inserted into a tutor prompt.

        Returns:
            A multi-line, human-readable summary.  It explains how confident
            the tutor should be in each estimate, rather than reporting only a
            percentage.  With no targets, it returns a short "not started"
            message.

        If students change this wording, they should also update the assertions
        in ``tests/test_knowledge_progress.py``.  Keep the distinction between
        an entry-test prior and mastery demonstrated during dialog: the tutor
        uses it to decide whether knowledge still needs confirmation.
        """
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
