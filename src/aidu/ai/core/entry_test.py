"""Apply standalone entry-test scoring to an ordered activity context."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from aidu.support.scoring import EntryTestScore, score_poll_test

from .context import ActivityContext
from .knowledge_progress import (
    EvidenceKnowledgeProgress,
    StudentKnowledgeProgress,
)


def populate_activity_context(
    previous: ActivityContext,
    score: EntryTestScore,
) -> ActivityContext:
    """Build ``ActivityContext(n)`` from ``ActivityContext(n-1)`` and a score.

    Only targets for which the submitted test produced evidence are replaced.
    Untested knowledge and the other two context elements are inherited.  The
    input context is never mutated.

    ``question_count`` is used as evidence mass because several option
    observations from one question are correlated.  This follows the scoring
    specification's recommendation to use question count as the simple measure
    of evidence breadth.
    """

    knowledge = dict(previous.knowledge.root)
    for target, prior in score.priors.items():
        evidence_mass = float(prior.question_count)
        knowledge[target] = EvidenceKnowledgeProgress(
            mastery=prior.prior,
            positive_evidence=prior.prior * evidence_mass,
            negative_evidence=(1.0 - prior.prior) * evidence_mass,
        )

    return ActivityContext(
        knowledge=StudentKnowledgeProgress(root=knowledge),
        belief=previous.belief.model_copy(deep=True),
        supervisor=previous.supervisor.model_copy(deep=True),
    )


def process_entry_test(
    previous: ActivityContext,
    raw_questions: Iterable[Mapping[str, Any]],
    poll_responses: Mapping[str, Any],
) -> ActivityContext:
    """Score an authored poll result and produce the next activity context."""

    score = score_poll_test(raw_questions, poll_responses)
    return populate_activity_context(previous, score)
