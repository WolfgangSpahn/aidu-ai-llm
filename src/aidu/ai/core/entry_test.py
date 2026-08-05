"""Apply standalone entry-test scoring to an ordered activity context."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from aidu.support.scoring import (
    EntryTestScore,
    initialize_from_entry_prior,
    score_poll_test,
)

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

    The understandable whole-test score becomes the conservative prior for all
    configured targets. Target annotations retain diagnostic evidence breadth
    but cannot inflate mastery above the score shown to learner and teacher.

    ``question_count`` is used as evidence mass because several option
    observations from one question are correlated.  This follows the scoring
    specification's recommendation to use question count as the simple measure
    of evidence breadth.
    """

    knowledge = dict(previous.knowledge.root)
    if not score.priors:
        return ActivityContext(
            knowledge=StudentKnowledgeProgress(root=knowledge),
            belief=previous.belief.model_copy(deep=True),
            supervisor=previous.supervisor.model_copy(deep=True),
        )
    for target in knowledge:
        if target in score.priors:
            continue
        knowledge[target] = EvidenceKnowledgeProgress.from_evidence_state(
            initialize_from_entry_prior(
                prior=score.overall_score,
                question_count=1,
            )
        )
    for target, prior in score.priors.items():
        knowledge[target] = EvidenceKnowledgeProgress.from_evidence_state(
            initialize_from_entry_prior(
                prior=prior.prior,
                question_count=prior.question_count,
            )
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
