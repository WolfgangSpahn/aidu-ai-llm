import pytest
from pydantic import ValidationError

from aidu.ai.core.context import ActivityContext
from aidu.ai.core.entry_test import process_entry_test
from aidu.ai.core.knowledge_progress import (
    EvidenceKnowledgeProgress,
    StudentKnowledgeProgress,
)
from aidu.ai.core.supervisor import SupervisorState


def test_neutral_activity_context_has_all_three_state_elements():
    activity_context = ActivityContext.neutral()

    assert activity_context.knowledge.root == {}
    assert activity_context.belief.engagement == 0.5
    assert activity_context.supervisor.factual_fit.fit == 0.5
    assert activity_context.supervisor.factual_fit.reason == (
        "No tutor response has been assessed yet."
    )
    assert set(activity_context.model_dump()) == {
        "knowledge",
        "belief",
        "supervisor",
    }


def test_activity_context_rejects_missing_state_elements():
    with pytest.raises(ValidationError):
        ActivityContext.model_validate({
            "knowledge": {},
            "belief": {},
        })


def test_activity_context_rejects_unknown_state_elements():
    with pytest.raises(ValidationError):
        ActivityContext.model_validate({
            "knowledge": {},
            "belief": {},
            "supervisor": SupervisorState.prior().model_dump(),
            "control": {},
        })


def test_entry_test_populates_next_context_without_mutating_previous():
    previous = ActivityContext.neutral()
    previous.knowledge = StudentKnowledgeProgress(
        root={
            "untested": EvidenceKnowledgeProgress(
                mastery=0.7,
                positive_evidence=2.0,
                negative_evidence=1.0,
            )
        }
    )
    questions = [
        {
            "id": "q01",
            "options": [
                {"text": "correct", "targets": ["tested"]},
                {"text": "wrong", "targets": ["tested"]},
            ],
            "solution": [0],
        }
    ]

    current = process_entry_test(
        previous,
        questions,
        {"q01": {"options": ["correct"], "skip": False}},
    )

    assert current is not previous
    assert set(current.knowledge.root) == {"untested", "tested"}
    assert current.knowledge.root["tested"].mastery == pytest.approx(0.7)
    assert current.knowledge.root["tested"].positive_evidence == pytest.approx(0.7)
    assert current.knowledge.root["tested"].negative_evidence == pytest.approx(0.3)
    assert previous.knowledge.root.get("tested") is None
    assert current.belief == previous.belief
    assert current.supervisor == previous.supervisor
    assert current.belief is not previous.belief
    assert current.supervisor is not previous.supervisor


def test_skipped_entry_test_carries_all_three_prior_states_forward():
    previous = ActivityContext.neutral()

    current = process_entry_test(previous, [], {})

    assert current == previous
    assert current is not previous
