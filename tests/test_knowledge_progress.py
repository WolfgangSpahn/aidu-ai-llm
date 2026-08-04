import pytest
from pydantic import ValidationError

from aidu.ai.core.knowledge_progress import StudentKnowledgeProgress


VALID_STATE = {
    "mastery": 0.4,
    "positive_evidence": 0.3,
    "negative_evidence": 0.45,
    "entry_prior": 0.4,
    "entry_weight": 0.75,
    "source_count": 1,
    "turn_assessment_count": 0,
    "last_updated_turn": None,
    "evidence_fingerprints": [],
}


def test_student_knowledge_progress_preserves_canonical_wire_shape():
    progress = StudentKnowledgeProgress.model_validate({"target-1": VALID_STATE})

    assert progress.model_dump() == {"target-1": VALID_STATE}


@pytest.mark.parametrize(
    "change",
    [
        {"mastery": 0.5},
        {"positive_evidence": -1.0},
        {"entry_weight": -1.0},
        {"source_count": -1},
    ],
)
def test_student_knowledge_progress_rejects_inconsistent_state(change):
    state = {**VALID_STATE, **change}
    with pytest.raises(ValidationError):
        StudentKnowledgeProgress.model_validate({"target-1": state})


def test_student_knowledge_progress_rejects_old_incomplete_contract():
    with pytest.raises(ValidationError):
        StudentKnowledgeProgress.model_validate(
            {
                "target-1": {
                    "mastery": 0.4,
                    "positive_evidence": 0.3,
                    "negative_evidence": 0.45,
                }
            }
        )


def test_student_knowledge_progress_describes_loaded_targets_to_tutor():
    empty = StudentKnowledgeProgress(root={})
    progress = StudentKnowledgeProgress.model_validate({"target-1": VALID_STATE})

    assert empty.to_tutor_text() == " - We have not started yet."
    tutor_text = progress.to_tutor_text()
    assert "Active learning targets: 1" in tutor_text
    assert "target-1: 40%" in tutor_text
    assert "tentative entry-test prior from 1 question" in tutor_text
    assert "not yet confirmed in dialog" in tutor_text
