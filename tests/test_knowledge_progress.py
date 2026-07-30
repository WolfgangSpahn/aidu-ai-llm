from aidu.ai.core.knowledge_progress import StudentKnowledgeProgress


def test_student_knowledge_progress_preserves_mapping_wire_shape():
    progress = StudentKnowledgeProgress.model_validate(
        {
            "target-1": {
                "mastery": 0.4,
                "positive_evidence": 2.0,
                "negative_evidence": 1.0,
            }
        }
    )

    assert progress.model_dump() == {
        "target-1": {
            "mastery": 0.4,
            "positive_evidence": 2.0,
            "negative_evidence": 1.0,
        }
    }


def test_student_knowledge_progress_clamps_its_numeric_state():
    progress = StudentKnowledgeProgress.model_validate(
        {
            "target-1": {
                "mastery": 1.4,
                "positive_evidence": -2.0,
                "negative_evidence": -1.0,
            }
        }
    )

    clamped = progress.clamped()

    assert clamped.root["target-1"].mastery == 1.0
    assert clamped.root["target-1"].positive_evidence == 0.0
    assert clamped.root["target-1"].negative_evidence == 0.0


def test_student_knowledge_progress_describes_loaded_targets_to_tutor():
    empty = StudentKnowledgeProgress(root={})
    progress = StudentKnowledgeProgress.model_validate(
        {
            "target-1": {
                "mastery": 0.0,
                "positive_evidence": 0.0,
                "negative_evidence": 4.0,
            }
        }
    )

    assert empty.to_tutor_text() == " - We have not started yet."
    assert progress.to_tutor_text() == (
        " - Learning targets loaded: 1. "
        "Each domain target progress starts at 0.0."
    )
