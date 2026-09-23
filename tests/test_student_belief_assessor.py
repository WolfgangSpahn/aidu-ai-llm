from aidu.ai.agents.student_belief_assessor import StudentBeliefAssessor
from aidu.ai.core.belief import StudentBeliefAssessment
def test_student_belief_assessor_extracts_speech_acts_not_belief_dimensions():
    prompt = StudentBeliefAssessor.prompt_template

    assert '"speech_act"' in prompt
    assert "express_uncertainty" in prompt
    assert "express_nonunderstanding" in prompt
    assert "{prior_belief}" not in prompt
    assert "{current_message}" in prompt
    assert '"evidence"' in prompt
    assert "Do not choose final belief-state values" in prompt


def test_belief_evidence_normalizes_common_model_speech_act_aliases():
    assessment = StudentBeliefAssessment.model_validate({
        "evidence": [{
            "speech_act": "answer",
            "strength": "strong",
            "confidence": 0.9,
            "quote": "Nitrogen",
        }],
        "review": False,
    })

    assert assessment.evidence[0].speech_act == "state"


def test_belief_assessment_recovers_misplaced_review_without_mutating_raw_result():
    from aidu.ai.core.belief import StudentBeliefAssessment
    import copy
    evidence = {"speech_act": "ask", "strength": "strong", "confidence": 0.9, "quote": "How?", "review": False}
    raw = {"evidence": [evidence]}
    original = copy.deepcopy(raw)
    parsed = StudentBeliefAssessment.model_validate(raw)
    assert parsed.review is False
    assert parsed.evidence[0].speech_act == "ask"
    assert raw == original
    assert "review" not in parsed.evidence[0].model_dump()
    evidence["review"] = True
    assert StudentBeliefAssessment.model_validate(raw).review is True
    raw["review"] = False
    assert StudentBeliefAssessment.model_validate(raw).review is True


def test_belief_assessment_still_rejects_invalid_review_and_other_extra_fields():
    from aidu.ai.core.belief import StudentBeliefAssessment
    from pydantic import ValidationError
    import pytest
    evidence = {"speech_act": "ask", "strength": "strong", "confidence": 0.9, "quote": "How?"}
    with pytest.raises(ValidationError):
        StudentBeliefAssessment.model_validate({"evidence": [{**evidence, "review": "false"}]})
    with pytest.raises(ValidationError):
        StudentBeliefAssessment.model_validate({"evidence": [{**evidence, "review": False, "unexpected": 1}]})
    with pytest.raises(ValidationError):
        StudentBeliefAssessment.model_validate({"evidence": [evidence]})


def test_belief_assessment_normalizes_assert_to_state():
    from aidu.ai.core.belief import StudentBeliefAssessment
    raw = {"evidence": [{"speech_act": "assert", "strength": "moderate",
                         "confidence": 0.9, "quote": "Now it is neutral"}], "review": False}
    parsed = StudentBeliefAssessment.model_validate(raw)
    assert parsed.evidence[0].speech_act == "state"
    assert raw["evidence"][0]["speech_act"] == "assert"
