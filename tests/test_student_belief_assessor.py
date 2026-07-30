from aidu.ai.agents.student_belief_assessor import StudentBeliefAssessor
from aidu.ai.core.belief import StudentBelief


def test_student_belief_assessor_uses_canonical_belief_dimensions():
    prompt = StudentBeliefAssessor.prompt_template

    for field_name in StudentBelief.model_fields:
        assert f'"{field_name}"' in prompt

    assert "{prior_belief}" in prompt
    assert "{current_message}" in prompt
    assert "subject-matter mastery" in prompt
