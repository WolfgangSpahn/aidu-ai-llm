from aidu.ai.agents.ai_supervisor import AiSupervisor


def test_ai_supervisor_assesses_tutor_response_against_learning_context():
    prompt = AiSupervisor.prompt_template

    for dimension in (
        "factual_fit",
        "goal_alignment",
        "knowledge_alignment",
        "belief_alignment",
        "scaffolding_fit",
    ):
        assert f'"{dimension}"' in prompt

    assert "{last_tutor_message}" in prompt
    assert "outcome evidence" in prompt
    assert "{teacher_targets}" in prompt
    assert "{student_knowledge_progress}" in prompt
    assert "{student_belief}" in prompt
    assert "{applet_state_at_tutor_turn}" in prompt
    assert "later human reviewer" in prompt
    assert "do not propose revisions or future actions" in prompt
    assert "intentional and is never a tutor-quality problem" in prompt
    assert "Do not require one message to complete the lesson" in prompt
    assert "one manageable applet action followed" in prompt
    assert "Do not lower it for pedagogical omissions" in prompt
    assert "suggested_action" not in prompt
