from aidu.ai.agents.ai_supervisor import AiSupervisor


def test_ai_supervisor_assesses_tutor_response_against_learning_context():
    prompt = AiSupervisor.prompt_template
    prompt_flat = " ".join(prompt.split())

    for dimension in (
        "factual_fit",
        "goal_alignment",
        "knowledge_alignment",
        "belief_alignment",
        "scaffolding_fit",
    ):
        assert f'"{dimension}"' in prompt

    assert "{last_tutor_message}" in prompt
    assert "subsequent response as evidence" in prompt
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
    assert "HISTORY ends with that tutor message" in prompt
    assert "CURRENT_STUDENT_MESSAGE" not in prompt
    assert "OUTCOME_EVIDENCE_AVAILABLE" not in prompt
    assert "every tutor action, claim, question, and topic" in prompt_flat
    assert "Do not lower the score merely because other targets" in prompt_flat
    assert "Relational or orienting support can be the correct immediate scaffold" in prompt_flat
    assert "{assessed_tutor_turn_index}" in prompt
    assert "{outcome_student_turn_index}" not in prompt
    assert "suggested_action" not in prompt


def test_supervisor_prompt_identifies_opening_turn_independently_of_recent_history():
    import json
    from aidu.ai.core.context import Context
    from aidu.ai.core.belief import StudentBelief
    from aidu.ai.core.session import SessionContext

    context = Context()
    session = SessionContext(on_air=False, domain_targets=[])
    context.state.data.update({
        "SessionContext": session,
        "StudentBelief": StudentBelief(),
        "StudentKnowledgeProgress": session.initial_student_knowledge_progress(),
        "AppletState": {},
        "LastTutorTurnIndex": 2,
    })
    for initial in (True, False, None):
        context.state.data["IsInitialTutorTurn"] = initial
        args = AiSupervisor.build_prompt_args(context=context)
        assert json.loads(args["is_initial_tutor_turn"]) is initial
        assert "IS_INITIAL_TUTOR_TURN:" in AiSupervisor.prompt_template.format(**args)
    assert "judge it as an opening invitation" in AiSupervisor.prompt_template


def test_supervisor_summarizes_snapshot_on_ordinary_learner_message_before_tutor():
    from aidu.ai.core.context import Context, Messages
    from aidu.ai.core.belief import StudentBelief
    from aidu.ai.core.session import SessionContext

    session = SessionContext(on_air=False, domain_targets=[])
    context = Context()
    context.state.data.update({"SessionContext": session, "StudentBelief": StudentBelief(),
                               "StudentKnowledgeProgress": session.initial_student_knowledge_progress()})
    context.trace.messages = Messages.model_validate([
        {"role": "user", "kind": "message", "content": "you are broken off", "applet_input": {
            "applet": "applet-build-an-atom", "infoStore": {
                "shorttext": "Placed: 2 protons, 1 neutrons, 2 inner electrons, 0 outer electrons.", "protonCount": 2,
            },
        }},
        {"role": "assistant", "content": "Apologies for that cutoff!"},
        {"role": "user", "content": "How should I now?", "applet_input": {
            "applet": "applet-build-an-atom", "infoStore": {"shorttext": "Placed: 3 protons.", "protonCount": 3},
        }},
    ]).cleaned_dialog()
    args = AiSupervisor.build_prompt_args(context=context)
    assert args["applet_state_summary"] == "Applet state: 2 protons, 1 neutrons, 2 inner electrons, 0 outer electrons."
    assert '"protonCount": 2' in args["applet_state_at_tutor_turn"]
    assert '"protonCount": 3' not in args["applet_state_at_tutor_turn"]
    assert "How should I now?" not in args["history"]
    assert "Apologies for that cutoff!" in args["history"]
    assert "current_student_message" not in args
    assert "time of its last change is unknown" in " ".join(AiSupervisor.prompt_template.split())
