import json

from aidu.ai.agents.ai_label_intervention import (
    INTERVENTION_LABELS,
    AiLabelIntervention,
)
from aidu.ai.core.artifacts import TextArtifact
from aidu.ai.core.belief import StudentBelief
from aidu.ai.core.context import Context
from aidu.ai.core.session import SessionContext
from aidu.ai.llm.agent import EndAgent


class NoProviderClient:
    def ask(self, *args, **kwargs):
        raise AssertionError("Off-air intervention labeler contacted the provider")


def test_prompt_constrains_label_to_manual_interventions():
    prompt = AiLabelIntervention.prompt_template

    assert set(INTERVENTION_LABELS) == {
        "CONTINUE", "PROBE", "RECALL", "ELICIT_EXPLANATION", "CHALLENGE",
        "CONTRAST", "HINT", "DIRECT_ATTENTION", "EXPLAIN", "MODEL_EXAMPLE",
        "TRANSFER", "META_REFLECT", "AFFECT_REGULATE", "ADJUST_DIFFICULTY",
    }
    assert all(label in prompt for label in INTERVENTION_LABELS)
    assert '"intervention"' in prompt
    assert '"reason"' in prompt
    assert "single dominant pedagogical intervention" in prompt
    assert "A question is not automatically PROBE" in prompt
    assert "not what it should have done" in prompt
    assert "reason must be one concise sentence" in prompt
    assert "LAST_TUTOR_MESSAGE" in prompt


def test_build_prompt_args_uses_preceding_tutor_context():
    session = SessionContext(on_air=False, domain_targets=[])
    context = Context()
    context.state.data.update({
        "SessionContext": session,
        "StudentBelief": StudentBelief(),
        "StudentKnowledgeProgress": session.initial_student_knowledge_progress(),
        "LastTutorTurnIndex": 3,
        "OutcomeStudentTurnIndex": 4,
    })

    args = AiLabelIntervention.build_prompt_args(
        context=context,
        current_student_message="Because the proton count changed.",
    )

    assert args["current_student_message"] == "Because the proton count changed."
    assert json.loads(args["assessed_tutor_turn_index"]) == 3
    assert json.loads(args["outcome_student_turn_index"]) == 4
    rendered = AiLabelIntervention.prompt_template.format(**args)
    assert "APPLET_STATE_AT_TUTOR_TURN:" in rendered
    assert '{"intervention":"CONTINUE","reason":""}' in rendered
    assert '{{"intervention"' not in rendered


def test_off_air_label_is_deterministic_and_valid():
    context = Context(on_air=False)
    agent = AiLabelIntervention(client=NoProviderClient(), target=EndAgent)

    result, returned_context = agent.run(
        TextArtifact(producer="test", step=0, content="Label this tutor turn."),
        context,
    )
    label = json.loads(result.content())

    assert returned_context is context
    assert label["intervention"] in INTERVENTION_LABELS
    assert "off-air test result" in label["reason"]


def test_prompt_distinguishes_answer_probe_from_explanation_request():
    prompt = AiLabelIntervention.prompt_template

    assert "PROBE is the fallback diagnostic category" in prompt
    assert "ELICIT_EXPLANATION requires an explicit request for reasoning" in prompt
    assert "justification, evidence, or how the learner knows" in prompt
