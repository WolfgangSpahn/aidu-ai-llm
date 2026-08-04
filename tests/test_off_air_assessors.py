import json

from aidu.ai.agents.ai_supervisor import AiSupervisor
from aidu.ai.agents.learning_target_assessor import LearningTargetAssessor
from aidu.ai.agents.student_belief_assessor import StudentBeliefAssessor
from aidu.ai.core.artifacts import TextArtifact
from aidu.ai.core.belief import StudentBelief, StudentBeliefAssessment
from aidu.ai.core.context import Context
from aidu.ai.core.supervisor import SUPERVISOR_DIMENSIONS, SupervisorState
from aidu.ai.llm.agent import EndAgent


class NoProviderClient:
    """Fail if an off-air assessor tries to contact its configured provider."""

    def ask(self, *args, **kwargs):
        raise AssertionError("Off-air assessment contacted the provider")


def run_off_air(agent, context: Context) -> dict:
    result, returned_context = agent.run(
        TextArtifact(producer="test", step=0, content="Assess this turn."),
        context,
    )

    assert returned_context is context
    return json.loads(result.content())


def test_learning_target_assessor_emits_valid_off_air_test_output():
    context = Context(on_air=False)

    assessment = run_off_air(
        LearningTargetAssessor(client=NoProviderClient(), target=EndAgent),
        context,
    )

    assert assessment == {"evidence": [], "review": True}


def test_student_belief_assessor_preserves_prior_belief_off_air():
    prior = StudentBelief(confidence=0.8, confusion=0.2)
    context = Context(on_air=False)
    context.state.data["StudentBelief"] = prior

    assessment = run_off_air(
        StudentBeliefAssessor(client=NoProviderClient(), target=EndAgent),
        context,
    )

    parsed = StudentBeliefAssessment.model_validate(assessment)
    assert parsed.belief.model_dump() == prior.model_dump()
    assert parsed.review is True


def test_ai_supervisor_emits_valid_neutral_off_air_test_output():
    context = Context(on_air=False)

    assessment = run_off_air(
        AiSupervisor(client=NoProviderClient(), target=EndAgent),
        context,
    )

    parsed = SupervisorState.model_validate(assessment)
    results = [getattr(parsed, dimension) for dimension in SUPERVISOR_DIMENSIONS]
    assert all(result.fit == 0.5 for result in results)
    assert all("off-air test result" in result.reason for result in results)
