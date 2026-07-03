from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.artifacts import AppletArtifact, TextArtifact
from aidu.ai.core.context import Context
from aidu.ai.core.recommendation import Recommendation
from aidu.ai.llm.agent import WorkflowAgent
from aidu.ai.llm.agent_runner import prepare_agent_context, run_agent_artifact_chat_turn, run_agent_chat_turn, run_agent_text_turn


class PromptAgent(WorkflowAgent):
    def build_system_prompt(self, prompt_params=None):
        return [{"role": "system", "content": f"Prompt: {prompt_params['topic']}"}]

    def run(self, artifact, context=None, agents=None, ask_params=None):
        return AgentResult(artifacts=[TextArtifact(producer=self.id, step=context.step, content=artifact.content)]), context


class FinalAgent(WorkflowAgent):
    def run(self, artifact, context=None, agents=None, ask_params=None):
        context.step += 1
        return AgentResult(artifacts=[TextArtifact(producer=self.id, step=context.step, content=f"final: {artifact.content}")]), context


class RoutingAgent(WorkflowAgent):
    target = FinalAgent

    def run(self, artifact, context=None, agents=None, ask_params=None):
        context.step += 1
        return (
            AgentResult(
                artifacts=[TextArtifact(producer=self.id, step=context.step, content=f"route: {artifact.content}")],
                recommendations=[Recommendation(target=FinalAgent, utility=1.0)],
            ),
            context,
        )


class AppletEchoAgent(WorkflowAgent):
    def run(self, artifact, context=None, agents=None, ask_params=None):
        context.step += 1
        return AgentResult(artifacts=[TextArtifact(producer=self.id, step=context.step, content=artifact.content["action"])]), context


def test_prepare_agent_context_builds_system_prompt():
    context = prepare_agent_context(PromptAgent(), prompt_params={"topic": "fractions"})

    assert context.trace.messages == [{"role": "system", "content": "Prompt: fractions"}]


def test_run_agent_text_turn_runs_starting_agent_only_by_default():
    result, context = run_agent_text_turn(
        starting_agent=RoutingAgent(),
        user_text="hello",
        agents=[RoutingAgent(), FinalAgent()],
    )

    assert context.step == 1
    assert result.artifacts[0].content == "route: hello"


def test_run_agent_text_turn_can_follow_recommendations():
    result, context = run_agent_text_turn(
        starting_agent=RoutingAgent(),
        user_text="hello",
        agents=[RoutingAgent(), FinalAgent()],
        max_hops=1,
    )

    assert context.step == 2
    assert result.artifacts[0].content == "final: route: hello"


def test_run_agent_chat_turn_returns_reply_text():
    reply, context = run_agent_chat_turn(
        starting_agent=RoutingAgent(),
        user_text="hello",
        agents=[RoutingAgent(), FinalAgent()],
        max_hops=1,
    )

    assert context.step == 2
    assert reply == "final: route: hello"


def test_run_agent_artifact_chat_turn_accepts_existing_artifact():
    reply, context = run_agent_artifact_chat_turn(
        starting_agent=AppletEchoAgent(),
        artifact=AppletArtifact(producer="user", step=0, content={"action": "you selected carbon"}),
    )

    assert context.step == 1
    assert reply == "you selected carbon"
