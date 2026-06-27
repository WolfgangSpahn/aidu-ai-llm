from aidu.ai.core.artifacts import TextArtifact
from aidu.ai.core.context import Context
from aidu.ai.llm.agent import BeginAgent, EndAgent


def test_begin_agent_passes_initial_artifact_to_target(monkeypatch):
    artifact = TextArtifact(producer="user", step=0, content="hello")
    context = Context()
    begin_agent = BeginAgent(target=EndAgent)

    monkeypatch.setattr(begin_agent, "_show_actor_input", lambda *args: None)

    result, next_context = begin_agent.run(
        artifact=artifact,
        context=context,
        agents=[begin_agent, EndAgent()],
    )

    assert next_context is context
    assert result.artifacts == [artifact]
    assert result.recommendations[0].target is EndAgent
