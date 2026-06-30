import sys
import types

from aidu.ai.core.artifacts import TextArtifact
from aidu.ai.core.context import Context, Trace
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


def test_begin_agent_trace_rows_truncate_content_and_mark_placeholders():
    rows = BeginAgent._trace_message_rows(
        [
            {
                "role": "system",
                "content": "You are a tutor for {level}. " + ("long " * 40),
            },
            {
                "role": "user",
                "content": "Hydrogen selected",
            },
        ],
        max_content_length=40,
    )

    assert rows[0] == (0, "system", "You are a tutor for {level}. long...", True)
    assert rows[1] == (1, "user", "Hydrogen selected", False)


def test_begin_agent_placeholder_detection_ignores_dict_repr_keys():
    assert BeginAgent._content_has_placeholders("{student_progress}")
    assert BeginAgent._content_has_placeholders("{primary_weight:.0%}")
    assert not BeginAgent._content_has_placeholders("{'elementSymbol': 'H'}")


def test_begin_agent_trace_rows_preview_structured_applet_payload():
    rows = BeginAgent._trace_message_rows(
        [
            {
                "role": "user",
                "kind": "applet",
                "content": "Applet event: applet-create-a-molecule",
                "applet_input": {
                    "applet": "applet-create-a-molecule",
                    "infoStore": {
                        "atoms": [{"element": "H"}, {"element": "O"}],
                        "bonds": [{"from": 0, "to": 1}],
                    },
                },
            },
        ],
    )

    assert rows[0] == (
        0,
        "user",
        "{'applet': 'applet-create-a-molecule', 'infoStore': {...}}",
        False,
    )


def test_begin_agent_trace_rows_default_to_100_character_preview():
    rows = BeginAgent._trace_message_rows(
        [
            {
                "role": "system",
                "content": "word " * 40,
            },
        ],
    )

    assert len(rows[0][2]) <= 100
    assert rows[0][2].endswith("...")


def test_begin_agent_trace_rows_mark_empty_content():
    rows = BeginAgent._trace_message_rows(
        [
            {
                "role": "system",
                "content": "",
            },
        ],
    )

    assert rows[0] == (0, "system", "<empty>", False)


def test_begin_agent_context_query_snapshot_is_json_serializable():
    context = Context(trace=Trace(messages=[{"role": "user", "content": "hello"}]))

    snapshot = BeginAgent._context_query_snapshot(context)

    assert snapshot["trace"]["messages"] == [{"role": "user", "content": "hello"}]
    assert "state" in snapshot


def test_begin_agent_runs_context_jq_query(monkeypatch):
    class FakeCompiledJq:
        def input(self, value):
            self.value = value
            return self

        def all(self):
            return [self.value["trace"]["messages"][0]["content"]]

    fake_jq = types.SimpleNamespace(compile=lambda query: FakeCompiledJq())
    monkeypatch.setitem(sys.modules, "jq", fake_jq)
    context = Context(trace=Trace(messages=[{"role": "user", "content": "hello"}]))

    assert BeginAgent._run_context_jq_query(context, ".trace.messages[0].content") == ["hello"]
