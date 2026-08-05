import pytest
from pydantic import ValidationError

from aidu.ai.core.context import Context, Message, Messages, PersistedTurn, Trace


def test_messages_returns_latest_persisted_learner_states():
    messages = Messages.model_validate(
        [
            {
                "role": "assistant",
                "content": "Assessment snapshot",
                "backend_knowledge_progress_state": {
                        "target-1": {
                            "mastery": 1.0,
                            "positive_evidence": 2.0,
                            "negative_evidence": 0.0,
                            "entry_prior": 0.8,
                            "entry_weight": 0.75,
                            "source_count": 2,
                            "turn_assessment_count": 1,
                            "last_updated_turn": 3,
                    }
                },
                "backend_belief_state": {
                    "confidence": 0.3,
                },
            }
        ]
    )

    assert messages.latest_knowledge_progress().root["target-1"].mastery == 1.0
    assert messages.latest_belief().confidence == 0.3


def test_messages_wire_shape_preserves_required_null_in_knowledge_snapshot():
    messages = Messages.model_validate(
        [{
            "role": "assistant",
            "content": "Prior state",
            "backend_knowledge_progress_state": {
                "target-1": {
                    "mastery": 0.4,
                    "positive_evidence": 0.3,
                    "negative_evidence": 0.45,
                    "entry_prior": 0.4,
                    "entry_weight": 0.75,
                    "source_count": 1,
                    "turn_assessment_count": 0,
                    "last_updated_turn": None,
                    "evidence_fingerprints": [],
                }
            },
        }]
    )

    serialized = messages.model_dump()

    assert serialized[0]["backend_knowledge_progress_state"]["target-1"]["last_updated_turn"] is None
    assert "avatar" not in serialized[0]


def test_messages_rejects_unknown_persisted_turn_fields_at_the_boundary():
    with pytest.raises(ValidationError, match="unexpected_metadata"):
        Messages.model_validate(
            [{"role": "user", "content": "Hello", "unexpected_metadata": True}]
        )


def test_trace_coerces_history_to_messages_and_preserves_wire_shape():
    trace = Trace(messages=[{"role": "user", "content": "Hello"}])
    trace.messages.append(Message(role="assistant", content="Hi"))

    assert isinstance(trace.messages, Messages)
    assert all(isinstance(turn, PersistedTurn) for turn in trace.messages)
    assert trace.model_dump()["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]


def test_messages_builds_dialog_history_from_applet_records():
    messages = Messages.model_validate(
        [
            {"role": "system", "content": "Internal prompt"},
            {
                "role": "user",
                "content": "I selected it",
                "kind": "applet",
                "applet_input": {
                    "applet": "periodic-table",
                    "infoStore": {"elementSymbol": "H"},
                },
            },
        ]
    )

    assert messages.dialog_history() == (
        " - user: Applet event: periodic-table with elementSymbol=H\n"
        "Student said: I selected it"
    )
    assert messages.cleaned_dialog().root[0].applet_input == {
        "applet": "periodic-table",
        "infoStore": {"elementSymbol": "H"},
    }


def test_context_creates_assessor_copy_without_stream_callback():
    callback = object()
    context = Context()
    context.control.data["stream_callback"] = callback
    context.state.data["value"] = 1

    assessor_context = context.for_assessor()

    assert "stream_callback" not in assessor_context.control.data
    assert context.control.data["stream_callback"] is callback
    assert assessor_context.state.data == context.state.data


def test_messages_returns_applet_state_before_last_tutor_turn():
    messages = Messages.model_validate(
        [
            {
                "role": "user",
                "kind": "applet",
                "applet_input": {
                    "applet": "build-an-atom",
                    "infoStore": {"protonCount": 0},
                },
            },
            {"role": "assistant", "content": "Add one proton."},
            {
                "role": "user",
                "kind": "applet",
                "applet_input": {
                    "applet": "build-an-atom",
                    "infoStore": {"protonCount": 1},
                },
            },
        ]
    )

    assert messages.applet_state_before_last_tutor_message() == {
        "applet": "build-an-atom",
        "infoStore": {"protonCount": 0},
    }
