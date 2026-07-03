from aidu.ai.core.context import Context, Message, Trace
from aidu.ai.llm.clients.openai import _chat_completion_message
from aidu.ai.llm.requester import LLMRequester


class CapturingClient:
    model = "fake-model"

    def __init__(self):
        self.messages = None

    def ask(self, message, context, config=None):
        self.messages = [*context.trace.messages, message]
        return {
            "role": "assistant",
            "content": "ok",
            "model": self.model,
        }


def test_ask_prepends_system_prompt_for_llm_without_mutating_dialog_trace():
    client = CapturingClient()
    requester = LLMRequester(
        client=client,
        prompt_template="System prompt for {topic}.",
    )
    context = Context(
        trace=Trace(
            messages=[
                {
                    "role": "assistant",
                    "content": "Welcome Anonymous to our Chemistry Periodic Table session.",
                },
                {
                    "role": "user",
                    "content": "Applet event: applet-periodic-table with elementName=Hydrogen",
                },
            ]
        )
    )

    response, next_context = requester.ask(
        {"role": "user", "content": "Why is hydrogen a gas?"},
        context,
        ask_params={"topic": "chemistry"},
    )

    assert response["content"] == "ok"
    assert next_context is context
    assert context.trace.messages[0]["role"] == "assistant"
    assert client.messages[0] == {
        "role": "system",
        "content": "System prompt for chemistry.",
    }
    assert client.messages[1:] == [
        *context.trace.messages,
        {"role": "user", "content": "Why is hydrogen a gas?"},
    ]


def test_ask_drops_duplicate_current_message_from_effective_llm_history():
    client = CapturingClient()
    requester = LLMRequester(
        client=client,
        prompt_template="System prompt.",
    )
    context = Context(
        trace=Trace(
            messages=[
                {
                    "role": "user",
                    "content": "Why is hydrogen a gas?",
                },
            ]
        )
    )

    requester.ask({"role": "user", "content": "Why is hydrogen a gas?"}, context)

    assert client.messages == [
        {
            "role": "system",
            "content": "System prompt.",
        },
        {
            "role": "user",
            "content": "Why is hydrogen a gas?",
        },
    ]


def test_ask_accepts_message_model_for_duplicate_detection_and_client_call():
    client = CapturingClient()
    requester = LLMRequester(
        client=client,
        prompt_template="System prompt.",
    )
    context = Context(
        trace=Trace(
            messages=[
                {
                    "role": "user",
                    "content": "Applet event: applet-build-an-atom",
                },
            ]
        )
    )

    requester.ask(Message(role="user", content="Applet event: applet-build-an-atom"), context)

    assert client.messages == [
        {
            "role": "system",
            "content": "System prompt.",
        },
        {
            "role": "user",
            "content": "Applet event: applet-build-an-atom",
        },
    ]


def test_openai_chat_message_drops_trace_only_applet_metadata():
    assert _chat_completion_message(
        {
            "role": "user",
            "content": "Applet event: applet-periodic-table",
            "kind": "applet",
            "applet_input": {
                "applet": "applet-periodic-table",
                "infoStore": {"elementName": "Hydrogen"},
            },
            "duration": 0.0,
        }
    ) == {
        "role": "user",
        "content": "Applet event: applet-periodic-table",
    }
