from types import SimpleNamespace

from aidu.ai.core.context import Context
from aidu.ai.llm.clients.openai import OpenAIClient


class Delta:
    def __init__(self, **values):
        self.values = values

    def model_dump(self, exclude_none=True):
        return self.values


def chunk(content="", *, finish_reason=None, usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=Delta(content=content), finish_reason=finish_reason)],
        usage=usage,
    )


def test_openai_client_streams_by_default_and_returns_complete_message():
    calls = []
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5, model_dump=lambda: {})
    completions = SimpleNamespace(
        create=lambda **kwargs: iter([
            chunk("Hel"),
            chunk("lo", finish_reason="stop", usage=usage),
        ])
    )
    client = object.__new__(OpenAIClient)
    client.model = "gpt-5-mini"
    client.config = {}
    client.stream = True
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    context = Context(on_air=True)
    context.control.data["stream_callback"] = calls.append

    response = client.ask({"role": "user", "content": "Hi"}, context)

    assert calls == ["Hel", "lo"]
    assert response["content"] == "Hello"
    assert response["finish_reason"] == "stop"
    assert response["total_tokens"] == 5
    assert response["time_to_first_token_seconds"] >= 0


def test_openai_client_can_disable_streaming():
    message = SimpleNamespace(model_dump=lambda: {"role": "assistant", "content": "complete"})
    response_obj = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=None,
    )
    completions = SimpleNamespace(create=lambda **kwargs: response_obj)
    client = object.__new__(OpenAIClient)
    client.model = "gpt-5-mini"
    client.config = {}
    client.stream = False
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    response = client.ask({"role": "user", "content": "Hi"}, Context(on_air=True))

    assert response["content"] == "complete"
