from types import SimpleNamespace

from aidu.ai.core.context import Context, Trace
from aidu.ai.llm.clients.google import GoogleClient


def test_google_client_streams_text_and_uses_system_instruction():
    deltas = []
    captured = {}

    def generate_content_stream(**kwargs):
        captured.update(kwargs)
        return iter([
            SimpleNamespace(text="Hel", candidates=[], usage_metadata=None),
            SimpleNamespace(text="lo", candidates=[], usage_metadata=None),
        ])

    client = object.__new__(GoogleClient)
    client.model = "gemini-3.6-flash"
    client.config = {"max_tokens": 1024, "thinking_level": "medium"}
    client.stream = True
    client.client = SimpleNamespace(
        models=SimpleNamespace(generate_content_stream=generate_content_stream)
    )
    context = Context(
        on_air=True,
        trace=Trace(messages=[{"role": "system", "content": "Tutor instructions"}]),
    )
    context.control.data["stream_callback"] = deltas.append

    response = client.ask({"role": "user", "content": "Hi"}, context)

    assert deltas == ["Hel", "lo"]
    assert response["content"] == "Hello"
    assert captured["config"]["system_instruction"] == "Tutor instructions"
    assert captured["config"]["max_output_tokens"] == 1024
    assert captured["config"]["thinking_config"] == {"thinking_level": "medium"}
