from aidu.ai.core.artifacts import AppletArtifact, TextArtifact


def test_text_artifact_returns_content_unchanged():
    artifact = TextArtifact(producer="user", step=0, content="Explain this.")

    assert artifact.to_text() == "Explain this."


def test_applet_artifact_returns_deterministic_json():
    artifact = AppletArtifact(
        producer="user",
        step=0,
        content={"value": 2, "applet": "example"},
    )

    assert artifact.to_text() == '{"applet": "example", "value": 2}'
