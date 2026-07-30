from aidu.ai.core.artifacts import (
    ActivityEventArtifact,
    AppletArtifact,
    TextArtifact,
    create_artifact,
    latest_display_artifact,
)


def test_applet_artifact_identifies_outbound_command():
    command = AppletArtifact(
        producer="tutor",
        step=1,
        content={"applet": "applet-build-an-atom", "command": {"action": "add_proton"}},
    )
    event = AppletArtifact(
        producer="user",
        step=1,
        content={"applet": "applet-build-an-atom", "infoStore": {"protonCount": 1}},
    )

    assert command.is_outbound_command() is True
    assert event.is_outbound_command() is False


def test_factory_creates_activity_event_artifact():
    artifact = create_artifact(
        "activity_event",
        id="event-1",
        producer="tutor",
        step=2,
        content={"type": "ai_activity_closed", "disposition": "finalize"},
    )

    assert isinstance(artifact, ActivityEventArtifact)


def test_latest_display_artifact_returns_most_recent_text():
    first = TextArtifact(producer="tutor", step=1, content="First")
    event = ActivityEventArtifact(
        producer="tutor",
        step=2,
        content={"type": "ai_activity_closed", "disposition": "finalize"},
    )
    latest = TextArtifact(producer="tutor", step=3, content="Latest")

    assert latest_display_artifact([first, event, latest]) is latest
    assert latest_display_artifact([event]) is None
