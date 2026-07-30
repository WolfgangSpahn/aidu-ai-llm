"""Read canonical turn evidence shared by assessment agents."""

import json

from aidu.ai.core.context import Context

MAX_HISTORY_TURNS = 10


def dialog_history(context: Context) -> str:
    """Serialize recent validated dialog messages for an assessment prompt."""
    return json.dumps(
        context.trace.messages[-MAX_HISTORY_TURNS:],
        ensure_ascii=False,
    )


def last_tutor_message(context: Context) -> str:
    """Return the most recent assistant message in the validated dialog trace."""
    return next(
        (
            f"Tutor: {message['content']}"
            for message in reversed(context.trace.messages[-MAX_HISTORY_TURNS:])
            if message["role"] == "assistant"
        ),
        "No previous tutor turn.",
    )


def activity_state(context: Context) -> str:
    """Serialize the canonical applet state stored for the current turn."""
    return json.dumps(context.state.data["AppletState"], ensure_ascii=False)


def tutor_activity_state(context: Context) -> str:
    """Serialize the applet state visible before the last tutor response."""
    return json.dumps(
        context.trace.messages.applet_state_before_last_tutor_message(),
        ensure_ascii=False,
    )
