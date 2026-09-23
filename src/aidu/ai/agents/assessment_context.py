"""Read canonical turn evidence shared by assessment agents."""

import json
import re

from aidu.ai.core.applet_info import AppletInfo
from aidu.ai.core.context import Context

MAX_HISTORY_TURNS = 10


def dialog_history(context: Context) -> str:
    """Serialize recent validated dialog messages for an assessment prompt."""
    return json.dumps(
        [turn.to_dict() for turn in context.trace.messages[-MAX_HISTORY_TURNS:]],
        ensure_ascii=False,
    )


def last_tutor_message(context: Context) -> str:
    """Return the most recent assistant message in the validated dialog trace."""
    return next(
        (
            f"Tutor: {message.content}"
            for message in reversed(context.trace.messages[-MAX_HISTORY_TURNS:])
            if message.role == "assistant"
        ),
        "No previous tutor turn.",
    )


def activity_state(context: Context) -> str:
    """Describe the current applet state and preserve its structured values."""
    payload = context.state.data.get("AppletState") or {}
    if not payload:
        return "No current applet state was reported."
    applet = AppletInfo.from_payload(payload)
    values = json.dumps(applet.selected_info(), ensure_ascii=False)
    return (
        f"{applet.state_summary()}\n{activity_change(context)}\n"
        f"Structured infoStore values: {values}"
    )


def activity_change(context: Context) -> str:
    """Narrate applet info-store changes since the latest tutor turn began."""
    previous_payload = context.trace.messages.applet_state_before_last_tutor_message()
    current_payload = context.state.data.get("AppletState") or {}
    if not previous_payload or not current_payload:
        return "No comparable before-and-after applet snapshots are available."

    previous = AppletInfo.from_payload(previous_payload)
    current = AppletInfo.from_payload(current_payload)
    if previous.applet != current.applet:
        return (
            f"The applet changed from {previous.applet} to {current.applet}; "
            "their infoStore fields cannot be compared."
        )

    before = previous.info_store
    after = current.info_store
    keys = sorted(before.keys() | after.keys())
    labels = {
        "protonCount": "proton count",
        "neutronCount": "neutron count",
        "innerElectronCount": "inner electron count",
        "outerElectronCount": "outer electron count",
        "atomicNumber": "atomic number",
        "massNumber": "mass number",
        "elementSymbol": "element symbol",
        "charge": "charge",
    }

    def label(key: str) -> str:
        return labels.get(key, re.sub(r"(?<!^)(?=[A-Z])", " ", key).lower())

    changes: list[str] = []
    for key in keys:
        if key == "shorttext" or before.get(key) == after.get(key):
            continue
        old, new = before.get(key), after.get(key)
        field_label = label(key)
        if (
            isinstance(old, (int, float)) and not isinstance(old, bool)
            and isinstance(new, (int, float)) and not isinstance(new, bool)
        ):
            delta = new - old
            changes.append(f"{field_label}: {old} → {new} ({delta:+g})")
        else:
            changes.append(f"{field_label}: {old!r} → {new!r}")

    stable_particles = [
        f"{label(key)} unchanged at {before[key]}"
        for key in keys
        if key in before and key in after and before[key] == after[key]
        and key in labels
        and key not in {"elementSymbol", "massNumber", "charge"}
    ]
    if not changes:
        changes.append("no infoStore values changed")
    if stable_particles:
        changes.extend(stable_particles)
    return (
        "Applet state change since the tutor's latest instruction: "
        + "; ".join(changes)
        + "."
    )


def tutor_activity_state(context: Context) -> str:
    """Serialize the applet state visible before the last tutor response."""
    return json.dumps(
        context.trace.messages.applet_state_before_last_tutor_message(),
        ensure_ascii=False,
    )
