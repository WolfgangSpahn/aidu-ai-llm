# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.


from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


class MessageRecord(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...


@dataclass(frozen=True)
class AppletInfo:
    """Structured view of a frontend applet infoStore message.

    Applet submissions usually arrive as a payload with an ``applet`` id and an
    ``infoStore`` object. This helper keeps that structure as the source of
    truth and only derives text when dialog history needs a compact summary.
    """

    applet: str = "applet"
    info_store: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AppletInfo":
        """Create an AppletInfo from a request payload dictionary."""
        applet = payload.get("applet")
        info_store = payload.get("infoStore")
        return cls(
            applet=str(applet or "applet"),
            info_store=info_store if isinstance(info_store, dict) else {},
            payload=payload,
        )

    @classmethod
    def from_snapshot(cls, snapshot: str) -> "AppletInfo":
        """Create an AppletInfo from a JSON snapshot string."""
        try:
            parsed = json.loads(snapshot)
        except json.JSONDecodeError:
            return cls(payload={"raw": snapshot})

        if isinstance(parsed, dict):
            return cls.from_payload(parsed)

        return cls(payload={"value": parsed})

    @classmethod
    def from_message(cls, message: MessageRecord) -> "AppletInfo | None":
        applet_input = message.get("applet_input")
        if isinstance(applet_input, dict) and applet_input:
            return cls.from_payload(applet_input)

        content = str(message.get("content") or "").strip()
        if content.startswith("Applet input:"):
            _, _, payload = content.partition("\n")
            return cls.from_snapshot(payload.strip())

        return None

    def to_state(self) -> dict[str, Any]:
        return self.payload

    def selected_info(self, keys: Iterable[str] | None = None) -> dict[str, Any]:
        source = self.info_store
        if keys is None:
            return {
                key: value
                for key, value in source.items()
                if value is not None
            }

        return {
            key: source[key]
            for key in keys
            if key in source and source[key] is not None
        }

    def state_summary(self) -> str:
        """Describe a recorded configuration without implying a recent action."""
        shorttext = self.info_store.get("shorttext")
        if isinstance(shorttext, str) and shorttext.strip():
            text = shorttext.strip()
            if text.startswith("Placed:"):
                text = text.removeprefix("Placed:").strip()
            if text == "No particles placed.":
                text = "No particles."
            return text if text.lower().startswith("applet state:") else f"Applet state: {text}"
        return f"Applet state: {self.applet} with " + json.dumps(self.selected_info(), ensure_ascii=False)

    def to_text(self, keys: Iterable[str] | None = None) -> str:
        details = ", ".join(
            f"{key}={value}"
            for key, value in self.selected_info(keys).items()
        )
        if details:
            return f"Applet event: {self.applet} with {details}"

        if "raw" in self.payload:
            return f"Applet event: {str(self.payload['raw'])[:500]}"

        if "value" in self.payload:
            return f"Applet event: {self.payload['value']!r}"

        return f"Applet event: {self.applet}"
