from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable


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
        applet = payload.get("applet")
        info_store = payload.get("infoStore")
        return cls(
            applet=str(applet or "applet"),
            info_store=info_store if isinstance(info_store, dict) else {},
            payload=payload,
        )

    @classmethod
    def from_snapshot(cls, snapshot: str) -> "AppletInfo":
        try:
            parsed = json.loads(snapshot)
        except json.JSONDecodeError:
            return cls(payload={"raw": snapshot})

        if isinstance(parsed, dict):
            return cls.from_payload(parsed)

        return cls(payload={"value": parsed})

    @classmethod
    def from_message(cls, message: dict[str, Any]) -> "AppletInfo | None":
        applet_input = message.get("applet_input")
        if message.get("kind") == "applet" and isinstance(applet_input, dict):
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
