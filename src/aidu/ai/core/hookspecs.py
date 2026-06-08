# src/aidu/ai/core/hookspecs.py

from __future__ import annotations

import pluggy

hookspec = pluggy.HookspecMarker("aidu")
hookimpl = pluggy.HookimplMarker("aidu")


class HookSpecs:
    @hookspec
    def get_assistants(self):
        """
        Return available chat-capable assistant classes.
        """
