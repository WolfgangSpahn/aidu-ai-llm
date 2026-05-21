# src/aidu/ai/core/hookspecs.py

from __future__ import annotations

import pluggy


hookspec = pluggy.HookspecMarker("aidu")
hookimpl = pluggy.HookimplMarker("aidu")


class HookSpecs:

    @hookspec
    def get_agents(self):
        """
        Return available chat-capable agent classes.
        """