# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
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
