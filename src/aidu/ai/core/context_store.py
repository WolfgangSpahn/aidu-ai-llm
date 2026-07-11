# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
"""
    Skeleton for a ContextStore implementation. This is meant to be implemented by the user of the library, 
    and can be used to store and retrieve context information for a given session.
"""


from aidu.ai.core.context import Context


class ContextStore:

    def load(
        self,
        session_id: str,
    ) -> Context:
        ...

    def save(
        self,
        session_id: str,
        context: Context,
    ):
        ...