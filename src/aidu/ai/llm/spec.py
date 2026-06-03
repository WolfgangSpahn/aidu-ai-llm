# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
Plugin specification definitions using pluggy for tool discovery and execution.
Defines the interface for getting tools and calling them with context management.
"""

import pluggy

hookspec = pluggy.HookspecMarker("app")


class ToolSpec:
    @hookspec
    def get_tools(self) -> list[dict]:
        """
        Return tool schemas (OpenAI format)
        """

    @hookspec
    def call_tool(self, name: str, arguments: dict, context):
        """
        Execute tool and return updated context
        """
