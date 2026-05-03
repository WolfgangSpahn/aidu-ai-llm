# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
Prompter encapsulates all system-prompt handling:
- templated prompts via PromptBuilder
- SafeFormat for partial placeholder filling
- build_system_prompt() and update_system_prompt() helpers
"""

import os

from .builder import PromptBuilder
from .client import Context
from .safeformat import SafeFormat


class Prompter:
    """
    Manages system prompt construction and updates.

    Usage patterns:
    - Template string: Prompter(prompt_template="You are a {subject} tutor.", prompt_args={"subject": "math"})
    - Template file:   Prompter(prompt_template="/path/to/prompt.txt")
    """

    def __init__(self, prompt_template=None, prompt_args=None):
        """
        Args:
            prompt_template: Template string or file path. Placeholders use {name} syntax.
            prompt_args: Default values for template placeholders (SafeFormat — unfilled
                         placeholders remain as {placeholder}).
        """
        self.prompt_args = prompt_args or {}

        template = prompt_template
        if template and os.path.isfile(template):
            with open(template, "r") as f:
                template = f.read()

        self.prompt_builder = PromptBuilder(template) if template else None

        assert self.prompt_builder, f"Failed to initialize PromptBuilder with the provided template: {prompt_template}"

    def build_system_prompt(self, prompt_params=None) -> list[dict]:
        """
        Return a one-element list containing the system message dict.

        Args:
            prompt_params: Extra values merged on top of prompt_args (take precedence).

        Returns:
            [{"role": "system", "content": "..."}] or [] when no prompt is configured.
        """
        merged = {**self.prompt_args, **(prompt_params or {})}

        if not self.prompt_builder:
            return []

        content = self.prompt_builder.build(prompt_params=merged or None)
        return [{"role": "system", "content": content}]

    def update_system_prompt(self, context: Context, prompt_params=None) -> Context:
        """
        Replace the first message in context.trace with a freshly built system message.
        Returns context unchanged if it has no leading system message.

        Args:
            context: Current conversation context.
            prompt_params: Extra values merged on top of prompt_args (take precedence).
        """
        if not context.trace.messages or context.trace.messages[0]["role"] != "system":
            return context

        new_system = self.build_system_prompt(prompt_params)
        if new_system:
            context.trace.messages = new_system + context.trace.messages[1:]
        return context
