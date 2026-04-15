# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See ../LICENSE for the full text.

"""
    Prompt builder for constructing and managing LLM prompts with parameter substitution.
    Handles placeholder replacement and template validation.
"""

import re
from .safeformat import SafeFormat
import logging

logger = logging.getLogger(__name__)

class PromptBuilder:
    """
    Handles prompt construction and placeholder replacement.
    """

    def __init__(self, template: str):
        self.template = template

    def build(self, prompt_params: dict = None) -> str:
        """
        Build the final prompt.

        Args:
            prompt_params: Parameters used initially (e.g. problem, context)

        Returns:
            Final formatted prompt string
        """
        prompt = self.template

        # apply prompt params safely
        if prompt_params:
            # assure that prompt_params does not contain keys that are not in the template 
            # to avoid confusion
            placeholders = self.extract_placeholders()
            for key in prompt_params.keys():
                if key not in placeholders:
                    logger.warning(f"Prompt param '{key}' is not a placeholder in the template.")
            prompt = prompt.format_map(SafeFormat(**prompt_params))

        return prompt

    def extract_placeholders(self) -> list[str]:
        """Return all placeholders in the template."""
        return re.findall(r'{(.*?)}', self.template)

# --------------------------------------------------------------------------------------------------------------
# smoke test
#


def run_smoke_test():
    template = "Solve the problem: {problem}. Use the following context: {context}."
    prompt_params = {
        "problem": "What is the capital of France?",
        "context": "France is a country in Europe. Its capital is Paris."
    }
    builder = PromptBuilder(template)
    prompt = builder.build(prompt_params)
    print(prompt)


if __name__ == "__main__":
    run_smoke_test()