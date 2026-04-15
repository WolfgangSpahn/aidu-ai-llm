# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
    Evaluation agent extending LLMRequester for correctness estimation and evaluation tasks.
    Specializes in analyzing and scoring LLM outputs against given criteria.
"""

import json
import logging

from aidu.ai.llm.requester import LLMRequester

logger = logging.getLogger(__name__)

class EvaluationAgent(LLMRequester):
    """
    Encapsulates LLM logic for correctness estimation.
    """

    def __init__(self, client, prompt_template=None, tools=None):

        super().__init__(
            client=client,
            prompt_template=prompt_template
        )

    def evaluate(self, user_input: str) -> float | None:
        try:
            system_messages = self.build_system_prompt(
                prompt_params={"user_input": user_input}
            )
            messages = system_messages + [
                {"role": "user", "content": "Evaluate correctness."}
            ]

            msg, _ = self.run( messages=messages, model="gpt-4o-mini", state={})

            content = msg.get("content", "")

            data = json.loads(content)

            logger.info(f"CorrectnessAgent output: {data}")

            correctness_value = data.get("correctness")
            if correctness_value is None: return None

            correctness = float(correctness_value)

            return max(0.0, min(1.0, correctness))

        except Exception as e:
            logger.error(f"CorrectnessAgent failed: {e}")
            raise



