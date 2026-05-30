# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
Evaluation agents extending LLMRequester for analyzing and scoring LLM outputs.
Provides base Evaluator class and specialized implementations for educational scenarios.
"""

import json
import logging

from aidu.ai.core.context import Context, Trace
from aidu.ai.core.config import AskConfig
from .requester import LLMRequester

logger = logging.getLogger(__name__)

class Evaluator(LLMRequester):
    """
    Evaluates the last student turn and returns a distribution over a Likert scale.
    
    The Likert scale has 5 points, and the evaluator returns a normalized probability 
    distribution across these points.
    
    Subclass and define prompt_template to create specific evaluators.
    See evaluators/ subdirectory for concrete implementations.
    """

    prompt_template = None
    likert_labels = [
        "Very likely",
        "Likely", 
        "Neutral",
        "Not likely",
        "Not at all"
    ]

    def __init__(self, client, prompt_template=None, prompt_args=None, tools=None):
        """Initialize evaluator with optional prompt template and arguments."""
        super().__init__(
            client=client,
            prompt_template=prompt_template,
            prompt_args=prompt_args,
            tools=tools
        )

    @staticmethod
    def _parse_json_response(response_text: str) -> dict:
        """Parse JSON response, accepting fenced markdown JSON blocks as fallback."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            if "```json" in response_text:
                json_str = response_text.split("```json", 1)[1].split("```", 1)[0].strip()
                return json.loads(json_str)
            raise

    def _run_chat(self, user_prompt: str, eval_params: dict, chat_config: AskConfig | None = None) -> tuple[dict, Context]:
        """Run one evaluator turn through the shared requester chat contract."""
        system_messages = self.build_system_prompt(prompt_params=eval_params)
        context = Context(trace=Trace(messages=system_messages))
        return self.ask(
            message={"role": "user", "content": user_prompt},
            context=context,
            chat_config=chat_config,
        )


    def evaluate(self, user_prompt="", eval_params: dict = None, enforce_json: bool = True) -> list[float] | None:
        """
        Evaluate using the provided parameters on a Likert scale.
        
        Args:
            user_prompt (str): User message to trigger evaluation
            eval_params (dict): Parameters for evaluation (structure defined by subclasses)
            enforce_json (bool): If True, enforce JSON response format (default True)
            
        Returns:
            list[float]: Probability distribution over 5 Likert points, or None on error
        """
        response_text = ""

        try:
            eval_params = eval_params or {}
            chat_config = AskConfig(json_mode=enforce_json)

            message, _ = self._run_chat(
                user_prompt=user_prompt,
                eval_params=eval_params,
                chat_config=chat_config,
            )
            
            response_text = message.get("content", "")
            result = self._parse_json_response(response_text)
            
            distribution = result.get("distribution")
            if not distribution or len(distribution) != 5:
                logger.error(f"Invalid distribution: {distribution}")
                return None
            
            # Normalize to probability distribution (sum to 1)
            total = sum(distribution)
            if total > 0:
                distribution = [x / total for x in distribution]
            
            logger.info(f"Evaluator distribution: {distribution}")
            return distribution

        except json.JSONDecodeError as e:
            logger.error(f"Evaluation failed (JSON parse): {e}")
            logger.debug(f"Response text: {response_text[:200]}")
            return None
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None



