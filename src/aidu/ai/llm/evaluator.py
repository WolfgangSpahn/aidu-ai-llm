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

from .requester import LLMRequester

logger = logging.getLogger(__name__)

class Evaluator(LLMRequester):
    """
    Evaluates the last student turn and returns a distribution over a Likert scale.
    
    The Likert scale has 5 points, and the evaluator returns a normalized probability 
    distribution across these points.
    
    Subclass and define system_prompt to create specific evaluators.
    See evaluators/ subdirectory for concrete implementations.
    """

    system_prompt = None  # Override in subclass
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
        try:
            eval_params = eval_params or {}
            system_messages = self.build_system_prompt(prompt_params=eval_params)
            
            # We need to add an explicit user prompt to trigger the evaluation, even if it's empty
            messages = system_messages + [{"role": "user", "content": user_prompt}]
            msg, _ = self.run(messages=messages, model="gpt-4o-mini", state={}, enforce_json=enforce_json)
            
            response_text = msg.get("content", "")
            result = json.loads(response_text)
            
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
            # Try to extract JSON from markdown code blocks
            if "```json" in response_text:
                try:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                    result = json.loads(json_str)
                    distribution = result.get("distribution")
                    if distribution and len(distribution) == 5:
                        total = sum(distribution)
                        if total > 0:
                            distribution = [x / total for x in distribution]
                        logger.info(f"Evaluator distribution (from markdown): {distribution}")
                        return distribution
                except Exception:
                    pass
            
            logger.error(f"Evaluation failed (JSON parse): {e}")
            logger.debug(f"Response text: {response_text[:200]}")
            return None
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None



