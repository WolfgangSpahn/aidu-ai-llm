# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
"""Assess the domain-independent student belief state from the current turn."""

import json
import logging
import textwrap
from typing import Any

from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.artifacts import TextArtifact
from aidu.ai.core.config import AskConfig
from aidu.ai.core.context import Context, Message
from aidu.ai.llm.agent import WorkflowAgent
from aidu.ai.llm.fc_requester import LLMFcRequester
from aidu.ai.agents.assessment_context import (
    activity_state,
    dialog_history,
    last_tutor_message,
)

logger = logging.getLogger(__name__)


class StudentBeliefAssessor(WorkflowAgent, LLMFcRequester):
    """Extract learner speech acts that can support a belief-state update."""

    prompt_template = textwrap.dedent("""\
        Extract observable evidence about the student's current learning state.
        Return ONLY JSON. Do not choose final belief-state values.

        CURRENT_MESSAGE is the primary evidence about the student now.
        LAST_MESSAGE, HISTORY, and ACTIVITY_STATE may clarify short or
        referential answers. Classify what the learner communicates before any
        mental-state inference.

        Output exactly this shape (the example speech_act is one value):
        {{"evidence":[{{"speech_act":"hypothesize",
        "strength":"weak|moderate|strong",
        "confidence":0.0,"quote":"exact learner quote"}}],"review":false}}

        Rules:
        - review is a required top-level boolean, alongside evidence.
          Never put review inside an evidence item. Each evidence item has only
          speech_act, strength, confidence, and quote.
        - Return at most four evidence items; an empty list is valid.
        - speech_act must be EXACTLY ONE value from this list:
          state, hypothesize, explain, report_observation, ask,
          ask_for_explanation, ask_for_hint, ask_for_confirmation, infer,
          predict, compare, justify, confirm, reject, identify_error,
          express_uncertainty, report_action, propose_action, commit_action,
          acknowledge, express_understanding, express_nonunderstanding,
          request_continue, request_topic_change, request_stop, off_topic.
        - Never combine values with "|", and never invent values such as answer.
          Use state for assertions; assert is not an allowed speech_act.
          express_surprise is not an allowed speech_act. Omit unsupported
          emotional labels; do not translate surprise into understanding.
        - Quote must be an exact substring of CURRENT_MESSAGE.
        - Use the speech-act ontology, not final learner-state dimensions.
        - Do not infer emotion, confidence, or engagement from correctness.
        - Python maps validated speech acts to numeric belief-state changes.
        - confidence is confidence that the quoted text supports this evidence.
        - strength describes how explicit and substantial the signal is.
        - Prefer the most specific act (for example ask_for_hint over ask).
        - hypothesize expresses possibility; it is not the same as guessing.
          For example, "Nitrogen, I guess" is hypothesize and may separately
          contain express_uncertainty; "answer" is not a speech act.
        - explain and justify require an expressed reason or relationship.
        - express_uncertainty and express_nonunderstanding must be explicit.
        - Set review:true only when the evidence cannot be interpreted reliably.

        HISTORY:
        {history}

        LAST_MESSAGE:
        {last_message}

        CURRENT_MESSAGE:
        {current_message}

        ACTIVITY_STATE:
        {activity_state}

        JSON:
        """).strip()

    @classmethod
    def build_prompt_args(
        cls,
        *,
        context: Context,
        current_message: str,
    ) -> dict[str, Any]:
        """Build this assessor's prompt values from the canonical turn context."""
        return {
            "history": dialog_history(context),
            "last_message": last_tutor_message(context),
            "current_message": current_message,
            "activity_state": activity_state(context),
        }

    def run(
        self,
        artifact: TextArtifact,
        context: Context,
        agents=None,
        *,
        ask_params: dict | None = None,
        ask_config: AskConfig | None = None,
    ) -> tuple[AgentResult, Context]:
        """Ask the configured model for observable acts in the current turn."""
        if not context.on_air:
            assessment = {
                "evidence": [],
                "review": True,
            }
            result = self.result(
                artifacts=[
                    TextArtifact(
                        producer=self.id,
                        step=context.step,
                        content=json.dumps(assessment),
                    )
                ]
            )
            logger.info("StudentBeliefAssessor off-air result: %s", assessment)
            return result, context

        if agents is not None:
            self.validate_target_continuations_against_agents(agents)

        result, context = self.ask(
            Message(role="user", content=artifact.content),
            context,
            ask_params=ask_params,
            ask_config=ask_config,
        )
        logger.info("StudentBeliefAssessor result: %s", result.content())
        return result, context
