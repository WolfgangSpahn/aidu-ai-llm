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
from aidu.ai.core.belief import StudentBelief
from aidu.ai.agents.assessment_context import (
    activity_state,
    dialog_history,
    last_tutor_message,
)

logger = logging.getLogger(__name__)


class StudentBeliefAssessor(WorkflowAgent, LLMFcRequester):
    """Estimate the canonical ``StudentBelief`` dimensions for one learner turn."""

    prompt_template = textwrap.dedent("""\
        Estimate the student's current learning state. Return ONLY JSON.

        CURRENT_MESSAGE is the primary evidence about the student now.
        LAST_MESSAGE, HISTORY, and ACTIVITY_STATE may clarify short or
        referential answers. PRIOR_BELIEF is the baseline from the preceding
        turn; retain it when the current turn provides no reason to change a
        dimension.

        Output exactly:
        {{"belief":{{"engagement":0.0,"confidence":0.0,"confusion":0.0,
        "frustration":0.0,"curiosity":0.0,"self_explanation":0.0,
        "guessing":0.0,"help_seeking":0.0}},"review":false}}

        Rules:
        - Every belief value must be a number from 0 to 1.
        - Return all eight dimensions.
        - Estimate observable state, not subject-matter mastery.
        - Do not infer emotion from correctness alone.
        - Change values conservatively from PRIOR_BELIEF.
        - Engagement: active participation and sustained effort.
        - Confidence: certainty expressed in the learner's own response.
        - Confusion: difficulty understanding or choosing a next step.
        - Frustration: irritation, discouragement, or repeated blocked effort.
        - Curiosity: interest, exploration, prediction, or conceptual questions.
        - Self-explanation: explaining relationships or reasoning in own words.
        - Guessing: unsupported answers or trial without stated reasoning.
        - Help-seeking: requests for hints, confirmation, or guidance.
        - Set review:true only when the evidence cannot be interpreted reliably.

        PRIOR_BELIEF:
        {prior_belief}

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
        belief: StudentBelief = context.state.data["StudentBelief"]
        return {
            "prior_belief": belief.model_dump_json(),
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
        """Ask the configured model for the current turn's belief estimate."""
        if not context.on_air:
            belief: StudentBelief = context.state.data["StudentBelief"]
            assessment = {
                "belief": belief.model_dump(),
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
