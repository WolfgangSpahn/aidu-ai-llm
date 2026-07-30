# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
"""Assess the quality and pedagogical fit of the AI tutor's current response."""

import logging
import textwrap
import json
from typing import Any

from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.artifacts import TextArtifact
from aidu.ai.core.config import AskConfig
from aidu.ai.core.context import Context, Message
from aidu.ai.llm.agent import WorkflowAgent
from aidu.ai.llm.fc_requester import LLMFcRequester
from aidu.ai.core.belief import StudentBelief
from aidu.ai.core.supervisor import SUPERVISOR_DIMENSIONS
from aidu.ai.agents.assessment_context import (
    dialog_history,
    last_tutor_message,
    tutor_activity_state,
)

logger = logging.getLogger(__name__)


class AiSupervisor(WorkflowAgent, LLMFcRequester):
    """Record retrospective fit signals for later human review."""

    prompt_template = textwrap.dedent("""\
        Assess the preceding AI tutor response for a later human reviewer.
        Return ONLY JSON.

        Evaluate LAST_TUTOR_MESSAGE in the context of CURRENT_STUDENT_MESSAGE,
        TEACHER_TARGETS, STUDENT_KNOWLEDGE_PROGRESS, STUDENT_BELIEF, recent
        HISTORY, and APPLET_STATE_AT_TUTOR_TURN.

        Output exactly:
        {{"factual_fit":{{"fit":0.0,"reason":""}},
        "goal_alignment":{{"fit":0.0,"reason":""}},
        "knowledge_alignment":{{"fit":0.0,"reason":""}},
        "belief_alignment":{{"fit":0.0,"reason":""}},
        "scaffolding_fit":{{"fit":0.0,"reason":""}}}}

        Rules:
        - Every fit must be a number from 0 to 1.
        - Return all five dimensions.
        - LAST_TUTOR_MESSAGE is the authoritative tutor turn to assess. HISTORY
          may contain the same turn for chronology; that duplication is
          intentional and is never a tutor-quality problem.
        - Assess only the quality of this one turn at its position in the
          dialog. Do not require one message to complete the lesson, address
          every teacher target, or contain every later scaffold.
        - A fit of 1.0 means fully appropriate for this turn and context, not a
          perfect or complete lesson.
        - Treat CURRENT_STUDENT_MESSAGE as outcome evidence: it may show whether
          the preceding tutor intervention was understandable and productive.
        - Give one concise reason for each fit score so a human reviewer can
          understand the signal later.
        - Assess what happened; do not propose revisions or future actions.
        - Do not discuss prompt fields, message selection, duplicated context,
          missing metadata, or how the supervisor input was assembled.
        - factual_fit: score only factual correctness and consistency with
          APPLET_STATE_AT_TUTOR_TURN. Do not lower it for pedagogical omissions.
        - goal_alignment: score whether this turn advances at least one relevant
          teacher target. It need not name the target or cover all targets.
        - knowledge_alignment: score whether the turn is manageable from the
          demonstrated target progress. Zero or absent mastery means novice;
          one concrete observable action is often appropriate.
        - belief_alignment: use only supported learner-state evidence. Default,
          mixed, or uncertain belief values do not establish confusion and
          should favor a neutral judgment.
        - scaffolding_fit: score the support needed for this immediate step.
          For an initial discovery turn, one manageable applet action followed
          by an observation question is valid scaffolding; the tutor need not
          explain or interpret the result before the student investigates it.

        TEACHER_TARGETS:
        {teacher_targets}

        STUDENT_KNOWLEDGE_PROGRESS:
        {student_knowledge_progress}

        STUDENT_BELIEF:
        {student_belief}

        HISTORY:
        {history}

        CURRENT_STUDENT_MESSAGE:
        {current_student_message}

        LAST_TUTOR_MESSAGE:
        {last_tutor_message}

        APPLET_STATE_AT_TUTOR_TURN:
        {applet_state_at_tutor_turn}

        JSON:
        """).strip()

    @classmethod
    def build_prompt_args(
        cls,
        *,
        context: Context,
        current_student_message: str,
    ) -> dict[str, Any]:
        """Build supervision values for the preceding tutor intervention."""
        belief: StudentBelief = context.state.data["StudentBelief"]
        progress = context.state.data["StudentKnowledgeProgress"]
        targets = context.state.data["SessionContext"].domain_targets
        teacher_targets = [{"id": target["id"], "text": target["text"]} for target in targets if target["id"] in progress.root]
        return {
            "teacher_targets": json.dumps(teacher_targets, ensure_ascii=False),
            "student_knowledge_progress": progress.model_dump_json(),
            "student_belief": belief.model_dump_json(),
            "history": dialog_history(context),
            "current_student_message": current_student_message,
            "last_tutor_message": last_tutor_message(context),
            "applet_state_at_tutor_turn": tutor_activity_state(context),
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
        """Ask the configured model to assess the current tutor response."""
        if not context.on_air:
            assessment = {
                dimension: {
                    "fit": 0.5,
                    "reason": ("Deterministic off-air test result; no live model assessment was performed."),
                }
                for dimension in SUPERVISOR_DIMENSIONS
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
            logger.info("AiSupervisor off-air result: %s", assessment)
            return result, context

        if agents is not None:
            self.validate_target_continuations_against_agents(agents)

        result, context = self.ask(
            Message(role="user", content=artifact.content),
            context,
            ask_params=ask_params,
            ask_config=ask_config,
        )
        logger.info("AiSupervisor result: %s", result.content())
        return result, context
