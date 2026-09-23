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
from aidu.ai.core.applet_info import AppletInfo
from aidu.ai.core.belief import StudentBelief
from aidu.ai.core.supervisor import SUPERVISOR_DIMENSIONS
from aidu.ai.agents.assessment_context import (
    MAX_HISTORY_TURNS,
    last_tutor_message,
    tutor_activity_state,
)

logger = logging.getLogger(__name__)


class AiSupervisor(WorkflowAgent, LLMFcRequester):
    """Record retrospective fit signals for later human review."""

    prompt_template = textwrap.dedent("""\
        Assess the preceding AI tutor response for a later human reviewer.
        Return ONLY JSON.

        Evaluate LAST_TUTOR_MESSAGE in the context of TEACHER_TARGETS,
        STUDENT_KNOWLEDGE_PROGRESS, STUDENT_BELIEF, prior-dialog HISTORY, and
        APPLET_STATE_AT_TUTOR_TURN.

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
        - IS_INITIAL_TUTOR_TURN identifies the opening tutor message in the full
          conversation, even when HISTORY contains only a recent excerpt.
          When true, judge it as an opening invitation: orientation, a manageable
          first action, and an observation question can be sufficient. Do not
          expect it to acknowledge earlier learner answers or provide feedback
          on an investigation that has not happened yet.
          When false, judge it as a continuation using the available prior context.
          When null, the opening position is unknown; do not infer it merely
          because the visible HISTORY is short.
        - Assess only the quality of this one turn at its position in the
          dialog. Do not require one message to complete the lesson, address
          every teacher target, or contain every later scaffold.
        - A fit of 1.0 means fully appropriate for this turn and context, not a
          perfect or complete lesson.
        - Assess the tutor using only information available at the time of
          LAST_TUTOR_MESSAGE. HISTORY ends with that tutor message and contains
          only earlier dialog; no later learner response or outcome is part of
          this assessment. Never use a student's subsequent response as evidence
          for or against the tutor's performance.
        - Give one concise reason for each fit score so a human reviewer can
          understand the signal later.
        - Every reason must evaluate only LAST_TUTOR_MESSAGE. HISTORY is context,
          not an alternative tutor turn. Before returning JSON, verify that every
          tutor action, claim, question, and topic named in a reason is actually
          present in LAST_TUTOR_MESSAGE or is a direct characterization of it.
          If a reason discusses a different question or later topic, discard it
          and reassess LAST_TUTOR_MESSAGE from scratch.
        - Assess what happened; do not propose revisions or future actions.
        - Do not discuss prompt fields, message selection, duplicated context,
          missing metadata, or how the supervisor input was assembled.
        - APPLET_STATE_SUMMARY describes the latest recorded applet configuration
          available before the assessed tutor response. It is machine-generated
          context, not learner-authored evidence or proof of an action on that
          turn. The configuration may be unchanged from earlier turns; the time
          of its last change is unknown. Do not infer a recent placement or
          penalize the tutor based on a later applet snapshot. An unavailable
          snapshot does not establish a factual error.
        - factual_fit: score only factual correctness and consistency with
          APPLET_STATE_AT_TUTOR_TURN. Do not lower it for pedagogical omissions.
        - goal_alignment: score whether this turn advances at least one relevant
          teacher target. It need not name the target or cover all targets.
          Do not lower the score merely because other targets are not addressed.
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
          Relational or orienting support can be the correct immediate scaffold
          when the learner expresses confusion, reluctance, or interface trouble;
          it need not advance chemistry content in the same message.

        TEACHER_TARGETS:
        {teacher_targets}

        STUDENT_KNOWLEDGE_PROGRESS:
        {student_knowledge_progress}

        STUDENT_BELIEF:
        {student_belief}

        HISTORY:
        {history}

        LAST_TUTOR_MESSAGE:
        {last_tutor_message}

        IS_INITIAL_TUTOR_TURN:
        {is_initial_tutor_turn}

        ASSESSED_TUTOR_TURN_INDEX:
        {assessed_tutor_turn_index}

        APPLET_STATE_SUMMARY:
        {applet_state_summary}

        APPLET_STATE_AT_TUTOR_TURN:
        {applet_state_at_tutor_turn}

        JSON:
        """).strip()

    @classmethod
    def build_prompt_args(
        cls,
        *,
        context: Context,
    ) -> dict[str, Any]:
        """Build supervision values for the preceding tutor intervention."""
        belief: StudentBelief = context.state.data["StudentBelief"]
        progress = context.state.data["StudentKnowledgeProgress"]
        targets = context.state.data["SessionContext"].domain_targets
        teacher_targets = [{"id": target["id"], "text": target["text"]} for target in targets if target["id"] in progress.root]
        recorded_state = context.trace.messages.applet_state_before_last_tutor_message()
        # The current learner turn triggers retrospective assessment, but it
        # follows the tutor message and must not be exposed to the supervisor.
        recent_messages = context.trace.messages[-MAX_HISTORY_TURNS:]
        tutor_position = next(
            (
                index
                for index in range(len(recent_messages) - 1, -1, -1)
                if recent_messages[index].role == "assistant"
            ),
            None,
        )
        prior_dialog_history = json.dumps(
            [message.to_dict() for message in recent_messages[: tutor_position + 1]]
            if tutor_position is not None else [],
            ensure_ascii=False,
        )
        return {
            "applet_state_summary": (
                AppletInfo.from_payload(recorded_state).state_summary()
                if recorded_state else "Applet state: No recorded snapshot available before this tutor response."
            ),
            "teacher_targets": json.dumps(teacher_targets, ensure_ascii=False),
            "student_knowledge_progress": progress.model_dump_json(),
            "student_belief": belief.model_dump_json(),
            "history": prior_dialog_history,
            "last_tutor_message": last_tutor_message(context),
            "is_initial_tutor_turn": json.dumps(
                context.state.data.get(
                    "IsInitialTutorTurn",
                    True if context.state.data.get("LastTutorTurnIndex") == 0 else None,
                )
            ),
            "assessed_tutor_turn_index": json.dumps(
                context.state.data.get("LastTutorTurnIndex")
            ),
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
