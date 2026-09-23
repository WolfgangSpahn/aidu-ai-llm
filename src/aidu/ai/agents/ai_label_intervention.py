# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
"""Label the main pedagogical intervention in the AI tutor's current response."""

import json
import logging
import textwrap
from typing import Any

from aidu.ai.agents.assessment_context import (
    dialog_history,
    last_tutor_message,
    tutor_activity_state,
)
from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.applet_info import AppletInfo
from aidu.ai.core.artifacts import TextArtifact
from aidu.ai.core.belief import StudentBelief
from aidu.ai.core.config import AskConfig
from aidu.ai.core.context import Context, Message
from aidu.ai.core.supervisor import InterventionLabel
from aidu.ai.llm.agent import WorkflowAgent
from aidu.ai.llm.fc_requester import LLMFcRequester

logger = logging.getLogger(__name__)


INTERVENTION_LABELS = (
    "CONTINUE",
    "PROBE",
    "RECALL",
    "ELICIT_EXPLANATION",
    "CHALLENGE",
    "CONTRAST",
    "HINT",
    "DIRECT_ATTENTION",
    "EXPLAIN",
    "MODEL_EXAMPLE",
    "TRANSFER",
    "META_REFLECT",
    "AFFECT_REGULATE",
    "ADJUST_DIFFICULTY",
)

assert set(INTERVENTION_LABELS) == set(
    InterventionLabel.model_fields["intervention"].annotation.__args__
)


class AiLabelIntervention(WorkflowAgent, LLMFcRequester):
    """Identify the dominant pedagogical intention of a completed tutor turn."""

    prompt_template = textwrap.dedent("""\
        Label the single dominant pedagogical intervention actually performed
        by LAST_TUTOR_MESSAGE. Return ONLY JSON.

        Use LAST_TUTOR_MESSAGE as the authoritative turn. Use the other context
        only to infer the pedagogical function of that turn.

        Output exactly:
        {{"intervention":"CONTINUE","reason":""}}

        intervention must be EXACTLY ONE of:
        CONTINUE, PROBE, RECALL, ELICIT_EXPLANATION, CHALLENGE, CONTRAST, HINT,
        DIRECT_ATTENTION, EXPLAIN, MODEL_EXAMPLE, TRANSFER, META_REFLECT,
        AFFECT_REGULATE, ADJUST_DIFFICULTY.

        Definitions:
        * CONTINUE: let productive work continue without adding a new pedagogical demand.
        * PROBE: obtain diagnostic evidence through a factual answer, observation,
          prediction, choice, description, or immediate inference when no more specific
          intervention applies.
        * RECALL: activate relevant prior or prerequisite knowledge for the current task.
        * ELICIT_EXPLANATION: explicitly ask the learner to justify an answer or claim,
          give reasons or evidence, explain why something is the case, or explain how
          they arrived at an answer.
        * CHALLENGE: stress-test understanding that already appears established or strong.
        * CONTRAST: ask the learner to compare or distinguish two or more cases, concepts,
          claims, representations, or outcomes.
        * HINT: provide a small scaffold while leaving the main reasoning to the learner.
        * DIRECT_ATTENTION: focus the learner on a relevant feature, change, relation, or
          piece of evidence.
        * EXPLAIN: directly provide missing conceptual information, reasoning, or a causal
          account.
        * MODEL_EXAMPLE: demonstrate a strategy, procedure, worked solution, or example.
        * TRANSFER: ask the learner to apply or generalize established knowledge to a
          meaningfully different case or context.
        * META_REFLECT: prompt reflection on the learner's reasoning, confidence, strategy,
          assumptions, or errors.
        * AFFECT_REGULATE: primarily support motivation, reduce frustration, or restore
          engagement.
        * ADJUST_DIFFICULTY: make the task easier or harder to better match the learner's
          current capability.

        Classification rules:
        - Label pedagogical FUNCTION, not grammatical form.
        - Choose the most specific applicable intervention before considering PROBE.

        - CONTRAST: use whenever the learner is explicitly asked to compare,
          distinguish, identify differences/similarities, or contrast two or more
          cases, concepts, claims, representations, or outcomes.
          CONTRAST takes precedence over PROBE and ELICIT_EXPLANATION.
        - "What stays the same and what changes?", "How do A and B differ?", and
          "A versus B" are CONTRAST.
          

        - DIRECT_ATTENTION: use when the main tutor action is to direct the learner
          toward a specific feature, observation, change, or piece of evidence.
          DIRECT_ATTENTION takes precedence over PROBE when noticing that feature is
          the main cognitive task.

        - ELICIT_EXPLANATION: use only when the learner is explicitly asked to provide
          reasons, justification, evidence, a causal explanation, or explain how they
          arrived at an answer.
          "What changed?", "What is different?", "How does X change Y?", "Describe X",
          and "What stays the same?" are NOT ELICIT_EXPLANATION.
        - Asking "How does X affect/change/determine Y?" asks for a relation and is
          PROBE, not ELICIT_EXPLANATION, unless the learner is explicitly asked to
          justify or explain why that relation holds.

        - PROBE: use for a factual answer, observation, prediction, choice,
          description, or immediate inference when none of the more specific
          interventions applies.

        - RECALL: use when prior or prerequisite knowledge is explicitly retrieved for
          use in the current task.

        - TRANSFER: use when established knowledge is applied to a meaningfully new
          case or context, not merely another instance of the same task.

        - CHALLENGE: use when apparently established understanding is deliberately
          stress-tested with a difficult case, counterexample, or demanding check.

        - HINT: use when a small clue is supplied while leaving the main inference to
          the learner.

        - EXPLAIN: use when the tutor directly supplies the missing conceptual
          relation, reasoning, or causal account.

        - MODEL_EXAMPLE: use when the tutor demonstrates a procedure, strategy,
          worked solution, or example.

        - META_REFLECT: use when the learner is asked to inspect their own reasoning,
          confidence, strategy, assumptions, or errors.

        - AFFECT_REGULATE: use when the main purpose is motivation, reassurance,
          frustration reduction, or restoring engagement.

        - ADJUST_DIFFICULTY: use when the task is deliberately made easier or harder.

        - CONTINUE: use when productive work is simply allowed or encouraged to continue
          without a new pedagogical demand.

        - When several acts occur in one turn, choose the intervention responsible for
          the main cognitive work.
        - Label what the tutor actually did, not what it should have done.
        - reason must be one concise sentence referring only to LAST_TUTOR_MESSAGE.



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
        current_student_message: str,
    ) -> dict[str, Any]:
        """Build labeling values for the preceding tutor intervention."""
        belief: StudentBelief = context.state.data["StudentBelief"]
        progress = context.state.data["StudentKnowledgeProgress"]
        targets = context.state.data["SessionContext"].domain_targets
        teacher_targets = [
            {"id": target["id"], "text": target["text"]}
            for target in targets
            if target["id"] in progress.root
        ]
        recorded_state = (
            context.trace.messages.applet_state_before_last_tutor_message()
        )
        return {
            "applet_state_summary": (
                AppletInfo.from_payload(recorded_state).state_summary()
                if recorded_state
                else "Applet state: No recorded snapshot available before this tutor response."
            ),
            "teacher_targets": json.dumps(teacher_targets, ensure_ascii=False),
            "student_knowledge_progress": progress.model_dump_json(),
            "student_belief": belief.model_dump_json(),
            "history": dialog_history(context),
            "current_student_message": current_student_message,
            "last_tutor_message": last_tutor_message(context),
            "assessed_tutor_turn_index": json.dumps(
                context.state.data.get("LastTutorTurnIndex")
            ),
            "outcome_student_turn_index": json.dumps(
                context.state.data.get(
                    "OutcomeStudentTurnIndex",
                    max(0, context.state.data.get("TurnIndex", 1) - 1),
                )
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
        """Ask the configured model to label the current tutor intervention."""
        if not context.on_air:
            label = {
                "intervention": "CONTINUE",
                "reason": (
                    "Deterministic off-air test result; no live model labeling was performed."
                ),
            }
            result = self.result(
                artifacts=[
                    TextArtifact(
                        producer=self.id,
                        step=context.step,
                        content=json.dumps(label),
                    )
                ]
            )
            logger.info("AiLabelIntervention off-air result: %s", label)
            return result, context

        if agents is not None:
            self.validate_target_continuations_against_agents(agents)

        result, context = self.ask(
            Message(role="user", content=artifact.content),
            context,
            ask_params=ask_params,
            ask_config=ask_config,
        )
        logger.info("AiLabelIntervention result: %s", result.content())
        return result, context
