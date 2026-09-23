# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
"""Generic assessor for teacher-defined learning targets."""

import os
import logging
import json
import textwrap
from typing import Any, Literal
from pprint import pformat
from dotenv import load_dotenv

from rich.console import Console
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.config import AskConfig
from aidu.ai.core.artifacts import TextArtifact
from aidu.support.filesystem.search import find_up
from aidu.ai.core.context import Context, Message
from aidu.ai.llm.clients.google import GoogleClient
from aidu.ai.llm.agent import EndAgent, WorkflowAgent
from aidu.ai.llm.fc_requester import LLMFcRequester
from aidu.ai.agents.assessment_context import (
    activity_change,
    activity_state,
    dialog_history,
    last_tutor_message,
)

logger = logging.getLogger(__name__)


class TargetEvidenceAssessment(BaseModel):
    """Strict LLM contract for one target-specific learner observation."""

    model_config = ConfigDict(extra="forbid")

    target: str = Field(min_length=1)
    direction: Literal["positive", "negative"]
    strength: Literal["weak", "moderate", "strong"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_type: Literal[
        "recall",
        "explanation",
        "application",
        "correction",
    ]
    response_mode: Literal["deliberate", "uncertain", "guess"]
    support_level: Literal[
        "independent",
        "small_prompt",
        "guided",
        "explicit_hint",
        "answer_revealed",
    ]
    quote: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_misplaced_response_mode(cls, value: Any) -> Any:
        """Move response-mode labels accidentally emitted as support levels."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        misplaced = normalized.get("support_level")
        if misplaced not in {"deliberate", "uncertain", "guess"}:
            return normalized
        normalized.setdefault("response_mode", misplaced)
        normalized["support_level"] = "independent"
        return normalized


class LearningTargetAssessment(BaseModel):
    """Complete strict response from the learning-target assessor."""

    model_config = ConfigDict(extra="forbid")

    evidence: list[TargetEvidenceAssessment] = Field(max_length=2)
    review: bool


def pretty_content(content: str):
    try:
        return pformat(json.loads(content), indent=2)
    except json.JSONDecodeError:
        return content


class LearningTargetAssessor(WorkflowAgent, LLMFcRequester):
    """Assess a learner message against teacher-defined learning targets."""

    prompt_template = textwrap.dedent("""\
        Assess learning evidence against teacher-defined targets. Return ONLY JSON.

        CURRENT_MESSAGE is the only source of learner evidence.
        TUTOR_QUESTION_OR_INSTRUCTION, HISTORY, and ACTIVITY_STATE provide context
        for interpreting short or referential answers, but they are not learner
        evidence themselves.

        quote rule:
        quote must be an exact substring of CURRENT_MESSAGE.
        Never quote the tutor, history, activity state, target text, or your own inference.
        Omit an evidence item if no exact learner quote exists.

        SOURCE OF TRUTH — APPLET STATE:
        The structured ACTIVITY_STATE / infoStore and its before-after change
        are the ground truth for what the learner actually did and what state
        the applet reached. They take precedence over a learner's claim about
        an action, selection, or result. If the learner says an object/value
        changed but the applet state shows it did not, treat that action/result
        as not having occurred. Never award positive evidence for completing
        that action or reaching that result based on the learner's conflicting
        claim. Score a learner's independently explained scientific reasoning
        separately; a bare expected result attached to the contradicted action
        is not an independent explanation. When the conflict leaves their
        understanding unclear, set review=true and omit positive evidence.

        Output:
        {{"evidence":[{{"target":"target-id","direction":"positive|negative",
        "strength":"weak|moderate|strong","confidence":0.0,
        "evidence_type":"recall|explanation|application|correction",
        "response_mode":"deliberate|uncertain|guess",
        "support_level":"independent|small_prompt|guided|explicit_hint|answer_revealed",
        "quote":"exact learner quote"}}],"review":false}}

        Rules:
        - Max 2 evidence items.
        - Use only IDs from LEARNING_TARGETS.
        - Interpret each ID only through its teacher-defined text.
        - Prefer the most specific matching target.
        - Do not duplicate broad+narrow evidence.
        - Omit targets with no direct learner evidence.
        - Do not reward information supplied only by the tutor, history, or activity.
        - First identify exactly what TUTOR_QUESTION_OR_INSTRUCTION asked the
          learner to say, predict, identify, or do. Assess CURRENT_MESSAGE and
          ACTIVITY_STATE only against that request; do not assess an unrelated
          fact merely because it appears in the applet state.
        - A short answer may be evidence when TUTOR_QUESTION_OR_INSTRUCTION makes
          its meaning clear.
        - Omit evidence when context is insufficient to decide whether the answer
          supports or contradicts the target.
        - Prefer no evidence over speculative evidence. A learner's successful
          action, compliance with an instruction, or report of what the applet
          displays does not by itself demonstrate the underlying concept.
        - ACTIVITY_STATE contains the same learner-facing applet summary plus
          structured values and the before/after change narrative. It may verify
          what happened, but must never add knowledge, reasoning, or particle
          identification that the learner did not express in CURRENT_MESSAGE.
        - Machine-generated status text and structured applet values are never
          learner-authored quotes, even when the UI displays them beside the
          learner message. Use them only to verify the learner's claim or action.
        - ACTIVITY_CHANGE explicitly compares the applet snapshot before the
          tutor's latest instruction with the current snapshot. Use this as the
          ground truth for what actually changed. If the tutor requested a
          change to a particular property but that property is unchanged while
          another property changed, the learner did not complete the requested
          applet action.
          Do not credit the claimed action as application evidence, and do not
          raise knowledge on the basis of a claimed result contradicted by this
          comparison. A separately stated concept may count only when the
          learner explains or justifies it independently of the failed action;
          an unsupported expected result in that mismatch context is not enough.
          Set review=true when this contradiction makes the learner's actual
          understanding uncertain.
        - A displayed value, name, symbol, or charge copied by the learner is at
          most weak evidence for a target that explicitly requires identifying or
          reading that displayed item. It is not evidence that the learner can
          explain the causal relationship, interpret a broader notation system,
          or calculate related quantities.
        - Evidence for a relationship target requires the learner to state,
          predict, compare, or apply that relationship. Merely observing the
          result after the tutor requested an applet action is not enough.
        - A learner's explicit causal calculation or comparison is conceptual
          application evidence for every target whose teacher-defined wording
          it directly satisfies, even when the learner does not use the target's
          preferred technical vocabulary. However, when ACTIVITY_CHANGE
          contradicts the learner's claimed applet action, do not credit an
          expected outcome tied to that uncompleted action as application
          evidence. Require a separate explanation of the causal relationship
          before applying the general rule above. Do not reduce valid reasoning
          to a copied applet value merely because displayed numbers are mentioned.
        - A failed applet attempt can be negative application evidence only when
          the failure itself directly demonstrates knowledge described by the
          target. Interface operation, dragging, placement, visibility, or motor
          difficulty alone is not evidence against a conceptual target. For
          example, "I cannot add an electron" does not contradict understanding
          of how electron number affects charge. Omit evidence unless the learner
          also states an incorrect relationship, prediction, identification, or
          interpretation. Never invent an explanation for why the attempt failed.
        - When CURRENT_MESSAGE and ACTIVITY_STATE conflict about whether an
          applet action succeeded, do not reward the claimed action as positive
          evidence. Apply the conceptual-evidence boundary above; otherwise omit.
        - Compare TUTOR_QUESTION_OR_INSTRUCTION's requested object or action with
          ACTIVITY_STATE's observed object or action. A mismatch must never receive positive
          evidence for either the requested or observed concept. It is negative
          application evidence only when choosing the wrong object directly tests
          a distinction named by a teacher-defined target; otherwise omit it.
          A vague claim such as "I added something" does not identify the object
          and cannot become positive evidence from applet telemetry.
        - An uncertainty response such as "I don't know" is negative evidence
          only for the specific target directly tested by TUTOR_QUESTION_OR_INSTRUCTION. Do not
          attach it to another target mentioned in HISTORY or ACTIVITY_STATE.
        - If LAST_MESSAGE revealed the answer, do not treat repetition or
          paraphrase as independent knowledge. Use support_level "answer_revealed"
          and no more than weak strength, or omit it when no knowledge beyond the
          revealed answer is demonstrated.
        - confidence measures confidence in this assessment, from 0.0 to 1.0.
        - support_level describes how much help preceded the demonstrated response.
        - evidence_type describes the cognitive performance demonstrated.
        - response_mode describes whether the answer is deliberate, uncertain,
          or explicitly presented as a guess.
        - uncertain and guess are response_mode values, never support_level values.
          Without preceding help, use support_level "independent".
        - Never output numeric weights or independence factors.

        Strength:
        - weak: a copied value, isolated observation, unsupported guess, or brief recognition without a relationship.
        - moderate: a correct relationship, prediction, comparison, or explanation in the current case.
        - strong: a general rule stated in the learner's own words, a justified explanation,
          or correct transfer of a relationship to a new case.
        Concise wording, spelling mistakes, and imperfect grammar do not reduce strength
        when the conceptual relationship is clear.

        LEARNING_TARGETS:
        {learning_targets}

        HISTORY:
        {history}

        TUTOR_QUESTION_OR_INSTRUCTION:
        {tutor_question}

        CURRENT_MESSAGE:
        {current_message}

        ACTIVITY_STATE:
        {activity_state}

        ACTIVITY_CHANGE_SINCE_TUTOR_INSTRUCTION:
        {activity_change}

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
        progress = context.state.data["StudentKnowledgeProgress"]
        targets = context.state.data["SessionContext"].domain_targets
        learning_targets = [{"id": target["id"], "text": target["text"]} for target in targets if target["id"] in progress.root]
        return {
            "learning_targets": json.dumps(learning_targets, ensure_ascii=False),
            "history": dialog_history(context),
            "tutor_question": last_tutor_message(context),
            "current_message": current_message,
            "activity_state": activity_state(context),
            "activity_change": activity_change(context),
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
        """Assess with the live model, or emit contract-valid off-air test data."""
        if not context.on_air:
            assessment = {"evidence": [], "review": True}
            result = self.result(
                artifacts=[
                    TextArtifact(
                        producer=self.id,
                        step=context.step,
                        content=json.dumps(assessment),
                    )
                ]
            )
            logger.info("LearningTargetAssessor off-air result: %s", assessment)
            return result, context

        # validate that our target and continuations are present in the provided agents list, if any
        if agents is not None:
            self.validate_target_continuations_against_agents(agents)

        # ask the LLM using standard LLMAgent patterns
        result, context = self.ask(
            Message(role="user", content=artifact.content),
            context,
            ask_params=ask_params,
            ask_config=ask_config,
        )
        logger.info("LearningTargetAssessor result: %s", result.content())
        return result, context


def smoke_test(console):
    console.rule("[bold cyan]LearningTargetAssessor Smoke Test[/bold cyan]")

    env_path = find_up(".env")
    logger.info("Loading environment variables from %s", env_path)
    load_dotenv(env_path)

    api_key = os.getenv("GOOGLE_API_KEY")
    assert api_key, "Missing GOOGLE_API_KEY in .env"

    client = GoogleClient("gemini-3.5-flash-lite", config={}, api_key=api_key)

    LearningTargetAssessor.target = EndAgent

    starting_agent = LearningTargetAssessor(client=client)
    agents = [
        starting_agent,
        EndAgent(),
    ]

    prompt_params = {
        "learning_targets": json.dumps(
            [
                {
                    "id": "compare-values",
                    "text": "Compare two values and explain which is greater.",
                },
            ]
        ),
        "history": " - assistant: Compare 3/4 and 2/3.\n - user: I will use a common denominator.",
        "tutor_question": "Tutor: Which fraction is greater, and why?",
        "current_message": "3/4 is greater because 9/12 is more than 8/12.",
        "activity_state": "{}",
    }

    # The artifact text is only the instruction; evidence is supplied separately.
    user_text = "Assess the current learning evidence."

    context = Context()
    result, context = starting_agent.run(
        artifact=TextArtifact(producer="user", step=0, content=user_text),
        context=context,
        agents=agents,
        ask_params=prompt_params,
        ask_config=AskConfig(
            json_mode=True,
            max_tokens=512,
            vendor_config={"reasoning": {"effort": "minimal"}, "verbosity": "low"},
        ),
    )

    return result, context


if __name__ == "__main__":
    console = Console()
    from rich.logging import RichHandler

    logging.basicConfig(level=logging.INFO, format="%(funcName)s - %(message)s", handlers=[RichHandler(console=console)])
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logging.getLogger("httpcore.connection").setLevel(logging.INFO)
    logging.getLogger("httpcore.http11").setLevel(logging.INFO)

    result, context = smoke_test(console)
    console.print("[bold green]LearningTargetAssessor Smoke Test Result:[/bold green]")
    console.print(pretty_content(result.content()))
    console.print("[bold green]Context after run:[/bold green]")
    console.print(pformat(context.control.model_dump(), indent=2))
