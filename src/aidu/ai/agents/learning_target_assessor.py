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
from pydantic import BaseModel, ConfigDict, Field

from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.config import AskConfig
from aidu.ai.core.artifacts import TextArtifact
from aidu.support.filesystem.search import find_up
from aidu.ai.core.context import Context, Message
from aidu.ai.llm.clients.openai import OpenAIClient
from aidu.ai.llm.agent import EndAgent, WorkflowAgent
from aidu.ai.llm.fc_requester import LLMFcRequester
from aidu.ai.agents.assessment_context import (
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
        "guess",
        "hinted_response",
    ]
    support_level: Literal[
        "independent",
        "small_prompt",
        "guided",
        "explicit_hint",
        "answer_revealed",
    ]
    quote: str = Field(min_length=1)


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

        Output:
        {{"evidence":[{{"target":"target-id","direction":"positive|negative",
        "strength":"weak|moderate|strong","confidence":0.0,
        "evidence_type":"recall|explanation|application|correction|guess|hinted_response",
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
        - ACTIVITY_STATE may disambiguate what the learner is referring to, but
          it must never add knowledge, reasoning, or particle identification that
          the learner did not express in CURRENT_MESSAGE.
        - Machine-generated status text and structured applet values are never
          learner-authored quotes, even when the UI displays them beside the
          learner message. Use them only to verify the learner's claim or action.
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
          preferred technical vocabulary. Do not reduce such reasoning to a
          copied applet value merely because displayed numbers are mentioned.
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
        - evidence_type describes what the learner actually demonstrated.

        Strength:
        - weak: a copied value, isolated observation, or brief recognition without a relationship.
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

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Missing OPENAI_API_KEY in .env"

    client = OpenAIClient("gpt-5-mini", config={}, api_key=api_key)

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
