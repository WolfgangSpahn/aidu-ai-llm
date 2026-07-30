# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
"""Generic assessor for teacher-defined learning targets."""

import os
import logging
import json
import textwrap
from typing import Any
from pprint import pformat
from dotenv import load_dotenv

from rich.console import Console

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
        LAST_MESSAGE, HISTORY, and ACTIVITY_STATE provide context for interpreting
        short or referential answers, but they are not learner evidence themselves.

        q rule:
        q must be an exact substring of CURRENT_MESSAGE.
        Never quote the tutor, history, activity state, target text, or your own inference.
        Use q:null if no exact quote exists.

        Output:
        {{"e":[{{"i":"indicator","p":"+|-|?","s":"w|m|s","q":"quote-or-null"}}],"review":false}}

        Rules:
        - Max 2 evidence items.
        - Use only IDs from LEARNING_TARGETS.
        - Interpret each ID only through its teacher-defined text.
        - Prefer the most specific matching target.
        - Do not duplicate broad+narrow evidence.
        - Omit targets with no direct learner evidence.
        - Do not reward information supplied only by the tutor, history, or activity.
        - A short answer may be evidence when LAST_MESSAGE makes its meaning clear.
        - Use p:? when context is insufficient to decide whether the answer supports
          or contradicts the target.

        p: +=understands, -=wrong, ?=unclear.
        Strength:
        - w: a copied value, isolated observation, or brief recognition without a relationship.
        - m: a correct relationship, prediction, comparison, or explanation in the current case.
        - s: a general rule stated in the learner's own words, a justified explanation,
          or correct transfer of a relationship to a new case.
        Concise wording, spelling mistakes, and imperfect grammar do not reduce strength
        when the conceptual relationship is clear.

        LEARNING_TARGETS:
        {learning_targets}

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
        progress = context.state.data["StudentKnowledgeProgress"]
        targets = context.state.data["SessionContext"].domain_targets
        learning_targets = [{"id": target["id"], "text": target["text"]} for target in targets if target["id"] in progress.root]
        return {
            "learning_targets": json.dumps(learning_targets, ensure_ascii=False),
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
        """Assess with the live model, or emit contract-valid off-air test data."""
        if not context.on_air:
            assessment = {"e": [], "review": True}
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
        "last_message": "Tutor: Which fraction is greater, and why?",
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
