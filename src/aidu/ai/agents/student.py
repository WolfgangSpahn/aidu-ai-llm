# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
"""Subject-neutral archetype-driven student agent."""

import logging
import textwrap

from aidu.ai.archetype.archetype import Archetype
from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.artifacts import TextArtifact
from aidu.ai.core.context import Context, Message
from aidu.ai.llm.agent import EndAgent, WorkflowAgent
from aidu.ai.llm.fc_requester import LLMFcRequester


logger = logging.getLogger(__name__)


class Student(WorkflowAgent, LLMFcRequester):
    """Common student identity, archetype mixing, and request flow."""

    target = EndAgent
    continuations = []
    student_knowledge = "* Knowledge is determined by the subject-specific student."
    student_missing_knowledge = "* Missing knowledge is determined by the subject-specific student."

    student_psych = textwrap.dedent("""\
        [STUDENT PSYCHOLOGICAL STATE]
        - Private inner state, never quote or describe it aloud: {inner_monologue}
        - Observable behavior: {behavioral_quirk}
        - Motivation: {motivation_level}
        """)

    narrative_block_mixin = textwrap.dedent("""\
        [PRIMARY ARCHETYPE]
        {primary_archetype}
        [SECONDARY ARCHETYPE]
        {secondary_archetype}
        Behave approximately {primary_weight:.0%} like the primary archetype and
        {secondary_weight:.0%} like the secondary archetype.
        """)

    prompt_template = textwrap.dedent("""\
        Act as a student in a classroom simulation.

        [KNOWLEDGE]
        {student_knowledge}

        [MISSING KNOWLEDGE]
        {student_missing_knowledge}

        [ARCHETYPE]
        {narrative_block}

        """).strip()

    def __init__(
        self,
        client,
        primary_anchor: Archetype,
        secondary_anchor: Archetype,
        primary_weight: float,
    ):
        if not isinstance(primary_anchor, Archetype):
            raise TypeError("primary_anchor must be an Archetype")
        if not isinstance(secondary_anchor, Archetype):
            raise TypeError("secondary_anchor must be an Archetype")
        if not 0.0 <= primary_weight <= 1.0:
            raise ValueError("primary_weight must be between 0.0 and 1.0")

        self.start_anchor = primary_anchor
        self.target_anchor = secondary_anchor
        self.primary_weight = primary_weight
        super().__init__(client, prompt_args=self.create_dynamic_ask_params(step=0))

    def mix_archetypes(
        self,
        primary_anchor: Archetype,
        secondary_anchor: Archetype,
        primary_weight: float,
    ) -> str:
        if primary_weight == 1.0:
            return self.student_psych.format(**primary_anchor.to_psych_state())
        if primary_weight == 0.0:
            return self.student_psych.format(**secondary_anchor.to_psych_state())
        if primary_weight < 0.5:
            primary_anchor, secondary_anchor = secondary_anchor, primary_anchor
            primary_weight = 1.0 - primary_weight
        secondary_weight = 1.0 - primary_weight
        return self.narrative_block_mixin.format(
            primary_archetype=self.student_psych.format(**primary_anchor.to_psych_state()),
            secondary_archetype=self.student_psych.format(**secondary_anchor.to_psych_state()),
            primary_weight=primary_weight,
            secondary_weight=secondary_weight,
        )

    def create_dynamic_ask_params(
        self,
        step: int,
        speed: float = 0.1,
        context: Context | None = None,
    ) -> dict[str, str]:
        del speed
        del context
        logger.debug("Student step=%s primary_weight=%.2f", step, self.primary_weight)
        return {
            "narrative_block": self.mix_archetypes(
                self.start_anchor,
                self.target_anchor,
                self.primary_weight,
            ),
            "student_knowledge": self.student_knowledge,
            "student_missing_knowledge": self.student_missing_knowledge,
        }

    def run(
        self,
        artifact: TextArtifact,
        context: Context,
        agents=None,
    ) -> tuple[AgentResult, Context]:
        if agents is not None:
            self.validate_target_continuations_against_agents(agents)
        ask_params = self.create_dynamic_ask_params(context.step, context=context)
        return self.ask(Message(role="user", content=artifact.content), context, ask_params=ask_params)
