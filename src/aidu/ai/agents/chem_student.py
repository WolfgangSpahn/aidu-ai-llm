# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
"""Archetype-driven chemistry student simulation agent."""

import logging
import re
import textwrap
from typing import Any

from aidu.ai.agents.student import Student
from aidu.ai.archetype.archetype import Archetype, archetype_dict
from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.artifacts import AppletArtifact, TextArtifact
from aidu.ai.core.config import AskConfig
from aidu.ai.core.context import Context, Message
from aidu.ai.core.recommendation import Recommendation
from aidu.ai.llm.agent import EndAgent
from aidu.backend.applets.registry import build_applet_info_store


logger = logging.getLogger(__name__)


class ChemStudent(Student):
    """Simulate a chemistry learner and the GUI applet actions they take."""

    APPLET_USAGE_STATE_KEY = "ChemStudentAppletUsage"
    APPLET_USAGE_TARGET = 0.25
    MAX_TEXT_ONLY_TURNS = 3

    student_knowledge = textwrap.dedent("""\
        * The student can use visible labels and controls in a chemistry applet.
        * Further chemistry knowledge must be supported by the student's own
          demonstrated understanding and the archetype.
        """)

    student_missing_knowledge = textwrap.dedent("""\
        * Charges cancel each other.
        * That matter is made of particles.
        * Do not assume the student understands connections between particle
          models and chemical symbols merely because they were shown.
        * Operating the applet does not by itself create understanding of atoms,
          ions, shells, isotopes, or chemical symbols.
        * Knowledge changes only when the student's responses demonstrate
          learning; otherwise preserve archetype uncertainty and misconceptions.
        """)

    prompt_template = textwrap.dedent("""\
        Act as a chemistry student in a classroom simulation. Respond naturally
        in at most two short sentences and 40 words. On the first turn, briefly
        greet the tutor in character.

        [STUDENT'S CHEMISTRY KNOWLEDGE]
        {student_knowledge}

        [STUDENT'S MISSING CHEMISTRY KNOWLEDGE]
        {student_missing_knowledge}

        [ARCHETYPE-SPECIFIC CHEMISTRY KNOWLEDGE]
        {archetype_chemistry_knowledge}

        [ARCHETYPE-SPECIFIC MISSING CHEMISTRY KNOWLEDGE]
        {archetype_chemistry_missing_knowledge}

        [APPLET USE OBSERVATION]
        {applet_usage_observation}

        Use fc_emit_build_an_atom_info_state when you think an
        applet action, when you say you will perform one, or when the observation
        above says applet use is low. Otherwise answer conceptual questions in
        plain text. With a function call, include one natural utterance of at
        most 20 words. Never mention functions, events, infoStore, field names,
        or other interface details.

        [ARCHETYPE]
        {narrative_block}

        The archetype is the final authority for the student's effort,
        correctness, uncertainty, mistakes, and willingness to explain or ask
        for help. Do not replace it with a uniformly helpful or expert student.

        """).strip()

    def __init__(
        self,
        client,
        primary_anchor: Archetype,
        secondary_anchor: Archetype,
        primary_weight: float,
    ):
        super().__init__(client, primary_anchor, secondary_anchor, primary_weight)

    def create_dynamic_ask_params(
        self,
        step: int,
        speed: float = 0.1,
        context: Context | None = None,
    ) -> dict[str, Any]:
        params = super().create_dynamic_ask_params(step, speed, context=context)
        domain_knowledge = self.start_anchor.domain_knowledge.get("chemistry")
        params["archetype_chemistry_knowledge"] = self._knowledge_lines(
            domain_knowledge.known if domain_knowledge else []
        )
        params["archetype_chemistry_missing_knowledge"] = self._knowledge_lines(
            domain_knowledge.missing if domain_knowledge else []
        )
        params["applet_usage_observation"] = self._applet_usage_observation(context)
        return params

    @staticmethod
    def _knowledge_lines(items: list[str]) -> str:
        return "\n".join(f"* {item}" for item in items) or "* No additional archetype-specific constraints."

    def _applet_usage_observation(self, context: Context | None) -> str:
        if context is None:
            return "Applet use is currently within the expected range."
        usage = context.state.data.get(self.APPLET_USAGE_STATE_KEY, {})
        response_turns = int(usage.get("response_turns", 0) or 0)
        applet_turns = int(usage.get("applet_turns", 0) or 0)
        text_only_streak = int(usage.get("text_only_streak", 0) or 0)
        rate = applet_turns / response_turns if response_turns else 0.0
        below_target = response_turns >= 4 and rate < self.APPLET_USAGE_TARGET
        if text_only_streak >= self.MAX_TEXT_ONLY_TURNS or below_target:
            return (
                f"Applet use is low ({applet_turns} of {response_turns} student turns). "
                "Use the applet in this turn to test or underline your answer when the "
                "prompt concerns particles, shells, charge, mass, or isotopes. Choose a "
                "relevant observable state yourself; do not wait for the tutor to request "
                "a specific manipulation."
            )
        return (
            f"Applet use is currently within the expected range "
            f"({applet_turns} of {response_turns} student turns)."
        )

    def _applet_use_is_overdue(self, context: Context) -> bool:
        usage = context.state.data.get(self.APPLET_USAGE_STATE_KEY, {})
        response_turns = int(usage.get("response_turns", 0) or 0)
        applet_turns = int(usage.get("applet_turns", 0) or 0)
        text_only_streak = int(usage.get("text_only_streak", 0) or 0)
        rate = applet_turns / response_turns if response_turns else 0.0
        return (
            text_only_streak >= self.MAX_TEXT_ONLY_TURNS
            or (response_turns >= 4 and rate < self.APPLET_USAGE_TARGET)
        )

    @staticmethod
    def _prompt_supports_atom_builder(text: str) -> bool:
        normalized = text.casefold()
        return any(term in normalized for term in (
            "atom", "proton", "neutron", "electron", "shell", "charge",
            "neutral", "ion", "mass", "isotope", "element", "octet",
        ))

    @staticmethod
    def _prompt_requests_applet_action(text: str) -> bool:
        normalized = " ".join(text.casefold().split())
        action = r"(?:build|select|add|remove|change|set|place|try|use)"
        return bool(
            re.match(rf"^(?:please\s+)?{action}\b", normalized)
            or re.search(rf"\b(?:can|could|would|will) you\s+{action}\b", normalized)
            or re.search(r"\bwould you like to\b.*\b(?:build|try|use)\b", normalized)
            or re.search(r"\bin (?:the )?(?:applet|atom builder)\b.*\b(?:add|remove|change|build|select)\b", normalized)
        )

    def run(
        self,
        artifact: TextArtifact,
        context: Context,
        agents=None,
    ) -> tuple[AgentResult, Context]:
        if agents is not None:
            self.validate_target_continuations_against_agents(agents)
        ask_params = self.create_dynamic_ask_params(context.step, context=context)
        explicit_applet_action = self._prompt_requests_applet_action(str(artifact.content or ""))
        require_applet = explicit_applet_action or (
            self._applet_use_is_overdue(context)
            and self._prompt_supports_atom_builder(str(artifact.content or ""))
        )
        effective_prompt = self.build_system_prompt(ask_params)[0]["content"]
        logger.info(
            "Virtual student prompt archetype=%s\n--- BEGIN STUDENT PROMPT ---\n%s\n--- END STUDENT PROMPT ---",
            self.start_anchor.id,
            effective_prompt,
        )
        result, context = self.ask(
            Message(role="user", content=artifact.content),
            context,
            ask_params=ask_params,
            ask_config=AskConfig(tool_choice="required" if require_applet else "none"),
        )
        used_applet = any(isinstance(item, AppletArtifact) for item in result.artifacts)
        usage = context.state.data.setdefault(self.APPLET_USAGE_STATE_KEY, {})
        usage["response_turns"] = int(usage.get("response_turns", 0) or 0) + 1
        usage["applet_turns"] = int(usage.get("applet_turns", 0) or 0) + int(used_applet)
        usage["text_only_streak"] = (
            0 if used_applet else int(usage.get("text_only_streak", 0) or 0) + 1
        )
        return result, context

    def emit_applet_info_state(
        self,
        context: Context,
        applet: str,
        info_store: dict[str, Any],
    ) -> tuple[AgentResult, Context]:
        """Emit the same structured infoStore payload produced by the GUI."""
        if not applet.strip():
            raise ValueError("applet must be a non-empty id")
        if not isinstance(info_store, dict):
            raise TypeError("info_store must be a dictionary")

        payload = {
            "applet": applet,
            "infoStore": dict(info_store),
        }
        context.state.data["LastAppletInfo"] = payload
        artifact = AppletArtifact(
            producer=f"{self.id}:fc_emit_applet_info_state",
            step=context.step,
            content=payload,
        )
        recommendation = Recommendation(
            target=EndAgent,
            continuations=[],
            utility=1.0,
            rationale="The simulated student emitted a GUI applet infoStore state.",
        )
        return self.result([artifact], [recommendation]), context

    def fc_emit_build_an_atom_info_state(
        self,
        context: Context,
        neutronCount: float,
        protonCount: float,
        innerElectronCount: float,
        outerElectronCount: float,
        utterance: str,
    ) -> tuple[AgentResult, Context]:
        """Emit the Build an Atom state produced by the simulated GUI.

        neutronCount (float): Neutrons currently placed in the nucleus.
        protonCount (float): Protons currently placed in the nucleus.
        innerElectronCount (float): Electrons currently in the inner shell.
        outerElectronCount (float): Electrons currently in the outer shell.
        utterance (str): A short natural student utterance describing what was changed or noticed; never empty.
        """
        if not utterance.strip():
            raise ValueError("utterance must be non-empty")
        info_store = build_applet_info_store(
            "applet-build-an-atom",
            neutron_count=int(neutronCount),
            proton_count=int(protonCount),
            inner_electron_count=int(innerElectronCount),
            outer_electron_count=int(outerElectronCount),
        )
        result, context = self.emit_applet_info_state(
            context,
            "applet-build-an-atom",
            info_store,
        )
        result.artifacts.append(
            TextArtifact(
                producer=f"{self.id}:fc_emit_build_an_atom_info_state",
                step=context.step,
                content=utterance.strip(),
            )
        )
        return result, context


ChemStudent.target = EndAgent
ChemStudent.continuations = []


def smoke_test() -> None:
    """Exercise the deterministic GUI-state contract without an LLM call."""

    class NoCallClient:
        pass

    student = ChemStudent(
        NoCallClient(),
        archetype_dict["balanced_student"],
        archetype_dict["curious_novice"],
        0.7,
    )
    result, _ = student.emit_applet_info_state(
        Context(),
        "applet-build-an-atom",
        {"protons": 1, "neutrons": 0, "electrons": 1},
    )
    print(result.artifacts[0].model_dump(mode="json"))


if __name__ == "__main__":
    smoke_test()
