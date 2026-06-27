"""
Domain- and applet-agnostic chemistry applet tutor agent.

This agent is the generic successor shape for ``chem_tutor.py``. It does not
hard-code Build-an-Atom, atomic structure, or a fixed applet state schema.
Instead, the director should provide the current domain and applet contract via
prompt/state placeholders.
"""

import json
import logging
import textwrap
from typing import Any

from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.artifacts import AppletArtifact, TextArtifact
from aidu.ai.core.context import Context, Message
from aidu.ai.llm.agent import EndAgent, UserInput, WorkflowAgent
from aidu.ai.llm.fc_requester import LLMFcRequester

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _compact_json(value: Any) -> str:
    """Serialize prompt placeholder values in a stable, readable form."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _applet_state_from_content(content: str) -> dict[str, Any]:
    if not content.startswith("Applet input:"):
        return {}

    _, _, payload = content.partition("\n")
    payload = payload.strip()
    if not payload:
        return {}

    parsed = _parse_json_object(payload)
    return parsed or {"raw": payload}


def _selected_object_name(info_store: dict[str, Any]) -> str:
    name = (
        info_store.get("elementName")
        or info_store.get("selectedName")
        or info_store.get("name")
        or info_store.get("moleculeName")
        or info_store.get("elementSymbol")
        or info_store.get("selected")
    )
    symbol = info_store.get("elementSymbol")
    if name and symbol and name != symbol:
        return f"{name} ({symbol})"
    return str(name or "this")


def _periodic_table_visual_feedback(selected: str, info_store: dict[str, Any]) -> str | None:
    valence = info_store.get("valenceElectrons")
    if isinstance(valence, int):
        if valence == 1:
            return f"I see you clicked {selected}; in the atom picture there is one electron on the outer ring. Why might that make this element reactive?"

        needed = max(0, 8 - valence)
        return f"I see you clicked {selected}; the atom picture shows {valence} electrons on the outer ring. How many more would fill that outer ring?"

    atomic_number = info_store.get("atomicNumber")
    if atomic_number is not None:
        return f"I see you clicked {selected}; its tile position and atom picture belong to atomic number {atomic_number}. What does that number count inside the atom?"

    return None


def build_deterministic_applet_feedback(applet_state: Any) -> str | None:
    state = _parse_json_object(applet_state)
    info_store = state.get("infoStore")
    if not isinstance(info_store, dict):
        return None

    selected = _selected_object_name(info_store)
    applet = state.get("applet")
    if applet == "applet-periodic-table":
        feedback = _periodic_table_visual_feedback(selected, info_store)
        if feedback:
            return feedback

    valence = info_store.get("valenceElectrons")
    if isinstance(valence, int):
        return f"I see you clicked {selected}. The picture shows {valence} electrons in the outer shell; what does that suggest about bonding?"

    atomic_number = info_store.get("atomicNumber")
    if atomic_number is not None:
        return f"I see you clicked {selected}. What does atomic number {atomic_number} tell you about this atom?"

    return f"I see you clicked {selected}. What pattern or property do you notice from that selection?"


class RespondToAppletInputAgent(WorkflowAgent):
    """
    Deterministic companion agent for applet-only input turns.

    It acknowledges the visual applet event and asks one focused reasoning
    question. LLM tutoring stays in ``ChemAppletTutor``.
    """

    target = EndAgent
    continuations = []

    def run(
        self,
        artifact: TextArtifact,
        context: Context,
        agents=None,
    ) -> tuple[AgentResult, Context]:
        if agents is not None:
            self.validate_target_continuations_against_agents(agents)

        applet_state = _applet_state_from_content(artifact.content)
        feedback = build_deterministic_applet_feedback(applet_state)
        if not feedback:
            feedback = "I see you changed the applet. What do you notice in the new picture?"

        response = TextArtifact(
            producer=self.id,
            step=context.step,
            content=feedback,
        )
        recommendation = self.register_recommendation(
            "default",
            target=EndAgent,
            continuations=[],
            utility=1.0,
            rationale="Applet input was handled by the deterministic applet-response agent.",
        )
        logger.warning("RespondToAppletInputAgent.response %s", feedback)
        return self.result([response], [recommendation]), context


class ChemAppletTutor(WorkflowAgent, LLMFcRequester):
    """
    A chemistry tutor for the currently selected curriculum domain and applet.

    The director is expected to update the prompt args/state whenever the user
    changes domain. That domain change should also select the corresponding
    applet and applet contract.
    """

    default_state = {
        "domain_id": "TODO_DOMAIN_ID",
        "domain_label": "TODO_DOMAIN_LABEL",
        "domain_description": "TODO_DOMAIN_DESCRIPTION",
        "learning_targets": "TODO_LEARNING_TARGETS",
        "applet_id": "TODO_APPLET_ID_FOR_DOMAIN",
        "applet_name": "TODO_APPLET_NAME",
        "applet_description": "TODO_APPLET_DESCRIPTION",
        "applet_remote_control": "TODO_APPLET_REMOTE_CONTROL_CONTRACT",
        "applet_info_store_schema": "TODO_APPLET_INFO_STORE_SCHEMA",
        "applet_state": "TODO_CURRENT_APPLET_INFO_STORE",
    }

    prompt_template = textwrap.dedent("""\
        You are a helpful and patient chemistry tutor {tutor_name}.

        You are tutoring the currently selected chemistry curriculum domain.

        Current domain:
        - id: {domain_id}
        - title: {domain_label}
        - description: {domain_description}
        - learning targets: {learning_targets}

        Current applet:
        - id: {applet_id}
        - name: {applet_name}
        - description: {applet_description}
        - remote-control contract: {applet_remote_control}
        - info-store schema: {applet_info_store_schema}

        Current applet state:
        {applet_state}

        Here is the summary of the task so far:
        {history}

        The student's progress:
        {student_progress}

        Our current belief of the student's internal state and misconceptions:
        {student_belief}

        Tutoring rules:
        - output at most 2 short sentences
        - ask only one question at a time
        - ask one focused next-step question, not a menu of options
        - you may add one brief remark before the question if it helps learning
        - do not list multiple possible activities unless the student explicitly asks for choices
        - when the latest user turn is applet input, start with "I see you clicked ..." or "I see you selected ..." and name only the selected object
        - after acknowledging the click or selection, do not repeat the full applet state or list properties that were clicked
        - use at most one observed value from the applet state if it is needed for the reasoning question
        - prefer conceptual questions that ask the student to reason from the visible applet state
        - use the applet state as shared evidence; do not ask the student to click merely to confirm a value already shown in the applet state
        - only ask the student to manipulate the applet when a new observation is needed for the next reasoning step
        - do not turn learning targets into a checklist of clicks or confirmations
        - avoid instructions like "click", "switch", "open", "select", or "change view" unless the student's next action must produce new evidence
        - do not offer to operate the applet for the student unless they explicitly ask you to and remote control is enabled
        - if remote control is disabled, never say "would you like me to switch/open/change" anything in the applet
        - if the applet already shows the relevant value, ask what the value means or predicts
        - when an element and valence count are visible, prefer a question about bonding, ion formation, group similarity, or reactivity over a question about changing display modes
        - bad response pattern: "Would you like me to switch the atom view so you can inspect it?"
        - bad response pattern: "I see Oxygen, atomic number 8, mass 16.00, nonmetal, gas, 6 valence electrons..."
        - better response pattern: "I see you clicked Oxygen. With 6 valence electrons, how many more would complete its outer shell?"
        - do not assume a specific applet schema; use the current applet contract
        - treat the current applet state as evidence; do not override it with a student's typed answer
        - compare the student's latest claim with the applet state and recent dialog before responding
        - if the student's answer conflicts with the applet state, name the mismatch explicitly and gently
        - do not praise an incorrect answer as correct; acknowledge the attempt, state the observed value, and ask a reasoning question that helps repair the idea
        - do not switch to a different element, molecule, or applet state just because the student typed a number or name
        - if the applet contract allows remote control, you may call the generic applet command function
        - let the student discover concepts through the simulation before giving explanations
        - use clear educational language appropriate for {level}
        """).strip()

    def run(
        self,
        artifact: TextArtifact,
        context: Context,
        agents=None,
    ) -> tuple[AgentResult, Context]:
        if agents is not None:
            self.validate_target_continuations_against_agents(agents)

        state = context.state.data.get(self.__class__.__name__, {})
        if not state:
            state = context.state.data.get(ChemAppletTutor.__name__, {})
        logger.warning(
            "ChemAppletTutor.run agent_class=%s state_keys=%s domain=%s:%s applet=%s:%s applet_state=%s artifact_prefix=%r",
            self.__class__.__name__,
            sorted(state.keys()),
            state.get("domain_id"),
            state.get("domain_label"),
            state.get("applet_id"),
            state.get("applet_name"),
            str(state.get("applet_state", ""))[:240],
            artifact.content[:240],
        )
        result, context = self.ask(Message(role="user", content=artifact.content), context)
        logger.warning("ChemAppletTutor.response %s", result.content())
        return result, context

    def _active_applet_id(self, context: Context) -> str:
        state = context.state.data.get(self.__class__.__name__, {})
        if not state:
            state = context.state.data.get(ChemAppletTutor.__name__, {})
        applet_id = state.get("applet_id") or self.default_state["applet_id"]
        logger.warning(
            "ChemAppletTutor.active_applet agent_class=%s applet_id=%s",
            self.__class__.__name__,
            applet_id,
        )
        return str(applet_id)

    def fc_change_active_applet(
        self,
        context: Context,
        command_kind: str,
        command_json: str = "{}",
    ) -> tuple[AgentResult, Context]:
        """
        Send a remote-control command to the currently selected applet.

        Args:
            command_kind (str): Generic command kind from the active applet remote-control contract.
            command_json (str): JSON object with command parameters defined by the active applet contract.
        """

        producer = f"{self.id}:fc_change_active_applet"
        try:
            if not command_kind:
                raise ValueError("Missing required parameter: command_kind")

            try:
                command_parameters = json.loads(command_json or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError("command_json must be a valid JSON object string") from exc

            if not isinstance(command_parameters, dict):
                raise ValueError("command_json must decode to a JSON object")

            result_content = {
                "applet": self._active_applet_id(context),
                "command": {
                    "kind": command_kind,
                    **command_parameters,
                },
            }
            logger.warning(
                "ChemAppletTutor.applet_command applet=%s command=%r",
                result_content["applet"],
                result_content["command"],
            )

            artifact = AppletArtifact(
                producer=producer,
                step=context.step,
                content=result_content,
            )
            recommendation = self.register_recommendation(
                "default",
                target=EndAgent,
                continuations=[],
                utility=1.0,
                rationale="Change active applet state requested",
            )
            logger.debug(
                "Routing to Applet with artifact: %s and recommendation: %s",
                artifact,
                recommendation,
            )

            return self.result([artifact], [recommendation]), context

        except Exception as exc:
            logger.exception("fc_change_active_applet failed")

            artifact = TextArtifact(producer=producer, step=context.step, content=str(exc))
            recommendation = self.register_recommendation(
                "error",
                target=EndAgent,
                continuations=[],
                utility=1.0,
                rationale="error in processing applet command",
            )
            logger.debug("Routing to EndAgent with artifact: %s", artifact)

            return self.result([artifact], [recommendation]), context


class ChemAppletUserInput(UserInput):
    target = ChemAppletTutor
    continuations = []
    state_key = ChemAppletTutor.__name__

    data_prompt = (
        "domain:{domain_id} | applet:{applet_id} | "
        "state:{applet_state} | "
    )


def build_chem_applet_prompt_args(
    *,
    tutor_name: str = "Marie",
    level: str = "beginner",
    history: str = " - Student just entered the GUI tutoring session.",
    student_progress: str = " - We have not started yet.",
    student_belief: str = " - No belief update yet.",
    domain: dict[str, Any] | None = None,
    applet: dict[str, Any] | None = None,
    applet_state: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """
    Build prompt args for the generic applet tutor.

    Placeholder contract for future director wiring:
    - ``domain`` should come from the selected curriculum domain.
    - ``applet`` should come from the applet registry entry selected by that domain.
    - ``applet_state`` should be the latest applet infoStore forwarded by the frontend.
    """

    domain = domain or {}
    applet = applet or {}

    args = {
        **ChemAppletTutor.default_state,
        "tutor_name": tutor_name,
        "level": level,
        "history": history,
        "student_progress": student_progress,
        "student_belief": student_belief,
        "domain_id": domain.get("id") or domain.get("value") or ChemAppletTutor.default_state["domain_id"],
        "domain_label": domain.get("label") or domain.get("name") or ChemAppletTutor.default_state["domain_label"],
        "domain_description": domain.get("description") or ChemAppletTutor.default_state["domain_description"],
        "learning_targets": _compact_json(
            domain.get("targets") or domain.get("learning_targets") or ChemAppletTutor.default_state["learning_targets"]
        ),
        "applet_id": applet.get("id") or ChemAppletTutor.default_state["applet_id"],
        "applet_name": applet.get("name") or ChemAppletTutor.default_state["applet_name"],
        "applet_description": applet.get("description") or ChemAppletTutor.default_state["applet_description"],
        "applet_remote_control": _compact_json(
            applet.get("remote_control") or ChemAppletTutor.default_state["applet_remote_control"]
        ),
        "applet_info_store_schema": _compact_json(
            applet.get("info_store_schema") or ChemAppletTutor.default_state["applet_info_store_schema"]
        ),
        "applet_state": _compact_json(
            applet_state if applet_state is not None else ChemAppletTutor.default_state["applet_state"]
        ),
    }
    return args


# late bind self-reference and other classes
ChemAppletTutor.target = ChemAppletUserInput
ChemAppletTutor.continuations = []
