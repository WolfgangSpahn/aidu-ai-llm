# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
"""
Domain- and applet-aware chemistry LLM tutor agent.

T
"""

import json
import logging
import re
import textwrap
from typing import Any

from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.applet_info import AppletInfo
from aidu.ai.core.artifacts import AppletArtifact, JsonArtifact, TextArtifact
from aidu.ai.core.context import Context, Message
from aidu.ai.llm.agent import EndAgent, UserInput, WorkflowAgent
from aidu.ai.llm.fc_requester import LLMFcRequester
from aidu.backend.applets.registry import derive_applet_payload, derive_info_analysis, update_student_progress

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_DEFAULT_CLOSE_DIALOG_FINAL_MESSAGE = (
    "No problem. We will stop here, and you can come back to the activity later."
)


def _compact_json(value: Any) -> str:
    """Serialize prompt placeholder values in a stable, readable form."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


_CLOSE_DIALOG_INTENT_PATTERNS = (
    re.compile(r"\b(i\s+)?(have|need|got)\s+to\s+(go|leave|run)\b"),
    re.compile(r"\bgotta\s+(go|leave|run)\b"),
    re.compile(r"\b(i'?m|i am)\s+(done|finished)\b"),
    re.compile(r"\b(finish|end|close|stop|quit)\s+(this\s+)?(activity|dialog|chat|session)\b"),
    re.compile(r"\b(go|come)\s+back\s+to\s+(welcome|activity|activities|lesson)\b"),
    re.compile(r"\breturn\s+to\s+(welcome|activity|activities|lesson)\b"),
    re.compile(r"\bresume\s+later\b"),
)


def _student_wants_to_close_dialog(text: str) -> bool:
    """
    Detect clear student requests to leave the AI activity.

    This deterministic guard catches high-confidence closing language before
    the LLM can answer conversationally instead of emitting the close event.
    """
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return False
    if re.search(r"\b(do not|don't|dont|not)\s+(close|end|finish|stop|quit)\b", normalized):
        return False
    if re.search(r"\bnot\s+(done|finished)\b", normalized):
        return False
    return any(pattern.search(normalized) for pattern in _CLOSE_DIALOG_INTENT_PATTERNS)


def analyze_applet_content(
    applet_content: dict[str, Any],
    last_applet_content: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return applet content enriched with backend-derived infoStore values."""

    try:
        return derive_applet_payload(applet_content, last_payload=last_applet_content)
    except Exception:
        logger.exception("Applet infoStore analysis failed")
        return applet_content


def _last_applet_payload_from_trace(
    context: Context,
    current_payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the most recent prior applet payload from trace, if available."""
    for message in reversed(context.trace.messages):
        if not isinstance(message, dict):
            continue
        applet_info = AppletInfo.from_message(message)
        if not applet_info:
            continue
        payload = applet_info.to_state()
        if isinstance(payload, dict) and payload != current_payload:
            return payload
    return None


def build_deterministic_applet_feedback(
    applet_content: dict[str, Any],
    analysis: dict[str, Any] | None = None,
    *,
    analyze: bool = True,
) -> str | None:
    """
    Build deterministic feedback for the given applet content.

    Args:
        applet_content: The current content/state of the applet.

    Returns:
        A string containing the feedback, or None if no feedback is applicable.
    """
    applet_content = analyze_applet_content(applet_content) if analyze else applet_content
    info_store = applet_content.get("infoStore")
    if not isinstance(info_store, dict):
        return "You have clicked this. What was your intent"
    if not isinstance(analysis, dict):
        analysis = derive_info_analysis(
            str(applet_content.get("applet") or ""),
            info_store,
        )
    if not isinstance(analysis, dict):
        return "You have clicked this. What was your intent"
    followup = analysis.get("followup")
    if not isinstance(followup, str) or not followup:
        return "You have clicked this. What was your intent"
    return followup


class AppletRuleResponder(WorkflowAgent):
    """
    Deterministic rule responder, when student only interacted with the applet.

    It acknowledges the visual applet event and asks one focused reasoning
    question. LLM tutoring stays in ``ChemLlmTutor``.
    """

    target = EndAgent
    continuations = []

    def run(
        self,
        artifact: AppletArtifact,
        context: Context,
        agents=None,
    ) -> tuple[AgentResult, Context]:
        """
        """
        if agents is not None:
            self.validate_target_continuations_against_agents(agents)
        else:
            logger.warning("AppletRuleResponder.run agents=None")

        last_applet_payload = _last_applet_payload_from_trace(context, artifact.content)
        applet_content = analyze_applet_content(
            applet_content=artifact.content,
            last_applet_content=last_applet_payload,
        )
        artifact.content.update(applet_content)
        applet_id = str(applet_content.get("applet") or "")
        current_info_store = applet_content.get("infoStore") if isinstance(applet_content.get("infoStore"), dict) else {}
        previous_info_store = (
            last_applet_payload.get("infoStore")
            if isinstance(last_applet_payload, dict) and isinstance(last_applet_payload.get("infoStore"), dict)
            else None
        )
        analysis = derive_info_analysis(
            applet_id,
            current_info_store,
            last_info_store=previous_info_store,
        )
        if analysis:
            context.state.data["LastAppletInfo"] = analysis
        student_progress = context.state.data.get("StudentProgress")
        if isinstance(student_progress, dict):
            update_student_progress(
                applet_id,
                student_progress,
                student_goal=context.state.data.get("StudentGoal", {}),
                info_store=current_info_store,
                last_info_store=previous_info_store,
            )
        feedback = build_deterministic_applet_feedback(applet_content, analysis=analysis, analyze=False)
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
            rationale="Applet input was handled by the deterministic rule responder.",
        )
        logger.warning("AppletRuleResponder.response %s", feedback)
        return self.result([response], [recommendation]), context


class ChemLlmTutor(WorkflowAgent, LLMFcRequester):
    """
    A chemistry tutor for the currently selected curriculum domain and applet.

    The director is expected to update the prompt args whenever the user
    changes domain. That domain change should also select the corresponding
    applet and applet contract.
    """

    default_args = {
        "subject_id": "TODO_SUBJECT_ID",
        "subject_label": "TODO_SUBJECT_LABEL",
        "domain_id": "TODO_DOMAIN_ID",
        "domain_label": "TODO_DOMAIN_LABEL",
        "context_summary": "TODO_ACTIVE_TUTORING_CONTEXT",
        "domain_description": "TODO_DOMAIN_DESCRIPTION",
        "learning_targets": "TODO_LEARNING_TARGETS",
        "applet_id": "TODO_APPLET_ID_FOR_DOMAIN",
        "applet_name": "TODO_APPLET_NAME",
        "applet_description": "TODO_APPLET_DESCRIPTION",
        "applet_remote_control": "TODO_APPLET_REMOTE_CONTROL_CONTRACT",
        "applet_info_store_schema": "TODO_APPLET_INFO_STORE_SCHEMA",
        "applet_tutor_instructions": "TODO_APPLET_TUTOR_INSTRUCTIONS",
        "applet_state": "TODO_CURRENT_APPLET_INFO_STORE",
    }

    prompt_template = textwrap.dedent("""\
        You are a helpful and patient chemistry tutor {tutor_name}.

        You are tutoring the currently selected chemistry curriculum domain.
        Active tutoring context: {context_summary}

        Current domain:
        - subject: {subject_label} ({subject_id})
        - id: {domain_id}
        - title: {domain_label}
        - description: {domain_description}
        - learning targets: {learning_targets}

        To allow the student to explore and discover concepts, you are tutoring with a chemistry applet.
                                      
        Current applet:
        - id: {applet_id}
        - name: {applet_name}
        - description: {applet_description}
        - remote-control contract: {applet_remote_control}
        - info-store schema: {applet_info_store_schema}

        Applet-specific tutoring instructions:
        {applet_tutor_instructions}

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
        - stay inside the active tutoring context unless the student explicitly asks to change topic
        - do not list multiple possible activities unless the student explicitly asks for choices
        - focus on probing the student's understanding, not checking isolated facts
        - avoid recall-style questions such as "what is the atomic number?" when the answer is visible or factual
        - prefer questions that ask the student to explain why, predict what will happen, compare two cases, or connect an observation to a concept
        - when a fact is needed, provide it briefly as shared evidence and ask what it implies
        - probe the student's mental model by asking for reasoning behind their prediction or choice
        - avoid micro-questioning: do not follow a correct explanation with the same tiny numeric variation
        - once the student shows the charge-count relationship, move to the broader pattern or consequence instead of asking "what if one more/two more?"
        - bad response pattern after the student explains one electron changes charge by one: "If you add two electrons, what net charge would it have?"
        - when the latest user turn is applet input, start with "You have clicked ..." or "You have selected ..." and name only the selected object
        - after acknowledging the click or selection, do not repeat the full applet state or list properties that were clicked
        - use at most one observed value from the applet state if it is needed for the reasoning question
        - prefer conceptual questions that ask the student to reason from the visible applet state rather than name or copy a displayed value
        - follow the applet-specific tutoring instructions for interpreting labels, schemas, and example response patterns
        - use the applet state as shared evidence; do not ask the student to click merely to confirm a value already shown in the applet state
        - when explaining an applet-specific label, first point to the visible label and connect it to the concept name
        - only ask the student to manipulate the applet when a new observation is needed for the next reasoning step
        - when the student states an intended applet action or prediction that is not yet reflected in the current applet state, make the next turn one simple move: either ask a reasoning question or invite one applet update, not both
        - if the next move is an applet update, give only that single applet action and do not append a reporting task such as "tell me what it shows" or "what do you observe afterward"
        - do not replace an applet-action next step with a purely conceptual follow-up question
        - do not turn learning targets into a checklist of clicks or confirmations
        - avoid instructions like "click", "switch", "open", "select", or "change view" unless the student's next action must produce new evidence
        - do not offer to operate the applet for the student unless they explicitly ask you to and remote control is enabled
        - if remote control is disabled, never say "would you like me to switch/open/change" anything in the applet
        - if the applet already shows the relevant value, ask what the value means or predicts
        - bad response pattern: "Would you like me to switch the atom view so you can inspect it?"
        - bad response pattern: "Remove the electron in the applet and tell me what net charge it shows afterward."
        - do not assume a specific applet schema; use the current applet contract
        - treat the current applet state as evidence; do not override it with a student's typed answer
        - compare the student's latest claim with the applet state and recent dialog before responding
        - if the student's answer conflicts with the applet state, name the mismatch explicitly and gently
        - do not praise an incorrect answer as correct; acknowledge the attempt, state the observed value, and ask a reasoning question that helps repair the idea
        - do not switch to a different element, molecule, or applet state just because the student typed a number or name
        - if the student says they need to leave, have to go, are done, want to stop, want to end/finish the activity, want to go back, or want to resume later, call fc_close_dialog immediately with one short friendly final_message instead of asking whether they want to resume later
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
            state = context.state.data.get(ChemLlmTutor.__name__, {})
        logger.warning(
            "ChemLlmTutor.run agent_class=%s state_keys=%s domain=%s:%s applet=%s:%s applet_state=%s artifact_prefix=%r",
            self.__class__.__name__,
            sorted(state.keys()),
            state.get("domain_id"),
            state.get("domain_label"),
            state.get("applet_id"),
            state.get("applet_name"),
            str(state.get("applet_state", ""))[:240],
            artifact.content[:240],
        )
        if _student_wants_to_close_dialog(artifact.content):
            logger.info(
                "ChemLlmTutor.close_dialog_intent_detected artifact_prefix=%r",
                artifact.content[:240],
            )
            return self.fc_close_dialog(context, final_message=_DEFAULT_CLOSE_DIALOG_FINAL_MESSAGE)

        result, context = self.ask(Message(role="user", content=artifact.content), context, ask_params=state)
        logger.warning("ChemLlmTutor.response %s", result.content())
        return result, context

    def _active_applet_id(self, context: Context) -> str:
        state = context.state.data.get(self.__class__.__name__, {})
        if not state:
            state = context.state.data.get(ChemLlmTutor.__name__, {})
        applet_id = state.get("applet_id") or self.default_args["applet_id"]
        logger.warning(
            "ChemLlmTutor.active_applet agent_class=%s applet_id=%s",
            self.__class__.__name__,
            applet_id,
        )
        return str(applet_id)

    def fc_close_dialog(
        self,
        context: Context,
        final_message: str = _DEFAULT_CLOSE_DIALOG_FINAL_MESSAGE,
    ) -> tuple[AgentResult, Context]:
        """
        Close and finalize the current AI activity.

        Use this when the student says they need to leave, have to go, are
        done, want to stop, want to end or finish the activity, want to go
        back to the activity list, or want to resume later. It emits the
        ai_activity_finalized event that tells the frontend to leave chat and
        return to the student's activity plan.

        Args:
            final_message (str): One short friendly goodbye message to show to
                the student before the frontend returns to the activity list.
        """
        producer = f"{self.id}:fc_close_dialog"
        safe_final_message = final_message.strip() or _DEFAULT_CLOSE_DIALOG_FINAL_MESSAGE
        farewell = TextArtifact(
            producer=producer,
            step=context.step,
            content=safe_final_message,
        )
        event = JsonArtifact(
            producer=producer,
            step=context.step,
            content={
                "type": "ai_activity_finalized",
                "reason": "dialog_closed",
                "message": "Closing dialog and returning to student activity plan.",
            },
        )
        recommendation = self.register_recommendation(
            "default",
            target=EndAgent,
            continuations=[],
            utility=1.0,
            rationale="User requested to close dialog and return to student activity plan.",
        )
        logger.info("ChemLlmTutor.fc_close_dialog event=%s", event.content)
        return self.result([event, farewell], [recommendation]), context

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
                "ChemLlmTutor.applet_command applet=%s command=%r",
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


class ChemLlmUserInput(UserInput):
    """
    ChemLlmUserInput is a user input agent that forwards the user's input from console to the ChemLlmTutor.
    """
    target = ChemLlmTutor
    continuations = []
    state_key = ChemLlmTutor.__name__

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

    ``domain`` should come from the selected curriculum domain, ``applet`` from
    the applet registry entry selected by that domain, and ``applet_state`` from
    the latest applet infoStore forwarded by the frontend.
    """

    domain = domain or {}
    applet = applet or {}
    subject_id = domain.get("subject") or domain.get("subject_id") or ChemLlmTutor.default_args["subject_id"]
    subject_label = domain.get("subject_label") or subject_id or ChemLlmTutor.default_args["subject_label"]
    domain_id = domain.get("id") or domain.get("value") or ChemLlmTutor.default_args["domain_id"]
    domain_label = domain.get("label") or domain.get("name") or ChemLlmTutor.default_args["domain_label"]
    context_parts = [
        str(part)
        for part in (subject_label, domain_label)
        if part and not str(part).startswith("TODO_")
    ]

    return {
        **ChemLlmTutor.default_args,
        "tutor_name": tutor_name,
        "level": level,
        "history": history,
        "student_progress": student_progress,
        "student_belief": student_belief,
        "subject_id": subject_id,
        "subject_label": subject_label,
        "domain_id": domain_id,
        "domain_label": domain_label,
        "context_summary": " / ".join(context_parts) or ChemLlmTutor.default_args["context_summary"],
        "domain_description": domain.get("description") or ChemLlmTutor.default_args["domain_description"],
        "learning_targets": _compact_json(
            domain.get("targets") or domain.get("learning_targets") or ChemLlmTutor.default_args["learning_targets"]
        ),
        "applet_id": applet.get("id") or ChemLlmTutor.default_args["applet_id"],
        "applet_name": applet.get("name") or ChemLlmTutor.default_args["applet_name"],
        "applet_description": applet.get("description") or ChemLlmTutor.default_args["applet_description"],
        "applet_remote_control": _compact_json(
            applet.get("remote_control") or ChemLlmTutor.default_args["applet_remote_control"]
        ),
        "applet_info_store_schema": _compact_json(
            applet.get("info_store_schema") or ChemLlmTutor.default_args["applet_info_store_schema"]
        ),
        "applet_tutor_instructions": _compact_json(
            applet.get("tutor_instructions") or ChemLlmTutor.default_args["applet_tutor_instructions"]
        ),
        "applet_state": _compact_json(
            applet_state if applet_state is not None else ChemLlmTutor.default_args["applet_state"]
        ),
    }


# late bind self-reference and other classes
ChemLlmTutor.target = ChemLlmUserInput
ChemLlmTutor.continuations = []
