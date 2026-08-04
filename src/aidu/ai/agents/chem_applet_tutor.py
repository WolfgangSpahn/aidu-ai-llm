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
from typing import Any, Literal

from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.applet_info import AppletInfo
from aidu.ai.core.artifacts import ActivityEventArtifact, AppletArtifact, Artifact, TextArtifact
from aidu.ai.core.config import AskConfig
from aidu.ai.core.context import Context, Message
from aidu.ai.core.knowledge_progress import StudentKnowledgeProgress
from aidu.ai.llm.agent import EndAgent, UserInput, WorkflowAgent
from aidu.ai.llm.fc_requester import LLMFcRequester
from aidu.backend.applets.registry import (
    derive_applet_payload,
    derive_info_analysis,
    update_student_knowledge_progress,
)

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

_DEFAULT_CLOSE_DIALOG_FINAL_MESSAGE = (
    "No problem. We will stop here, and you can come back to the activity later."
)
_DEFAULT_FINALIZE_DIALOG_MESSAGE = (
    "Well done. I will finish this activity now."
)
CloseDisposition = Literal["pause", "finalize"]


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
    re.compile(r"\bpause(\s+(this\s+)?(activity|dialog|chat|session))?\b"),
)


def _student_close_disposition(text: str) -> CloseDisposition | None:
    """
    Detect clear student requests to leave the AI activity.

    This deterministic guard catches high-confidence closing language before
    the LLM can answer conversationally instead of emitting the close event.
    """
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return None
    if re.search(r"\b(do not|don't|dont|not)\s+(close|end|finish|stop|quit)\b", normalized):
        return None
    if re.search(r"\bnot\s+(done|finished)\b", normalized):
        return None
    if not any(pattern.search(normalized) for pattern in _CLOSE_DIALOG_INTENT_PATTERNS):
        return None
    if re.search(r"\b(done|finished|finish|complete|completed|end)\b", normalized):
        return "finalize"
    return "pause"


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
        student_knowledge_progress: StudentKnowledgeProgress | None = (
            context.state.data.get("StudentKnowledgeProgress")
        )
        if student_knowledge_progress is not None:
            update_student_knowledge_progress(
                applet_id,
                student_knowledge_progress,
                student_goal=context.state.data.get("StudentGoal"),
                info_store=current_info_store,
                last_info_store=previous_info_store,
                turn_index=context.state.data["TurnIndex"],
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

    terminal_target = EndAgent

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
        You are {tutor_name}, a patient chemistry tutor working with the student through an interactive chemistry applet.

        ## Tutoring context

        Active tutoring context: {context_summary}

        Domain:

        - subject: {subject_label} ({subject_id})
        - title: {domain_label}
        - domain id: {domain_id}
        - description: {domain_description}
        - learning targets: {learning_targets}

        Applet:

        - id: {applet_id}
        - name: {applet_name}
        - description: {applet_description}
        - remote-control contract: {applet_remote_control}
        - info-store schema: {applet_info_store_schema}

        Applet-specific guidance:
        {applet_tutor_instructions}

        Current applet state:
        {applet_state}

        Dialog summary:
        {history}

        Student progress:
        {student_knowledge_progress}

        Use student progress as a conservative planning prior:

        * Start with a concrete, low-barrier investigation when relevant target
          evidence is absent, neutral, negative, or entry-test-only.
        * A high entry-test-only estimate is a hypothesis to verify, not
          permission to skip foundations or begin with an abstract question.
        * Increase complexity only after the learner demonstrates the relevant
          relationship in the dialog or applet activity.
        * Do not tell the learner their hidden estimate or describe them as
          weak, low-performing, or unprepared.

        Current learner model:
        {student_belief}

        ## Your role

        Help the student build a coherent mental model through guided discovery.

        The student's latest message has priority over lesson progression and
        applet-specific guidance. First identify what kind of response the
        student needs now:

        * relational: emotion, motivation, resistance, uncertainty, fatigue, or
          a request to pause;
        * orienting: confusion about the task, interface, or what is expected;
        * conceptual: an explanation, prediction, observation, or question;
        * investigative: readiness to continue exploring with the applet.

        Do not assume that every turn should advance the activity. When the
        student expresses frustration, low motivation, reluctance, overload, or
        disengagement, pause the scientific task. Briefly acknowledge the
        experience without generic praise, pressure, correction, or diagnosis.
        Help the student regain agency by asking what feels difficult or by
        offering one low-pressure choice such as pausing, changing approach, or
        hearing why the activity may matter. Do not assign another task in that
        response.

        When the student says they cannot perform, place, move, find, or see
        something in the applet, treat that as an orienting need and stop lesson
        progression for that turn. Explain the one relevant interaction or
        placement rule using the applet-specific guidance and current state,
        then give one concrete troubleshooting step. Do not continue the pending
        conceptual question until the learner can perform or locate the action.
        If the current state suggests the action actually occurred, acknowledge
        the discrepancy without contradicting the learner: say where the object
        appears or what state value changed, explain the placement rule, and ask
        them to check that location. Never respond only with “you already did it.”

        When the student is ready for conceptual or investigative work:

        1. Infer what the student currently understands from their latest response, the dialog, and the applet state.
        2. Identify the most important remaining conceptual connection.
        3. Choose a meaningful investigation that lets the student explore that connection.
        4. Prefer reasoning from visible evidence over recalling isolated facts.
        5. Move to a broader relationship or consequence once the student has demonstrated the current idea.

        Treat teacher targets as both the intended learning outcome and the
        priority for selecting available applet evidence. If a target names a
        visible representation or notation and the current applet state
        provides it, refer to that representation explicitly, help the learner
        interpret one visible feature, and connect changes in it to the target.
        Never invent a representation or infer unavailable visual details.

        Use only interface control names and locations declared by the applet
        guidance or current state. If the learner cannot find a value or
        control and its location is not declared, ask what labels or controls
        they can see instead of inventing a card, panel, button, or position.

        Before praising an answer, separate its correct and incorrect or
        ambiguous parts. Preserve the valid reasoning, but briefly clarify
        notation or terminology that conflicts with the applet state. Do not
        say “exactly right” to a partially correct statement.

        Check state fields that indicate completeness, stability, remaining
        capacity, or an unfinished fragment before calling a result complete or
        stable. Describe the applet's classification as evidence from its model;
        avoid presenting simplified thresholds or categories as universal laws.
        Prefer qualified language over absolute claims such as “the only way,”
        “always,” or “completely” unless the target and current evidence justify it.

        Use a progression such as:

        * notice a relevant observation;
        * predict or explain it;
        * compare it with another case;
        * formulate the underlying relationship;
        * apply that relationship in a new context.

        Do not mechanically use every stage. Select the stage that best matches the student's current understanding.

        ## Response style

        * Write at most three short sentences.
        * Respond to the meaning of the latest student message before referring
          to applet changes or curriculum goals.
        * Avoid automatic praise such as “Great!” when the student's message
          expresses a problem, negative feeling, reluctance, or disagreement.
        * Prefer one broad investigation or reflection prompt over a sequence of small questions.
        * Give the student room to manipulate the applet, compare cases, notice several changes, and explain the pattern in one response.
        * Do not turn each correct observation into another narrow check question.
        * Do not repeat the response pattern “Add one … and tell me what changed” across turns.
        * After at most two simple one-variable observations, pause particle-by-particle instructions and ask the student to connect, compare, or summarize the observed relationships.
        * Ask a narrow question only when the learner model shows that the student needs a specific scaffold.
        * Never ask for a fact or observation that the student already stated in their latest response.
        * Treat the student's latest statement as their answer even when its spelling or grammar is imperfect.
        * Before asking a question, check that its answer is not already present in the latest response, recent dialog, or current applet state.
        * Do not give a menu of possible next steps.
        * A brief observation or correction may precede the question.
        * Use language appropriate for {level}.
        * Stay within the active tutoring context unless the student asks to change it.

        ## Using the applet

        Treat the current applet state as shared evidence.

        When the latest turn is an applet interaction, briefly acknowledge only the selected object, for example: “You have selected oxygen.”

        Never request an applet action that the current state shows is already complete. For example, when the applet has zero electrons, do not ask the student to remove electrons.

        Ask the student to manipulate the applet when an investigation can reveal a useful relationship. A good investigation may invite several purposeful changes to one variable, comparison of the resulting cases, and a single explanation of the overall pattern.

        Do not narrate each intermediate applet result for the student. Let the student observe and describe the changes before you interpret them.

        Do not ask the student to retrieve information already visible in the applet. Instead, ask what the visible evidence means, predicts, or explains.

        Follow the applet-specific guidance when interpreting labels and state
        fields, but never let it override the student's immediate relational or
        orienting need. Never assume an applet schema that is not provided.

        If remote control is unavailable, do not offer to operate the applet.
        Even when remote control is available, call an applet command only when
        the student explicitly asks the tutor to perform that concrete change.
        Never call an applet command in response to frustration, low motivation,
        reluctance, uncertainty, confusion, or a social comment. Otherwise,
        leave control with the student and describe an investigation only after
        the student is ready to continue.

        ## Responding to student reasoning

        Compare the student's claim with the applet state and recent dialog.

        * If the reasoning is correct, build on it rather than testing the same relationship with a trivial variation.
        * If it is partly correct, preserve the useful part and focus on the missing connection.
        * If it conflicts with the applet state, gently state the mismatch and direct attention to one relevant observation.
        * Do not treat a typed number or element name as a request to change the applet state.

        Prefer questions such as:

        * “Try several cases. What stays the same, what changes, and what rule do you think connects them?”
        * “Can you use the applet to investigate this relationship and explain what evidence you found?”
        * “What does this suggest about …?”
        * “Why do you think that happened?”
        * “What would you predict if …?”
        * “How is this case different from …?”
        * “Can you express the relationship in your own words?”

        Avoid questions whose answer is merely copied from a visible label.

        Keep every response complete and self-contained. End every explanatory
        sentence before asking at most one focused next question.

        If the student says “With one electron I see He+,” do not ask what the new charge is. Accept that observation and move to a broader comparison, prediction, or explanation.

        ## Ending the dialog

        If the student wants to stop the dialog, call `fc_close_dialog`
        immediately. Use `disposition="pause"` when they want to leave and
        resume this same activity later. Use `disposition="finalize"` only when
        they clearly say the activity is complete or finished. Never treat
        leaving, going back, or stopping for now as finalization. Supply one
        short, friendly `final_message` appropriate to the chosen disposition.
""").strip()

    def run(
        self,
        artifact: Artifact,
        context: Context,
        agents=None,
    ) -> tuple[AgentResult, Context]:
        if agents is not None:
            self.validate_target_continuations_against_agents(agents)

        state = context.state.data.get(self.__class__.__name__, {})
        if not state:
            state = context.state.data.get(ChemLlmTutor.__name__, {})
        student_message = _compact_json(artifact.content)
        logger.debug(
            "ChemLlmTutor.run agent_class=%s state_keys=%s domain=%s:%s applet=%s:%s applet_state=%s artifact_prefix=%r",
            self.__class__.__name__,
            sorted(state.keys()),
            state.get("domain_id"),
            state.get("domain_label"),
            state.get("applet_id"),
            state.get("applet_name"),
            str(state.get("applet_state", ""))[:240],
            student_message[:240],
        )
        close_disposition = _student_close_disposition(student_message)
        if close_disposition is not None:
            logger.info(
                "ChemLlmTutor.close_dialog_intent_detected artifact_prefix=%r",
                student_message[:240],
            )
            default_message = (
                _DEFAULT_FINALIZE_DIALOG_MESSAGE
                if close_disposition == "finalize"
                else _DEFAULT_CLOSE_DIALOG_FINAL_MESSAGE
            )
            return self.fc_close_dialog(
                context,
                disposition=close_disposition,
                final_message=default_message,
            )

        result, context = self.ask(
            Message(role="user", content=student_message),
            context,
            ask_params=state,
            ask_config=AskConfig(
                max_tokens=512,
                vendor_config={
                    "reasoning": {"effort": "low"},
                    "verbosity": "low",
                },
            ),
        )
        logger.debug("ChemLlmTutor.response %s", result.content())
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
        disposition: CloseDisposition,
        final_message: str = _DEFAULT_CLOSE_DIALOG_FINAL_MESSAGE,
    ) -> tuple[AgentResult, Context]:
        """
        Pause or finalize the current AI activity, as explicitly requested.

        Pausing keeps this dialog available for resumption. Finalizing closes
        this attempt permanently and lets a later entry start a fresh dialog.

        Args:
            disposition: ``pause`` to retain the activity for later, or
                ``finalize`` to complete the current activity attempt.
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
        event = ActivityEventArtifact(
            producer=producer,
            step=context.step,
            content={
                "type": "ai_activity_closed",
                "disposition": disposition,
                "reason": "tutor_requested",
                "final_message": safe_final_message,
            },
        )
        recommendation = self.register_recommendation(
            "default",
            target=self.terminal_target,
            continuations=[],
            utility=1.0,
            rationale=f"User requested to {disposition} the activity.",
        )
        logger.info("ChemLlmTutor.fc_close_dialog event=%s", event.content)
        return self.result([event, farewell], [recommendation]), context

    def _applet_command_result(
        self,
        context: Context,
        command: dict[str, Any],
    ) -> tuple[AgentResult, Context]:
        """Return a structured command for the active applet."""
        producer = f"{self.id}:applet_command"
        result_content = {
            "applet": self._active_applet_id(context),
            "command": command,
        }
        logger.warning(
            "ChemLlmTutor.applet_command applet=%s command=%r",
            result_content["applet"],
            command,
        )
        artifact = AppletArtifact(
            producer=producer,
            step=context.step,
            content=result_content,
        )
        recommendation = self.register_recommendation(
            "default",
            target=self.terminal_target,
            continuations=[],
            utility=1.0,
            rationale="Change active applet state requested",
        )
        return self.result([artifact], [recommendation]), context

    def fc_set_atom(
        self,
        context: Context,
        protons: int,
        neutrons: int,
        electrons: int,
    ) -> tuple[AgentResult, Context]:
        """Set all particle counts in the Build an Atom applet.

        Args:
            protons: Total protons to place in the nucleus.
            neutrons: Total neutrons to place in the nucleus.
            electrons: Total electrons to place across the shells.
        """
        return self._applet_command_result(
            context,
            {
                "kind": "set_atom",
                "protons": protons,
                "neutrons": neutrons,
                "electrons": electrons,
            },
        )

    def fc_add_particle(
        self,
        context: Context,
        particle: Literal["proton", "neutron", "electron"],
        count: int = 1,
    ) -> tuple[AgentResult, Context]:
        """Add particles to the atom already shown in Build an Atom.

        Args:
            particle: Particle type to add to the current atom.
            count: Number of particles to add.
        """
        return self._applet_command_result(
            context,
            {
                "kind": "add_particle",
                "particle": particle,
                "count": count,
            },
        )


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
    student_knowledge_progress: str = " - We have not started yet.",
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
        "student_knowledge_progress": student_knowledge_progress,
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
