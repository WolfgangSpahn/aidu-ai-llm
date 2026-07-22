from aidu.ai.agents.chem_student import ChemStudent
from aidu.ai.agents.math_student import MathStudent
from aidu.ai.agents.student import Student
from aidu.ai.archetype.archetype import archetype_dict
from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.artifacts import AppletArtifact, TextArtifact
from aidu.ai.core.context import Context


class NoCallClient:
    pass


def build_student() -> ChemStudent:
    return ChemStudent(
        NoCallClient(),
        archetype_dict["balanced_student"],
        archetype_dict["curious_novice"],
        0.7,
    )


def test_chem_student_uses_archetypes_in_prompt():
    prompt = build_student().build_system_prompt()[0]["content"]

    assert "chemistry student" in prompt
    assert "balanced_student" not in prompt
    assert "Private inner state" in prompt


def test_subject_students_are_siblings_and_chemistry_prompt_is_fully_resolved():
    student = build_student()
    prompt = student.build_system_prompt()[0]["content"]

    assert issubclass(ChemStudent, Student)
    assert issubclass(MathStudent, Student)
    assert not issubclass(ChemStudent, MathStudent)
    assert "matter is made of particles" in prompt
    assert "Operating the applet does not by itself" in prompt
    assert "responses demonstrate" in prompt
    assert "quadratic" not in prompt.casefold()
    assert "factor expressions" not in prompt.casefold()
    assert "x represents an unknown" not in prompt.casefold()
    assert "{" not in prompt
    assert "}" not in prompt


def test_configured_archetype_weight_does_not_drift_with_dialog_steps():
    student = ChemStudent(
        NoCallClient(),
        archetype_dict["disengaged_guesser"],
        archetype_dict["self_improving_learner"],
        0.8,
    )

    narrative = student.create_dynamic_ask_params(25)["narrative_block"]

    assert "approximately 80% like the primary archetype" in narrative
    assert "inconsistent effort" in narrative


def test_disengaged_guesser_behavior_is_defined_by_final_archetype_block():
    student = ChemStudent(
        NoCallClient(),
        archetype_dict["disengaged_guesser"],
        archetype_dict["disengaged_guesser"],
        1.0,
    )

    prompt = student.build_system_prompt()[0]["content"]

    assert prompt.index("[ARCHETYPE]") > prompt.index("[APPLET USE OBSERVATION]")
    assert "Sometimes give a plausible guess without checking it" in prompt
    assert "simple or familiar answers may still be correct" in prompt
    assert "Do not announce disengagement" in prompt
    assert "Private inner state, never quote or describe it aloud" in prompt
    assert "Does not initially understand proton and electron charges" in prompt
    assert "Does not initially know that proton count determines element identity" in prompt
    assert "Hearing the tutor state a fact" in prompt
    assert "Avoid eager greetings" in prompt


def test_chem_student_prompt_starts_naturally_and_hides_interface_details():
    prompt = " ".join(build_student().build_system_prompt()[0]["content"].split())

    assert "On the first turn, briefly greet the tutor in character" in prompt
    assert "Otherwise answer conceptual questions in plain text" in prompt
    assert "Never mention functions, events, infoStore, field names" in prompt
    assert "at most two short sentences and 40 words" in prompt
    assert "The archetype is the final authority" in prompt
    assert "at most 20 words" in prompt
    assert "p=6, n=7" not in prompt
    assert "{dialogue_history}" not in prompt
    assert "{teacher_prompt}" not in prompt


def test_chem_student_injects_applet_nudge_after_text_only_streak():
    student = build_student()
    context = Context()
    context.state.data[student.APPLET_USAGE_STATE_KEY] = {
        "response_turns": 3,
        "applet_turns": 0,
        "text_only_streak": 3,
    }

    params = student.create_dynamic_ask_params(3, context=context)

    assert "Applet use is low (0 of 3 student turns)" in params["applet_usage_observation"]
    assert "Use the applet in this turn" in params["applet_usage_observation"]


def test_chem_student_does_not_nudge_when_applet_usage_is_healthy():
    student = build_student()
    context = Context()
    context.state.data[student.APPLET_USAGE_STATE_KEY] = {
        "response_turns": 4,
        "applet_turns": 1,
        "text_only_streak": 1,
    }

    params = student.create_dynamic_ask_params(4, context=context)

    assert "within the expected range (1 of 4 student turns)" in params["applet_usage_observation"]


def test_chem_student_requires_applet_when_usage_is_overdue_and_prompt_is_observable(monkeypatch, caplog):
    student = build_student()
    context = Context()
    context.state.data[student.APPLET_USAGE_STATE_KEY] = {
        "response_turns": 3,
        "applet_turns": 0,
        "text_only_streak": 3,
    }
    captured = {}

    def fake_ask(message, returned_context, ask_params=None, ask_config=None):
        captured["ask_config"] = ask_config
        return AgentResult(artifacts=[TextArtifact(producer="student", step=0, content="I tested it.")]), returned_context

    monkeypatch.setattr(student, "ask", fake_ask)
    with caplog.at_level("INFO", logger="aidu.ai.agents.chem_student"):
        student.run(TextArtifact(producer="tutor", step=0, content="How does electron count affect charge?"), context)

    assert captured["ask_config"].tool_choice == "required"
    assert "Virtual student prompt archetype=balanced_student" in caplog.text
    assert "--- BEGIN STUDENT PROMPT ---" in caplog.text
    assert "[ARCHETYPE]" in caplog.text


def test_chem_student_disables_applet_for_conceptual_prompt(monkeypatch):
    student = build_student()
    context = Context()
    context.state.data[student.APPLET_USAGE_STATE_KEY] = {
        "response_turns": 3,
        "applet_turns": 0,
        "text_only_streak": 3,
    }
    captured = {}

    def fake_ask(message, returned_context, ask_params=None, ask_config=None):
        captured["ask_config"] = ask_config
        return AgentResult(artifacts=[TextArtifact(producer="student", step=0, content="I am unsure.")]), returned_context

    monkeypatch.setattr(student, "ask", fake_ask)
    student.run(TextArtifact(producer="tutor", step=0, content="How confident do you feel?"), context)

    assert captured["ask_config"].tool_choice == "none"


def test_chem_student_requires_applet_for_explicit_action(monkeypatch):
    student = build_student()
    captured = {}

    def fake_ask(message, returned_context, ask_params=None, ask_config=None):
        captured["ask_config"] = ask_config
        return AgentResult(artifacts=[TextArtifact(producer="student", step=0, content="Done.")]), returned_context

    monkeypatch.setattr(student, "ask", fake_ask)
    student.run(TextArtifact(producer="tutor", step=0, content="Please remove one electron in the applet."), Context())

    assert captured["ask_config"].tool_choice == "required"


def test_chem_student_emits_gui_applet_info_store():
    student = build_student()
    context = Context()

    result, returned_context = student.emit_applet_info_state(
        context,
        "applet-build-an-atom",
        {"protons": 6, "neutrons": 6, "electrons": 6},
    )

    assert returned_context is context
    assert len(result.artifacts) == 1
    assert isinstance(result.artifacts[0], AppletArtifact)
    assert result.artifacts[0].content == {
        "applet": "applet-build-an-atom",
        "infoStore": {"protons": 6, "neutrons": 6, "electrons": 6},
    }
    assert context.state.data["LastAppletInfo"] == result.artifacts[0].content


def test_build_an_atom_tool_requires_and_emits_student_utterance():
    student = build_student()

    result, _ = student.fc_emit_build_an_atom_info_state(
        Context(),
        neutronCount=7,
        protonCount=6,
        innerElectronCount=2,
        outerElectronCount=4,
        utterance="I made carbon-13. It still looks neutral.",
    )

    assert isinstance(result.artifacts[0], AppletArtifact)
    assert isinstance(result.artifacts[1], TextArtifact)
    assert result.artifacts[0].content["infoStore"]["shorttext"] == (
        "Placed: 6 protons, 7 neutrons, 2 inner electrons, 4 outer electrons."
    )
    assert set(result.artifacts[0].content["infoStore"]) == {
        "shorttext", "lewisChemfig", "electronShellSchema", "neutronCount",
        "protonCount", "innerElectronCount", "outerElectronCount", "charge",
        "mass", "isotope", "isStable", "elementSymbol", "elementVisibility",
        "legendMode",
    }
    assert result.artifacts[1].content == "I made carbon-13. It still looks neutral."
