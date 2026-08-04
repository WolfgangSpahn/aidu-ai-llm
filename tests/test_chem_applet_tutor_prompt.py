from aidu.ai.agents.chem_applet_tutor import (
    ChemLlmTutor,
    build_chem_applet_prompt_args,
    build_deterministic_applet_feedback,
)
from aidu.backend.applets.registry import build_applet_info_store


def test_chem_applet_tutor_exposes_typed_atom_commands():
    schemas = {
        tool["function"]["name"]: tool["function"]
        for tool in ChemLlmTutor.schema()
    }

    assert "fc_change_active_applet" not in schemas
    assert schemas["fc_set_atom"]["parameters"]["required"] == [
        "protons",
        "neutrons",
        "electrons",
    ]
    assert schemas["fc_add_particle"]["parameters"]["properties"]["particle"]["enum"] == [
        "proton",
        "neutron",
        "electron",
    ]


def test_chem_applet_tutor_routes_by_current_learner_need_before_progressing():
    prompt = ChemLlmTutor.prompt_template

    assert "latest message has priority over lesson progression" in prompt
    assert "Do not assume that every turn should advance the activity" in prompt
    assert "relational: emotion, motivation, resistance" in prompt
    assert "orienting: confusion about the task" in prompt


def test_chem_applet_tutor_pauses_tasks_for_affective_needs():
    prompt = ChemLlmTutor.prompt_template

    assert "frustration, low motivation, reluctance, overload" in prompt
    assert "pause the scientific task" in prompt
    assert "Help the student regain agency" in prompt
    assert "Do not assign another task in that" in prompt


def test_chem_applet_tutor_preserves_student_control_of_the_applet():
    prompt = ChemLlmTutor.prompt_template

    assert "the student explicitly asks the tutor" in prompt
    assert "Never call an applet command in response to frustration" in prompt
    assert "leave control with the student" in prompt


def test_chem_applet_tutor_responds_to_meaning_before_applet_state():
    prompt = ChemLlmTutor.prompt_template

    assert "Respond to the meaning of the latest student message" in prompt
    assert "Avoid automatic praise such as “Great!”" in prompt
    assert "never let it override the student's immediate relational" in prompt


def test_chem_applet_tutor_troubleshoots_applet_placement_before_progressing():
    prompt = " ".join(ChemLlmTutor.prompt_template.split())

    assert "treat that as an orienting need and stop lesson progression" in prompt
    assert "give one concrete troubleshooting step" in prompt
    assert "Do not continue the pending conceptual question" in prompt
    assert "Never respond only with “you already did it.”" in prompt


def test_chem_applet_tutor_uses_discovery_when_student_is_ready():
    prompt = ChemLlmTutor.prompt_template

    assert "When the student is ready for conceptual or investigative work" in prompt
    assert "Choose a meaningful investigation" in prompt
    assert "Prefer reasoning from visible evidence" in prompt


def test_chem_applet_tutor_prompt_prefers_holistic_investigations():
    prompt = ChemLlmTutor.prompt_template

    assert "Prefer one broad investigation or reflection prompt" in prompt
    assert "compare cases, notice several changes, and explain the pattern" in prompt
    assert "Do not turn each correct observation into another narrow check question" in prompt
    assert "Do not narrate each intermediate applet result for the student" in prompt


def test_chem_applet_tutor_does_not_repeat_already_answered_questions():
    prompt = ChemLlmTutor.prompt_template

    assert "Never ask for a fact or observation that the student already stated" in prompt
    assert "Treat the student's latest statement as their answer" in prompt
    assert "do not ask what the new charge is" in prompt
    assert "do not ask the student to remove electrons" in prompt


def test_chem_applet_tutor_breaks_repetitive_action_cycles():
    prompt = ChemLlmTutor.prompt_template

    assert "Do not repeat the response pattern" in prompt
    assert "After at most two simple one-variable observations" in prompt
    assert "connect, compare, or summarize" in prompt


def test_chem_applet_tutor_prompt_uses_applet_specific_instruction_placeholder():
    prompt = ChemLlmTutor.prompt_template

    assert "Applet-specific guidance:" in prompt
    assert "{applet_tutor_instructions}" in prompt
    assert "Follow the applet-specific guidance" in prompt


def test_chem_applet_tutor_prompt_args_include_applet_specific_instructions():
    args = build_chem_applet_prompt_args(
        applet={
            "id": "applet-build-an-atom",
            "name": "Build an Atom",
            "tutor_instructions": [
                "The visible label 'A:<mass>' is the mass number.",
                "Better response pattern: 'The applet shows A:3; A is the mass number.'",
            ],
        }
    )

    assert "A:<mass>" in args["applet_tutor_instructions"]
    assert "The applet shows A:3" in args["applet_tutor_instructions"]


def test_applet_rule_feedback_derives_build_an_atom_followup_from_infostore():
    feedback = build_deterministic_applet_feedback(
        {
            "applet": "applet-build-an-atom",
            "infoStore": build_applet_info_store(
                "applet-build-an-atom",
                proton_count=1,
                neutron_count=0,
                inner_electron_count=0,
                outer_electron_count=0,
            ),
        }
    )

    assert feedback == (
        "Use the applet to add electrons until the charge becomes zero. "
        "How many electrons do you need for 1 proton?"
    )
