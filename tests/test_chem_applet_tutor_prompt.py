from aidu.ai.agents.chem_applet_tutor import (
    ChemLlmTutor,
    build_chem_applet_prompt_args,
    build_deterministic_applet_feedback,
)


def test_chem_applet_tutor_prompt_motivates_applet_updates_after_text_predictions():
    prompt = ChemLlmTutor.prompt_template

    assert "make the next turn one simple move" in prompt
    assert "either ask a reasoning question or invite one applet update, not both" in prompt
    assert "do not append a reporting task" in prompt
    assert "Remove the electron in the applet and tell me what net charge it shows afterward." in prompt
    assert "do not replace an applet-action next step" in prompt
    assert "Correct — now add that electron in the atom builder" not in prompt


def test_chem_applet_tutor_prompt_probes_understanding_instead_of_facts():
    prompt = ChemLlmTutor.prompt_template

    assert "focus on probing the student's understanding" in prompt
    assert "avoid recall-style questions" in prompt
    assert "ask the student to explain why, predict what will happen, compare two cases" in prompt
    assert "rather than name or copy a displayed value" in prompt


def test_chem_applet_tutor_prompt_avoids_micro_questioning():
    prompt = ChemLlmTutor.prompt_template

    assert "avoid micro-questioning" in prompt
    assert "do not follow a correct explanation with the same tiny numeric variation" in prompt
    assert "move to the broader pattern or consequence" in prompt
    assert "If you add two electrons, what net charge would it have?" in prompt


def test_chem_applet_tutor_prompt_uses_applet_specific_instruction_placeholder():
    prompt = ChemLlmTutor.prompt_template

    assert "Applet-specific tutoring instructions:" in prompt
    assert "{applet_tutor_instructions}" in prompt
    assert "follow the applet-specific tutoring instructions" in prompt


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
            "infoStore": {
                "protonCount": 1,
                "neutronCount": 0,
                "innerElectronCount": 0,
                "outerElectronCount": 0,
                "charge": 1,
                "mass": 1,
                "elementSymbol": "H",
                "isStable": True,
            },
        }
    )

    assert feedback == (
        "Use the applet to add electrons until the charge becomes zero. "
        "How many electrons do you need for 1 proton?"
    )
