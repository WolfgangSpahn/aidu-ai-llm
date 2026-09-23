from aidu.ai.agents.assessment_context import activity_change, activity_state
from aidu.ai.agents.learning_target_assessor import (
    LearningTargetAssessor,
    TargetEvidenceAssessment,
)
from aidu.ai.core.context import Context


def test_misplaced_response_mode_is_recovered_from_support_level():
    evidence = TargetEvidenceAssessment.model_validate({
        "target": "proton-identity",
        "direction": "negative",
        "strength": "weak",
        "confidence": 0.9,
        "evidence_type": "recall",
        "support_level": "uncertain",
        "quote": "Nitrogen I guess",
    })

    assert evidence.response_mode == "uncertain"
    assert evidence.support_level == "independent"


def test_assessment_context_exposes_applet_changes_since_tutor_instruction():
    context = Context()
    previous = {
        "applet": "applet-build-an-atom",
        "infoStore": {"protonCount": 1, "neutronCount": 1},
    }
    current = {
        "applet": "applet-build-an-atom",
        "infoStore": {"protonCount": 1, "neutronCount": 2},
    }
    context.trace.messages.append({"role": "assistant", "content": "Welcome."})
    context.trace.messages.append({"role": "user", "content": "Applet state", "applet_input": previous})
    context.trace.messages.append({"role": "assistant", "content": "Add a proton."})
    context.trace.messages.append({"role": "user", "content": "I added a proton.", "applet_input": current})
    context.state.data["AppletState"] = current

    change = activity_change(context)

    assert "neutron count: 1 → 2 (+1)" in change
    assert "proton count unchanged at 1" in change
    state = activity_state(context)
    assert "neutronCount" in state
    assert "protonCount" in state
    assert "neutron count: 1 → 2 (+1)" in state


def test_current_applet_state_is_supplied_as_display_text_and_structured_values():
    context = Context()
    context.state.data["AppletState"] = {
        "applet": "applet-build-an-atom",
        "infoStore": {
            "shorttext": "Placed: 1 protons, 2 neutrons, 0 inner electrons, 0 outer electrons.",
            "protonCount": 1,
            "neutronCount": 2,
            "innerElectronCount": 0,
            "outerElectronCount": 0,
        },
    }

    state = activity_state(context)

    assert "Applet state: 1 protons, 2 neutrons, 0 inner electrons, 0 outer electrons." in state
    assert '"protonCount": 1' in state
    assert '"neutronCount": 2' in state


def test_learning_target_prompt_requires_telemetry_to_verify_applet_actions():
    prompt = LearningTargetAssessor.prompt_template

    assert "SOURCE OF TRUTH — APPLET STATE" in prompt
    assert "are the ground truth for what the learner actually did" in prompt
    assert "ACTIVITY_CHANGE explicitly compares the applet snapshot" in prompt
    assert "Do not credit the claimed action as application evidence" in prompt
    assert "ACTIVITY_CHANGE_SINCE_TUTOR_INSTRUCTION" in prompt
