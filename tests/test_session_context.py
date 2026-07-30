from aidu.ai.core.session import SessionContext


def test_session_context_exposes_domain_prompt_metadata():
    context = SessionContext(
        on_air=False,
        subject="chemistry",
        subject_label="Chemistry",
        domain="atomic-structure",
        domain_label="Atomic Structure",
        domain_description="How atoms are built.",
        domain_targets=[{"id": "protons", "text": "Identify protons."}],
    )

    assert context.domain_prompt_metadata() == {
        "subject": "chemistry",
        "subject_label": "Chemistry",
        "id": "atomic-structure",
        "label": "Atomic Structure",
        "description": "How atoms are built.",
        "targets": [{"id": "protons", "text": "Identify protons."}],
    }


def test_session_context_prefers_complete_applet_metadata():
    applet = {"id": "build-an-atom", "remote_control": {"enabled": True}}
    context = SessionContext(
        on_air=False,
        applet=applet,
        applet_id="fallback",
    )

    assert context.applet_prompt_metadata() == applet


def test_session_context_initializes_teacher_target_progress():
    context = SessionContext(
        on_air=False,
        domain_targets=[
            {"id": "target-1", "text": "A learning target."},
            {"id": "progress_update_count", "text": "Internal metadata."},
        ],
    )

    progress = context.initial_student_knowledge_progress()

    assert set(progress.root) == {"target-1"}
    assert progress.root["target-1"].mastery == 0.0
    assert progress.root["target-1"].negative_evidence == 4.0
