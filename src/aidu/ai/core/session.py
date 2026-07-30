# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
from __future__ import annotations

import logging

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aidu.ai.core.context import Message, Messages
from aidu.ai.core.knowledge_progress import (
    EvidenceKnowledgeProgress,
    StudentKnowledgeProgress,
)

logger = logging.getLogger(__name__)


class SessionContext(BaseModel):
    """Validated session context shared from backend through assessment agents."""

    model_config = ConfigDict(extra="allow")

    on_air: bool
    username: str = ""
    class_name: str = ""
    class_voucher: str = ""
    lesson_id: str = ""
    activity_id: str = ""
    subject: str = ""
    subject_label: str = ""
    domain: str = ""
    domain_label: str = ""
    domain_description: str = ""
    domain_targets: list[dict[str, Any]] = Field(default_factory=list)
    applet_id: str = ""
    applet_name: str = ""
    applet_description: str = ""
    applet: dict[str, Any] = Field(default_factory=dict)

    def domain_prompt_metadata(self) -> dict[str, Any]:
        """Return the active curriculum domain in the tutor prompt shape."""
        return {
            "subject": self.subject,
            "subject_label": self.subject_label,
            "id": self.domain,
            "label": self.domain_label,
            "description": self.domain_description,
            "targets": self.domain_targets,
        }

    def applet_prompt_metadata(self) -> dict[str, Any]:
        """Return applet identity and capabilities, excluding its live state."""
        if self.applet:
            return self.applet
        return {
            "id": self.applet_id,
            "name": self.applet_name,
            "description": self.applet_description,
        }

    def initial_student_knowledge_progress(self) -> StudentKnowledgeProgress:
        """Initialize evidence state for the teacher-defined domain targets."""
        metadata_keys = {
            "progress_update_count",
            "progress_update_indicator",
        }
        return StudentKnowledgeProgress(
            root={
                target["id"]: EvidenceKnowledgeProgress(
                    mastery=0.0,
                    positive_evidence=0.0,
                    negative_evidence=4.0,
                )
                for target in self.domain_targets
                if target["id"] not in metadata_keys
            }
        )


class SessionInfo(BaseModel):
    """
    Runtime/session envelope around a core Message.

    This contains backend/session data needed to interpret or route the
    message, but not the message itself.

    In particular:

    - session_id identifies the backend session.
    - session_context contains the enriched backend context.
    - applet_input contains the current or paired applet state.
    - messages contains recent chat turns for short-term dialog context.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    session_id: str
    session_context: SessionContext

    applet_input: dict[str, Any] | None = None
    messages: Messages | None = None


class SessionResponse(BaseModel):
    """
    Complete session-level response.

    `message` is the clean core AI message.
    `info` is the surrounding backend/session context.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    message: Message
    info: SessionInfo

    def to_director_payload(self) -> dict[str, Any]:
        """
        Transitional compatibility payload.

        This flattens the clean core message and the session envelope into the
        old dict shape:

            role
            content
            actor
            kind
            session_id
            session_context
            applet_input
            messages

        Keep this only while Director/Actor code still expects session data
        inside the incoming message object.
        """
        return {
            **self.message.model_dump(exclude_none=True),
            **self.info.model_dump(exclude_none=True),
        }


class RoutedMessage(BaseModel):
    """
    Director event envelope.

    This is deliberately separate from the core AI ``Message``. It carries the
    routing fields needed by frontend/backend listeners without making those
    fields valid input-message metadata.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    role: str | None = None
    content: str | dict[str, Any] | list[Any] | None = None
    source_actor: str
    recipient_actor: str
    session_id: str | None = None
    applet: str | None = None
    applet_command: dict[str, Any] | None = None
    activity_event: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
    backend_belief_state: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
    backend_knowledge_progress_state: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
    backend_supervision_state: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
