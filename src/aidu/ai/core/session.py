from __future__ import annotations

import logging

from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict, Field

from aidu.ai.core.context import Message

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    session_context: dict[str, Any]

    applet_input: dict[str, Any] | None = None
    messages: list[dict[str, Any]] | None = None


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
    backend_belief_state: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
    backend_progress_state: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
