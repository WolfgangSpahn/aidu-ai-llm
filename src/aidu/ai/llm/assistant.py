# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.
#

"""
LLMAssistant extends LLMRequester with automatic schema generation for function calls.

Methods prefixed with 'fc_' are automatically discovered and converted to OpenAI function schemas.
Supports Google-style docstrings for parameter descriptions and Pydantic models for complex types.
"""

import logging

from typing import get_origin, get_args

from pydantic import BaseModel


from aidu.ai.core.context import Context, Message
from aidu.ai.core.agent_result import AgentResult
from .fc_requester import LLMFcRequester

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LLMAssistant(LLMFcRequester):
    """
    LLMAssistant extends LLMFcRequester with automatic schema generation for function calls.

    Methods prefixed with 'fc_' are automatically discovered and converted to OpenAI function schemas.
    Supports Google-style docstrings for parameter descriptions and Pydantic models for complex types.

    Prompting pattern (inherited from LLMRequester):
    - Define class-level system_prompt for fixed prompts:
      class MyTutor(LLMAssistant):
          system_prompt = "You are a math tutor."

    - Or define class-level prompt_template for templated prompts with {placeholders}:
      class MyTutor(LLMAssistant):
          prompt_template = "You are a {subject} tutor."

    - Use prompt_args in __init__ to fill placeholders (supports SafeFormat):
      tutor = MyTutor(client, prompt_args={"subject": "math"})
      # Unfilled placeholders remain as {placeholder} for later customization

    - Override template at instantiation:
      tutor = MyTutor(client, prompt_template="Override template", prompt_args={...})

    Example usage:
        class MyTutor(LLMAssistant):
            system_prompt = "You are a {subject} tutor{level}."

            def fc_solve_problem(self, context: Context, problem: str) -> tuple[Message, Context]:
                '''
                Solves a problem.

                Args:
                    problem (str): The problem to solve
                '''
                context.state.data['result'] = f"Solution to {problem}"
                return "Solved!", context

        # Use with class-level prompt
        tutor = MyTutor(client)
        # Prompt: "You are a {subject} tutor{level}." (placeholders remain)

        # Fill some placeholders
        tutor2 = MyTutor(client, prompt_args={"subject": "math"})
        # Prompt: "You are a math tutor{level}." (subject filled, level unfilled)

        # Use function calls and schema
        tools = tutor.schema()  # Auto-generated from fc_* methods
        fnames = tutor.fnames()  # ["solve_problem"]
    """

    result_type = Message

    @property
    def id(self) -> str:
        return self.__class__.__name__

    capability_specs: dict[str, type | object | None] = {}

    def chat_turn(self, user_message: dict, context: Context) -> tuple[str, Context]:
        """
        Execute a complete conversational turn.

        This method:

        1. Sends the user message to the model.
        2. Extracts a user-visible reply.
        3. Stores both user message and model response in the conversation trace.
        4. Returns the reply text and updated context.

        Parameters
        ----------
        user_message:
            User message for the current turn.

        context:
            Conversation context.

        Returns
        -------
        tuple[str, Context]
            The reply text and the updated context.

        Notes
        -----
        If the model requests a function call, a placeholder reply such as
        ``Executing <function>...`` may be returned instead of assistant text.

        Unlike ``chat()``, this method always persists the turn into the
        conversation history.
        """
        message, context = self.ask(message=user_message, context=context)

        reply = message.get("content", "")
        if not reply and message.get("_fc_message"):
            fc_msg = message.get("_fc_message")
            reply = fc_msg.get("content", "") if isinstance(fc_msg, dict) else fc_msg
        if not reply and message.get("function_call"):
            fc = message.get("function_call")
            reply = f"Executing {fc['name']}..."

        context = self.store_turn(
            context=context,
            user_message=user_message,
            response=message,
        )

        return reply, context

    def interactive_chat(self, context, on_display_header, on_get_user_input, on_session_end, on_display_response, console=None):
        """
        Run an interactive chat session with I/O handlers.

        Args:
            context (Context): Context object with initial trace (system prompt, etc.)
            on_display_header (callable): Handler() to display header
            on_get_user_input (callable): Handler() -> str | None for user input (None exits)
            on_session_end (callable): Handler() when session ends
            on_display_response (callable): Handler(response_text) to display assistant response

        Returns:
            context: Final context after session
        """
        # Display header
        on_display_header()

        # Chat loop
        while True:
            # Get user input
            user_text = on_get_user_input()
            if user_text is None:
                break
            if not user_text:
                continue

            user_message = {"role": "user", "content": user_text}
            reply, context = self.chat_turn(user_message=user_message, context=context)
            # logger.debug(f"LLM reply: {reply}, updated context: {context.pretty(console)}")
            on_display_response(reply)

        # Session end
        on_session_end()

        return context
