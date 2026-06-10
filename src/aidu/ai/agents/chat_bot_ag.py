"""
Chat bot assistant that can have a conversation with the user. It uses a simple prompt template and can be extended with more complex behavior as needed.
"""

import logging
import textwrap

from aidu.ai.llm.assistant import LLMAssistant

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ChatBot(LLMAssistant):
    """A chat bot assistant which talks to the user."""

    # System prompt with flexible placeholders that can be filled via prompt_args
    # Unfilled placeholders will remain as {placeholder} for later customization

    prompt_template = textwrap.dedent("""\
        You are a helpful and patient chat bot.
                  
        """).strip()

    id: str = "chat_bot"
    target: str = "input"
