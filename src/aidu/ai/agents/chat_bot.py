"""
Chat bot assistant that can have a conversation with the user. It uses a simple prompt template and can be extended with more complex behavior as needed.
"""

import logging
import os
import sys
import textwrap
from dotenv import load_dotenv
from rich.console import Console

from aidu.support.filesystem.search import find_up
from aidu.ai.core.context import Context, Message, Trace
from aidu.ai.llm.agent import UserInput
from aidu.ai.llm.assistant import LLMAssistant
from aidu.ai.llm.clients.openai import OpenAIClient, make_openai_client

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ChatBot(LLMAssistant):
    """A chat bot assistant which talks to the user."""

    result_type = Message

    # System prompt with flexible placeholders that can be filled via prompt_args
    # Unfilled placeholders will remain as {placeholder} for later customization

    prompt_template = textwrap.dedent("""\
        You are a helpful and patient chat bot.
                  
        """).strip()


def smoke_test(console: Console):
    """
    Smoke test for the ChatBot assistant, which uses simple Messages as input and output.
    This is a basic test to ensure the assistant can process a simple conversation without errors.
    """

    console.rule("[bold cyan]ChatBot Smoke Test[/bold cyan]")

    # ---------------------------------------------------------------------------
    # setup a client interface to API provider (e.g. OpenAI)
    # ---------------------------------------------------------------------------

    client = make_openai_client()

    # ----------------------------------------------------------------------------
    # setup chat bot with one turn conversation
    # ----------------------------------------------------------------------------

    chat_bot = ChatBot(client=client, prompt_args={})

    # test chat bot with a sample conversation
    user_input = "Hi, how are you?"

    # we define the system prompt as part of the context trace,
    # which allows us to manage the prompt building logic in the chat bot
    response, context = chat_bot.ask(Message(role="user", content=user_input), Context(trace=Trace(messages=chat_bot.build_system_prompt())))

    console.print(f"[bold green]User:[/bold green] {user_input}")
    console.print(f"[bold blue]ChatBot:[/bold blue] {response['content']}")


if __name__ == "__main__":
    from aidu.ai.core.context import Context

    # rich logging setup
    console = Console()
    from rich.logging import RichHandler

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(console=console)])

    smoke_test(console)
