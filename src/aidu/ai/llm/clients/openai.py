# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
OpenAI LLM client implementation.
"""

import logging
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aidu.support.filesystem.search import find_up
from aidu.ai.core.context import Context, Trace
from aidu.ai.core.config import AskConfig
from ..client import Client, clean_message

logger = logging.getLogger(__name__)

# USD per 1M tokens (input/output)
MODEL_COSTS_USD_PER_1M = {
    # GPT-4o family
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    # o-series reasoning models
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 1.10, "output": 4.40},
    # updated o3 pricing
    "o3": {"input": 2.00, "output": 8.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    # GPT-4.1 family
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    # GPT-5 family
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    # optional newer nano
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    # GPT-4 Turbo (legacy)
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-2024-04-09": {"input": 10.00, "output": 30.00},
}


def _estimate_cost_usd(model: str | None, prompt_tokens: int, completion_tokens: int) -> float:
    rates = MODEL_COSTS_USD_PER_1M.get(model or "")
    if not rates:
        return 0.0
    return ((prompt_tokens * rates["input"]) + (completion_tokens * rates["output"])) / 1_000_000


class OpenAIClient(Client):
    def __init__(self, model=None, config={}, api_key=None):

        if model is None:
            logger.warning("No model specified for OpenAIClient, defaulting to gpt-4o-mini")
            model = "gpt-4o-mini"

        if api_key is None:
            env_path = find_up(".env")
            logger.warning("No API key provided, loading environment variables from %s", env_path)
            load_dotenv(env_path)
            api_key = os.getenv("OPENAI_API_KEY")
            assert api_key, "Missing OPENAI_API_KEY in .env"

        super().__init__(model=model, config=config)
        self.client = OpenAI(api_key=api_key)

    def ask(self, message, context, config: AskConfig | None = None):
        """
        Send a message to the LLM and return the assistant response.

        Parameters
        ----------
        message:
            New user, assistant, tool, or system message to append to the
            conversation for this request.

        context:
            Conversation context containing the trace of previous messages.
            The trace is read but not modified.

        config:
            Optional per-request configuration. May override JSON mode,
            available tools, and tool selection behavior.

        Returns
        -------
        Message
            Normalized assistant message.

            The returned message may contain:

            - ``content``: assistant text response
            - ``function_call``: normalized function call request
            - ``tool_calls``: provider-native tool call data
            - token usage information:
                - ``prompt_tokens``
                - ``completion_tokens``
                - ``total_tokens``
                - ``cost_usd``
            - ``model``: model identifier used for the request

        Behavior
        --------
        - Sends ``context.trace.messages + [message]`` to the model.
        - Optionally enables tool calling.
        - Optionally enables JSON response mode.
        - Normalizes provider-specific responses into a common message format.
        - Converts the first tool call into a legacy ``function_call`` field.
        - Adds token usage and estimated cost information when available.

        Notes
        -----
        This method does not modify ``context``.
        Tool calls are not executed automatically; they are returned to the
        caller for handling.
        """
        json_mode = config.json_mode if config else self.config.get("enforce_json", False)
        response_format = {"type": "json_object"} if json_mode else None


        kwargs = {
            "model": self.model,
            "messages": context.trace.messages + [message],
        }

        tools = config.tools if config else None
        if tools:
            kwargs["tools"] = tools
            logger.debug(f"Tools {tools} enabled for this request.")
            if config.tool_choice:
                kwargs["tool_choice"] = config.tool_choice

        if response_format:
            kwargs["response_format"] = response_format

        # Respect a configured timeout (seconds) if provided in client config
        timeout = self.config.get("timeout", 30)
        logger.debug(f"Sending request to OpenAI with model={self.model}, timeout={timeout}s, json_mode={json_mode}, tools={'enabled' if tools else 'none'}")
        logger.debug(f"Request messages kwargs: {kwargs.keys()}")
        logger.debug(f"Request messages: {kwargs['messages'][1:]}")

        try:
            response = self.client.chat.completions.create(timeout=timeout, **kwargs)
        except Exception as exc:
            logger.exception("OpenAI client request failed")
            # Return a safe assistant message indicating the error so the caller can proceed
            return {"role": "assistant", "content": f"Error calling LLM: {exc}", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "model": self.model}

        message = response.choices[0].message.model_dump()
        usage = response.usage

        if usage is not None:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            if total_tokens <= 0:
                total_tokens = prompt_tokens + completion_tokens
            message["prompt_tokens"] = prompt_tokens
            message["completion_tokens"] = completion_tokens
            message["total_tokens"] = total_tokens
            message["cost_usd"] = _estimate_cost_usd(self.model, prompt_tokens, completion_tokens)
        message["model"] = self.model

        # --- normalize tool call ---
        if "tool_calls" in message and message["tool_calls"]:
            tool = message["tool_calls"][0]
            message["function_call"] = {
                "name": tool["function"]["name"],
                "arguments": tool["function"]["arguments"],
            }

        message = clean_message(message)
        return message


def make_openai_client(model="gpt-4o-mini", config={}):
    """
    Helper function to create an OpenAIClient with environment variable loading.
    """

    env_path = find_up(".env")
    if not env_path:
        logger.error("No .env file, even up to root directory, found. Make sure to create one following the .env_example.")
        sys.exit(1)
    else:
        logger.info("Loading environment variables from %s", env_path)
    load_dotenv(env_path)

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Missing OPENAI_API_KEY in .env"

    return OpenAIClient("gpt-4o-mini", config=config, api_key=api_key)


# --------------------------------------------------------------------------------------------------------------
# smoke test - basic ask


def run_smoke_test_ask(console):

    env_path = find_up(".env")
    logger.info("Loading environment variables from %s", env_path)
    load_dotenv(env_path)

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Missing OPENAI_API_KEY in .env"

    client = OpenAIClient("gpt-4o-mini", config={}, api_key=api_key)

    context = Context(
        trace=Trace(
            messages=[
                {"role": "system", "content": "You are a helpful assistant and meeting Henry. Address Henry by his name"},
            ]
        )
    )
    message = {"role": "user", "content": "Hi you."}

    system_prompt = context.trace.messages[0]["content"]
    console.rule("System Prompt")
    console.print(Panel(system_prompt, expand=False))

    response = client.ask(
        message=message,
        context=context,
    )

    # Build full dialog flow: system -> user -> assistant
    full_dialog = context.trace.messages + [message, response]

    console.rule("Messages exchanged in the conversation")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("role", style="magenta", width=10)
    table.add_column("content", style="white")

    for i, turn in enumerate(full_dialog, start=1):
        if i == len(full_dialog):
            table.add_section()
        table.add_row(str(i), str(turn.get("role", "")), str(turn.get("content", "")))

    console.print(table)

    # --- basic checks ---
    assert "role" in response
    assert response["role"] == "assistant"
    assert "content" in response

    console.print("[bold green]Smoke test passed![/bold green]")


if __name__ == "__main__":
    from rich.logging import RichHandler

    console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console)],
    )

    run_smoke_test_ask(console)
