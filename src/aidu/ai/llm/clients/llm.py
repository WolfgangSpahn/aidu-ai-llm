# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
OpenAI LLM client implementation.
"""

import os
import json

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..client import Client, Context, Trace, clean_message


class LLMClient(Client):
    def __init__(self, model, config, api_key, tools=None):
        super().__init__(model=model, config=config)
        self.client = OpenAI(api_key=api_key)
        self.tools = tools

        # Response format is controlled by config only.
        # Priority: explicit config["response_format"], then enforce_json shortcut.
        self.response_format = self.config.get("response_format")
        if self.response_format is None and self.config.get("enforce_json"):
            self.response_format = {"type": "json_object"}

    def chat(self, message, context):
        kwargs = {
            "model": self.model,
            "messages": context.trace.messages + [message],
        }

        if self.tools:
            kwargs["tools"] = self.tools

        if self.response_format:
            kwargs["response_format"] = self.response_format

        response = self.client.chat.completions.create(**kwargs)

        message = response.choices[0].message.model_dump()

        # --- normalize tool call ---
        if "tool_calls" in message and message["tool_calls"]:
            tool = message["tool_calls"][0]
            message["function_call"] = {
                "name": tool["function"]["name"],
                "arguments": tool["function"]["arguments"],
            }

        message = clean_message(message)
        return message


# --------------------------------------------------------------------------------------------------------------
# smoke test - basic chat

def run_smoke_test_chat():
    console = Console()

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Missing OPENAI_API_KEY in .env"

    client = LLMClient("gpt-4o-mini", config={}, api_key=api_key)

    context = Context(trace=Trace(messages=[
        {"role": "system", "content": "You are a helpful assistant and meeting Henry. Address Henry by his name"
        },
    ]))
    message = {"role": "user", "content": "Hi you."}

    system_prompt = context.trace.messages[0]["content"]
    console.rule("System Prompt")
    console.print(Panel(system_prompt, expand=False))

    response = client.chat(
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
    run_smoke_test_chat()

