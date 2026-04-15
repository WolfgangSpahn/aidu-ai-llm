# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.
#

"""
    LLMRequester manages prompts, tools, and interactions with the LLM client.
    Provides methods to build system prompts, update them dynamically, and run agents
    with messages and state, including support for function calls from LLM responses.
"""

import os
import json
import logging

from rich.logging import RichHandler
from rich.console import Console

from dotenv import load_dotenv
from rich.rule import Rule

from .builder import PromptBuilder
from .client import LLMClient, clean_message
from .safeformat import SafeFormat

logger = logging.getLogger(__name__)

class LLMRequester:
    """
      LLMRequester is responsible for managing prompts, tools, and interactions with the LLM client.
      It provides methods to build system prompts, update them dynamically, and run the agent with given messages and state.
      It also supports function calls from the LLM response to modify the state.

      Example usage:
            client = LLMClient(api_key)
            prompt = "You are a {subject} tutor. Problem: {problem}"
            tools = [...]  # define tools if needed
            agent = LLMRequester(client, prompt_template=prompt, tools=tools)

            agent.register("add_numbers", add_numbers_fn)
            system_messages = agent.build_system_prompt({"subject": "math", "problem": "2 + 3"})
            user_messages = [{"role": "user", "content": "What is 2 + 3? Use the tool."}]
            state = {}
            msg, state = agent.run(system_messages + user_messages, model="gpt-4o-mini", state=state)
    """ 
    def __init__(self, client, prompt_template=None, tools=None):
        self.client = client
        self.tools = tools or []
        self.function_lookup = {}
        # load prompt from yaml if prompt_template is a path, else use it as a string
        if prompt_template and os.path.isfile(prompt_template):
            with open(prompt_template, "r") as f:
                prompt_template = f.read()
        self.prompt_builder = (
            PromptBuilder(prompt_template) if prompt_template else None
        )

    def register(self, name, fn):
        self.function_lookup[name] = fn

    def build_system_prompt(self, prompt_params=None):
        """
        Build system message once (compile step).
        Returns a prepared message list.
        """
        if not self.prompt_builder:
            return []

        system_prompt = self.prompt_builder.build(prompt_params=prompt_params)
        return [{"role": "system", "content": system_prompt}]

    def update_system_prompt(self, messages, prompt_params=None):
        """
        Update system message with new params (e.g. subject).
        Returns updated message list.
        """
        if not self.prompt_builder or not messages or messages[0]["role"] != "system":
            return messages

        system_prompt = self.prompt_builder.build(prompt_params=prompt_params)
        system_message = {"role": "system", "content": system_prompt}
        return [system_message] + messages[1:]

    def run(self, messages, model, state, run_params=None):
        """
        Run the agent with given messages, model, and state.
        - messages: list of message dicts (role/content)
        - model: LLM model name (e.g. "gpt-4o-mini")
        - state: dict representing current state (can be modified by tools)
        - run_params: optional dict for dynamic prompt updates (e.g. {"subject": "math", "problem": "2 + 3"})
        """
        if run_params:
            messages = self.update_system_prompt(messages, prompt_params=run_params)
        msg = self.client.chat(model, messages, tools=self.tools)

        fc = msg.get("function_call")
        if fc:
            fn = self.function_lookup.get(fc["name"])
            args = json.loads(fc["arguments"])

            if fn:
                state = fn(state=state, **args)

        return msg, state
    
    def talk(self, messages, model, state, run_params=None):
        """
        Convenience method to run and append response to messages.
        """
        effective_messages = (
            self.update_system_prompt(messages, prompt_params=run_params)
            if run_params
            else messages
        )
        msg, state = self.run(
            messages=effective_messages,
            model=model,
            state=state,
            run_params=None,
        )

        return effective_messages + clean_message(msg), state
    

# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# smoke test - function call
#

def add_numbers(state, a: int, b: int):
    result = a + b
    state["result"] = result
    return state

def run_smoke_test_fn_call():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Missing OPENAI_API_KEY in .env"

    # --- setup ---
    client = LLMClient(api_key)

    prompt = """
    You are a {subject} tutor.

    Problem:
    {problem}

    If needed, call the function.
    """

    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["a", "b"]
                }
            }
        }
    ]

    # setup the agent
    agent = LLMRequester(client, prompt_template=prompt, tools=tools)
    agent.register("add_numbers", add_numbers)

    # setup messages
    system_messages = agent.build_system_prompt({
        "problem": "2 + 3",
        "subject": "math"
    })
    user_messages = [
        {"role": "user", "content": "What is 2 + 3? Use the tool."}
    ]
  
    # initial state (can be empty or pre-populated)
    state = {}

    # run the agent
    msg, state = agent.run(
        messages=system_messages + user_messages,
        model="gpt-4o-mini",
        state=state
    )

    # print results
    print("\n--- RESPONSE ---")
    print(json.dumps(msg, indent=2))

    print("\n--- STATE ---")
    print(json.dumps(state, indent=2))

    # assert response structure
    assert "role" in msg
    assert msg["role"] == "assistant"

    # tool may or may not be called (LLM decision), so check safely
    if "function_call" in msg:
        assert "result" in state
        print("\n✅ Tool execution verified")

    print("\n✅ Smoke test passed!")

# --------------------------------------------------------------------------------------------------------------
# smoke test - basic chat (no tool call)


def run_smoke_test_chat(console):
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Missing OPENAI_API_KEY in .env"

    client = LLMClient(api_key)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    console.print(Rule("Interactive Smoke Chat"))
    console.print("Type your message and press Enter. Type 'exit' to quit.\n")

    while True:
        try:
            user_text = console.input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nSession ended.")
            break

        if user_text.lower() in {"exit", "quit", "q"}:
            console.print("Session ended.")
            break

        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})

        msg = client.chat(
            model="gpt-4o-mini",
            messages=messages,
        )

        # --- basic checks ---
        assert "role" in msg
        assert msg["role"] == "assistant"
        assert "content" in msg

        console.print("assistant>", msg["content"])
        messages.append(clean_message(msg))

    console.print("\n✅ Smoke chat finished.")
    

if __name__ == "__main__":

    # Setup logging
    console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(funcName)s - %(message)s",
        handlers=[RichHandler(console=console)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger.info("Run smoke tests...")

    # run_smoke_test_fn_call()
    run_smoke_test_chat(console)
