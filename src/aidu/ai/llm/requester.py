# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.
#

"""
    LLMRequester manages prompts, tools, and interactions with the LLM client.
    Provides methods to build system prompts, update them dynamically, and run agents
    with messages and context, including support for function calls from LLM responses.
"""

import os
import json
import logging

from rich.logging import RichHandler
from rich.console import Console

from dotenv import load_dotenv
from rich.rule import Rule

from .client import Context, Trace, clean_message
from .clients.llm import LLMClient
from .prompter import Prompter

logger = logging.getLogger(__name__)

class LLMRequester:
    """
      LLMRequester is responsible for managing prompts, tools, and interactions with the LLM client.
    It provides methods to build system prompts, update them dynamically, and run the agent with given messages and context.
    It also supports function calls from the LLM response to modify the context.

      Prompt handling pattern:
        - Define prompt_template as a class variable (optional)
        - Override during __init__ with prompt_template parameter
        - Use prompt_args to pass parameters to the template

      Example usage:
            client = LLMClient(api_key)
            prompt = "You are a {subject} tutor. Problem: {problem}"
            tools = [...]  # define tools if needed
            agent = LLMRequester(client, prompt_template=prompt, tools=tools)

            agent.register("add_numbers", add_numbers_fn)
            system_messages = agent.build_system_prompt({"subject": "math", "problem": "2 + 3"})
            user_messages = [{"role": "user", "content": "What is 2 + 3? Use the tool."}]
            context = Context(trace=Trace(messages=system_messages))
            message, context = agent.run(message=user_messages[0], context=context)
    """
    # Class-level prompt template (optional - can be overridden in subclasses)
    prompt_template = None
    
    def __init__(self, client, prompt_template=None, prompt_args=None, tools=None):
        """
        Initialize LLMRequester.
        
        Args:
            client: LLM client instance
            prompt_template: Override the class-level prompt_template (can be string or file path)
            prompt_args: Dict of arguments to pass to the prompt template via .format()
            tools: List of OpenAI tool definitions (auto-generated in LLMActor if None)
        """
        self.client = client
        self.tools = tools or []
        self.function_lookup = {}

        resolved_template = prompt_template if prompt_template is not None else self.prompt_template
        assert resolved_template, "No prompt template provided for LLMRequester"
                    
        self.prompter = Prompter(
            prompt_template=resolved_template,
            prompt_args=prompt_args,
        )
        assert self.prompter.prompt_builder, "Failed to initialize Prompter with the provided template."

    def register(self, name, fn):
        self.function_lookup[name] = fn

    def build_system_prompt(self, prompt_params=None):
        """Build and return the system message list."""
        return self.prompter.build_system_prompt(prompt_params)

    def update_system_prompt(self, context, prompt_params=None):
        """Replace the system message in context with a freshly built one."""
        return self.prompter.update_system_prompt(context, prompt_params)

    def chat(self, message, context, chat_params=None):
        """
        Run the agent with given message and context.
        - context: Context object with trace of message dicts (role/content)
        - message: current input message dict (role/content)
        - chat_params: optional dict for dynamic prompt updates (e.g. {"subject": "math", "problem": "2 + 3"})
        
        Function calls can return either context or (message, context) tuple.
        Returned msg dict will contain _fc_message key if a function was called.
        """
        if chat_params:
            context = self.update_system_prompt(context, prompt_params=chat_params)

        response = self.client.chat(message, context)

        fc = response.get("function_call")
        if fc:
            fn = self.function_lookup.get(fc["name"])
            args = json.loads(fc["arguments"])

            if fn:
                result = fn(context=context, **args)
                # Handle functions returning (message, context) or just context
                if isinstance(result, tuple) and len(result) == 2:
                    fc_message, context = result
                    # Store the function call message for the caller
                    response["_fc_message"] = fc_message
                    # Append function call message to context
                    if fc_message:
                        context.trace.messages.append({"role": "assistant", "content": fc_message})
                else:
                    context = result

        return response, context
    
    def talk(self, message, context, run_params=None):
        """
        Convenience method to run and append response to context.trace.
        """
        effective_context = (
            self.update_system_prompt(context, prompt_params=run_params)
            if run_params
            else context
        )
        response, effective_context = self.chat(
            message=message,
            context=effective_context,
            chat_params=None,
        )

        effective_context.trace.messages.append(clean_message(message))
        effective_context.trace.messages.append(clean_message(response))
        return response, effective_context
    

# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# smoke test - function call
#

def add_numbers(context, a: int, b: int):
    result = a + b
    context.state.data["result"] = result
    return context

def run_smoke_test_fn_call():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Missing OPENAI_API_KEY in .env"

    # --- setup ---
    client = LLMClient("gpt-4o-mini", config={'enforce_json': False}, api_key=api_key)

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
  
    # initial context
    context = Context(trace=Trace(messages=system_messages))

    # run the agent
    message, context = agent.chat(
        message=user_messages[0],
        context=context,
    )

    # print results
    print("\n--- RESPONSE ---")
    print(json.dumps(message, indent=2))

    print("\n--- CONTEXT ---")
    print(json.dumps(context.state.data, indent=2))

    # assert response structure
    assert "role" in message
    assert message["role"] == "assistant"

    # tool may or may not be called (LLM decision), so check safely
    if "function_call" in message:
        assert "result" in context.state.data
        print("\n✅ Tool execution verified")

    print("\n✅ Smoke test passed!")

# --------------------------------------------------------------------------------------------------------------
# smoke test - basic chat (no tool call)


def run_smoke_test_chat(console):
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Missing OPENAI_API_KEY in .env"

    client = LLMClient("gpt-4o-mini", config={}, api_key=api_key)

    context = Context(trace=Trace(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
    ]))

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

        user_message = {"role": "user", "content": user_text}
        message = client.chat(
            message=user_message,
            context=context,
        )
        context.trace.messages.append(user_message)

        # --- basic checks ---
        assert "role" in message
        assert message["role"] == "assistant"
        assert "content" in message

        console.print("assistant>", message["content"])
        context.trace.messages.append(clean_message(message))

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
