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
import time

from rich.logging import RichHandler
from rich.console import Console

from dotenv import load_dotenv
from rich.rule import Rule

from .client import Context, Trace, clean_message
from .clients.llm import LLMClient
from .prompter import Prompter

logger = logging.getLogger(__name__)

# USD per 1M tokens (input/output)
MODEL_COSTS_USD_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
}

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

    @staticmethod
    def _clean_message_for_storage(msg: dict) -> dict:
        """Keep only stable fields suitable for reuse in future chat completions."""
        cleaned = {}
        if "role" in msg:
            cleaned["role"] = msg["role"]
        if "content" in msg and msg["content"]:
            cleaned["content"] = msg["content"]
        if "duration" in msg and msg["duration"] is not None:
            cleaned["duration"] = msg["duration"]
        if "prompt_tokens" in msg and msg["prompt_tokens"] is not None:
            cleaned["prompt_tokens"] = msg["prompt_tokens"]
        if "completion_tokens" in msg and msg["completion_tokens"] is not None:
            cleaned["completion_tokens"] = msg["completion_tokens"]
        if "total_tokens" in msg and msg["total_tokens"] is not None:
            cleaned["total_tokens"] = msg["total_tokens"]
        if "cost_usd" in msg and msg["cost_usd"] is not None:
            cleaned["cost_usd"] = msg["cost_usd"]
        if "model" in msg and msg["model"] is not None:
            cleaned["model"] = msg["model"]
        if "timestamp" in msg and msg["timestamp"] is not None:
            cleaned["timestamp"] = msg["timestamp"]
        if "function_call" in msg and msg["function_call"]:
            cleaned["function_call"] = msg["function_call"]
        return cleaned

    @staticmethod
    def _estimate_cost_usd(model: str | None, prompt_tokens: int, completion_tokens: int) -> float:
        rates = MODEL_COSTS_USD_PER_1M.get(model or "")
        if not rates:
            return 0.0

        return (
            (prompt_tokens * rates["input"]) +
            (completion_tokens * rates["output"])
        ) / 1_000_000

    @staticmethod
    def _resolve_stored_assistant_message(response: dict, duration: float | None = None, timestamp: float | None = None) -> dict:
        """Normalize the assistant response into the stored trace representation."""
        stored_assistant = LLMRequester._clean_message_for_storage(response)
        if not stored_assistant.get("content") and response.get("_fc_message"):
            if isinstance(response["_fc_message"], dict):
                stored_assistant["content"] = response["_fc_message"].get("content", "")
            else:
                stored_assistant["content"] = response["_fc_message"]
        if duration is not None:
            stored_assistant["duration"] = duration
        if timestamp is not None:
            stored_assistant["timestamp"] = timestamp
        return stored_assistant

    def store_turn(self, context: Context, user_message: dict, response: dict) -> Context:
        """Append one user/assistant turn to context.trace.messages."""
        stored_user = clean_message(user_message)
        stored_user["duration"] = 0.0
        stored_user["prompt_tokens"] = 0
        stored_user["completion_tokens"] = 0
        stored_user["total_tokens"] = 0
        stored_user["cost_usd"] = 0.0
        response_timestamp = time.time()
        stored_user["timestamp"] = response_timestamp - context.control.duration
        context.trace.messages.append(stored_user)
        context.trace.messages.append(
            self._resolve_stored_assistant_message(
                response,
                duration=context.control.duration,
                timestamp=response_timestamp,
            )
        )
        return context

    def chat(self, message, context, chat_params=None):
        """
        Run the agent with given message and context.
        - context: Context object with trace of message dicts (role/content)
        - message: current input message dict (role/content)
        - chat_params: optional dict for dynamic prompt updates (e.g. {"subject": "math", "problem": "2 + 3"})

        Function calls must return (message, context).
        Returned msg dict will contain _fc_message key if a function was called.
        This method does not append messages to context.trace; callers own storage.
        """
        if chat_params:
            context = self.update_system_prompt(context, prompt_params=chat_params)

        _t0 = time.perf_counter()
        response = self.client.chat(message, context)
        context.control.duration = time.perf_counter() - _t0

        prompt_tokens = int(response.get("prompt_tokens", 0) or 0)
        completion_tokens = int(response.get("completion_tokens", 0) or 0)
        total_tokens = int(response.get("total_tokens", 0) or 0)
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens
        model = response.get("model", getattr(self.client, "model", None))
        cost_usd = self._estimate_cost_usd(model, prompt_tokens, completion_tokens)

        response["prompt_tokens"] = prompt_tokens
        response["completion_tokens"] = completion_tokens
        response["total_tokens"] = total_tokens
        response["cost_usd"] = cost_usd
        if model is not None:
            response["model"] = model

        context.control.data["usage"] = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
        }

        fc = response.get("function_call")
        if fc:
            fn = self.function_lookup.get(fc["name"])
            args = json.loads(fc["arguments"])

            if fn:
                result = fn(context=context, **args)
                assert isinstance(result, tuple) and len(result) == 2, (
                    f"Function call '{fc['name']}' must return (message, context)"
                )
                fc_message, context = result
                response["_fc_message"] = fc_message

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

        effective_context = self.store_turn(
            context=effective_context,
            user_message=message,
            response=response,
        )
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
    agent = LLMRequester(client, prompt_template="You are a helpful assistant.")

    context = Context(trace=Trace(messages=agent.build_system_prompt()))

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
        message, context = agent.talk(
            message=user_message,
            context=context,
        )

        # --- basic checks ---
        assert "role" in message
        assert message["role"] == "assistant"
        assert "content" in message

        console.print("assistant>", message["content"])

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
