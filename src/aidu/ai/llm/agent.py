# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.
#

"""
LLMAgent extends LLMRequester with automatic schema generation for function calls.

Methods prefixed with 'fc_' are automatically discovered and converted to OpenAI function schemas.
Supports Google-style docstrings for parameter descriptions and Pydantic models for complex types.
"""
import logging
import inspect
import re
import textwrap
from typing import get_origin, get_args

from pydantic import BaseModel


from aidu.ai.core.context import Context, Message, Trace
from aidu.ai.core.agent_result import AgentResult
from .requester import LLMRequester


logger = logging.getLogger(__name__)

def parse_docstring(func):
    """Extracts parameter descriptions from a function's Google-style docstring."""
    docstring = func.__doc__ or ""
    param_descriptions = {}

    # Regex pattern for Google-style docstrings (e.g., "param name (type): description")
    pattern = re.findall(r"(\w+)\s*\((.*?)\):\s*(.+)", docstring)

    for name, _, desc in pattern:
        param_descriptions[name] = desc.strip()

    return docstring.strip(), param_descriptions


def get_openai_function_schema(func, make_all_required=False):
    """Dynamically extract function name, docstring, and parameter schema from function definition."""
    docstring = func.__doc__ or ""
    signature = inspect.signature(func)

    parameters = {}
    required_fields = []

    for name, param in signature.parameters.items():
        if name in ["self", "context"]:
            continue  # Ignore self and context

        annotation = param.annotation
        description = "No description available"

        # Extract description from docstring if available
        doc_lines = docstring.split("\n")
        for line in doc_lines:
            if line.strip().startswith(f"{name} ("):
                description = line.split(":", 1)[-1].strip()

        # Handle list of Pydantic models (e.g., list[Phrase])
        if get_origin(annotation) == list:
            item_type = get_args(annotation)[0]
            if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                parameters[name] = {
                    "type": "array",
                    "items": item_type.model_json_schema(),  # Resolve Pydantic model
                    "description": description
                }
            else:
                parameters[name] = {
                    "type": "array",
                    "items": {"type": "string"},  # Assume list of strings if not a Pydantic model
                    "description": description
                }
        
        # Handle single Pydantic models (e.g., StudentInfo, Phrase)
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            parameters[name] = {
                **annotation.model_json_schema(),  # Resolve full Pydantic schema
                "description": description
            }
        
        # Handle basic types
        else:
            type_mapping = {
                str: "string",
                int: "integer",
                bool: "boolean",
                float: "number"
            }
            param_type = type_mapping.get(annotation, "object")
            parameters[name] = {
                "type": param_type,
                "description": description
            }

        if param.default == inspect.Parameter.empty or make_all_required:
            required_fields.append(name)

    return {
        "name": func.__name__,
        "description": docstring.strip().split("\n")[0] if docstring else func.__name__,  # Use the first line of the docstring as summary
        "parameters": {
            "type": "object",
            "properties": parameters,
            "required": required_fields
        }
    }


class LLMAgent(LLMRequester):
    """
    LLMAgent extends LLMRequester with automatic schema generation for function calls.
    
    Methods prefixed with 'fc_' are automatically discovered and converted to OpenAI function schemas.
    Supports Google-style docstrings for parameter descriptions and Pydantic models for complex types.
    
    Prompting pattern (inherited from LLMRequester):
    - Define class-level system_prompt for fixed prompts:
      class MyTutor(LLMAgent):
          system_prompt = "You are a math tutor."
    
    - Or define class-level prompt_template for templated prompts with {placeholders}:
      class MyTutor(LLMAgent):
          prompt_template = "You are a {subject} tutor."
    
    - Use prompt_args in __init__ to fill placeholders (supports SafeFormat):
      tutor = MyTutor(client, prompt_args={"subject": "math"})
      # Unfilled placeholders remain as {placeholder} for later customization
    
    - Override template at instantiation:
      tutor = MyTutor(client, prompt_template="Override template", prompt_args={...})
    
    Example usage:
        class MyTutor(LLMAgent):
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

    capability_specs: dict[str, type | object | None] = {}

    @classmethod
    def schema(cls, make_all_required=False, prefix="fc_"):
        """Extracts all function call methods and generates OpenAI function schemas."""
        functions = [
            func for name, func in inspect.getmembers(cls, predicate=inspect.isfunction)
            if name.startswith(prefix)
        ]
        return [
            {
                "type": "function",
                "function": get_openai_function_schema(func, make_all_required=make_all_required)
            }
            for func in functions
        ]

    @classmethod
    def fnames(cls, prefix="fc_"):
        """Extracts all function call method names starting with the prefix from the class."""
        l = len(prefix)
        return [
            name[l:] for name, func in inspect.getmembers(cls, predicate=inspect.isfunction)
            if name.startswith(prefix)
        ]
    
    def __init__(self, client, prompt_template=None, prompt_args=None, tools=None, capability_overrides=None):
        """
        Initialize LLMAgent with optional template and argument overrides.
        - If tools is None, automatically generates from schema
        - Inherits prompt_template from class variable if not overridden
        - Supports prompt_args to parameterize the template
        - Automatically sets client.tools so OpenAI API can trigger function calls
        """
        if tools is None:
            tools = self.schema()
        
        # Set tools on the client so OpenAI API can trigger function calls
        client.tools = tools
        
        super().__init__(client, prompt_template=prompt_template, prompt_args=prompt_args, tools=tools)
        
        # Auto-register all fc_* methods with their full function name
        for name, func in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith("fc_"):
                self._assert_fc_contract(name, func)
                self.register(name, self._wrap_fc_method(name, func))

        # Store capabilities for use in function calls
        resolved = {}

        for name, provider in self.capability_specs.items():
            resolved[name] = provider() if isinstance(provider, type) else provider

        if capability_overrides:
            resolved.update(capability_overrides)

        self.capabilities = resolved

    @staticmethod
    def _assert_fc_contract(name: str, method) -> None:
        """Validate the fc_* method signature before exposing it to the LLM."""
        signature = inspect.signature(method)
        params = list(signature.parameters.values())

        assert params, f"{name} must accept a 'context' argument as its first parameter"
        assert params[0].name == "context", f"{name} must declare 'context' as its first parameter"
        assert params[0].annotation is Context, f"{name} must annotate 'context' as Context"

        for param in params:
            assert param.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ), f"{name} must not use *args or **kwargs"

        assert (
            signature.return_annotation == tuple[Message, Context]
        ), f"{name} must declare return type tuple[Message, Context]"

    @staticmethod
    def _wrap_fc_method(name: str, method):
        """Assert the runtime fc_* return contract on every invocation."""
        def wrapped(*args, **kwargs):
            result = method(*args, **kwargs)
            assert isinstance(result, tuple) and len(result) == 2, (
                f"{name} must return a tuple of (message, context)"
            )

            message, context = result
            assert isinstance(message, dict | str), (
                f"{name} must return a message as str or Message-compatible dict"
            )
            assert isinstance(context, Context), f"{name} must return Context as second tuple item"
            return result

        return wrapped

    @staticmethod
    def _clean_message_for_storage(msg: dict) -> dict:
        """Keep only stable fields suitable for reuse in future chat completions."""
        cleaned = {}
        if "role" in msg:
            cleaned["role"] = msg["role"]
        if "content" in msg and msg["content"]:
            cleaned["content"] = msg["content"]
        if "function_call" in msg and msg["function_call"]:
            cleaned["function_call"] = msg["function_call"]
        return cleaned

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

    def interactive_chat(self, context,
                        on_display_header, on_get_user_input,
                        on_session_end, on_display_response, console=None):
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

    def run(
        self,
        context: Context,
    ) -> AgentResult:
        raise NotImplementedError

