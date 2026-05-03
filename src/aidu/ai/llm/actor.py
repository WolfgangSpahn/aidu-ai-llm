# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.
#

"""
LLMActor extends LLMRequester with automatic schema generation for function calls.

Methods prefixed with 'fc_' are automatically discovered and converted to OpenAI function schemas.
Supports Google-style docstrings for parameter descriptions and Pydantic models for complex types.
"""
import logging
import inspect
import re
import textwrap
from typing import get_origin, get_args

from pydantic import BaseModel


from .client import Context, Trace
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


class LLMActor(LLMRequester):
    """
    LLMActor extends LLMRequester with automatic schema generation for function calls.
    
    Methods prefixed with 'fc_' are automatically discovered and converted to OpenAI function schemas.
    Supports Google-style docstrings for parameter descriptions and Pydantic models for complex types.
    
    Prompting pattern (inherited from LLMRequester):
    - Define class-level system_prompt for fixed prompts:
      class MyTutor(LLMActor):
          system_prompt = "You are a math tutor."
    
    - Or define class-level prompt_template for templated prompts with {placeholders}:
      class MyTutor(LLMActor):
          prompt_template = "You are a {subject} tutor."
    
    - Use prompt_args in __init__ to fill placeholders (supports SafeFormat):
      tutor = MyTutor(client, prompt_args={"subject": "math"})
      # Unfilled placeholders remain as {placeholder} for later customization
    
    - Override template at instantiation:
      tutor = MyTutor(client, prompt_template="Override template", prompt_args={...})
    
    Example usage:
        class MyTutor(LLMActor):
            system_prompt = "You are a {subject} tutor{level}."
            
            def fc_solve_problem(self, context, problem: str):
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
    
    def __init__(self, client, prompt_template=None, prompt_args=None, tools=None):
        """
        Initialize LLMActor with optional template and argument overrides.
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
                self.register(name, func)

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

    def chat_turn(self, context: Context, user_message: dict) -> tuple[str, Context]:
        """
        Run a full user turn: call model, resolve reply text, and persist turn in context trace.

        Returns:
            tuple[str, Context]: (reply_text, updated_context)
        """
        message, context = self.chat(message=user_message, context=context)

        reply = message.get("content", "")
        if not reply and message.get("_fc_message"):
            reply = message.get("_fc_message")
        if not reply and message.get("function_call"):
            fc = message.get("function_call")
            reply = f"Executing {fc['name']}..."

        context.trace.messages.append(user_message)
        stored_assistant = self._clean_message_for_storage(message)
        if not stored_assistant.get("content") and message.get("_fc_message"):
            stored_assistant["content"] = message.get("_fc_message")
        context.trace.messages.append(stored_assistant)

        return reply, context

    def interactive_chat(self, context, 
                        on_display_header, on_get_user_input, 
                        on_session_end, on_display_response):
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
            
            # Call model with the current user turn
            user_message = {"role": "user", "content": user_text}
            message, context = self.chat(user_message, context)
            logger.warning(f"LLM response: {message}, updated context: {context.state}")
            
            # Display response: text content, tool call message, or context
            if message.get("content"):
                on_display_response(message.get("content"))
            elif message.get("_fc_message"):
                # Display the actual function call result message
                on_display_response(message.get("_fc_message"))
            elif message.get("function_call"):
                # Fallback: display notification if no message was returned
                fc = message.get("function_call")
                on_display_response(f'  Calling {fc["name"]} with: {fc["arguments"]}')
            else:
                on_display_response(f"context updated: {context.state}")
            
            # Append user and response to context trace for next turn
            context.trace.messages.append(user_message)
            context.trace.messages.append({"role": message.get("role"), "content": message.get("content", "")})
        
        # Session end
        on_session_end()
        
        return context



