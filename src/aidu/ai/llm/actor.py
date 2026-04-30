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
from typing import get_origin, get_args

from pydantic import BaseModel


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
        if name in ["self", "state"]:
            continue  # Ignore self and state

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
    
    - Dynamically override system prompt:
      tutor.set_system_prompt("New system prompt")
    
    - Override template at instantiation:
      tutor = MyTutor(client, prompt_template="Override template", prompt_args={...})
    
    Example usage:
        class MyTutor(LLMActor):
            system_prompt = "You are a {subject} tutor{level}."
            
            def fc_solve_problem(self, state, problem: str):
                '''
                Solves a problem.
                
                Args:
                    problem (str): The problem to solve
                '''
                state['result'] = f"Solution to {problem}"
                return "Solved!", state
        
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
        """
        if tools is None:
            tools = self.schema()
        super().__init__(client, prompt_template=prompt_template, prompt_args=prompt_args, tools=tools)
        
        # Auto-register all fc_* methods with their full function name
        for name, func in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith("fc_"):
                self.register(name, func)

    def interactive_chat(self, model, initial_messages, state, 
                        on_display_header, on_get_user_input, 
                        on_session_end, on_display_response):
        """
        Run an interactive chat session with I/O handlers.
        
        Args:
            model (str): LLM model name (e.g., "gpt-4o-mini")
            initial_messages (list): Initial message list (system prompt, etc.)
            state (dict): Initial state dict
            on_display_header (callable): Handler() to display header
            on_get_user_input (callable): Handler() -> str | None for user input (None exits)
            on_session_end (callable): Handler() when session ends
            on_display_response (callable): Handler(response_text) to display assistant response
        
        Returns:
            messages, state: Final message history and state
        """
        messages = initial_messages
        
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
            
            # Add user message and get response
            messages.append({"role": "user", "content": user_text})
            msg, state = self.run(messages=messages, model=model, state=state)
            logger.warning(f"LLM response: {msg}, updated state: {state}")
            
            # Display response: text content, tool call message, or state
            if msg.get("content"):
                on_display_response(msg.get("content"))
            elif msg.get("_fc_message"):
                # Display the actual function call result message
                on_display_response(msg.get("_fc_message"))
            elif msg.get("function_call"):
                # Fallback: display notification if no message was returned
                fc = msg.get("function_call")
                on_display_response(f'  Calling {fc["name"]} with: {fc["arguments"]}')
            else:
                on_display_response(f"state updated: {state}")
            
            # Append response to messages for next turn
            messages.append({"role": msg.get("role"), "content": msg.get("content", "")})
        
        # Session end
        on_session_end()
        
        return messages, state


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# smoke test - LLMActor with automatic schema generation
#

def run_smoke_test_actor(console):
    """
    Smoke test for LLMActor demonstrating schema generation and interactive chat.
    """
    import json
    from dotenv import load_dotenv
    import os
    from .client import LLMClient
    from .actors import MathTutor
    from rich.console import Console
    from rich.rule import Rule
    from rich.markdown import Markdown
    

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Missing OPENAI_API_KEY in .env"

    # -----------------------------------------------------------------------------------------------------
    # Initialize the tutor
    # The system_prompt is automatically used from the MathTutor class definition

    client = LLMClient(api_key)
    tutor = MathTutor(client)  # Uses MathTutor.system_prompt automatically

    # -----------------------------------------------------------------------------------------------------
    # Display schema generation test

    console.print(Rule("LLMActor Schema Generation Test"))

    schemas = MathTutor.schema()
    fnames = MathTutor.fnames()

    console.print(f"\nDiscovered functions: {fnames}")
    console.print(f"Generated schemas: {len(schemas)} function(s)")
    
    assert len(schemas) == 2, f"Expected 2 schemas, got {len(schemas)}"
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] in ["fc_solve_math_problem", "fc_student_completed"]
    assert "parameters" in schemas[0]["function"]
    console.print("✅ Schema generation verified\n")

    # -----------------------------------------------------------------------------------------------------
    # Interactive chat test

    turn_count = [0]

    def header():
        console.print(Rule("Math Tutor Chat"))

    def get_input():
        turn_count[0] += 1
        if turn_count[0] == 1:
            text = "What is the derivative of 7x^2 + 3x - 5? Just the result, no explanation yet."
        elif turn_count[0] == 2:
            text = "Can you explain how?"
        else:
            return None
        
        indented = textwrap.indent(text, "  ")
        console.print(f"[yellow][user>[/]\n{indented}")
        return text

    def display_response(text):
        console.print(f"[cyan][tutor>[/]")
        console.print(Markdown(text))

    def on_end():
        console.print("\n[green]✓ Session Complete[/]")

    # Run interactive chat using the system prompt defined in MathTutor class
    messages, state = tutor.interactive_chat(
        model="gpt-4o-mini",
        initial_messages=[{"role": "system", "content": MathTutor.system_prompt}],
        state={},
        on_display_header=header,
        on_get_user_input=get_input,
        on_display_response=display_response,
        on_session_end=on_end
    )

    print("\n✅ LLMActor smoke test passed!")


if __name__ == "__main__":
    # setup up rich logging
    from rich.logging import RichHandler
    from rich.console import Console
    console = Console()
    logging.basicConfig(level=logging.WARNING, format="%(message)s", handlers=[RichHandler(console=console, rich_tracebacks=True)])

    

    run_smoke_test_actor(console)
