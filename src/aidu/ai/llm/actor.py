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
from sympy import symbols, solve, diff, sympify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


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
    
    Example usage:
        class MyAgent(LLMActor):
            def fc_my_function(self, state, param1: str, param2: int):
                '''
                Performs an operation.
                
                Args:
                    param1 (str): First parameter description.
                    param2 (int): Second parameter description.
                '''
                state['result'] = param1 + str(param2)
                return state
        
        agent = MyAgent(client, prompt_template=prompt)
        tools = agent.schema()
        function_names = agent.fnames()
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
    
    def __init__(self, client, prompt_template=None, tools=None):
        """Initialize LLMActor. If tools is None, automatically generates from schema."""
        if tools is None:
            tools = self.schema()
        super().__init__(client, prompt_template=prompt_template, tools=tools)
        
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
# Utility function for solving math problems
#

def solve_math_problem_with_sympy(problem: str) -> str:
    """
    Solves a mathematical problem using SymPy.
    
    Supports multiple syntaxes:
    - "diff(expr,x)" for derivatives (e.g., "diff(7x^2 + 3x - 5, x)")
    - "solve(expr,x)" to solve for x in an expression (e.g., "solve(2x + 3, x)")
    - "2x + 3 = 7" for equations
    - "7x^2 + 3x - 5" for expression evaluation
    
    Args:
        problem (str): The math problem string
        
    Returns:
        str: The solution result
        
    Raises:
        ValueError: If the syntax is invalid
    """
    x = symbols('x')  # Default variable
    
    if problem.startswith('diff('):
        match = re.match(r'diff\((.+),\s*(\w+)\)', problem)
        if match:
            expr_str, var_name = match.groups()
            var = symbols(var_name)
            expr = parse_expr(expr_str, transformations=(standard_transformations + (implicit_multiplication_application,)))
            result = diff(expr, var)
            solution = f"diff({expr_str}, {var_name}) = {result}"
        else:
            raise ValueError("Invalid diff syntax. Use: diff(expression, variable)")
    elif problem.startswith('solve('):
        # Try to parse as a solve command (e.g., "solve(expr, x)")
        match = re.match(r'solve\((.+),\s*(\w+)\)', problem)
        if match:
            expr_str, var_name = match.groups()
            var = symbols(var_name)
            expr = parse_expr(expr_str, transformations=(standard_transformations + (implicit_multiplication_application,)))
            solutions = solve(expr, var)
            solution = f"Solutions for {var_name} in {expr_str}: {solutions}"
        else:
            raise ValueError("Invalid solve syntax. Use: solve(expression, variable)")
    # Try to parse as an equation to solve
    elif '=' in problem:
        lhs, rhs = problem.split('=')
        expr = parse_expr(lhs.strip()) - parse_expr(rhs.strip())
        solutions = solve(expr, x)
        solution = f"Solutions to {problem}: {solutions}"
    else:
        # Try to evaluate the expression
        expr = parse_expr(problem, transformations=(standard_transformations + (implicit_multiplication_application,)))
        solution = f"{problem} = {expr}"
    
    return solution


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# smoke test - LLMActor with automatic schema generation
#

# Define a simple Pydantic model for structured data
from pydantic import BaseModel, Field

class StudentInfo(BaseModel):
    """Student information."""
    name: str = Field(..., description="Student's full name")
    age: int = Field(..., description="Student's age")

# Create a custom actor with function calls
class MathTutor(LLMActor):
    """A math tutor agent with function calls."""

    def fc_solve_math_problem(self, state, problem: str):
        """
        Solves a mathematical problem using SymPy.

        Args:
            problem (str): The math problem to solve (e.g., "2x + 3 = 7"). Use sympy syntax for more complex problems, such as:
            - "diff(expr,x)" for derivatives (e.g., "diff(7x^2 + 3x - 5, x)")
            - "solve(expr,x )" to solve for x in an expression (e.g., "solve(2x + 3 = 7, x)")
            
        Returns:
            tuple: (message, state) where message describes the solution for context
        """
        try:
            solution = solve_math_problem_with_sympy(problem)
            state["solution"] = solution
            message = f"Solved: {solution}"
        except Exception as e:
            state["solution"] = f"Error solving equation: {str(e)}"
            message = f"Error solving equation: {str(e)}"
        
        logger.warning(f"State updated in fc_solve_math_problem: {state}")
        return message, state

    def fc_student_completed(self, state, student: StudentInfo):
        """
        Mark that a student has completed an exercise.

        Args:
            student (StudentInfo): The student who completed the exercise.
            
        Returns:
            tuple: (message, state) where message describes the completion for context
        """
        state["completed_by"] = student.name
        message = f"Student {student.name} (age {student.age}) has completed the exercise."
        logger.warning(f"State updated in fc_student_completed: {state}")
        return message, state

def run_smoke_test_actor(console):
    """
    Smoke test for LLMActor demonstrating schema generation and interactive chat.
    """
    import json
    from dotenv import load_dotenv
    import os
    from .client import LLMClient
    from rich.console import Console
    from rich.rule import Rule
    from rich.markdown import Markdown
    import textwrap
    

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Missing OPENAI_API_KEY in .env"

    # -----------------------------------------------------------------------------------------------------
    # Initialize the tutor

    client = LLMClient(api_key)
    tutor = MathTutor(client, prompt_template="You are a helpful math tutor.")

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

    # Run interactive chat
    system_prompt = textwrap.dedent("""\
                                    You are a helpful math tutor. Format your responses using markdown with:
                                    - Headers for sections
                                    - **Bold** for important terms
                                    - Lists for steps
                                    - Code blocks for equations when needed""")
    
    messages, state = tutor.interactive_chat(
        model="gpt-4o-mini",
        initial_messages=[{"role": "system", "content": system_prompt}],
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
