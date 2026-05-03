# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
SymPy-based client that solves mathematical problems locally without an LLM.
"""

import re

from sympy import symbols, solve, diff, latex
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from ..client import (
    Client,
    Context,
    Message,
)

TRANSFORMATIONS = standard_transformations


def _preprocess(expr_str: str) -> str:
    """Convert implicit math notation to Python-compatible syntax."""
    expr_str = expr_str.replace('^', '**')
    expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
    return expr_str

def solve_math_problem_with_sympy(problem: str) -> dict:
    """
    Solves a mathematical problem using SymPy and formats with natural wording and LaTeX.
    
    Supports multiple syntaxes:
    - "diff(expr,x)" for derivatives (e.g., "diff(7x^2 + 3x - 5, x)")
    - "solve(expr,x)" to solve for x in an expression (e.g., "solve(2x + 3, x)")
    - "2x + 3 = 7" for equations
    - "7x^2 + 3x - 5" for expression evaluation
    
    Args:
        problem (str): The math problem string
        
    Returns:
        dict: Contains 'type', 'expression', 'result', 'latex', and 'message' keys
        
    Raises:
        ValueError: If the syntax is invalid
    """
    # Create symbolic variable 'x' - tells SymPy that 'x' is a mathematical variable
    x = symbols('x')  # Default variable
    
    # CASE 1: Derivative using diff(expr, variable)
    if problem.startswith('diff('):
        # Extract expression and variable using regex pattern: diff(..., ...)
        match = re.match(r'diff\((.+),\s*(\w+)\)', problem)
        if match:
            # Get expression string and variable name from the regex match
            expr_str, var_name = match.groups()
            # Create symbolic variable for differentiation
            var = symbols(var_name)
            # Parse string into SymPy expression (e.g., "7x^2" -> mathematical object)
            expr = parse_expr(_preprocess(expr_str), transformations=TRANSFORMATIONS)
            # Calculate derivative: d/dx of expr
            result = diff(expr, var)
            # Convert to LaTeX format and remove spaces for clean output
            expr_latex = latex(expr).replace(' ', '')
            result_latex = latex(result).replace(' ', '')
            # Create human-readable message with LaTeX delimiters ($...$)
            message = f"When we differentiate ${expr_latex}$, we get ${result_latex}$ from SymPy."
            # Return result with type indicator and all components
            return {
                'type': 'derivative',
                'expression': str(expr),
                'result': str(result),
                'latex': f"${expr_latex}$ → ${result_latex}$",
                'message': message
            }
        else:
            raise ValueError("Invalid diff syntax. Use: diff(expression, variable)")
    # CASE 2: Solve using solve(expr, variable) - finds where expr = 0
    elif problem.startswith('solve('):
        # Extract expression and variable using regex pattern
        match = re.match(r'solve\((.+),\s*(\w+)\)', problem)
        if match:
            # Get expression and variable from regex match
            expr_str, var_name = match.groups()
            # Create symbolic variable
            var = symbols(var_name)
            # Parse expression string into SymPy object
            expr = parse_expr(_preprocess(expr_str), transformations=TRANSFORMATIONS)
            # Solve: find all values of 'var' that make expr = 0
            solutions = solve(expr, var)
            # Convert to LaTeX, removing spaces
            expr_latex = latex(expr).replace(' ', '')
            # Format multiple solutions as comma-separated LaTeX strings
            solutions_latex = ', '.join([latex(sol).replace(' ', '') for sol in solutions])
            # Create explanation message showing the problem and solution
            message = f"Solving ${expr_latex} = 0$ for ${var_name}$: ${var_name} = {solutions_latex}$ with SymPy."
            # Return with type and all components
            return {
                'type': 'solve',
                'expression': str(expr),
                'result': str(solutions),
                'latex': solutions_latex,
                'message': message
            }
        else:
            raise ValueError("Invalid solve syntax. Use: solve(expression, variable)")
    # CASE 3: Equation solving (contains = sign, e.g., "2x + 3 = 7")
    elif '=' in problem:
        # Split at = sign to get left and right sides
        lhs, rhs = problem.split('=')
        # Rearrange to standard form: lhs - rhs = 0 for solving
        expr = parse_expr(_preprocess(lhs.strip())) - parse_expr(_preprocess(rhs.strip()))
        # Solve the rearranged equation
        solutions = solve(expr, x)
        # Convert both sides to LaTeX for display
        lhs_latex = latex(parse_expr(_preprocess(lhs.strip()))).replace(' ', '')
        rhs_latex = latex(parse_expr(_preprocess(rhs.strip()))).replace(' ', '')
        # Format solutions as LaTeX string
        solutions_latex = ', '.join([latex(sol).replace(' ', '') for sol in solutions])
        # Create message showing original equation and solution
        message = f"Solving ${lhs_latex} = {rhs_latex}$ for $x$: $x = {solutions_latex}$ with SymPy."
        # Return with equation type
        return {
            'type': 'equation',
            'expression': problem,
            'result': str(solutions),
            'latex': solutions_latex,
            'message': message
        }
    # CASE 4: Expression formatting (no operation, just format the math)
    else:
        # Parse and format the expression without solving anything
        expr = parse_expr(_preprocess(problem), transformations=TRANSFORMATIONS)
        # Convert to LaTeX, removing spaces
        expr_latex = latex(expr).replace(' ', '')
        # Create description message with SymPy attribution
        message = f"The expression ${expr_latex}$ evaluates to ${expr}$ with SymPy."
        # Return with expression type (not solved, just formatted)
        return {
            'type': 'expression',
            'expression': str(expr),
            'result': str(expr),
            'latex': expr_latex,
            'message': message
        }


class SymPyClient(Client):
    """
    A chat-compatible client that resolves math problems using SymPy.

    The ``message`` content is passed directly to ``solve_math_problem_with_sympy``.
    The ``model`` parameter is accepted for interface compatibility but ignored.

    Supported problem syntaxes:
    - ``diff(expr, x)``  — derivative
    - ``solve(expr, x)`` — solve for x
    - ``lhs = rhs``      — equation
    - ``expr``           — expression formatting
    """

    def __init__(self, model, config, process):
        super().__init__(model=model, config=config)
        self.process = process

    def chat(self, message: Message, context: Context) -> Message:
        """
        Solve the math problem contained in *message* and return an assistant message.

        Args:
            message: A message dict with ``role`` and ``content`` keys.
                     The ``content`` string is passed to SymPy as the problem.
            context: Conversation context (not mutated here).

        Returns:
            A normalized message dict with ``role: "assistant"`` and the
            SymPy result serialized as JSON in ``content``.
        """
        problem = message.get("content", None)
        assert problem, "SymPyClient requires a 'content' field with the math problem string."
        
        result = self.process(problem)
        return {
            "role": "solver",
            "content": result["message"],
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    client = SymPyClient('sympy', config={}, process=solve_math_problem_with_sympy)

    problems = [
        "diff(7x^2 + 3x - 5, x)",
        "solve(2x + 3, x)",
        "2x + 3 = 7",
        "7x^2 + 3x - 5",
    ]

    context = Context()

    console.rule("[bold cyan]SymPyClient Smoke Test[/bold cyan]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("problem", style="yellow")
    table.add_column("response", style="white")

    for problem in problems:
        message = {"role": "user", "content": problem}
        response = client.chat( message, context)
        table.add_row(problem, response["content"])

    console.print(table)


if __name__ == "__main__":
    run_smoke_test()
