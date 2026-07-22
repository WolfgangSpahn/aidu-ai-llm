# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
import logging
import re

from rich.console import Console

from sympy import symbols, solve, diff, latex, nsimplify
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from json import dumps

from aidu.ai.core.artifacts import SymbolicArtifact
from aidu.ai.core.context import Context
from aidu.ai.llm.agent import UtilityAgent
from aidu.ai.core.agent_result import AgentResult


from aidu.ai.symbolic.engine import Engine

logger = logging.getLogger(__name__)


TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

SUPERSCRIPT_MAP = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁺": "+",
        "⁻": "-",
    }
)

DERIVATIVE_PREFIXES = (
    "derivate ",
    "derive ",
    "differentiate ",
    "derivative of ",
)


def _preprocess(expr_str: str) -> str:
    """Convert implicit math notation to Python-compatible syntax."""
    expr_str = expr_str.translate(SUPERSCRIPT_MAP)
    expr_str = expr_str.replace("^", "**")
    expr_str = expr_str.replace("×", "*")
    expr_str = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", expr_str)
    return expr_str


def _normalize_problem(problem: str) -> str:
    """Normalize casual math phrasing into the compact forms handled below."""
    normalized = problem.strip().translate(SUPERSCRIPT_MAP)
    lowered = normalized.lower()

    for prefix in DERIVATIVE_PREFIXES:
        if lowered.startswith(prefix):
            expression = normalized[len(prefix) :].strip()
            return f"diff({expression}, x)"

    return normalized


def parse_math(expr_str: str):
    """Parse a math expression and raise a user-facing error on failure."""
    prepared = _preprocess(expr_str)

    try:
        return parse_expr(prepared, transformations=TRANSFORMATIONS)
    except Exception as exc:
        raise ValueError(f"I couldn't parse that math expression '{expr_str}'. Use forms like '4x^3', 'diff(4x^3, x)', or '2x + 3 = 7'.") from exc


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
    problem = _normalize_problem(problem)

    # Create symbolic variable 'x' - tells SymPy that 'x' is a mathematical variable
    x = symbols("x")  # Default variable

    # CASE 1: Derivative using diff(expr, variable)
    if problem.startswith("diff("):
        # Extract expression and variable using regex pattern: diff(..., ...)
        match = re.match(r"diff\((.+),\s*(\w+)\)", problem)
        if match:
            # Get expression string and variable name from the regex match
            expr_str, var_name = match.groups()
            if var_name != "x":
                raise ValueError("Only variable 'x' is supported for differentiation in this version.")
            if not expr_str.strip():
                raise ValueError("Expression for differentiation cannot be empty.")

            # Create symbolic variable for differentiation
            var = symbols(var_name)
            # Parse string into SymPy expression (e.g., "7x^2" -> mathematical object)
            expr = parse_math(expr_str)
            # Calculate derivative: d/dx of expr
            result = diff(expr, var)
            # Convert to LaTeX format and remove spaces for clean output
            expr_latex = latex(expr).replace(" ", "")
            result_latex = latex(result).replace(" ", "")
            # Create human-readable message with LaTeX delimiters ($...$)
            message = f"When we differentiate ${expr_latex}$, we get ${result_latex}$ from SymPy."
            # Return result with type indicator and all components
            return {"type": "derivative", "expression": str(expr), "result": str(result), "latex": result_latex, "message": message}
        else:
            raise ValueError(f"Invalid diff syntax {problem}. Use: diff(expression, variable)")
    # CASE 2: Solve using solve(expr, variable) - finds where expr = 0
    elif problem.startswith("solve("):
        # Extract expression and variable using regex pattern
        match = re.match(r"solve\((.+),\s*(\w+)\)", problem)
        if match:
            # Get expression and variable from regex match
            expr_str, var_name = match.groups()
            if not expr_str.strip():
                raise ValueError("Expression for solving cannot be empty.")
            if var_name != "x":
                raise ValueError("Only variable 'x' is supported for solving in this version.")
            # Create symbolic variable
            var = symbols(var_name)
            # Parse expression string into SymPy object
            expr = parse_math(expr_str)
            # Keep coefficients exact so symbolic roots (e.g., sqrt(3)) are preserved.
            expr = nsimplify(expr, rational=True)
            # Solve: find all values of 'var' that make expr = 0
            solutions = solve(expr, var, rational=True)
            solutions = [nsimplify(sol, rational=True) for sol in solutions]
            # Convert to LaTeX, removing spaces
            expr_latex = latex(expr).replace(" ", "")
            # Format multiple solutions as comma-separated LaTeX strings
            solutions_latex = ", ".join([latex(sol).replace(" ", "") for sol in solutions])
            # Create explanation message showing the problem and solution
            message = f"Solving ${expr_latex} = 0$ for ${var_name}$: ${var_name} = {solutions_latex}$ with SymPy."
            # Return with type and all components
            return {"type": "solve", "expression": str(problem), "result": str(solutions), "latex": solutions_latex, "message": message}
        else:
            raise ValueError("Invalid solve syntax. Use: solve(expression, variable)")
    # CASE 3: Equation solving (contains = sign, e.g., "2x + 3 = 7")
    elif "=" in problem:
        # Split at = sign to get left and right sides
        lhs, rhs = problem.split("=")
        # Rearrange to standard form: lhs - rhs = 0 for solving
        lhs_expr = parse_math(lhs.strip())
        rhs_expr = parse_math(rhs.strip())
        expr = lhs_expr - rhs_expr
        # Solve the rearranged equation
        solutions = solve(expr, x)
        # Convert both sides to LaTeX for display
        lhs_latex = latex(lhs_expr).replace(" ", "")
        rhs_latex = latex(rhs_expr).replace(" ", "")
        # Format solutions as LaTeX string
        solutions_latex = ", ".join([latex(sol).replace(" ", "") for sol in solutions])
        # Create message showing original equation and solution
        message = f"Solving ${lhs_latex} = {rhs_latex}$ for $x$: $x = {solutions_latex}$ with SymPy."
        # Return with equation type
        return {"type": "equation", "expression": problem, "result": str(solutions), "latex": solutions_latex, "message": message}
    # CASE 4: Expression formatting (no operation, just format the math)
    else:
        # Parse and format the expression without solving anything
        expr = parse_math(problem)
        # Convert to LaTeX, removing spaces
        expr_latex = latex(expr).replace(" ", "")
        # Create description message with SymPy attribution
        message = f"The expression ${expr_latex}$ evaluates to ${expr}$ with SymPy."
        # Return with expression type (not solved, just formatted)
        return {"type": "expression", "expression": str(expr), "result": str(expr), "latex": expr_latex, "message": message}


class SymbolicSolver(UtilityAgent, Engine):
    process = staticmethod(solve_math_problem_with_sympy)  # Engine pattern

    def run(self, artifact: SymbolicArtifact, context: Context, agents=None) -> tuple[AgentResult, Context]:  # UtilityAgent pattern
        logger.debug(f"input Artifact: {artifact}")
        # if content is empty raise an error
        if not artifact.content.strip():
            raise ValueError("The problem statement cannot be empty.")
        output = self.process(artifact.content)

        output_str = dumps(output, indent=2)
        logger.debug(f"SymbolicSolver output str: {output_str}")

        result = output.get("result", "no result key")

        context.step = context.step + 1
        artifact = SymbolicArtifact(producer=self.id, step=context.step, content=result)

        return self.result([artifact]), context


def smoke_test(solver, problem):
    """Run a simple smoke test to verify the MathSolver works end-to-end."""

    result, context = solver.run(artifact=SymbolicArtifact(producer="test", step=0, content=problem), context=Context())
    logger.debug(f"Run result: {result}")
    #
    # artifact = SymbolicArtifact(producer="test", step=0, content=result.content)

    # # we should get a structured response with the solution to the math problem
    # # rendering the json in content
    return result.artifacts, result.recommendations


if __name__ == "__main__":
    console = Console()
    # rich logging setup
    from rich.logging import RichHandler

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(console=console)])

    console.rule("[bold cyan]SymbolicSolver Smoke Test[/bold cyan]")
    solver = SymbolicSolver()
    polynomial = "7x^2 + 3x - 5"
    problem = f"solve({polynomial}, x)"
    console.print(f"Testing SymbolicSolver with problem: [bold yellow]{problem}[/bold yellow]")
    artifacts, recommendations = smoke_test(solver, problem=problem)
    logger.debug(f"Raw smoke test result: {artifacts}")
    console.rule(f"Smoke Test Result: [bold green]{artifacts[0].content}[/bold green]")
    # result = set(raw.get("result", []))
    # console.print(f"Solver result: [bold cyan]{result}[/bold cyan]")
