# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
Math tutor agent with automatic function calling for solving math problems and tracking student progress.
"""

import logging
import re
import textwrap
from pydantic import BaseModel, Field
from sympy import symbols, solve, diff, latex
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

from ..actor import LLMActor

logger = logging.getLogger(__name__)


class StudentInfo(BaseModel):
    """Student information."""
    name: str = Field(..., description="Student's full name")
    age: int = Field(..., description="Student's age")


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
            expr = parse_expr(expr_str, transformations=(standard_transformations + (implicit_multiplication_application,)))
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
            expr = parse_expr(expr_str, transformations=(standard_transformations + (implicit_multiplication_application,)))
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
        expr = parse_expr(lhs.strip()) - parse_expr(rhs.strip())
        # Solve the rearranged equation
        solutions = solve(expr, x)
        # Convert both sides to LaTeX for display
        lhs_latex = latex(parse_expr(lhs.strip())).replace(' ', '')
        rhs_latex = latex(parse_expr(rhs.strip())).replace(' ', '')
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
        expr = parse_expr(problem, transformations=(standard_transformations + (implicit_multiplication_application,)))
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


class MathTutor(LLMActor):
    """A math tutor agent with function calls for solving problems and tracking student progress."""

    # System prompt with flexible placeholders that can be filled via prompt_args
    # Unfilled placeholders will remain as {placeholder} for later customization
    # 
    # Usage examples:
    #   # Use with defaults (unfilled placeholders remain as {placeholder})
    #   tutor = MathTutor(client)
    #   
    #   # Customize specific fields
    #   tutor = MathTutor(client, prompt_args={"student_name": " for Alice", "level": " in algebra"})
    #   
    #   # Override at prompt building time
    #   messages = tutor.build_system_prompt(prompt_params={"focus_areas": " - focus on calculus"})
    system_prompt = textwrap.dedent("""\
        You are a helpful and patient math tutor{student_name}.
        Your goal is to help students{level} understand mathematical concepts and solve problems step by step.

        When responding:
        - Use clear, educational language appropriate for the student's level
        - Format your responses using markdown with:
          - **Bold** for important mathematical terms
          - Headers for major sections or steps
          - Lists for step-by-step solutions
          - LaTeX expressions for equations (wrapped in $ or $$ delimiters)
        - Explain the reasoning behind each step, not just the answer
        - When solving math problems, show your work and reasoning
        - Encourage students to think critically and ask questions
        - Be supportive and patient with students who are learning{focus_areas}
        """).strip()

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
            # Call the solver function to handle all types of math problems
            result = solve_math_problem_with_sympy(problem)
            # Store the full result in state (type, expression, result, latex, message)
            state["solution"] = result
            # Extract the natural language message with LaTeX for sending to user
            message = result['message']
            # Log the solution for debugging
            logger.warning(f"Math solution message: {message}")
        except Exception as e:
            # Handle any parsing or math errors gracefully
            state["solution"] = f"Error solving equation: {str(e)}"
            message = f"Error solving equation: {str(e)}"
            logger.error(f"Math error: {message}")
        
        # Log updated state for tracking conversation history
        logger.warning(f"State updated in fc_solve_math_problem: {state}")
        # Return both message (for user) and state (for LLM context)
        return message, state

    def fc_student_completed(self, state, student: StudentInfo):
        """
        Mark that a student has completed an exercise.

        Args:
            student (StudentInfo): The student who completed the exercise.
            
        Returns:
            tuple: (message, state) where message describes the completion for context
        """
        # Record the student's name in state to track who completed the exercise
        state["completed_by"] = student.name
        # Create confirmation message with student info
        message = f"Student {student.name} (age {student.age}) has completed the exercise."
        # Log the state update
        logger.warning(f"State updated in fc_student_completed: {state}")
        # Return confirmation and updated state
        return message, state
