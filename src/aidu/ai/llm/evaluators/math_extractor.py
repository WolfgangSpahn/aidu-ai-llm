"""
Math Expression Extractor Evaluator

Extracts mathematical expressions from student messages and returns them as a structured list.
Useful for parsing and analyzing mathematical content submitted by students.
"""

import json
import textwrap
from src.aidu.ai.llm.evaluator import Evaluator


class MathExtractorEvaluator(Evaluator):
    """
    Extracts mathematical expressions from user messages.

    Returns a list of extracted expressions in LaTeX or standard mathematical notation.
    Example: ["3x^2", "2x + 5", "d/dx(x^3)"]
    """

    prompt_template = textwrap.dedent("""
        You are a mathematics expression extractor. Your task is to identify and extract
        all mathematical expressions from the given student message.
        
        Return a JSON object with the following structure:
        {
            "expressions": ["expr1", "expr2", "expr3"],
            "count": 3
        }
        
        Guidelines:
        - Extract ALL mathematical expressions, including variables, equations, and functions
        - Use LaTeX notation where appropriate (e.g., "\\frac{1}{2}", "x^2")
        - Include both explicit expressions and implicit ones (e.g., "x" in "let x be a number")
        - If no mathematical expressions are found, return {"expressions": [], "count": 0}
        - Do not include explanatory text, only the expressions themselves
    """).strip()

    def extract(self, user_message: str, enforce_json: bool = True) -> list[str]:
        """
        Extract mathematical expressions from a user message.

        Args:
            user_message: The student's message containing math expressions
            enforce_json: Whether to enforce JSON response format (default: True)

        Returns:
            A list of extracted mathematical expressions
        """
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": f"Extract math expressions from this message:\n\n{user_message}"}]

        response = self.client.chat(model="gpt-4o-mini", messages=messages, tools=None, response_format={"type": "json_object"} if enforce_json else None)

        response_text = response.get("content", "")

        # Parse JSON, handling markdown code blocks
        try:
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            data = json.loads(response_text)
            return data.get("expressions", [])
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Error parsing response: {e}")
            return []


def run_smoke_test():
    """Run smoke tests for the MathExtractorEvaluator."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from src.aidu.ai.llm.clients.openai import OpenAIClient
    import os
    from dotenv import load_dotenv

    console = Console()

    # Load environment variables from .env file
    load_dotenv()

    test_cases = [
        {"message": "The derivative of 3x^2 is 6x", "description": "Simple algebraic expression with derivative"},
        {"message": "Solve the equation 2x + 5 = 13 for x, and then find the integral of 3x^2 dx", "description": "Multiple expressions with equation and calculus"},
        {"message": "I'm not sure about the limit of (1/x) as x approaches infinity", "description": "Expression with limit notation"},
        {"message": "This is just a regular sentence without any math", "description": "No mathematical expressions"},
    ]

    console.print(Panel("Math Expression Extractor Smoke Test", style="bold blue"))

    # Create LLM client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY not set[/red]")
        return

    client = OpenAIClient(api_key=api_key)
    extractor = MathExtractorEvaluator(client=client)

    for i, test in enumerate(test_cases, 1):
        console.print(f"\n[bold]Test {i}: {test['description']}[/bold]")
        console.print(f"Message: {test['message']}")

        expressions = extractor.extract(test["message"])

        # Create results table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Extracted Expressions", width=50)

        if expressions:
            for expr in expressions:
                table.add_row(expr)
        else:
            table.add_row("[dim]No expressions found[/dim]")

        console.print(table)


if __name__ == "__main__":
    run_smoke_test()
