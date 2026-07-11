# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""LLM-based MathSolver using the shared LLMRequester flow."""

import logging
import os
from dotenv import load_dotenv
import json
from rich.console import Console

from ..requester import LLMRequester
from ..clients.openai import OpenAIClient
from aidu.ai.core.context import Context, Trace
from aidu.ai.core.config import AskConfig

logger = logging.getLogger(__name__)


class MathSolver(LLMRequester):
    """One-shot math solver using LLMRequester base flow."""

    prompt_template = (
        "You are a deterministic math solver. "
        "Solve the given math problem and return strict JSON with exactly these keys: "
        "type, expression, result, latex, message. "
        "Do not add extra keys or markdown."
    )


def smoke_test(client, solver, problem="diff(7x^2 + 3x - 5, x)"):
    """Run a simple smoke test to verify the MathSolver works end-to-end."""

    response, _ = solver.ask(
        message={"role": "user", "content": problem},
        context=Context(trace=Trace(messages=solver.build_system_prompt())),
        ask_config=AskConfig(json_mode=True),
    )
    # we should get a structured response with the solution to the math problem
    # rendering the json in content
    return json.loads(response.get("content", "{}"))


def generate_polynomial(degree=3):
    """
    Generate a random polynomial of the given degree, via getting random a_n coefficients. (x-a_0)(x-a_1)⋯(x-a_n)
    and multiplying it out with sympy.
    Coefficients are random integers between -5 and 5, excluding 0 to ensure the term is present.
    """

    import random

    terms = []
    coeffs = []
    for i in range(degree):
        a_n = random.choice([j for j in range(-5, 6) if j != 0]) + 1 / 3.0
        coeffs.append(a_n)
        terms.append(f"(x - {a_n})")
    polynomial = " * ".join(terms)
    logger.info(f"Generated polynomial (factored form): {polynomial}")
    # multiplied out with sympy
    from sympy import symbols, expand

    x = symbols("x")  # noqa: F841
    polynomial = expand(eval(polynomial))
    return polynomial, set(coeffs)


if __name__ == "__main__":
    console = Console()
    # rich logging setup
    from rich.logging import RichHandler

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Missing OPENAI_API_KEY in .env"

    client = OpenAIClient("gpt-4o-mini", config={}, api_key=api_key)
    solver = MathSolver(client=client)

    N = 20
    passed = 0
    failed = 0
    errors = 0

    for i in range(1, N + 1):
        console.rule(f"[bold]Run {i}/{N}[/bold]")
        polynomial, coeffs = generate_polynomial(degree=2)
        console.print(f"Generated polynomial: [bold magenta]{polynomial}[/bold magenta]")
        console.print(f"Expected roots (coefficients): [bold cyan]{coeffs}[/bold cyan]")
        try:
            raw = smoke_test(client, solver, problem=f"solve({polynomial}, x)")
            result = set(raw.get("result", []))
            console.print(f"Solver result: [bold cyan]{result}[/bold cyan]")
            if all(any(abs(r - c) < 0.5 for r in result) for c in coeffs):
                console.print("[bold green]PASS[/bold green]")
                passed += 1
            else:
                console.print("[bold red]FAIL[/bold red]")
                failed += 1
        except Exception as exc:
            console.print(f"[bold yellow]ERROR: {exc}[/bold yellow]")
            errors += 1

    console.rule("[bold]Statistics[/bold]")
    total = passed + failed + errors
    console.print(f"Runs:   {total}")
    console.print(f"Passed: [bold green]{passed}[/bold green] ({100 * passed / total:.0f}%)")
    console.print(f"Failed: [bold red]{failed}[/bold red] ({100 * failed / total:.0f}%)")
    if errors:
        console.print(f"Errors: [bold yellow]{errors}[/bold yellow] ({100 * errors / total:.0f}%)")
