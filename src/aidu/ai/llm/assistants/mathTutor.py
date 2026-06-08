# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
Math tutor agent with automatic function calling for solving math problems and tracking student progress.
"""

import logging
import textwrap
from pydantic import BaseModel, Field

from aidu.ai.symbolic.engines.SymbolicSolver import SymbolicSolver
from aidu.ai.core.context import Context, Message

from ..assistant import LLMAssistant

logger = logging.getLogger(__name__)


class StudentInfo(BaseModel):
    """Student information."""

    name: str = Field(..., description="Student's full name")
    age: int = Field(..., description="Student's age")


class MathTutor(LLMAssistant):
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
    prompt_template = textwrap.dedent("""\
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

    capability_specs = {
        "symbolic_engine": SymbolicSolver,
    }

    def fc_solve_math_problem(self, context: Context, problem: str) -> tuple[Message, Context]:
        """
        Solves a mathematical problem using SymPy.

        Args:
            problem (str): The math problem to solve (e.g., "2x + 3 = 7"). Use sympy syntax for more complex problems, such as:
            - "diff(expr,x)" for derivatives (e.g., "diff(7x^2 + 3x - 5, x)")
            - "solve(expr,x )" to solve for x in an expression (e.g., "solve(2x + 3 = 7, x)")

        Returns:
            tuple: (message, context) where message describes the solution for context
        """
        try:
            # Call the solver function to handle all types of math problems
            engine = self.capabilities.get("symbolic_engine")
            if not engine:
                raise ValueError("Symbolic engine not available")

            result, context = engine.ask(
                {"role": "solver", "content": problem},
                context=context,
            )

            # Store the full result in context (type, expression, result, latex, message)
            context.state.data["solution"] = result
            # Wrap the natural language message in a Message dict
            message = {"role": "assistant", "content": result["message"]}
            logger.info(f"Math solution message: {message['content']}")
        except Exception as e:
            # Handle any parsing or math errors gracefully
            context.state.data["solution"] = f"Math input error: {str(e)}"
            message = {"role": "assistant", "content": f"I couldn't solve that yet. {str(e)}"}
            logger.error(f"Math error: {message['content']}")

        # Log updated context for tracking conversation history
        logger.info(f"Context updated in fc_solve_math_problem: {context.state.data}")
        # Return both message (for user) and context (for LLM context)
        return message, context

    def fc_student_completed(self, context: Context, student: StudentInfo) -> tuple[Message, Context]:
        """
        Mark that a student has completed an exercise.

        Args:
            student (StudentInfo): The student who completed the exercise.

        Returns:
            tuple: (message, context) where message describes the completion for context
        """
        # Record the student's name in context to track who completed the exercise
        context.state.data["completed_by"] = student.name
        # Create confirmation message with student info
        message = {"role": "assistant", "content": f"Student {student.name} (age {student.age}) has completed the exercise."}
        # Log the context update
        logger.info(f"Context updated in fc_student_completed: {context.state.data}")
        # Return confirmation and updated context
        return message, context


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# Smoke test
#


def run_smoke_test(console):
    """Smoke test for MathTutor demonstrating schema generation and interactive chat."""
    import os
    from dotenv import load_dotenv
    from rich.rule import Rule
    from rich.markdown import Markdown
    from aidu.ai.core.context import Context, Trace
    from ..clients.openai import OpenAIClient

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Missing OPENAI_API_KEY in .env"

    client = OpenAIClient("gpt-4o-mini", config={}, api_key=api_key)
    tutor = MathTutor(client)

    # Schema generation
    console.print(Rule("MathTutor Schema Generation Test"))
    schemas = MathTutor.schema()
    fnames = MathTutor.fnames()
    console.print(f"\nDiscovered functions: {fnames}")
    console.print(f"Generated schemas: {len(schemas)} function(s)")
    assert len(schemas) == 2, f"Expected 2 schemas, got {len(schemas)}"
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] in ["fc_solve_math_problem", "fc_student_completed"]
    assert "parameters" in schemas[0]["function"]
    console.print("✅ Schema generation verified\n")

    # Interactive chat
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
        console.print("[cyan][tutor>[/]")
        console.print(Markdown(text))

    def on_end():
        console.print("\n[green]✓ Session Complete[/]")

    tutor.interactive_chat(
        context=Context(trace=Trace(messages=tutor.build_system_prompt())),
        on_display_header=header,
        on_get_user_input=get_input,
        on_display_response=display_response,
        on_session_end=on_end,
        console=console,
    )

    print("\n✅ MathTutor smoke test passed!")

    def run_smoke_test_new(console):
        """Smoke test for MathTutor demonstrating schema generation, AgentResult generation and interactive chat."""

        import os
        import textwrap

        from dotenv import load_dotenv
        from rich.rule import Rule
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.pretty import Pretty

        from aidu.ai.core.context import Context, Trace
        from aidu.ai.core.agent_result import ProcessorResult as AgentResult

        from ..clients.openai import OpenAIClient
        from aidu.support.filesystem.search import find_up

        env_path = find_up(".env")
        logger.info("Loading environment variables from %s", env_path)
        load_dotenv(env_path)

        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key, "Missing OPENAI_API_KEY in .env"

        client = OpenAIClient(
            "gpt-4o-mini",
            config={},
            api_key=api_key,
        )

        tutor = MathTutor(client)

        # --------------------------------------------------------------
        # Schema generation
        # --------------------------------------------------------------

        console.print(Rule("MathTutor Schema Generation Test"))

        schemas = MathTutor.schema()
        fnames = MathTutor.fnames()

        console.print(f"\nDiscovered functions: {fnames}")
        console.print(f"Generated schemas: {len(schemas)} function(s)")

        assert len(schemas) == 2
        assert schemas[0]["type"] == "function"
        assert "parameters" in schemas[0]["function"]

        console.print("[green]✓ Schema generation verified[/green]\n")

        # --------------------------------------------------------------
        # AgentResult generation
        # --------------------------------------------------------------

        console.print(Rule("MathTutor AgentResult Test"))

        context = Context()

        result = tutor.fc_solve_math_problem(
            context=context,
            problem="diff(7*x**2 + 3*x - 5, x)",
        )

        assert isinstance(result, AgentResult)

        console.print(
            Panel.fit(
                Pretty(result),
                title="Solve Problem Result",
                border_style="green",
            )
        )

        assert len(result.artifacts) > 0
        assert len(result.recommendations) > 0

        console.print(f"[green]✓ Produced {len(result.artifacts)} artifact(s)[/green]")

        console.print(f"[green]✓ Produced {len(result.recommendations)} recommendation(s)[/green]")

        completion_result = tutor.fc_student_completed(
            context=context,
            student=StudentInfo(
                name="Alice",
                age=15,
            ),
        )

        assert isinstance(completion_result, AgentResult)

        console.print(
            Panel.fit(
                Pretty(completion_result),
                title="Completion Result",
                border_style="yellow",
            )
        )

        assert len(completion_result.artifacts) > 0
        assert len(completion_result.recommendations) > 0

        console.print("[green]✓ AgentResult generation verified[/green]\n")

        # --------------------------------------------------------------
        # Interactive chat
        # --------------------------------------------------------------

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

            console.print("[yellow][user>[/]")
            console.print(indented)

            return text

        def display_response(text):

            console.print("[cyan][tutor>[/]")

            if text:
                console.print(Markdown(text))

        def on_end():

            console.print("\n[green]✓ Session Complete[/green]")

        context = tutor.interactive_chat(
            context=Context(trace=Trace(messages=tutor.build_system_prompt())),
            on_display_header=header,
            on_get_user_input=get_input,
            on_display_response=display_response,
            on_session_end=on_end,
            console=console,
        )

        # console.print()
        # console.print(
        #     Panel.fit(
        #         Pretty(context),
        #         title="Final Context",
        #         border_style="blue",
        #     )
        # )

        # console.print(
        #     "\n[bold green]✓ MathTutor smoke test passed[/bold green]"
        # )


if __name__ == "__main__":
    from rich.logging import RichHandler
    from rich.console import Console

    console = Console()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    run_smoke_test(console)
