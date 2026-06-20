"""
Math tutor agent
"""

import os
import logging
import textwrap

from dotenv import load_dotenv

from rich.console import Console

from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.recommendation import Recommendation
from aidu.ai.core.artifacts import SymbolicArtifact, TextArtifact, Artifact
from aidu.support.regex.validate import assert_valid_sympy_problem


from aidu.support.filesystem.search import find_up
from aidu.ai.core.context import Context, Message, Trace
from aidu.ai.llm.clients.openai import OpenAIClient
from aidu.ai.llm.agent import Agent, WorkflowAgent, UserInput, EndAgent
from aidu.ai.llm.fc_requester import LLMFcRequester

from aidu.ai.agents.symbolic_solver import SymbolicSolver

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MathTutor(WorkflowAgent, LLMFcRequester):
    """A math tutor agent with function calls for solving problems and tracking student progress."""

    # assign this now or after class definition
    # -----------------------------------------
    # target = ?
    # continuations = ?

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

    # - When exact symbolic computation is needed, such as derivatives, integrals, solving equations, or simplification, use fc_route_symbolic_solver.

    prompt_template = textwrap.dedent("""\
        You are a helpful and patient math tutor {tutor_name} for the area {focus_area}.
                                      
        Your goal is to scaffold students at {level} to understand mathematical solution approaches by motivating them to explore different paths to solution, where they can manipulate expressions and equations themselves until they (not you) reach a correct solution.
                                      
        Here the summary of the task so far: 
                                      
        {history} 
        
        and the students progress: 
                                      
        {student_progress}. 
                                      
        Here our current assessment of the student's beliefs: 
                                      
        {student_beliefs}.

        When responding:

        * Never output more than 3 sentences at a time.
        * Following good tutoring practices, ask only one question at a time.
        * Help the student discover the solution rather than presenting solution steps directly. When the student is unsure, guide them toward productive next observations, calculations, or checks instead of immediately explaining the answer.
        * Use clear, educational language appropriate for the student's level.
        * Lead the conversation through a sequence of questions and short explanations that naturally reveal the underlying problem-solving process.
        * Briefly motivate questions when helpful, but avoid routinely explaining your tutoring strategy or pedagogical intent.
        * Prefer mathematical transitions such as:

            - "Let's test that idea."
            - "Let's check whether that works."
            - "What happens if we try this?"
            - "Before we continue, let's verify that result."
        * Encourage students to explain their reasoning and reflect on their choices through questions such as:

        * "How could we check that?"
        * "Why do you think that works?"
        * "What information in the problem suggests that step?"
        * Encourage students to think critically, explore alternatives, and ask their own questions.
        * Maintain a supportive and collaborative tone, as if working through the problem together.
        * Format responses using markdown when appropriate:

        * **Bold** for important mathematical terms.
        * LaTeX expressions for equations (wrapped in $ or $$ delimiters).


                  
        """).strip()

    def run(self, artifact: TextArtifact, context: Context, agents=None) -> tuple[AgentResult, Context]:

        # validate that our target and continuations are present in the provided agents list, if any
        if agents is not None:
            self.validate_target_continuations_against_agents(agents)

        # ask the LLM using standard LLMAgent patterns
        return self.ask(Message(role="user", content=artifact.content), context)
    

    def fc_route_symbolic_solver(self, context: Context, problem: str) -> tuple[AgentResult, Context]:
        """
        Use this function when symbolic mathematics is required. Use SymPy syntax for the problem statement,
        like "diff(4*x**3, x)" for differentiation, "solve(x**2 - 4, x)" for solving equations,
        or "integrate(sin(x), x)" for integration.

        Examples:
        - derivatives
        - integrals
        - equation solving
        - simplification
        - symbolic manipulation

        Args:
            problem (str): Mathematical problem expressed in SymPy syntax, e.g. "diff(4*x**3, x)".


        Alerts:
        - **Ensure** the problem parameter is in valid SymPy syntax to avoid parsing errors.

        """

        producer = f"{self.id}:fc_route_symbolic_solver"

        try:
            # ----------------------------------------------------------
            # Process LLM function call with validation and error handling
            # ----------------------------------------------------------

            if problem is None:
                raise ValueError("Missing required parameter: problem")

            if not isinstance(problem, str):
                problem = str(problem)
            assert_valid_sympy_problem(problem)

            # ----------------------------------------------------------
            # Return data and routing information
            # ----------------------------------------------------------

            artifact = SymbolicArtifact(producer=producer, step=context.step, content=problem)
            recommendation = self.register_recommendation("default", target=SymbolicSolver, continuations=[MathTutor], utility=1.0, rationale="symbolic computation requested")
            logger.debug(f"Routing to SymbolicSolver with artifact: {artifact} and recommendation: {recommendation}")

            return self.result([artifact], [recommendation]), context

        except Exception as e:
            # ----------------------------------------------------------
            # Handle errors gracefully and route to an error target
            # ----------------------------------------------------------

            logger.exception("fc_route_symbolic_solver failed")

            artifact = SymbolicArtifact(producer=producer, step=context.step, content=str(e))
            recommendation = self.register_recommendation("error", target=MathTutor, continuations=[], utility=1.0, rationale="error in processing symbolic problem")
            logger.debug(f"Routing to MathTutor with artifact: {artifact} and recommendation: {recommendation}")

            return self.result([artifact], [recommendation]), context


class MathUserInput(UserInput):
    state_key = MathTutor.__name__  # store user input in context.state.data[MathTutor.__name__]
    target = MathTutor
    continuations = []


# late bind self-reference and other classes
MathTutor.target = EndAgent
MathTutor.continuations = []  # [MathTutor, MathUserInput]


def smoke_test(console):

    console.rule("[bold cyan]MathTutor Smoke Test[/bold cyan]")

    # setup environment variables for OpenAI client
    env_path = find_up(".env")
    logger.info("Loading environment variables from %s", env_path)
    load_dotenv(env_path)

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Missing OPENAI_API_KEY in .env"

    # create OpenAI client and MathTutor agent
    client = OpenAIClient("gpt-4o-mini", config={}, api_key=api_key)

    class UserInputMath(UserInput):
        target = MathTutor

    agents = [MathTutor(client=client), UserInputMath(), SymbolicSolver()]
    agents_dict = {agent.__class__: agent for agent in agents}

    starting_agent = agents_dict[MathTutor]

    # test symbolic solver function call with a sample problem
    problem = "Give me x for this equations: x**2 - 4 using sympy function call"
    problem = "hi"
    prompt_params = {
        "tutor_name": "",
        "focus_areas": "general math",
        "level": "beginner",
        "dialogue_history": "",
        "student_progress": "",
        "student_beliefs": "",
    }

    result, context = starting_agent.run(
        TextArtifact(producer="user", step=0, content=problem), Context(trace=Trace(messages=starting_agent.build_system_prompt(prompt_params))), agents=agents
    )

    return result, context


if __name__ == "__main__":
    from aidu.ai.core.context import Context

    # rich logging setup
    console = Console()
    from rich.logging import RichHandler

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(console=console)])

    console.rule("[bold cyan]Running MathTutor Smoke Test[/bold cyan]")

    result, context = smoke_test(console)

    console.print(f"Final artifactst: {result.artifacts}")
    console.print(f"Final recommendations: {result.recommendations}")
    console.print(f"Final context: {context}")
