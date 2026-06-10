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
from aidu.ai.core.artifacts import SymbolicArtifact, TextArtifact
from aidu.support.regex.validate import assert_valid_sympy_problem


from aidu.support.filesystem.search import find_up
from aidu.ai.core.context import Context, Trace
from aidu.ai.llm.clients.openai import OpenAIClient
from aidu.ai.llm.agent import WorkflowAgent, UserInput
from aidu.ai.llm.fc_requester import LLMFcRequester

from aidu.ai.agents.symbolic_solver import SymbolicSolver


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MathTutor(WorkflowAgent, LLMFcRequester):
    """A math tutor agent with function calls for solving problems and tracking student progress."""
    # result_type = AgentResult
    # Use class references for target/continuations; assigned after class definition

    # will be assigned after class definition to allow for self-reference and other classes
    # target = SymbolicSolver
    # continuations = [] 

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
        You are a helpful and patient math tutor {tutor_name} for the area {focus_areas}.
                                      
        Your goal is to help students at {level}to understand mathematical concepts and solve problems step by step.
                                      
        Here the summary of the task so far: {dialogue_history} and the students progress: {student_progress}. Here our current assessment of the student's beliefs: {student_beliefs}.

        When responding:
        - never output more than 3 sentences at a time                 
        - Use clear, educational language appropriate for the student's level
        - Format your responses using markdown with:
            - **Bold** for important mathematical terms
            - Headers for major sections or steps
            - Lists for step-by-step solutions
            - LaTeX expressions for equations (wrapped in $ or $$ delimiters)
        - Explain the reasoning behind each step, not just the answer
        - When exact symbolic computation is needed, such as derivatives, integrals, solving equations, or simplification, use fc_route_symbolic_solver.
        - Encourage students to think critically and ask questions
                  
        """).strip()



    def run(self, artifact: SymbolicArtifact, context: Context, agents=None) -> tuple[AgentResult, Context]:

        # convert from instance to class if agents are passed as instances; otherwise assume they are already classes
        if agents is not None:
            agents = [agent.__class__ if isinstance(agent, WorkflowAgent) else agent for agent in agents]

        if agents is not None:
            self.validate_agents(agents)

        logger.debug(f"MathTutor received artifact: {artifact}")
        # Build system prompt with dynamic content from context
        message = {"role": "user", "content": artifact.content}
        result, context = self.ask(message, context)
        logger.debug(f"returns artifacts: {result.artifacts}")
        logger.debug(f"returns recommendations: {result.recommendations}")

        return result, context

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

            default_target = SymbolicSolver
  

            # ----------------------------------------------------------
            # Build Artifact and Recommendation for routing to the symbolic solver
            # ----------------------------------------------------------

            artifact = SymbolicArtifact(producer=producer, step=context.step, content=problem)
            recommendation = Recommendation(target=default_target, continuations=[], utility=1.0, rationale="symbolic computation requested")

            # Return an AgentResult containing the artifact and recommendation
            return self.result([artifact], [recommendation]), context

        except Exception as e:
            # ----------------------------------------------------------
            # Handle errors gracefully and route to an error target
            # ----------------------------------------------------------

            logger.exception("fc_route_symbolic_solver failed")
            error_target = MathTutor

            artifact = SymbolicArtifact(producer=producer, step=context.step, content=str(e))
            recommendation = Recommendation(target=error_target, continuations=[], utility=1.0, rationale="error in processing symbolic problem")

            return self.result([artifact], [recommendation]), context

# late bind self-reference and other classes
MathTutor.target = UserInput
MathTutor.continuations = [MathTutor, UserInput]

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
    agents_dict = {     agent.__class__: agent  for agent in agents }

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
        TextArtifact(producer="user", step=0, content=problem), 
        Context(trace=Trace(messages=starting_agent.build_system_prompt(prompt_params))), 
        agents=agents)

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


# Assign class references for target and continuations to use identity-based validation
MathTutor.target = MathTutor
MathTutor.continuations = [SymbolicSolver]
