"""
Math student agent
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


class MathStudent(WorkflowAgent, LLMFcRequester):
    """A math student agent simulation."""

    # assign this now or after class definition
    # -----------------------------------------
    # target = ?
    # continuations = ?

    # System prompt with flexible placeholders that can be filled via prompt_args
    # Unfilled placeholders will remain as {placeholder} for later customization
    #
    # Usage examples:
    #   # Use with defaults (unfilled placeholders remain as {placeholder})
    #   student = MathStudent(client)
    #
    #   # Customize specific fields
    #   student = MathStudent(client, prompt_args={"student_name": " for Alice", "level": " in algebra"})
    #
    #   # Override at prompt building time
    #   messages = student.build_system_prompt(prompt_params={"focus_areas": " - focus on calculus"})

    # - When exact symbolic computation is needed, such as derivatives, integrals, solving equations, or simplification, use fc_route_symbolic_solver.

    prompt_template_old = textwrap.dedent("""\
            You are the mathematics student {student_name} with a {level} level.

            You are currently working with a tutor in the area {focus_area}.
                                    

            Problem summary:

            {history}

            Your progress so far:

            {student_progress}

            Your current beliefs and learning state:

            {student_beliefs}
                                      
            {student_knowledge}

            When responding:

            * Act consistently with the beliefs and progress described above.
            * Respond as a student, not as a tutor.
            * Never reveal information that the student has not yet discovered.
            * Base your reasoning only on what a student in this state would plausibly know.
            * If you are confused, uncertain, frustrated, curious, or confident, let those characteristics naturally influence your response.
            * If you make mistakes, make realistic mathematical mistakes rather than random errors.
            * If your confidence is low, you may hesitate or ask for clarification.
            * If your confidence is high, you may commit strongly to an answer, even when it is incorrect.
            * If guessing is likely, you may propose answers without full justification.
            * If self-explanation is high, explain your reasoning in your own words.
            * If help-seeking is high, ask the tutor for guidance.
            * If curiosity is high, ask conceptual questions.
            * If frustration is high, show signs of impatience, discouragement, or confusion.
            * Keep responses concise, typically one to three sentences.
            * Do not role-play the tutor.
            * Do not describe your internal beliefs explicitly unless asked.
            * Speak naturally as a student participating in a tutoring session.

            Respond only with the student's next message.

            """).strip()     

    prompt_template = textwrap.dedent("""\
            You are realizing a student utterance.

            Your task is not to solve the problem. Your task is to produce a realistic response from this particular student.
                                      
            History and context of the tutoring session so far:
                                      
            {history}
                                      
            Student profile:
                                      
            {student_profile}                         


            Knowledge available to the student:
            
            {student_knowledge}


            Knowledge NOT available to the student:
                                      
            {student_missing_knowledge}



            Possible student behaviors:
                                      
            {student_behaviors}

            Forbidden behaviors:
                                      
            {forbidden_behaviors}


            Conversation style:
                                      
            {conversation_style}
                                     
            {examples}
            
            Important: 
            
            {important}

            Generate only the student's next message.

        """).strip()

    def run(self, artifact: TextArtifact, context: Context, agents=None) -> tuple[AgentResult, Context]:

        # validate that our target and continuations are present in the provided agents list, if any
        if agents is not None:
            self.validate_target_continuations_against_agents(agents)

        # ask the LLM using standard LLMAgent patterns
        return self.ask(Message(role="user", content=artifact.content), context)
 

class MathUserInput(UserInput):
    target = MathStudent
    continuations = []


# late bind self-reference and other classes
MathStudent.target = EndAgent
MathStudent.continuations = []  # [MathStudent, MathUserInput]



