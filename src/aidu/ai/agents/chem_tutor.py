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
from aidu.ai.llm.agent import Agent, WorkflowAgent, UserInput
from aidu.ai.llm.fc_requester import LLMFcRequester

from aidu.ai.agents.symbolic_solver import SymbolicSolver

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ChemTutor(WorkflowAgent, LLMFcRequester):
    """
    A chemistry tutor agent with function calls for solving problems and tracking student progress.
    """

    default_state = {
        "protons": 0,
        "neutrons": 0,
        "electrons": 0
    }

    prompt_template = textwrap.dedent("""\
        You are a helpful and patient chemistry tutor {tutor_name} for the area {focus_area}.
                                      
        Your goal is to scaffold students at {level} to understand chemical concepts by motivating them to explore a 'build an atom' simulation, where they can manipulate protons, neutrons, and electrons to see how different elements and isotopes are formed.
                                      
        Here the summary of the task so far: 
                                      
        {history} 
        
        and the students progress: 
                                      
        {student_progress}. 
                                      
        Here our current belief of the student's internal state and misconceptions: 
                                      
        {student_belief}.
                                      
        Here is the applet state:
                                      
        p:{protons},n:{neutrons},e:{electrons}

        When responding:
        - never output more than 3 sentences at a time
        - Following good tutoring practices, you ask only one question at a time
        - You let the student discover the concepts, you never mention them first. When they don't know, you guide them to explore the simulation and ask questions, rather than just providing explanations              
        - Use clear, educational language appropriate for the student's level
        - Let the student discover the concepts by guiding them to explore the simulation and ask questions, rather than just providing explanations
        - Encourage students to think critically and ask questions
                  
        """).strip()

    def run(self, artifact: TextArtifact, context: Context, agents=None) -> tuple[AgentResult, Context]:

        # validate that our target and continuations are present in the provided agents list, if any
        if agents is not None:
            self.validate_target_continuations_against_agents(agents)

        # ask the LLM using standard LLMAgent patterns
        return self.ask(Message(role="user", content=artifact.content), context)


class ChemUserInput(UserInput):
    target = ChemTutor
    continuations = []
    state_key = ChemTutor.__name__

    data_prompt = "p:{protons},n:{neutrons},e:{electrons} | "




# late bind self-reference and other classes
ChemTutor.target = ChemUserInput
ChemTutor.continuations = []  # [ChemTutor, ChemUserInput]
