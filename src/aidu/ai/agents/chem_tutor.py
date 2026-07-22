# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
"""
Chem tutor agent
"""

import os
import logging
import textwrap
from pprint import pformat
from dotenv import load_dotenv

from rich.console import Console

from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.recommendation import Recommendation
from aidu.ai.core.artifacts import SymbolicArtifact, TextArtifact, Artifact, AppletArtifact
from aidu.support.regex.validate import assert_valid_sympy_problem


from aidu.support.filesystem.search import find_up
from aidu.ai.core.context import Context, Message, Trace
from aidu.ai.llm.clients.openai import OpenAIClient
from aidu.ai.llm.agent import Agent, EndAgent, WorkflowAgent, UserInput
from aidu.ai.llm.fc_requester import LLMFcRequester

from aidu.ai.agents.symbolic_solver import SymbolicSolver

logger = logging.getLogger(__name__)


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
        result, context = self.ask(Message(role="user", content=artifact.content), context)
        logger.warning(result.content())
        return result, context

    def fc_change_build_an_atom_applet(self, context: Context, protons: int, neutrons: int, electrons: int) -> tuple[AgentResult, Context]:
        """
        Use this function call to change the state of the 'build an atom' applet in response to student input.
        """

        producer = f"{self.id}:fc_change_build_an_atom_applet"
        try:
            # ----------------------------------------------------------
            # Process LLM function call with validation and error handling
            # ----------------------------------------------------------


            if protons is None or neutrons is None or electrons is None:
                raise ValueError("Missing required parameter: protons, neutrons, or electrons")

            if not isinstance(protons, int) or not isinstance(neutrons, int) or not isinstance(electrons, int):
                raise ValueError("Parameters must be integers: protons, neutrons, electrons")

            # ----------------------------------------------------------
            # Return data and routing information aidu-frontend-assets/assets/images/periodic_table_dark.png
            # ----------------------------------------------------------

            result_content = {
                "applet": "applet-build-an-atom",
                "command": {
                    "kind": "set_atom",
                    "protons": protons,
                    "neutrons": neutrons,
                    "electrons": electrons,
                },
            }

            artifact = AppletArtifact(producer=producer, step=context.step, content=result_content)
            # TODO: register_recommendation contract assumes only registration of fn call routes, but this is a direct artifact return.
            #       We should consider a more general approach for registering recommendations for direct artifact returns.
            #       Or renaming register_recommendation to register_function_call_recommendation to clarify its purpose.
            # recommendation = self.register_recommendation("default", 
            #                                               target=EndAgent, continuations=[], 
            #                                               utility=1.0, 
            #                                               rationale="Change build an atom applet state requested")
            recommendation = Recommendation(target=EndAgent, 
                                            continuations=[], 
                                            utility=1.0, 
                                            rationale="Change build an atom applet state requested")
            logger.debug(f"Routing to Applet with artifact: {artifact} and recommendation: {recommendation}")

            return self.result([artifact], [recommendation]), context

        except Exception as e:
            # ----------------------------------------------------------
            # Handle errors gracefully and route to an error target
            # ----------------------------------------------------------

            logger.exception("fc_change_build_an_atom_applet failed")

            artifact = TextArtifact(producer=producer, step=context.step, content=str(e))
            recommendation = self.register_recommendation("error", target=EndAgent, continuations=[], utility=1.0, rationale="error in processing symbolic problem")
            logger.debug(f"Routing to EndAgent with artifact: {artifact} and recommendation: {recommendation}")

            return self.result([artifact], [recommendation]), context


class ChemUserInput(UserInput):
    target = ChemTutor
    continuations = []
    state_key = ChemTutor.__name__

    data_prompt = "p:{protons},n:{neutrons},e:{electrons} | "




# late bind self-reference and other classes
ChemTutor.target = ChemUserInput # NExt Agent, None for last agent
ChemTutor.continuations = []  # [ChemTutor, ChemUserInput]
