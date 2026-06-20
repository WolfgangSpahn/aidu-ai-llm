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
from aidu.ai.archetype.archetype import Archetype

from aidu.ai.agents.symbolic_solver import SymbolicSolver
from aidu.ai.agents.math_tutor import MathTutor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Example of Anchor #42 (High Frustration / Low Engagement)
anchor_42 = {
        "vector": [0.8, 0.1, 0.9, 0.2, 0.3, 0.1, 0.2, 0.1],
        "inner_monologue": "I am completely overwhelmed, angry, and honestly just want to close the laptop and quit.",
        "behavioral_quirk": "Giving passive-aggressive, monosyllabic answers, intentionally ignoring the teacher's main question, and avoiding punctuation.",
        "motivation_level": "Extremely low. Survival mode. Zero interest in learning right now."
    }

anchor_12 = {
    "vector": [0.2, 0.4, 0.4, 0.2, 0.7, 0.2, 0.4, 0.2],

    "inner_monologue":
        "This is kind of interesting, but not interesting enough to put in real effort. "
        "Maybe I'll pay attention if something catches my eye.",

    "behavioral_quirk":
        "Occasionally asks unexpected questions about side topics, "
        "but rarely follows through and often drifts back into passive observation.",

    "motivation_level":
        "Low to moderate. Not actively resisting learning, but not investing much energy either."
}

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



    # ------------------------------------------------------------------------------------------
    # Math student knowledge state
    # ------------------------------------------------------------------------------------------


    student_knowledge = textwrap.dedent("""\
            * The student knows that x represents an unknown value.
            """)

    student_missing_knowledge = textwrap.dedent("""\
            * How to solve quadratic equations.
            * How to factor expressions.
            * Standard equation-solving procedures.
            * Algebraic techniques that have not already been introduced by the tutor.
            """)


    # ----------------------------------------------------------------------------------------
    # Math tutor physical and emotional state
    # ----------------------------------------------------------------------------------------

    student_psych = textwrap.dedent("""\
            [STUDENT PSYCHOLOGICAL STATE]
            - Inner Monologue: Mostly feeling like "{inner_monologue}" .
            - Communication Script: You must express this state by {behavioral_quirk}.
            - Motivation Level: {motivation_level} Do not break character.
            """)
    
    narrative_block_mixin = textwrap.dedent("""\
            [PRIMARY ARCHETYPE]
            {primary_archetype}                            
            [SECONDARY ARCHETYPE]
            {secondary_archetype}                                
            [MIX BOTH ARCHETYPES]
            When generating a response, behave approximately {primary_weight:.0%} like the primary archetype and {secondary_weight:.0%} like the secondary archetype.
            """)

    prompt_template = textwrap.dedent("""\
            You are acting as a student in a classroom simulation. You must strictly adopt the psychological narrative provided below. 
            Do not try to be a "good student" or helpful if your profile dictates otherwise.

            {narrative_block}

            [STUDENT'S KNOWLEDGE]     
            {student_knowledge}
                                      
            [STUDENT'S MISSING KNOWLEDGE]
            {student_missing_knowledge}                          
            
            [DIALOGUE HISTORY]
            {dialogue_history}

            [TEACHER'S PROMPT]
            {teacher_prompt}

            [STUDENT RESPONSE (Stay deeply in character, reflect your inner monologue):]
            """).strip()

    def __init__(self, client: OpenAIClient, primary_anchor, secondary_anchor, primary_weight):
        """
        Initialize the MathStudent agent with a client, and archetype anchors.
        """

        assert isinstance(primary_anchor, Archetype), "primary_anchor must be an instance of Archetype"
        assert isinstance(secondary_anchor, Archetype), "secondary_anchor must be an instance of Archetype"
        assert 0.0 <= primary_weight <= 1.0, "primary_weight must be between 0.0 and 1.0"

        self.start_anchor = primary_anchor
        self.target_anchor = secondary_anchor
        self.primary_weight = primary_weight

        dynamic_prompt_args = self.create_dynamic_ask_params(step=0)

        prompt_args={
            "student_name": "Bob",
            "focus_area": "general math",
        }
        prompt_args.update(**dynamic_prompt_args)

        super().__init__(client, prompt_args=prompt_args)

    def mix_archetypes(self, primary_anchor, secondary_anchor, primary_weight=0.7):
        """Mix two archetypes based on the primary weight, into a narrative block for the system prompt."""
        if primary_weight == 1.0:
            # return only the primary archetype
            return self.student_psych.format(**primary_anchor.to_psych_state())
        elif primary_weight == 0.0:
            # return only the secondary archetype
            return self.student_psych.format(**secondary_anchor.to_psych_state())
        elif primary_weight < 0.5:
            # if the primary weight is less than 0.5, swap the primary and secondary archetypes
            primary_anchor, secondary_anchor = secondary_anchor, primary_anchor
            primary_weight = 1.0 - primary_weight

        # if the primary weight is greater than 0.5, return a mix of both archetypes
        secondary_weight = 1.0 - primary_weight
        return self.narrative_block_mixin.format(
            primary_archetype=self.student_psych.format(**primary_anchor.to_psych_state()), 
            secondary_archetype=self.student_psych.format(**secondary_anchor.to_psych_state()),
            primary_weight=primary_weight,
            secondary_weight=secondary_weight
        )

       
    def create_dynamic_ask_params(self, step:int ,speed: float = 0.1) -> dict:
        """Create ask parameters to create a system prompt for the math student agent from behavior path.
        
        Args:
            step (int): The current step in the behavior path, starting from 0.
            speed (float): A value between 0.0 and 0.1 (after 10 steps primary weight is 90%) that controls how quickly the student transitions from the start to the target archetype.
        Returns:"""

        # calculate the primary weight based on the speed x steps and context.step.
        primary_weight = max(0.0, min(1.0, 1.0 - (step * speed )))

        logger.debug(f"Step: {step}, speed: {speed}, primary_weight: {primary_weight:.2f}")

        return {
            "narrative_block": self.mix_archetypes(self.start_anchor, self.target_anchor, primary_weight=primary_weight),
            "student_knowledge": self.student_knowledge,
            "student_missing_knowledge": self.student_missing_knowledge,
        }




    def run(self, artifact: TextArtifact, context: Context, agents=None) -> tuple[AgentResult, Context]:

        # validate that our target and continuations are present in the provided agents list, if any
        if agents is not None:
            self.validate_target_continuations_against_agents(agents)

        ask_params = self.create_dynamic_ask_params(context.step)

        # ask the LLM using standard LLMAgent patterns
        result, context = self.ask(Message(role="user", content=artifact.content), context, ask_params=ask_params)

        return result, context


class MathUserInput(UserInput):
    state_key = MathTutor.__name__  # store user input in context.state.data[MathTutor.__name__]
    target = MathStudent
    continuations = []


# late bind self-reference and other classes
MathStudent.target = EndAgent
MathStudent.continuations = []  # [MathStudent, MathUserInput]



