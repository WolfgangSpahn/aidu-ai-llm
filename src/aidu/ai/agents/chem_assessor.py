# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
"""
Chem assessor agent
"""

import os
import logging
import json
import textwrap
from pprint import pformat
from aidu.ai.llm.agent_runner import run_agent_text_turn
from dotenv import load_dotenv

from rich.console import Console

from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.config import AskConfig
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
logger.setLevel(logging.DEBUG)


def pretty_content(content: str):
    try:
        return pformat(json.loads(content), indent=2)
    except json.JSONDecodeError:
        return content


class ChemAssessor(WorkflowAgent, LLMFcRequester):
    """
    A chemistry assessor agent for evaluating student understanding and tracking progress.
    """

    prompt_template = textwrap.dedent("""\
        Assess chemistry learning evidence. Return ONLY JSON.

        Use CURRENT_TURN as evidence.
        LAST_TURN and APPLET are context only.
        Judge student understanding only.

        q rule:
        q must be an exact substring of the student's CURRENT_TURN answer.
        Never quote tutor text, applet state, or your own inference.
        Use q:null if no exact quote exists.

        Output:
        {{"e":[{{"i":"indicator","p":"+|-|?","s":"w|m|s","q":"quote-or-null"}}],"review":false}}

        Rules:
        - Max 2 evidence items.
        - Use only listed indicators.
        - Prefer the most specific indicator.
        - Do not duplicate broad+narrow evidence.
        - Omit indicators with no direct student evidence.
        - Do not reward tutor explanations or applet state.

        Indicators:
        {valid_indicators}

        Map:
        atomic-number-mass-isotopes=Z/A, A=p+n, Z=p;
        isotope-notation=H-2/C-12 notation;
        proton-identity=element from proton count;
        neutron-identity=recognizing neutron count and that neutrons are counted separately from protons/electrons;
        neutron-isotopes=isotopes differ by neutrons;
        electron-ions=electrons determine charge/neutrality;
        electron-arrangement=K/L/M shells;
        valence-periodic-position=valence/periodic position.

        Prefer:
        A=p+n or neutron from A,p -> atomic-number-mass-isotopes.
        identifying which displayed number is the neutron count -> neutron-identity.
        isotope difference by neutrons -> neutron-isotopes.
        notation meaning H-2/C-12 -> isotope-notation.

        p: +=understands, -=wrong, ?=unclear.
        s: w=hint, m=correct in context, s=explanation/transfer.

        LAST_TURN:
        {last_turn}

        CURRENT_TURN:
        {current_turn}

        APPLET:
        {current_applet_state}

        JSON:
        """).strip()
    def run(
        self,
        artifact: TextArtifact,
        context: Context,
        agents=None,
        ask_config: AskConfig | None = None,
    ) -> tuple[AgentResult, Context]:

        # validate that our target and continuations are present in the provided agents list, if any
        if agents is not None:
            self.validate_target_continuations_against_agents(agents)

        # ask the LLM using standard LLMAgent patterns
        result, context = self.ask(Message(role="user", content=artifact.content), context, ask_config=ask_config)
        return result, context
    
def smoke_test(console):
    console.rule("[bold cyan]ChemAssessor Smoke Test[/bold cyan]")

    env_path = find_up(".env")
    logger.info("Loading environment variables from %s", env_path)
    load_dotenv(env_path)

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Missing OPENAI_API_KEY in .env"

    client = OpenAIClient(
        "gpt-5-mini",
        config={},
        api_key=api_key
    )

    ChemAssessor.target = EndAgent


    starting_agent = ChemAssessor(client=client)
    agents = [
        starting_agent,
        EndAgent(),
    ]

    valid_indicators = [
        "atomic-number-mass-isotopes",
        "isotope-notation",
        "proton-identity",
        "neutron-identity",
        "neutron-isotopes",
        "electron-ions",
        "electron-arrangement",
        "valence-periodic-position",
    ]

    last_turn = textwrap.dedent("""\
        Tutor: Correct - neutral hydrogen has 1 proton and 1 electron. The applet shows A:2; A is the mass number - which two particle counts add up to 2?
        Student: the protons and the neutrons
    """).strip()

    current_turn = textwrap.dedent("""\
        Tutor: Correct - protons plus neutrons make the mass number. The applet shows A:2 and there is 1 proton; how many neutrons does that imply?
        Student: 1
    """).strip()

    current_applet_state = {
        "infoStore": {
            "lewisChemfig": "",
            "electronShellSchema": "K:1 L:0",
            "neutronCount": 1,
            "protonCount": 1,
            "innerElectronCount": 1,
            "outerElectronCount": 0,
            "charge": 0,
            "mass": 2,
            "isotope": "H-2",
            "isStable": True,
            "elementSymbol": "H",
            "elementVisibility": True,
            "legendMode": "legend",
        }
    }

    prompt_params = {
        "valid_indicators": valid_indicators,
        "last_turn": last_turn,
        "current_turn": current_turn,
        "current_applet_state": current_applet_state,
    }

    # The user_text can be minimal because all relevant information is in prompt_params.
    # If your run_agent_text_turn requires non-empty user text, keep this.
    user_text = "Evaluate the current chemistry learning evidence."

    result, context = run_agent_text_turn(
        starting_agent=starting_agent,
        user_text=user_text,
        agents=agents,
        prompt_params=prompt_params,
        ask_config=AskConfig(
            json_mode=True,
            max_tokens=512,
            vendor_config={"reasoning": {"effort": "minimal"}, "verbosity": "low"},
        ),
    )

    return result, context

if __name__ == "__main__":
    console = Console()
    from rich.logging import RichHandler
    logging.basicConfig(
        level=logging.INFO,
        format="%(funcName)s - %(message)s",
        handlers=[RichHandler(console=console)]
    )
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logging.getLogger("httpcore.connection").setLevel(logging.INFO)
    logging.getLogger("httpcore.http11").setLevel(logging.INFO)
    
    result, context = smoke_test(console)
    console.print("[bold green]ChemAssessor Smoke Test Result:[/bold green]")
    console.print(pretty_content(result.content()))
    console.print("[bold green]Context after run:[/bold green]")
    console.print(pformat(context.control.model_dump(), indent=2))
