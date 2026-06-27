# src/aidu/ai/core/processor.py
from __future__ import annotations

import logging
import inspect

from abc import ABC, abstractmethod
from uuid import uuid4

from pydantic import BaseModel, Field

from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.artifacts import Artifact, EndArtifact, SymbolicArtifact, TextArtifact
from aidu.ai.core.context import Context, Message, Trace
from aidu.ai.core.recommendation import Recommendation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class UtilityResult(BaseModel):
    artifacts: list[Artifact] = Field(default_factory=list)


class WorkflowResult(UtilityResult):
    recommendations: list[Recommendation] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Specialized agent types
# --------------------------------------------------------------------------


class Agent(ABC):
    result_type = AgentResult

    @property
    def id(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def run(self, artifact, context=None, agents: list[Agent] | None = None, ask_params=None) -> tuple[AgentResult, Context]:
        pass

    def validate_target_continuations_against_agents(self, agents: list | None = None):
        """
        Validate that agent's target and continuations are present in the provided agents list, and that the
        agents list is consistent with this agent's target and continuations.
        """

        def is_agent_class(a):
            return isinstance(a, type) and issubclass(a, Agent)

        def is_agent_instance(a):
            return not isinstance(a, type) and isinstance(a, Agent)

        self.agents = []

        for agent in agents or []:
            if isinstance(agent, Agent):
                agent = agent.__class__

            if not (isinstance(agent, type) and issubclass(agent, Agent)):
                raise TypeError(f"Expected Agent instance or Agent class, got {agent!r}")

            self.agents.append(agent)

        # -------------------------------------------------------------------------------------------------------------------------
        # Validate that this agent's class-level target is present in the agents list
        # -------------------------------------------------------------------------------------------------------------------------

        cls_target = getattr(self.__class__, "target", None)
        if cls_target is not None:
            # Enforce that class-level `target` is an Agent subclass (class type only)
            if not (isinstance(cls_target, type) and issubclass(cls_target, Agent)):
                raise TypeError(f"Agent {self.id} class-level 'target' must be an Agent subclass (class), got {cls_target!r}")

            matches = []
            for a in self.agents:
                if is_agent_class(a):
                    if a is cls_target or issubclass(a, cls_target):
                        matches.append(a)
                else:  # instance
                    if isinstance(a, cls_target):
                        matches.append(a)

            if not matches:
                raise ValueError(f"Agent {self.id} declares target={cls_target.__name__!r} but no matching agent found in provided agents list")
        # -------------------------------------------------------------------------------------------------------------------------
        # Validate continuations declared on the class point to available agents
        # -------------------------------------------------------------------------------------------------------------------------

        cls_continuations = getattr(self.__class__, "continuations", None)
        if cls_continuations:
            if not isinstance(cls_continuations, (list, tuple)):
                raise TypeError(f"Agent {self.id} class-level 'continuations' must be a list/tuple of Agent subclasses (classes), got {type(cls_continuations)!r}")

            missing = []
            for cont in cls_continuations:
                # Each continuation must be an Agent subclass (class type)
                if not (isinstance(cont, type) and issubclass(cont, Agent)):
                    raise TypeError(f"Agent {self.id} continuation entries must be Agent subclasses (classes), got {cont!r}")

                found = False
                for a in self.agents:
                    if is_agent_class(a):
                        if a is cont or issubclass(a, cont):
                            found = True
                            break
                    else:  # instance
                        if isinstance(a, cont):
                            found = True
                            break
                if not found:
                    missing.append(cont.__name__)
            if missing:
                raise ValueError(
                    f"Agent {self.id} continuations={[c.__name__ for c in cls_continuations]} are not fully overlapping with agents = {[agent.__name__ for agent in self.agents]}"
                )

        # -------------------------------------------------------------------------------------------------------------------------
        # Also check whether any provided agents declare this agent in their continuations
        # -------------------------------------------------------------------------------------------------------------------------

        incoming = []
        for a in self.agents:
            if is_agent_class(a) or is_agent_instance(a):
                conts = getattr(a, "continuations", []) or []
                for c in conts:
                    if isinstance(c, type) and issubclass(c, Agent):
                        if c is self.__class__ or issubclass(self.__class__, c):
                            incoming.append(a)
                            break

        if not incoming:
            logger.debug(f"No provided agents list {self.id} in their continuations (no agent continues to '{self.id}')")


class UtilityAgent(Agent):
    """
    A utility agent is a specialized problem solver.

    It performs a specific task and produces artifacts describing
    the result. Utility agents are unaware of the surrounding
    workflow and therefore never emit recommendations.
    """

    def result(self, artifacts) -> AgentResult:
        assert isinstance(artifacts, list), "UtilityAgent result must be a list of artifacts"
        logger.debug(f"{self.id} produced artifacts: {artifacts}")
        return AgentResult(
            artifacts=artifacts,
            recommendations=[],
        )


class WorkflowAgent(Agent):
    """
    A workflow agent operates at a semantic level.

    In addition to producing artifacts it generates
    recommendations with target and continuations describing how processing should continue.
    """

    # target and continuations for non function-call driven agent results; function-call specify their
    # own targets and continuations in the content of the fc_.... definitions
    target: type[Agent] | None = None
    continuations: list[type[Agent]] = []
    discovered_fn_routes = set()

    def __init_subclass__(cls):
        """
        Create a separate route registry for every WorkflowAgent subclass.

        Without this, all agents would share the same route set inherited
        from WorkflowAgent.
    """
        super().__init_subclass__()
        cls.discovered_fn_routes = set()


    def result(
        self,
        artifacts=None,
        recommendations=None
    ) -> AgentResult:

        return AgentResult(
            artifacts=artifacts or [],
            recommendations=recommendations or [],
        )

    def register_recommendation(
        self,
        mode,
        target,
        continuations=None,
        utility=1.0,
        rationale=""
    ):

        self.__class__.discovered_fn_routes.add(
            (
                inspect.currentframe().f_back.f_code.co_name,
                mode,
                target,
                tuple(continuations or []),
            )
        )

        return Recommendation(
            target=target,
            continuations=continuations or [],
            utility=utility,
            rationale=rationale,
        )


class BeginAgent(WorkflowAgent):
    """Inspect and log the initial actor input before normal routing begins.

    ``BeginAgent`` is useful as an actor startup agent. It does not interpret
    the input. It shows the initial artifact, context, and available agents,
    then forwards the same artifact to the configured target.

    When ``interactive`` is enabled, it pauses on the server console after
    printing the inspection panel. This is intentionally a server-side debugging
    affordance: the actor request waits until the operator presses Enter.
    """

    target: type[Agent] | None = None
    continuations: list[type[Agent]] = []

    def __init__(
        self,
        target: type[Agent] | None = None,
        interactive: bool = False,
    ):
        self.target_agent = target
        self.interactive = interactive

    def run(
        self,
        artifact: Artifact,
        context: Context,
        agents: list[Agent] | None = None,
    ) -> tuple[AgentResult, Context]:
        target = self.target_agent or self.__class__.target
        if target is None:
            raise ValueError("BeginAgent requires a target agent.")

        available_agents = [
            agent.__class__ if isinstance(agent, Agent) else agent
            for agent in (agents or [])
        ]
        if available_agents and target not in available_agents:
            raise ValueError(
                f"BeginAgent target {target.__name__!r} is not registered in this actor."
            )

        self._show_actor_input(artifact, context, agents or [], target)
        if self.interactive:
            from rich import get_console

            get_console().input("[bold green]BeginAgent> press Enter to continue[/bold green] ")

        recommendation = self.register_recommendation(
            "begin",
            target=target,
            continuations=[],
            utility=1.0,
            rationale="Initial actor input inspected; continue to actor router.",
        )
        return self.result(artifacts=[artifact], recommendations=[recommendation]), context

    def _show_actor_input(
        self,
        artifact: Artifact,
        context: Context,
        agents: list[Agent],
        target: type[Agent],
    ) -> None:
        from rich import get_console
        from rich.panel import Panel
        from rich.pretty import Pretty
        from rich.table import Table

        table = Table(show_header=False, box=None)
        table.add_row("[bold cyan]Target[/bold cyan]", target.__name__)
        table.add_row("[bold cyan]Artifact[/bold cyan]", artifact.__class__.__name__)
        table.add_row("[bold cyan]Producer[/bold cyan]", artifact.producer)
        table.add_row("[bold cyan]Content[/bold cyan]", Pretty(artifact.content))
        table.add_row("[bold cyan]Step[/bold cyan]", str(context.step))
        table.add_row("[bold yellow]Trace messages[/bold yellow]", str(len(context.trace.messages)))
        table.add_row("[bold yellow]State keys[/bold yellow]", ", ".join(sorted(context.state.data.keys())))
        table.add_row(
            "[bold yellow]Agents[/bold yellow]",
            ", ".join(agent.__class__.__name__ for agent in agents),
        )

        get_console().print(
            Panel(
                table,
                title=f"[bold magenta]{self.id}[/bold magenta]",
                expand=False,
            )
        )

# --------------------------------------------------------------------------
# Specialized agents for testing and demonstration purposes
# --------------------------------------------------------------------------


from rich.panel import Panel
from rich.table import Table
from rich.pretty import Pretty
from rich import get_console


class DebugAgent(UtilityAgent):
    def run(
        self,
        artifact: Artifact,
        context: Context,
        agents: list[Agent] | None = None,
    ) -> tuple[AgentResult, Context]:

        console = get_console()

        table = Table(show_header=False, box=None)

        # ------------------------------------------------------------------
        # Artifact
        # ------------------------------------------------------------------

        table.add_row(
            "[bold cyan]Artifact[/bold cyan]",
            artifact.__class__.__name__,
        )

        table.add_row(
            "[bold cyan]Producer[/bold cyan]",
            artifact.producer,
        )

        table.add_row(
            "[bold cyan]Content[/bold cyan]",
            str(artifact.content),
        )

        table.add_row(
            "[bold cyan]Step[/bold cyan]",
            str(context.step),
        )

        # ------------------------------------------------------------------
        # Trace
        # ------------------------------------------------------------------

        table.add_row("", "")
        table.add_row(
            "[bold yellow]Trace[/bold yellow]",
            f"{len(context.trace.messages)} message(s)",
        )

        for i, msg in enumerate(context.trace.messages):
            role = msg.get("role", "?")
            content = msg.get("content", "")

            if len(content) > 120:
                content = content[:120] + "..."

            table.add_row(
                f"  [{i}] {role}",
                content,
            )

        # ------------------------------------------------------------------
        # Context
        # ------------------------------------------------------------------

        table.add_row("", "")
        table.add_row(
            "[bold yellow]State[/bold yellow]",
            Pretty(context.state),
        )

        table.add_row(
            "[bold yellow]Control[/bold yellow]",
            Pretty(context.control),
        )

        table.add_row(
            "[bold yellow]Artifacts[/bold yellow]",
            str(len(context.artifacts)),
        )

        # ------------------------------------------------------------------
        # Agents
        # ------------------------------------------------------------------

        table.add_row("", "")

        table.add_row(
            "[bold yellow]Agents[/bold yellow]",
            ", ".join(agent.__name__ if isinstance(agent, type) else agent.__class__.__name__ for agent in (agents or [])),
        )

        console.print(
            Panel(
                table,
                title=f"[bold magenta]{self.id}[/bold magenta]",
                expand=False,
            ),
        )

        # ------------------------------------------------------------------
        # Pass artifact through unchanged
        # ------------------------------------------------------------------

        context.step += 1

        output_artifact = TextArtifact(
            producer=self.id,
            step=context.step,
            content=str(artifact.content),
        )

        return (
            self.result(
                artifacts=[output_artifact],
            ),
            context,
        )


from prompt_toolkit import prompt
from aidu.support.regex.micro_parse import compile_prompt_parser

class UserInput(WorkflowAgent):
    data_prompt = ""

    def __new__(cls, *args, **kwargs):
        if cls is UserInput:
            raise TypeError(
                "UserInput is abstract and cannot be instantiated directly"
            )
        return super().__new__(cls)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls.state_key is None:
            raise TypeError(
                f"{cls.__name__} must define state_key"
            )

    
    def build_prompt_context(self, context: Context) -> dict:

        if self.state_key in context.state.data:
            return context.state.data[self.state_key]

        available_keys = list(context.state.data.keys())
        logger.error(f"Expected context.state.data to contain key '{self.state_key}' for {self.__class__.__name__}, but it was not found. Available keys: {available_keys}")
        raise KeyError(self.state_key)

    def update_state_from_user_input(self, user_input: str, context: Context) -> None:
        parser = compile_prompt_parser(self.data_prompt)
        match = parser.match(user_input)
        if match:
            context.state.data[self.state_key].update({k: v for k, v in match.groupdict().items() if k != "text"})
        else:
            from rich import get_console

            console = get_console()
            if console is not None:
                console.print(f"[bold red]Failed to parse user input with prompt template[/bold red]: {self.data_prompt}")

    def sync_target_system_prompt(self, context: Context, agents: list[Agent] | None = None) -> None:
        if not context.trace.messages or context.trace.messages[0].get("role") != "system":
            return

        target = getattr(self, "target", None)
        if not isinstance(target, type):
            return

        for agent in agents or []:
            if isinstance(agent, target) and hasattr(agent, "build_system_prompt"):
                context.trace.messages[0] = agent.build_system_prompt(
                    prompt_params=context.state.data[self.state_key],
                )[0]
                return
        

    def run(self, artifact: Artifact, context: Context, agents: list[Agent] | None = None) -> tuple[AgentResult, Context]:

        user_input = context.control.data.pop(
            "user_input",
            None,
        )
        from rich import get_console

        console = get_console()
        if user_input is None:
            if console is None:
                raise ValueError("UserInput requires either context.control.data['user_input'] or a console.")

            console.print(
                f"[bold green]{self.__class__.__name__}>[/bold green] ",
                end="",
            )

            prompt_text = self.data_prompt.format(**self.build_prompt_context(context))
            # user_input = console.input(prompt_text + "> ")
            user_input = prompt("", default=prompt_text)

        self.update_state_from_user_input(user_input, context)
        self.sync_target_system_prompt(context, agents)

        context.step += 1
        context.trace.messages.append(Message(role="user", content=user_input))

        logger.debug(f"COntext is now: {context}")

        # console.print(f"\nUserInput Messages: {context.trace.messages}") # NOT SHOWN WHY

        artifact = TextArtifact(
            producer=self.id,
            step=context.step,
            content=user_input,
        )

        return (
            self.result(
                artifacts=[artifact],
                recommendations=[
                    Recommendation(
                        target=self.target,
                        continuations=self.continuations,
                        utility=1.0,
                        rationale="processing user input",
                    )
                ],
            ),
            context,
        )


class DummyAgent(WorkflowAgent):
    def run(self, artifact: SymbolicArtifact, context: Context, agents: list[Agent] | None = None) -> tuple[AgentResult, Context]:

        value = int(artifact.content) + 1

        context.step += 1

        artifact = SymbolicArtifact(
            producer=self.id,
            step=context.step,
            content=str(value),
        )

        return (
            self.result(
                artifacts=[artifact],
                recommendations=[
                    Recommendation(
                        target=self.target,
                        continuations=self.continuations,
                        utility=1.0,
                        rationale="increment again",
                    )
                ],
            ),
            context,
        )


class EchoAgent(WorkflowAgent):
    def run(self, artifact: TextArtifact, context: Context, agents: list[Agent] | None = None) -> tuple[AgentResult, Context]:

        context.step += 1

        artifact = TextArtifact(
            producer=self.id,
            step=context.step,
            content=f"you said: {artifact.content}",
        )

        return (
            self.result(
                artifacts=[artifact],
                recommendations=[
                    Recommendation(
                        target=self.target,
                        continuations=self.continuations,
                        utility=1.0,
                        rationale="echo input",
                    )
                ],
            ),
            context,
        )

class EndAgent(WorkflowAgent):
    def run(self, artifact: TextArtifact, context: Context, agents=None) -> tuple[AgentResult, Context]:

        context.step += 1

        artifact = EndArtifact(
            producer=self.id,
            step=context.step,
            content=f"{artifact.content}",
        )

        return (
            self.result(
                artifacts=[artifact],
                recommendations=[],
            ),
            context,
        )
    
