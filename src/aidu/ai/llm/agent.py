# src/aidu/ai/core/processor.py
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from uuid import uuid4

from pydantic import BaseModel, Field

from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.artifacts import Artifact, SymbolicArtifact, TextArtifact
from aidu.ai.core.context import Context
from aidu.ai.core.recommendation import Recommendation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# class Agent_old(ABC):
#     """
#     Base class for all processing units.

#     A processor consumes an artifact and produces:

#         - artifacts
#         - recommendations

#     The controller owns execution flow.
#     Agents never invoke each other directly.
#     """

#     id: str = "processor"
#     name: str = "Base Agent"

#     @abstractmethod
#     def run(self, step: int, artifact: Artifact, context: Context = None, console=None) -> AgentResult:
#         """
#         Process a single artifact.

#         Parameters
#         ----------
#         artifact:
#             Input artifact.

#         Returns
#         -------
#         AgentResult
#             Produced artifacts and recommendations.
#         """
#         raise NotImplementedError

#     def _to_agent_result(self, response: dict, context: Context, producer: str, step: int) -> AgentResult:
#         """
#         Convert an agent response to a AgentResult, handling both direct content and route messages.
#         """

#         logger.debug(f"Processing response: {infer_schema(response)}")
#         # we should see '_fc_message' when we see 'function_call'
#         assert "function_call" not in response or "_fc_message" in response, "Response contains 'function_call' but no '_fc_message'"

#         # check if response['content'] contains already the keys 'artifacts' and 'recommendations'
#         if "artifacts" in response.get("content", {}):
#             logger.debug("Response content contains 'artifacts'. Checking for 'recommendations'.")
#             if "recommendations" in response.get("content", {}):
#                 logger.debug("Response content contains 'artifacts' and 'recommendations'. Using it directly.")
#                 return AgentResult(
#                     artifacts=[
#                         create_artifact(
#                             a["type"],
#                             id=a["id"],
#                             producer=a["producer"],
#                             step=a["step"],
#                             content=a["content"],
#                         )
#                         for a in response["content"].get("artifacts", [])
#                     ],
#                     recommendations=[Recommendation.model_validate(r) for r in response["content"].get("recommendations", [])],
#                 )
#             else:
#                 logger.debug("Response content contains 'artifacts' but no 'recommendations'. Using artifacts and empty recommendations.")
#                 return AgentResult(
#                     artifacts=[
#                         create_artifact(
#                             a["type"],
#                             id=a["id"],
#                             producer=a["producer"],
#                             step=a["step"],
#                             content=a["content"],
#                         )
#                         for a in response["content"].get("artifacts", [])
#                     ],
#                     recommendations=[],
#                 )

#         # check if response is a fc_message and contains a route message and process it if present
#         fc_message = response.get("_fc_message")
#         if fc_message and fc_message.get("type") == "route":
#             logger.debug("Response contains a route message. Processing route content.")
#             content = fc_message["content"]

#             return AgentResult(
#                 artifacts=[
#                     create_artifact(
#                         a["type"],
#                         id=a["id"],
#                         producer=a["producer"],
#                         step=a["step"],
#                         content=a["content"],
#                     )
#                     for a in content.get("artifacts", [])
#                 ],
#                 recommendations=[Recommendation.model_validate(r) for r in content.get("recommendations", [])],
#             )

#         logger.debug("Response does not contain a route message.")
#         return AgentResult(
#             artifacts=[
#                 TextArtifact(
#                     id="response",
#                     producer=producer,
#                     step=step,
#                     content=response["content"],
#                 )
#             ],
#             recommendations=[Recommendation.model_validate(r) for r in content.get("recommendations", [])],
#         )


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
    def run(self, artifact, context=None) -> tuple[AgentResult, Context]:
        pass

    def validate_agents(self, agents: list | None = None):
        # Accept Agent instances, Agent classes, or string names; normalize list
        self.agents = agents or []

        def is_agent_class(obj):
            return isinstance(obj, type) and issubclass(obj, Agent)

        def is_agent_instance(obj):
            return isinstance(obj, Agent)

        # Only allow Agent instances or Agent classes (no raw strings)
        normalized = []
        for a in self.agents:
            if is_agent_instance(a) or is_agent_class(a):
                normalized.append(a)
            else:
                raise TypeError(f"Invalid entry in agents list: {a!r}. Expected Agent instance or Agent class.")

        self.agents = normalized

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
                raise ValueError(
                    f"Agent {self.id} declares target={cls_target.__name__!r} but no matching agent found in provided agents list"
                )

        # Validate continuations declared on the class point to available agents
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

        # Also check whether any provided agents declare this agent in their continuations
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
            logger.info(
                f"No provided agents list {self.id} in their continuations (no agent continues to '{self.id}')"
            )


class UtilityAgent(Agent):
    """
    A utility agent is a specialized problem solver.

    It performs a specific task and produces artifacts describing
    the result. Utility agents are unaware of the surrounding
    workflow and therefore never emit recommendations.
    """

    def result(self, *artifacts) -> AgentResult:
        return AgentResult(
            artifacts=list(artifacts),
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

    def result(
        self,
        artifacts=None,
        recommendations=None,
    ) -> AgentResult:

        return AgentResult(
            artifacts=artifacts or [],
            recommendations=recommendations or [],
        )


class DummyAgent:
    """
    A processor that increments the content of a SymbolicArtifact by 1.
    """

    def run(self, step: int, artifact: SymbolicArtifact, context: Context, console=None) -> tuple[int, AgentResult]:
        logger.debug(f"DummyAgent received artifact: {artifact}")

        value = int(artifact.content) + 1

        next_step = step + 1

        context.step = next_step

        result = AgentResult(
            artifacts=[
                SymbolicArtifact(
                    id=f"{uuid4()}",
                    producer=self.id,
                    step=next_step,
                    content=value,
                )
            ],
            recommendations=[
                Recommendation(
                    target="input",
                    utility=1.0,
                    rationale="increment again",
                )
            ],
        )

        logger.debug(f"DummyAgent result: {result}")

        return result, context


class EchoAgent(Agent):
    """
    A processor that echoes the content of a TextArtifact back as a SymbolicArtifact.
    """

    def __init__(self, target: str = "input"):
        self.target = target

    def run(self, step: int, artifact: TextArtifact, context: Context, console=None) -> tuple[int, AgentResult]:
        logger.debug(f"EchoAgent received artifact: {artifact}")

        value = "you said, " + artifact.content

        context.step = step + 1

        result = AgentResult(
            artifacts=[
                SymbolicArtifact(
                    id=f"{uuid4()}",
                    producer=self.id,
                    step=context.step,
                    content=value,
                )
            ],
            recommendations=[
                Recommendation(
                    target=self.target,
                    utility=1.0,
                    rationale="echo input",
                )
            ],
        )

        logger.debug(f"EchoAgent result: {result}")

        return result, context


class UserInput(WorkflowAgent):
    """
    A processor that gets user input from the console or context and produces a TextArtifact.
    """

    # def __init__(self, target: Agent):
    #     # if target is instance, get its class; if it's a class, use it directly; otherwise raise error
    #     if isinstance(target, Agent):
    #         self.target = target.__class__
    #     elif isinstance(target, type) and issubclass(target, Agent):
    #         self.target = target
    #     else:
    #         raise ValueError("Invalid target for UserInputAgent")

    def run(self, step: int, artifact: TextArtifact, context: Context, console=None) -> tuple[int, AgentResult]:
        logger.debug(f"UserInputAgent received artifact: {artifact}")

        # ----------------------------------------------------------
        # Get user input
        # ----------------------------------------------------------
        user_input = context.control.data.pop("user_input", None)
        if user_input is None:
            if console is None:
                raise ValueError("UserInputAgent requires input in context.control.data['user_input'] when no console is provided")
            console.print("[bold green]user>[/bold green] ", end="")
            user_input = console.input()

        context.step = step + 1

        # if user_input.lower() == "exit":
        #     target = "exit"
        # else:
        #     target = self.target

        result = AgentResult(
            artifacts=[
                SymbolicArtifact(
                    id=f"{uuid4()}",
                    producer=self.id,
                    step=context.step,
                    content=user_input,
                )
            ],
            recommendations=[
                Recommendation(
                    target=self.target,
                    utility=1.0,
                    rationale="processing user input",
                )
            ],
        )

        return result, context




class ExitAgent(Agent):
    """
    A processor that signals the controller to stop execution when reached.
    """

    pass
