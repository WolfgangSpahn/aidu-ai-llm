# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.

"""
Helpers for running agent workflows from plain user text.

The core ``Agent`` contract stays artifact-oriented:

    run(artifact, context, agents=...) -> (AgentResult, Context)

This module provides the small adapter layer needed by smoke tests and web
chat endpoints: create the initial ``TextArtifact``, prepare the system prompt
for LLM-backed agents, optionally follow recommendations, and extract a
user-visible reply string.
"""

from __future__ import annotations

import inspect

from aidu.ai.core.agent_result import AgentResult
from aidu.ai.core.artifacts import Artifact, TextArtifact
from aidu.ai.core.config import AskConfig
from aidu.ai.core.context import Context, Trace
from aidu.ai.core.recommendation import Recommendation
from aidu.ai.llm.agent import Agent


def agent_map(agents: list[Agent]) -> dict[type[Agent], Agent]:
    """Return agents keyed by their concrete class."""
    return {agent.__class__: agent for agent in agents}


def find_agent(agents: list[Agent], agent_cls: type[Agent]) -> Agent:
    """Find an agent instance by exact class or subclass compatibility."""
    for agent in agents:
        if isinstance(agent, agent_cls):
            return agent
    raise KeyError(f"No agent instance found for {agent_cls.__name__!r}.")


def prepare_agent_context(agent: Agent, context: Context | None = None, prompt_params: dict | None = None) -> Context:
    """
    Ensure a context exists and has a system prompt when the agent can build one.
    """
    context = context or Context()
    has_system_prompt = bool(context.trace.messages and context.trace.messages[0].get("role") == "system")
    if not has_system_prompt and hasattr(agent, "build_system_prompt"):
        context.trace = Trace(messages=agent.build_system_prompt(prompt_params))  # type: ignore[attr-defined]
    return context


def select_recommendation(result: AgentResult) -> Recommendation | None:
    """Choose the highest-utility recommendation, if any."""
    if not result.recommendations:
        return None
    return max(result.recommendations, key=lambda recommendation: recommendation.utility)


def select_artifact(result: AgentResult) -> Artifact:
    """
    Choose the artifact to pass to the next recommended agent.

    Function-call routes typically put the routed artifact first. This also
    avoids trailing empty text artifacts that may be attached as model metadata.
    """
    if not result.artifacts:
        raise ValueError("AgentResult has no artifacts to route.")
    return result.artifacts[0]


def result_text(result: AgentResult) -> str:
    """Extract a compact user-visible text from an AgentResult."""
    parts = [artifact.content for artifact in result.artifacts if isinstance(artifact.content, str) and artifact.content]
    return "\n".join(parts)


def run_agent(
    agent: Agent,
    *,
    artifact: Artifact,
    context: Context,
    agents: list[Agent],
    ask_config: AskConfig | None = None,
) -> tuple[AgentResult, Context]:
    """Run an agent, passing optional LLM ask configuration when supported."""
    signature = inspect.signature(agent.run)
    if "ask_config" in signature.parameters:
        return agent.run(artifact=artifact, context=context, agents=agents, ask_config=ask_config)
    return agent.run(artifact=artifact, context=context, agents=agents)


def run_agent_text_turn(
    *,
    starting_agent: Agent,
    user_text: str,
    context: Context | None = None,
    agents: list[Agent] | None = None,
    prompt_params: dict | None = None,
    ask_config: AskConfig | None = None,
    max_hops: int = 0,
) -> tuple[AgentResult, Context]:
    """
    Run one user-text turn through an agent workflow.

    ``max_hops=0`` preserves the current smoke-test behavior: run only the
    starting agent. Increase it to follow recommendations to other agents.
    """
    context = context or Context()
    artifact = TextArtifact(producer="user", step=context.step, content=user_text)
    return run_agent_artifact_turn(
        starting_agent=starting_agent,
        artifact=artifact,
        context=context,
        agents=agents,
        prompt_params=prompt_params,
        ask_config=ask_config,
        max_hops=max_hops,
    )


def run_agent_artifact_turn(
    *,
    starting_agent: Agent,
    artifact: Artifact,
    context: Context | None = None,
    agents: list[Agent] | None = None,
    prompt_params: dict | None = None,
    ask_config: AskConfig | None = None,
    max_hops: int = 0,
) -> tuple[AgentResult, Context]:
    """
    Run one artifact turn through an agent workflow.

    This is the contract-level adapter: callers decide whether the input is a
    ``TextArtifact``, ``AppletArtifact``, or another artifact type.
    """
    agents = agents or [starting_agent]
    context = prepare_agent_context(starting_agent, context=context, prompt_params=prompt_params)
    artifact.step = context.step

    result, context = run_agent(
        starting_agent,
        artifact=artifact,
        context=context,
        agents=agents,
        ask_config=ask_config,
    )

    for _ in range(max_hops):
        recommendation = select_recommendation(result)
        if recommendation is None or recommendation.target is None:
            break
        next_agent = find_agent(agents, recommendation.target)
        artifact = select_artifact(result)
        result, context = run_agent(
            next_agent,
            artifact=artifact,
            context=context,
            agents=agents,
            ask_config=ask_config,
        )

    return result, context


def run_agent_artifact_chat_turn(
    *,
    starting_agent: Agent,
    artifact: Artifact,
    context: Context | None = None,
    agents: list[Agent] | None = None,
    prompt_params: dict | None = None,
    ask_config: AskConfig | None = None,
    max_hops: int = 0,
) -> tuple[str, Context]:
    """
    Run one artifact turn and return the web-chat friendly ``(reply, context)``.
    """
    result, context = run_agent_artifact_turn(
        starting_agent=starting_agent,
        artifact=artifact,
        context=context,
        agents=agents,
        prompt_params=prompt_params,
        ask_config=ask_config,
        max_hops=max_hops,
    )
    return result_text(result), context


def run_agent_chat_turn(
    *,
    starting_agent: Agent,
    user_text: str,
    context: Context | None = None,
    agents: list[Agent] | None = None,
    prompt_params: dict | None = None,
    max_hops: int = 0,
) -> tuple[str, Context]:
    """
    Run one agent turn and return the web-chat friendly ``(reply, context)``.
    """
    result, context = run_agent_text_turn(
        starting_agent=starting_agent,
        user_text=user_text,
        context=context,
        agents=agents,
        prompt_params=prompt_params,
        max_hops=max_hops,
    )
    return result_text(result), context
