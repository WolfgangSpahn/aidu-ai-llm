# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
# src/aidu/ai/llm/plugin.py
import logging

from dotenv import load_dotenv
import os
import pluggy
from .clients.openai import OpenAIClient
from aidu.ai.core.context import Context, Trace
from aidu.ai.core.config import AskConfig
import asyncio

from aidu.ai.llm.assistants.mathAssistent_ass import MathAssistent
from aidu.ai.llm.solver.MathSolver import MathSolver
from aidu.ai.core.hookspecs import hookimpl, HookSpecs
from aidu.support.filesystem.search import find_up

logger = logging.getLogger(__name__)


class LLMPlugin:
    @hookimpl
    def get_assistants(self):

        return [
            MathAssistent,
            MathSolver,
        ]


plugin = LLMPlugin()


# -------------------------------------------------------------------
# Smoke Test
# -------------------------------------------------------------------


async def _smoke_test():

    env_path = find_up(".env")
    logger.info("Loading environment variables from %s", env_path)
    load_dotenv(env_path)

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key

    pm = pluggy.PluginManager("aidu")
    pm.add_hookspecs(HookSpecs)
    pm.register(LLMPlugin())

    results = pm.hook.get_assistants()

    assistants = []

    for group in results:
        assistants.extend(group)

    client = OpenAIClient(
        "gpt-4o-mini",
        config={},
        api_key=api_key,
    )

    message = {
        "role": "user",
        "content": "What is 2 + 2?",
    }

    for agent_cls in assistants:
        print(f"\n- {agent_cls.__name__}")

        try:
            agent = agent_cls(client=client)

            context = Context(trace=Trace(messages=agent.build_system_prompt()))

            response, context = agent.ask(
                message=message,
                context=context,
                ask_config=AskConfig(json_mode=issubclass(agent_cls, MathSolver)),
            )

            print(response)

        except Exception as e:
            print(f"Failed: {e}")


if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console)],
    )

    asyncio.run(_smoke_test())
