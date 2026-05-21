# src/aidu/ai/llm/plugin.py
from dotenv import load_dotenv
import os
import pluggy
from .clients.openai import OpenAIClient
from aidu.ai.core.context import Context, Trace, State, Control
from aidu.ai.core.config import ChatConfig
import asyncio
import pluggy

from aidu.ai.llm.actors.mathTutor import MathTutor
from aidu.ai.llm.solver.MathSolver import MathSolver
from aidu.ai.core.hookspecs import hookimpl, HookSpecs

class LLMPlugin:

    @hookimpl
    def get_agents(self):

        return [
            MathTutor,
            MathSolver,
        ]


plugin = LLMPlugin()

# -------------------------------------------------------------------
# Smoke Test
# -------------------------------------------------------------------

async def _smoke_test():

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    assert api_key

    pm = pluggy.PluginManager("aidu")

    pm.add_hookspecs(HookSpecs)

    pm.register(LLMPlugin())

    results = pm.hook.get_agents()

    agents = []

    for group in results:
        agents.extend(group)

    client = OpenAIClient(
        "gpt-4o-mini",
        config={},
        api_key=api_key,
    )

    

    message = {
        "role": "user",
        "content": "What is 2 + 2?",
    }

    for agent_cls in agents:

        print(f"\n- {agent_cls.__name__}")

        try:

            agent = agent_cls(client=client)

            context = Context( trace=Trace(
                messages=agent.build_system_prompt()
            ))

            response, context = agent.chat(
                message=message,
                context=context,
                chat_config=ChatConfig(json_mode=issubclass(agent_cls, MathSolver)),
            )

            print(response)

        except Exception as e:

            print(f"Failed: {e}")

if __name__ == "__main__":

    asyncio.run(_smoke_test())