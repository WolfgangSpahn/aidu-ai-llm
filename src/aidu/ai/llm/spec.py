import pluggy

hookspec = pluggy.HookspecMarker("app")


class ToolSpec:
    @hookspec
    def get_tools(self) -> list[dict]:
        """
        Return tool schemas (OpenAI format)
        """

    @hookspec
    def call_tool(self, name: str, arguments: dict, state):
        """
        Execute tool and return updated state
        """