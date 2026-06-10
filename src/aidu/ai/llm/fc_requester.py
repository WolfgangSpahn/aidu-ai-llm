# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.
#

"""
LLMFcRequester: An LLMRequester subclass that automatically exposes methods prefixed with 'fc_' as OpenAI function-call tools.
"""

import inspect
import re
from typing import get_origin, get_args
from pydantic import BaseModel

from aidu.ai.core.context import Context, Message
from aidu.ai.core.agent_result import AgentResult
from aidu.ai.llm.requester import LLMRequester


def parse_docstring(func):
    """Extracts parameter descriptions from a function's Google-style docstring."""
    docstring = func.__doc__ or ""
    param_descriptions = {}

    # Regex pattern for Google-style docstrings (e.g., "param name (type): description")
    pattern = re.findall(r"(\w+)\s*\((.*?)\):\s*(.+)", docstring)

    for name, _, desc in pattern:
        param_descriptions[name] = desc.strip()

    return docstring.strip(), param_descriptions


def get_openai_function_schema(func, make_all_required=False):
    """Dynamically extract function name, docstring, and parameter schema from function definition."""
    docstring = func.__doc__ or ""
    signature = inspect.signature(func)

    parameters = {}
    required_fields = []

    for name, param in signature.parameters.items():
        if name in ["self", "context"]:
            continue  # Ignore self and context

        annotation = param.annotation
        description = "No description available"

        # Extract description from docstring if available
        doc_lines = docstring.split("\n")
        for line in doc_lines:
            if line.strip().startswith(f"{name} ("):
                description = line.split(":", 1)[-1].strip()

        # Handle list of Pydantic models (e.g., list[Phrase])
        if get_origin(annotation) is list:
            item_type = get_args(annotation)[0]
            if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                parameters[name] = {"type": "array", "items": item_type.model_json_schema(), "description": description}  # Resolve Pydantic model
            else:
                parameters[name] = {"type": "array", "items": {"type": "string"}, "description": description}  # Assume list of strings if not a Pydantic model

        # Handle single Pydantic models (e.g., StudentInfo, Phrase)
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            parameters[name] = {**annotation.model_json_schema(), "description": description}  # Resolve full Pydantic schema

        # Handle basic types
        else:
            type_mapping = {str: "string", int: "integer", bool: "boolean", float: "number"}
            param_type = type_mapping.get(annotation, "object")
            parameters[name] = {"type": param_type, "description": description}

        if param.default == inspect.Parameter.empty or make_all_required:
            required_fields.append(name)

    return {
        "name": func.__name__,
        "description": docstring.strip().split("\n")[0] if docstring else func.__name__,  # Use the first line of the docstring as summary
        "parameters": {"type": "object", "properties": parameters, "required": required_fields},
    }

class LLMFcRequester(LLMRequester):
    """
    LLMRequester with automatic OpenAI function-call support.

    Methods prefixed with ``fc_`` are automatically:

    - discovered,
    - converted to OpenAI tool schemas,
    - registered as callable functions,
    - validated against a common contract.

    Example
    -------

    class MathTutor(WorkflowAgent, LLMFcRequester):

        def fc_solve(
            self,
            context: Context,
            expression: str,
        ) -> tuple[Message, Context]:
            ...

    """

    capability_specs: dict[str, type | object | None] = {}

    @classmethod
    def schema(cls, make_all_required: bool = False, prefix: str = "fc_") -> list[dict]:
        """
        Generate OpenAI tool schemas from fc_* methods.
        """
        functions = [
            func
            for name, func in inspect.getmembers(
                cls,
                predicate=inspect.isfunction,
            )
            if name.startswith(prefix)
        ]

        return [
            {
                "type": "function",
                "function": get_openai_function_schema(
                    func,
                    make_all_required=make_all_required,
                ),
            }
            for func in functions
        ]
    @classmethod
    def fnames(cls, prefix="fc_"):
        return [
            name
            for name, func in inspect.getmembers(
                cls,
                predicate=inspect.isfunction,
            )
            if name.startswith(prefix)
        ]

    def __init__(self, client, prompt_template=None, prompt_args=None, tools=None, capability_overrides=None, target: str = None):
        """
        Initialize requester and expose fc_* methods.
        """

        if tools is None:
            tools = self.schema()

        client.tools = tools

        super().__init__(
            client=client,
            prompt_template=prompt_template,
            prompt_args=prompt_args,
            tools=tools,
            target=target,
        )

        #
        # Register fc_* methods
        #
        for name, func in inspect.getmembers(
            self,
            predicate=inspect.ismethod,
        ):
            if name.startswith("fc_"):
                self._assert_fc_contract(name, func)
                self.register(
                    name,
                    self._wrap_fc_method(name, func),
                )

        #
        # Resolve capabilities
        #
        resolved = {}

        for name, provider in self.capability_specs.items():
            resolved[name] = provider() if isinstance(provider, type) else provider

        if capability_overrides:
            resolved.update(capability_overrides)

        self.capabilities = resolved

    @staticmethod
    def _assert_fc_contract(name: str, method) -> None:
        """
        Validate fc_* signature.
        """

        signature = inspect.signature(method)
        params = list(signature.parameters.values())

        assert params, f"{name} must accept 'context' as first parameter"

        assert params[0].name == "context", f"{name} must declare 'context' as first parameter"

        assert params[0].annotation is Context, f"{name} must annotate context as Context"

        for param in params:
            assert param.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ), f"{name} must not use *args or **kwargs"

        # Allow either (Message, Context) or (AgentResult, Context)
        allowed_returns = (tuple[Message, Context], tuple[AgentResult, Context])
        assert signature.return_annotation in allowed_returns, f"{name} must declare return type tuple[Message, Context] or tuple[AgentResult, Context]"

    @staticmethod
    def _wrap_fc_method(name: str, method):
        """
        Validate runtime return values.
        """

        def wrapped(*args, **kwargs):

            result = method(*args, **kwargs)

            assert isinstance(result, tuple) and len(result) == 2, f"{name} must return (message, context)"

            message, context = result

            assert isinstance(
                message,
                (dict, str, AgentResult),
            ), f"{name} must return str, Message-compatible dict, or AgentResult"

            assert isinstance(
                context,
                Context,
            ), f"{name} must return Context as second item"

            return result

        return wrapped

    @staticmethod
    def clean_message_for_storage(msg: dict) -> dict:
        """
        Keep only fields reusable in later
        chat-completion requests.
        """

        # Accept either a dict-like Message or an AgentResult model
        if isinstance(msg, AgentResult):
            return msg.model_dump()

        cleaned = {}

        if "role" in msg:
            cleaned["role"] = msg["role"]

        if msg.get("content"):
            cleaned["content"] = msg["content"]

        if msg.get("function_call"):
            cleaned["function_call"] = msg["function_call"]

        return cleaned
def _smoke_test():

    class DummyClient:
        def __init__(self):
            self.tools = None

    class DummyRequester(LLMFcRequester):

        prompt_template = "Smoke test"

        capability_specs = {
            "value": 42,
        }

        def fc_echo(
            self,
            context: Context,
            text: str,
        ) -> tuple[Message, Context]:
            """
            Echo text.

            Args:
                text (str): Text to echo.
            """
            return text, context

    # --------------------------------------------------
    # schema generation
    # --------------------------------------------------

    tools = DummyRequester.schema()

    assert len(tools) == 1

    tool = tools[0]

    assert tool["type"] == "function"
    assert tool["function"]["name"] == "fc_echo"

    properties = tool["function"]["parameters"]["properties"]

    assert "text" in properties
    assert properties["text"]["type"] == "string"

    # --------------------------------------------------
    # function names
    # --------------------------------------------------

    assert DummyRequester.fnames() == ["fc_echo"]

    # --------------------------------------------------
    # instance creation
    # --------------------------------------------------

    client = DummyClient()

    requester = DummyRequester(
        client=client,
        target="dummy",
    )

    # --------------------------------------------------
    # tools propagated to client
    # --------------------------------------------------

    assert client.tools is not None
    assert len(client.tools) == 1

    # --------------------------------------------------
    # capability resolution
    # --------------------------------------------------

    assert requester.capabilities["value"] == 42

    # --------------------------------------------------
    # function registration
    # --------------------------------------------------

    assert "fc_echo" in requester.function_lookup

    # --------------------------------------------------
    # wrapped execution
    # --------------------------------------------------

    ctx = Context()

    fn = requester.function_lookup["fc_echo"]

    message, ctx2 = fn(
        context=ctx,
        text="hello",
    )

    assert message == "hello"
    assert ctx2 is ctx

    print("LLMFcRequester smoke test passed.")

if __name__ == "__main__":
    _smoke_test()