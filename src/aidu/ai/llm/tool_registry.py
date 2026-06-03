# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
Tool registry for mapping and executing LLM function calls to Python methods.
Handles function discovery, signature normalization, and safe execution.
"""

import inspect
import json
import logging

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Maps LLM function calls to Python methods.

    Responsibilities:
    - discover functions (fc_*)
    - normalize signatures
    - execute safely
    """

    def __init__(self):
        self._tools = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_from_instance(self, instance):
        """
        Register all fc_* methods from an object.
        """

        for name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if not name.startswith("fc_"):
                continue

            self._tools[name] = self._wrap(method)

    # ------------------------------------------------------------------
    # Signature normalization (your key idea)
    # ------------------------------------------------------------------

    def _wrap(self, method):
        """
        Normalize different function signatures into one callable.
        """

        signature = inspect.signature(method)
        params = list(signature.parameters.keys())

        if params == ["arguments", "context"]:
            return lambda args, context: method(args, context)

        elif "context" in params:
            return lambda args, context: method(context=context, **args)

        else:
            logger.error(f"Invalid signature for {method.__name__}: {params}")
            raise Exception(f"Invalid signature: {method.__name__}")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, function_call, context):
        """
        Execute a function call from the LLM.
        """

        name = function_call.get("name")

        if name not in self._tools:
            logger.error(f"Unknown function: {name}")
            return context

        try:
            args = json.loads(function_call.get("arguments", "{}"))
        except Exception:
            logger.error(f"Invalid arguments for {name}")
            args = {}

        try:
            return self._tools[name](args, context)
        except Exception as e:
            logger.error(f"Error executing {name}: {e}")
            return context
