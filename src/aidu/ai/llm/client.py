# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
    OpenAI client wrapper providing utilities for interacting with OpenAI's API.

    This module includes:
    - LLMClient: Wrapper around OpenAI's API for chat completions
    - clean_message: Utility function for recursive cleaning of message objects
    - JSON handling helpers for robust API interaction
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import json

def clean_message(obj):
    """
    Recursively remove:
    - None
    - empty lists []
    - empty dicts {}
    """
    if isinstance(obj, dict):
        cleaned = {
            k: clean_message(v)
            for k, v in obj.items()
            if v is not None
        }

        # remove empty containers
        return {
            k: v
            for k, v in cleaned.items()
            if v not in ({}, [])
        }

    elif isinstance(obj, list):
        cleaned = [
            clean_message(v)
            for v in obj
            if v is not None
        ]

        return [v for v in cleaned if v not in ({}, [])]

    return obj
class LLMClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def chat(
        self,
        model,
        messages,
        tools=None,
        response_format=None,
    ):
        kwargs = {
            "model": model,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools

        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)

        msg = response.choices[0].message.model_dump()
        
        # --- normalize tool call ---
        if "tool_calls" in msg and msg["tool_calls"]:
            tool = msg["tool_calls"][0]
            msg["function_call"] = {
                "name": tool["function"]["name"],
                "arguments": tool["function"]["arguments"],
            }

        msg = clean_message(msg)
        return msg
    
# --------------------------------------------------------------------------------------------------------------
# smoke test - basic chat

def run_smoke_test_function_call():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Missing OPENAI_API_KEY in .env"

    client = LLMClient(api_key)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one short sentence."}
    ]

    msg = client.chat(
        model="gpt-4o-mini",
        messages=messages
    )

    print("\n--- Smoke Test Result ---")
    print(json.dumps(msg, indent=2))

    # --- basic checks ---
    assert "role" in msg
    assert msg["role"] == "assistant"
    assert "content" in msg

    print("\n✅ Smoke test passed!")


if __name__ == "__main__":
    run_smoke_test_function_call()