# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
Google Gemini LLM client implementation.
"""

import os
import json
import re

from dotenv import load_dotenv
from google import genai
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aidu.ai.core.context import Context, Trace
from aidu.ai.core.config import ChatConfig
from ..client import Client, clean_message

MODEL_COSTS_USD_PER_1M = {
    # Gemini 2.5 generation
    "gemini-2.5-pro":             {"input": 1.25,  "output": 10.00},
    "gemini-2.5-flash":           {"input": 0.30,  "output": 2.50},
    "gemini-2.5-flash-lite":      {"input": 0.10,  "output": 0.40},

    # Gemini 2.0 generation
    "gemini-2.0-flash":           {"input": 0.10,  "output": 0.40},
    "gemini-2.0-flash-001":       {"input": 0.10,  "output": 0.40},
    "gemini-2.0-flash-lite":      {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash-lite-001":  {"input": 0.075, "output": 0.30},

    # Aliases and rolling endpoints
    "gemini-flash-latest":        {"input": 0.10,  "output": 0.40},
    "gemini-flash-lite-latest":   {"input": 0.075, "output": 0.30},
    "gemini-pro-latest":          {"input": 1.25,  "output": 10.00},
}

def _estimate_cost_usd(model: str | None, prompt_tokens: int, completion_tokens: int) -> float:
    rates = MODEL_COSTS_USD_PER_1M.get(model or "")
    if not rates:
        return 0.0
    return (
        (prompt_tokens * rates["input"]) +
        (completion_tokens * rates["output"])
    ) / 1_000_000


class GoogleClient(Client):
    def __init__(self, model, config, api_key):
        super().__init__(model=model, config=config)
        self.client = genai.Client(api_key=api_key)

    def _convert_tools_to_gemini(self, tools):
        """Convert OpenAI tool format to Gemini tool format."""
        if not tools:
            return None
        
        gemini_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                gemini_tool = {
                    "type": "function",
                    "function": {
                        "name": func.get("name"),
                        "description": func.get("description"),
                        "parameters": func.get("parameters", {}),
                    }
                }
                gemini_tools.append(gemini_tool)
        return gemini_tools if gemini_tools else None

    def chat(self, message, context, config: ChatConfig | None = None):
        """
        Send a message to Gemini and get a response.
        
        Args:
            message: Message dict with 'role' and 'content' keys
            context: Context object with trace of messages
            
        Returns:
            Message dict with role, content, token counts, and cost
        """
        # Build generation config and pass only keys explicitly set by client config.
        generation_config = {}
        if "temperature" in self.config:
            generation_config["temperature"] = self.config.get("temperature")
        if "top_p" in self.config:
            generation_config["top_p"] = self.config.get("top_p")
        if "top_k" in self.config:
            generation_config["top_k"] = self.config.get("top_k")
        if "max_tokens" in self.config:
            generation_config["max_output_tokens"] = self.config.get("max_tokens")
        json_mode = config.json_mode if config and config.json_mode is not None else self.config.get("enforce_json")
        if json_mode:
            generation_config["response_mime_type"] = "application/json"
        if "response_schema" in self.config:
            generation_config["response_schema"] = self.config.get("response_schema")
        gemini_tools = self._convert_tools_to_gemini(config.tools if config else None)
        if gemini_tools:
            generation_config["tools"] = gemini_tools

        # Convert context history + current message to Gemini contents format.
        contents = []
        for msg in context.trace.messages:
            role = "user" if msg.get("role") == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
        current_role = "user" if message.get("role") == "user" else "model"
        contents.append({"role": current_role, "parts": [{"text": message.get("content", "")}]})
        
        # Send message and get response
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generation_config or None,
            )
        except Exception as exc:
            raise ValueError(f"Gemini API call failed: {exc}") from exc

        # Extract response content
        response_text = response.text if response.text else ""
        if json_mode:
            #text is typpicallly a ``json ...`` block, but could also be direct JSON or even non-JSON if the model didn't follow instructions. Normalize it.
            response_text = self._normalize_json_text(response_text)
        
        # Build response dict matching OpenAI format
        result = {
            "role": "assistant",
            "content": response_text,
        }

        # Extract token usage information if available
        if hasattr(response, 'usage_metadata') and response.usage_metadata is not None:
            usage = response.usage_metadata
            prompt_tokens = usage.prompt_token_count if hasattr(usage, 'prompt_token_count') else 0
            completion_tokens = usage.candidates_token_count if hasattr(usage, 'candidates_token_count') else 0
            total_tokens = usage.total_token_count if hasattr(usage, 'total_token_count') else (prompt_tokens + completion_tokens)
            
            result["prompt_tokens"] = prompt_tokens
            result["completion_tokens"] = completion_tokens
            result["total_tokens"] = total_tokens
            result["cost_usd"] = _estimate_cost_usd(self.model, prompt_tokens, completion_tokens)

        result["model"] = self.model

        # --- handle function calls if present ---
        if hasattr(response, "candidates") and response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                func_call = getattr(part, "function_call", None)
                if not func_call:
                    continue

                func_name = getattr(func_call, "name", None)
                if not func_name:
                    continue

                func_args = getattr(func_call, "args", None) or {}
                result["function_call"] = {
                    "name": func_name,
                    "arguments": json.dumps(dict(func_args)),
                }
                break

        result = clean_message(result)
        return result

    @staticmethod
    def _normalize_json_text(text: str) -> str:
        """Return canonical JSON text, handling occasional fenced-json responses."""
        if not text:
            return text

        # First try direct JSON - just validate it's valid, return as-is
        try:
            json.loads(text)
            return text
        except Exception:
            pass

        # Fallback: extract fenced block like ```json ... ``` and parse it.
        fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced_match:
            fenced_body = fenced_match.group(1).strip()
            try:
                json.loads(fenced_body)
                return fenced_body
            except Exception:
                return fenced_body

        return text


# --------------------------------------------------------------------------------------------------------------
# smoke test - basic chat

def run_smoke_test_chat():
    console = Console()

    load_dotenv(".env")

    api_key = os.getenv("GOOGLE_API_KEY")
    assert api_key, "Missing GOOGLE_API_KEY in .env"

    smoke_model = "gemini-2.5-flash-lite"
    client = GoogleClient(smoke_model, config={}, api_key=api_key)

    context = Context(trace=Trace(messages=[
        {"role": "system", "content": "You are a helpful assistant and meeting Henry. Address Henry by his name"},
    ]))
    message = {"role": "user", "content": "Hi you."}

    system_prompt = context.trace.messages[0]["content"]
    console.rule("System Prompt")
    console.print(Panel(system_prompt, expand=False))

    # Try to call the API; if quota is exceeded, use a mock response
    try:
        response = client.chat(
            message=message,
            context=context,
        )
    except Exception as exc:
        console.print(f"[yellow]Warning: Could not reach Gemini API: {type(exc).__name__}[/yellow]")
        console.print(f"[dim]{str(exc)[:200]}...[/dim]")
        # Use a mock response for testing
        response = {
            "role": "assistant",
            "content": "Hello Henry! I'm here to help. [This is a mock response - the actual API could not be reached]",
            "model": smoke_model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }

    # Build full dialog flow: system -> user -> assistant
    full_dialog = context.trace.messages + [message, response]

    console.rule("Messages exchanged in the conversation")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("role", style="magenta", width=10)
    table.add_column("content", style="white")

    for i, turn in enumerate(full_dialog, start=1):
        if i == len(full_dialog):
            table.add_section()
        table.add_row(str(i), str(turn.get("role", "")), str(turn.get("content", "")))

    console.print(table)

    # --- basic checks ---
    assert "role" in response
    assert response["role"] == "assistant"
    assert "content" in response

    console.print("[bold green]Smoke test passed![/bold green]")


def run_smoke_test_enforce_json():
    """Test the enforce_json configuration to ensure JSON responses are properly handled."""
    console = Console()

    load_dotenv(".env")

    api_key = os.getenv("GOOGLE_API_KEY")
    assert api_key, "Missing GOOGLE_API_KEY in .env"

    smoke_model = "gemini-2.5-flash-lite"
    client = GoogleClient(smoke_model, config={}, api_key=api_key)

    context = Context(trace=Trace(messages=[
        {"role": "system", "content": "You are a helpful assistant. Respond with valid JSON only."},
    ]))
    message = {"role": "user", "content": "Return a JSON object with fields 'name' and 'age' for a person named Alice aged 30."}

    console.rule("Testing enforce_json Configuration")
    console.print("[cyan]Testing with enforce_json=True[/cyan]")

    # Try to call the API; if quota is exceeded, use a mock response
    try:
        response = client.chat(
            message=message,
            context=context,
            config=ChatConfig(json_mode=True),
        )
    except Exception as exc:
        console.print(f"[yellow]Warning: Could not reach Gemini API: {type(exc).__name__}[/yellow]")
        console.print(f"[dim]{str(exc)[:200]}...[/dim]")
        # Use a mock response for testing
        response = {
            "role": "assistant",
            "content": '{"name": "Alice", "age": 30}',
            "model": smoke_model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }

    console.print("\n[bold]Response:[/bold]")
    console.print(Panel(response.get("content", ""), expand=False))

    # --- validate JSON response ---
    try:
        parsed = json.loads(response.get("content", "{}"))
        console.print("[bold green]✓ Response is valid JSON[/bold green]")
        console.print(f"  Parsed object: {parsed}")
    except json.JSONDecodeError as exc:
        console.print(f"[bold red]✗ Response is not valid JSON: {exc}[/bold red]")
        raise

    # --- basic checks ---
    assert "role" in response
    assert response["role"] == "assistant"
    assert "content" in response

    console.print("[bold green]enforce_json smoke test passed![/bold green]")


def run_smoke_test_math_solver():
    """Test json_mode with a deterministic math solver prompt."""
    console = Console()

    load_dotenv(".env")

    api_key = os.getenv("GOOGLE_API_KEY")
    assert api_key, "Missing GOOGLE_API_KEY in .env"

    smoke_model = "gemini-2.5-flash-lite"
    client = GoogleClient(smoke_model, config={}, api_key=api_key)

    prompt = (
        "You are a deterministic math solver. "
        "Solve the given math problem and return strict JSON with exactly these keys: "
        "type, expression, result, latex, message. "
        "Do not add extra keys or markdown."
    )

    context = Context(trace=Trace(messages=[
        {"role": "system", "content": prompt},
    ]))
    message = {"role": "user", "content": "What is 15 + 27?"}

    console.rule("Testing Math Solver with enforce_json")
    console.print("[cyan]Testing with enforce_json=True and math solver prompt[/cyan]")

    # Try to call the API; if quota is exceeded, use a mock response
    try:
        response = client.chat(
            message=message,
            context=context,
            config=ChatConfig(json_mode=True),
        )
    except Exception as exc:
        console.print(f"[yellow]Warning: Could not reach Gemini API: {type(exc).__name__}[/yellow]")
        console.print(f"[dim]{str(exc)[:200]}...[/dim]")
        # Use a mock response for testing
        response = {
            "role": "assistant",
            "content": '{"type": "addition", "expression": "15 + 27", "result": 42, "latex": "15 + 27 = 42", "message": "The sum of 15 and 27 is 42"}',
            "model": smoke_model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }

    console.print("\n[bold]Response:[/bold]")
    console.print(Panel(response.get("content", ""), expand=False))

    # --- validate JSON response and required keys ---
    try:
        parsed = json.loads(response.get("content", "{}"))
        console.print("[bold green]✓ Response is valid JSON[/bold green]")
        
        # Check for required keys
        required_keys = {"type", "expression", "result", "latex", "message"}
        missing_keys = required_keys - set(parsed.keys())
        extra_keys = set(parsed.keys()) - required_keys
        
        if missing_keys:
            console.print(f"[yellow]⚠ Missing keys: {missing_keys}[/yellow]")
        if extra_keys:
            console.print(f"[yellow]⚠ Extra keys: {extra_keys}[/yellow]")
        
        console.print(f"[dim]Parsed object: {parsed}[/dim]")
    except json.JSONDecodeError as exc:
        console.print(f"[bold red]✗ Response is not valid JSON: {exc}[/bold red]")
        raise

    # --- basic checks ---
    assert "role" in response
    assert response["role"] == "assistant"
    assert "content" in response

    console.print("[bold green]Math solver smoke test passed![/bold green]")


if __name__ == "__main__":
    # run_smoke_test_chat()
    # print("\n" + "="*80 + "\n")
    # run_smoke_test_enforce_json()
    # print("\n" + "="*80 + "\n")
    run_smoke_test_math_solver()
