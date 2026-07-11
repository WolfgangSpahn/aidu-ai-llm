# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
# src/aidu/ai/core/config.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AskConfig:
    # Response behavior
    json_mode: bool = False

    # Returns route message style
    route_mode: bool = False

    # Sampling
    temperature: float | None = None
    max_tokens: int | None = None

    # Tool calling
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None

    # Vendor escape hatch
    vendor_config: dict[str, Any] | None = None


