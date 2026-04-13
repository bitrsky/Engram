"""
think_adapter.py — ThinkFn implementation backed by EchoCodeClient.

Provides a ThinkFn-compatible callable that uses EchoCodeClient in headless
mode to process prompts. Used by benchmarks and any code that needs an LLM
backend for engram's ThinkFn protocol.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_think_fn(model: str = "anthropic/claude-haiku-3-5-20241022"):
    """Create an async ThinkFn backed by EchoCodeClient headless mode.

    Returns an async callable: (prompt, system, **kw) -> Optional[str]

    The returned function creates a fresh EchoCodeClient per call and
    supports multi-turn agent interaction (tool use).
    """
    from echo_code import EchoCodeClient

    async def think_fn(prompt: str, system: str = "", **kw) -> Optional[str]:
        client = EchoCodeClient(
            model=model,
            system_prompt=system,
            headless=True,
        )
        try:
            return await client.prompt_collect(prompt)
        except Exception as exc:
            logger.debug("think_fn failed: %s", exc)
            return None
        finally:
            await client.aclose()

    return think_fn
