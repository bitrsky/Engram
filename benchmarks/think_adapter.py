"""
think_adapter.py — Create a ThinkFn backed by EchoCodeClient for benchmarks.

Wraps EchoCodeClient.prompt_collect() into a sync callable that satisfies
the engram ThinkFn protocol.

Usage::

    from benchmarks.think_adapter import create_think_fn

    think = create_think_fn("anthropic/claude-sonnet-4-20250514")
    result = think("Rank these documents ...", system="You are a judge.")
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_think_fn(model: str = "anthropic/claude-sonnet-4-20250514"):
    """Create a sync ThinkFn backed by EchoCodeClient.

    Args:
        model: LiteLLM model identifier.

    Returns:
        A callable ``(prompt, system="", **kw) -> Optional[str]``
        satisfying the engram ThinkFn protocol.
    """
    from echo_code import EchoCodeClient

    def think_fn(prompt: str, system: str = "", **kw) -> Optional[str]:
        # system prompt gets baked into EchoCodeClient at construction
        client = EchoCodeClient(
            model=model,
            system_prompt=system,
            headless=True,
        )

        coro = client.prompt_collect(prompt)

        # Bridge async → sync
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=kw.get("timeout", 120))
        else:
            return asyncio.run(coro)

    return think_fn
