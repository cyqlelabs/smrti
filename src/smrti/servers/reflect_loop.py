"""Periodic background reflect() for all server modes."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Callable, Sequence

from smrti import Smrti

logger = logging.getLogger("smrti.reflect_loop")

try:
    REFLECT_INTERVAL = int(os.environ.get("SMRTI_REFLECT_INTERVAL", "60"))
except ValueError:
    REFLECT_INTERVAL = 60


async def run_reflect_loop(get_instances: Callable[[], Sequence[Smrti]]) -> None:
    """Periodically call reflect() on all active Smrti instances.

    Args:
        get_instances: callable returning the current list of Smrti instances.
    """
    if REFLECT_INTERVAL <= 0:
        return

    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(REFLECT_INTERVAL)
        for mem in get_instances():
            try:
                await loop.run_in_executor(None, mem.reflect)
            except Exception:
                logger.exception(
                    "reflect failed for tenant=%s space=%s",
                    mem.tenant_id,
                    mem.write_space,
                )
