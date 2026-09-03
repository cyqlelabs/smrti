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


def _was_used(mem: Smrti) -> bool:
    """Whether the space saw any memory operation since its last epoch.

    An epoch is one unit of decay, and the unit has to mean something. Run on
    a timer alone it meant "one minute of server uptime": a memory lost 98%
    of its attention in an hour whether or not anyone spoke to the agent,
    and lost nothing while the server was down. Gating on use makes an epoch
    a unit of the agent's own activity — an idle space, served or not, does
    not age — which is also what a consolidation cycle is for: there is
    nothing to consolidate when nothing happened.

    Instances that do not report activity (test doubles, older subclasses)
    are consolidated every interval as before.
    """
    ops = getattr(mem, "ops_since_reflect", None)
    return not isinstance(ops, int) or ops > 0


async def run_reflect_loop(get_instances: Callable[[], Sequence[Smrti]]) -> None:
    """Periodically call reflect() on every active Smrti instance that was used.

    Args:
        get_instances: callable returning the current list of Smrti instances.
    """
    if REFLECT_INTERVAL <= 0:
        return

    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(REFLECT_INTERVAL)
        for mem in get_instances():
            if not _was_used(mem):
                continue
            try:
                await loop.run_in_executor(None, mem.reflect)
            except Exception:
                logger.exception(
                    "reflect failed for tenant=%s space=%s",
                    mem.tenant_id,
                    mem.write_space,
                )
