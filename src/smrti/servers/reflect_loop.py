"""Periodic background reflect() for all server modes."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Callable, Sequence

from smrti import Smrti
from smrti.servers.common import epoch_owed

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


def _has_pending(mem: Smrti) -> bool:
    """Whether the instance reports activity no epoch has consolidated yet."""
    ops = getattr(mem, "ops_since_reflect", None)
    return isinstance(ops, int) and ops > 0


def _evicted_with_pending(seen: list[Smrti], current: list[Smrti]) -> list[Smrti]:
    """The instances seen before that are no longer served and still hold activity.

    One whose owed epoch is still queued or running is left to it: its count
    is reset only when that epoch ends, so a pass in between would run a
    second epoch behind the first. Once the owed epoch ends, the server shows
    the instance again whenever it holds activity (see
    ``common.evicted_instances``).
    """
    live = {id(mem) for mem in current}
    return [
        mem for mem in seen
        if id(mem) not in live and _has_pending(mem) and not epoch_owed(mem)
    ]


async def run_reflect_loop(get_instances: Callable[[], Sequence[Smrti]]) -> None:
    """Periodically call reflect() on every active Smrti instance that was used.

    The activity count lives on the instance, so a server that evicts an
    instance from its cache between passes used to drop the count with it:
    what was done through that instance since its last epoch was never
    consolidated unless the space happened to be used again. The loop keeps
    the instances it saw on its last pass and consolidates any that left the
    cache with activity pending, holding on to one only until it has.

    Args:
        get_instances: callable returning the current list of Smrti instances.
    """
    if REFLECT_INTERVAL <= 0:
        return

    loop = asyncio.get_running_loop()
    seen: list[Smrti] = []
    while True:
        await asyncio.sleep(REFLECT_INTERVAL)
        current = list(get_instances())
        evicted = _evicted_with_pending(seen, current)
        for mem in (*current, *evicted):
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
        seen = current + [mem for mem in evicted if _has_pending(mem)]
