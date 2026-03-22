"""Fire-and-forget Smrti extraction for town engine events."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smrti import Smrti
    from smrti_town.llm import LLMClient

logger = logging.getLogger("smrti_town.extractor")


def _get_upstream(llm_client: "LLMClient | None") -> str:
    from smrti.servers import config as cfg
    if llm_client is None:
        return cfg.EXTRACT_URL.rstrip("/")
    base = llm_client.settings.base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base or cfg.EXTRACT_URL.rstrip("/")


def _get_model(llm_client: "LLMClient | None") -> str:
    from smrti.servers import config as cfg
    if cfg.EXTRACT_MODEL:
        return cfg.EXTRACT_MODEL
    return llm_client.settings.model if llm_client else ""


async def _run_extraction(
    episode_id: str,
    content: str,
    mem: "Smrti",
    llm_client: "LLMClient | None",
) -> None:
    from smrti.extraction.extract import extract_and_link_serialized
    from smrti.servers import config as cfg
    try:
        await extract_and_link_serialized(
            episode_id=episode_id,
            content=content,
            mem=mem,
            auth="",
            model=_get_model(llm_client),
            upstream=_get_upstream(llm_client),
            source="user",
            mode=cfg.EXTRACT_MODE,
        )
    except Exception as exc:
        logger.debug("Extraction failed for episode %s: %s", episode_id, exc)


def fire_extraction(
    episode_id: str,
    content: str,
    mem: "Smrti",
    llm_client: "LLMClient | None",
    bg_tasks: set,
) -> None:
    """Schedule extraction as a fire-and-forget background task."""
    from smrti.servers import config as cfg
    if not cfg.EXTRACT or not episode_id:
        return
    task = asyncio.create_task(_run_extraction(episode_id, content, mem, llm_client))
    bg_tasks.add(task)
    task.add_done_callback(bg_tasks.discard)
