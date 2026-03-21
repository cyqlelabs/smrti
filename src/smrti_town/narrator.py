"""Converts structured Actions into natural language strings for Smrti ingestion."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from smrti_town.config import (
    ACTION_EAT,
    ACTION_INTERACT,
    ACTION_MOVE,
    ACTION_PROPOSE,
    ACTION_REPRODUCE,
    ACTION_SLEEP,
    ACTION_STUDY,
    ACTION_TALK,
    ACTION_WAIT,
    ACTION_WANDER,
    ACTION_WORK,
)

if TYPE_CHECKING:
    from smrti_town.agent import Action, Agent


class Narrator:
    """Converts structured actions into natural language for Smrti ingestion."""

    def narrate_action(self, agent: Agent, action: Action) -> str:
        """Generate a natural-language narrative for an agent's action."""
        name = agent.name
        loc = agent.location
        target = action.target
        dialogue = action.dialogue

        if action.type == ACTION_MOVE:
            return self._narrate_move(name, loc, target)
        if action.type == ACTION_TALK:
            return self._narrate_talk(name, target, loc, dialogue)
        if action.type == ACTION_EAT:
            return self._narrate_eat(name, loc)
        if action.type == ACTION_SLEEP:
            return self._narrate_sleep(name, loc)
        if action.type == ACTION_WORK:
            return self._narrate_work(name, loc)
        if action.type == ACTION_STUDY:
            return self._narrate_study(name, loc)
        if action.type == ACTION_INTERACT:
            return self._narrate_interact(name, loc, dialogue)
        if action.type == ACTION_WANDER:
            return self._narrate_wander(name, loc, target)
        if action.type == ACTION_PROPOSE:
            return self._narrate_propose(name, target, loc)
        if action.type == ACTION_REPRODUCE:
            return self._narrate_reproduce(name, target)
        if action.type == ACTION_WAIT:
            return self._narrate_wait(name, loc)

        return f"{name} did something at {loc}."

    def narrate_conversation(
        self,
        speaker: str,
        listener: str,
        location: str,
        content: str,
    ) -> dict[str, str]:
        """Generate epistemic wrappers for a conversation.

        Returns dict with keys 'place', 'speaker', 'listener' — the narrative
        string to remember in each space.
        """
        return {
            "place": f"{speaker} said to {listener}: '{content}'",
            "speaker": f"I told {listener}: '{content}'",
            "listener": f"{speaker} told me at {location}: '{content}'",
        }

    def narrate_sporadic(self, description: str) -> str:
        return description

    def narrate_milestone(self, detail: str) -> str:
        return detail

    def narrate_birthday(self, agent_name: str, age: int) -> str:
        return f"Today is {agent_name}'s birthday! They turned {age}."

    def narrate_death(self, agent_name: str) -> str:
        return f"{agent_name} has passed away."

    def narrate_birth(self, child_name: str, parent_a: str, parent_b: str) -> str:
        return f"A child named {child_name} was born to {parent_a} and {parent_b}."

    def narrate_relationship(self, agent_a: str, agent_b: str, state: str) -> str:
        if state == "married":
            return f"{agent_a} and {agent_b} got married."
        if state == "romantic":
            return f"{agent_a} and {agent_b} have become romantically involved."
        return f"{agent_a} and {agent_b} are now {state}s."

    # ── Private narrators ────────────────────────────────────────────

    def _narrate_move(self, name: str, from_loc: str, to_loc: str | None) -> str:
        templates = [
            f"{name} walked from {from_loc} to {to_loc}.",
            f"{name} headed over to {to_loc}.",
            f"{name} left {from_loc} and went to {to_loc}.",
        ]
        return random.choice(templates)

    def _narrate_talk(self, name: str, target: str | None, loc: str, dialogue: str) -> str:
        if not target:
            return f"{name} muttered to themselves at {loc}."
        if dialogue:
            return f"{name} said to {target} at {loc}: \"{dialogue}\""
        templates = [
            f"{name} chatted with {target} at {loc}.",
            f"{name} had a conversation with {target} at {loc}.",
        ]
        return random.choice(templates)

    def _narrate_eat(self, name: str, loc: str) -> str:
        templates = [
            f"{name} had a meal at {loc}.",
            f"{name} ate something at {loc}.",
            f"{name} grabbed a bite to eat at {loc}.",
        ]
        return random.choice(templates)

    def _narrate_sleep(self, name: str, loc: str) -> str:
        templates = [
            f"{name} fell asleep at {loc}.",
            f"{name} went to sleep at {loc}.",
            f"{name} rested at {loc}.",
        ]
        return random.choice(templates)

    def _narrate_work(self, name: str, loc: str) -> str:
        templates = [
            f"{name} worked diligently at {loc}.",
            f"{name} focused on their tasks at {loc}.",
            f"{name} put in a productive shift at {loc}.",
        ]
        return random.choice(templates)

    def _narrate_study(self, name: str, loc: str) -> str:
        templates = [
            f"{name} studied at {loc}.",
            f"{name} read and learned something new at {loc}.",
            f"{name} spent time learning at {loc}.",
        ]
        return random.choice(templates)

    def _narrate_interact(self, name: str, loc: str, dialogue: str) -> str:
        if dialogue:
            return f"{name} at {loc}: {dialogue}"
        return f"{name} interacted with something at {loc}."

    def _narrate_wander(self, name: str, from_loc: str, to_loc: str | None) -> str:
        if to_loc:
            return f"{name} wandered from {from_loc} toward {to_loc}."
        return f"{name} wandered around {from_loc}."

    def _narrate_propose(self, name: str, target: str | None, loc: str) -> str:
        if target:
            return f"{name} proposed to {target} at {loc}."
        return f"{name} thought about proposing."

    def _narrate_reproduce(self, name: str, target: str | None) -> str:
        if target:
            return f"{name} and {target} decided to start a family."
        return f"{name} thought about starting a family."

    def _narrate_wait(self, name: str, loc: str) -> str:
        templates = [
            f"{name} waited quietly at {loc}.",
            f"{name} relaxed at {loc}.",
            f"{name} took a moment to themselves at {loc}.",
        ]
        return random.choice(templates)
