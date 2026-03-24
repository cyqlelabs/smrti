"""Council — governance system with meetings, proposals, and debates."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field

from smrti_town.config import (
    BUILDING_CATALOG,
    COUNCIL_MEETING_INTERVAL_HOURS,
    COUNCIL_ROLES,
)


@dataclass
class CouncilMember:
    name: str
    role: str  # "mayor", "sheriff", "superintendent", "doctor", "treasurer"
    domain: str  # "governance", "security", "education", "health", "finances"
    personality: str
    governing_style: str


@dataclass
class Proposal:
    action_type: str  # "build", "tax_change", "event", "policy"
    building_key: str | None
    description: str
    cost: int
    proposed_by: str  # role
    arguments: dict[str, str] = field(default_factory=dict)  # role -> argument text


@dataclass
class CouncilMeeting:
    meeting_id: str
    tick_number: int
    debate_transcript: list[dict]  # [{role, name, argument}]
    proposal: Proposal
    status: str = "pending"  # "pending", "approved", "rejected", "countered"


# ── Need-to-building mappings for fallback meeting generation ──────────

_DOMAIN_BUILDING_PRIORITIES: dict[str, list[str]] = {
    "governance": ["courthouse", "town_hall"],
    "security": ["constabulary", "jail", "fire_station"],
    "education": ["school", "library", "university"],
    "health": ["clinic", "hospital", "well", "water_tower"],
    "finances": ["trading_post", "market", "warehouse", "granary"],
}

_NEED_BUILDING_MAP: dict[str, list[str]] = {
    "housing": ["cottage", "house", "apartment", "inn"],
    "food": ["farm", "bakery", "butcher", "market", "granary"],
    "safety": ["constabulary", "jail", "fire_station", "courthouse"],
    "education": ["school", "library", "university"],
    "health": ["clinic", "hospital", "well", "water_tower"],
    "culture": ["park", "theater", "museum", "festival_grounds", "church"],
    "infrastructure": ["well", "water_tower", "warehouse", "trading_post"],
    "commerce": ["general_store", "bakery", "market", "tavern", "blacksmith"],
}

# Governing style templates for fallback debate generation.
_STYLE_ARGUMENTS: dict[str, str] = {
    "pragmatic": "We must prioritise what benefits the most citizens right now.",
    "cautious": "We should be careful with spending. Can we afford this?",
    "progressive": "This investment will pay off in the long term. We need progress.",
    "conservative": "Let us not rush. The current state of affairs is manageable.",
    "populist": "The people demand action. We must respond to their needs.",
}


class Council:
    """Town council: convenes meetings, debates proposals, approves/rejects."""

    def __init__(self, members: list[CouncilMember] | None = None) -> None:
        self.members: list[CouncilMember] = list(members or [])
        self.meetings: list[CouncilMeeting] = []
        self.last_meeting_tick: int = -COUNCIL_MEETING_INTERVAL_HOURS  # allow immediate first meeting

    # ── membership ──────────────────────────────────────────────────────

    def add_member(self, member: CouncilMember) -> None:
        # Replace existing member in the same role.
        self.members = [m for m in self.members if m.role != member.role]
        self.members.append(member)

    def remove_member(self, name: str) -> None:
        self.members = [m for m in self.members if m.name != name]

    def get_member_by_role(self, role: str) -> CouncilMember | None:
        for m in self.members:
            if m.role == role:
                return m
        return None

    # ── scheduling ──────────────────────────────────────────────────────

    def should_convene(self, tick_number: int) -> bool:
        """Returns True if enough time has passed since the last meeting and
        there are no unresolved pending meetings."""
        if self.get_pending_meeting() is not None:
            return False
        elapsed = tick_number - self.last_meeting_tick
        return elapsed >= COUNCIL_MEETING_INTERVAL_HOURS

    # ── fallback meeting generation ─────────────────────────────────────

    def generate_fallback_meeting(self, town_state: dict) -> CouncilMeeting:
        """Rule-based meeting when LLM is unavailable.

        *town_state* should contain:
            - population: int
            - treasury: int
            - existing_buildings: list[str]  (building_keys)
            - unmet_needs: dict[str, float]  (need_category -> urgency 0-1)
            - petitions: list[dict]
            - tick_number: int
        """
        tick_number = town_state.get("tick_number", 0)
        population = town_state.get("population", 0)
        treasury = town_state.get("treasury", 0)
        existing = set(town_state.get("existing_buildings", []))
        unmet = town_state.get("unmet_needs", {})
        petitions = town_state.get("petitions", [])

        # 1. Find highest-urgency need.
        proposal = self._pick_proposal(unmet, petitions, existing, population, treasury)

        # 2. Generate debate transcript from each council member.
        transcript: list[dict] = []
        for member in self.members:
            argument = self._generate_argument(member, proposal, town_state)
            transcript.append({
                "role": member.role,
                "name": member.name,
                "argument": argument,
            })
            proposal.arguments[member.role] = argument

        meeting = CouncilMeeting(
            meeting_id=uuid.uuid4().hex[:12],
            tick_number=tick_number,
            debate_transcript=transcript,
            proposal=proposal,
            status="pending",
        )
        self.meetings.append(meeting)
        self.last_meeting_tick = tick_number
        return meeting

    def _pick_proposal(
        self,
        unmet: dict[str, float],
        petitions: list[dict],
        existing: set[str],
        population: int,
        treasury: int,
    ) -> Proposal:
        """Select the best proposal based on urgency and feasibility."""

        # Check petitions with building suggestions first.
        for pet in sorted(petitions, key=lambda p: p.get("urgency", 0), reverse=True):
            bkey = pet.get("building_suggestion")
            if bkey and bkey in BUILDING_CATALOG and bkey not in existing:
                bdef = BUILDING_CATALOG[bkey]
                if bdef.unlock_population <= population and bdef.cost <= treasury:
                    return Proposal(
                        action_type="build",
                        building_key=bkey,
                        description=f"Build {bkey.replace('_', ' ')} ({pet.get('category', 'general')})",
                        cost=bdef.cost,
                        proposed_by="mayor",
                    )

        # Fall back to highest unmet need.
        if unmet:
            top_need = max(unmet, key=lambda k: unmet[k])
            candidates = _NEED_BUILDING_MAP.get(top_need, [])
            for bkey in candidates:
                if bkey in existing:
                    continue
                bdef = BUILDING_CATALOG.get(bkey)
                if not bdef:
                    continue
                if bdef.unlock_population > population:
                    continue
                # Check unlock building requirements.
                if bdef.unlock_buildings and not all(ub in existing for ub in bdef.unlock_buildings):
                    continue
                if bdef.cost <= treasury:
                    return Proposal(
                        action_type="build",
                        building_key=bkey,
                        description=f"Build {bkey.replace('_', ' ')} to address {top_need}",
                        cost=bdef.cost,
                        proposed_by="mayor",
                    )

        # No suitable building — propose tax adjustment if treasury is low.
        if treasury < 10000:
            return Proposal(
                action_type="tax_change",
                building_key=None,
                description="Raise business tax by 2% to replenish treasury",
                cost=0,
                proposed_by="treasurer",
            )

        # Default: community event to boost morale.
        return Proposal(
            action_type="event",
            building_key=None,
            description="Organise a town festival to boost citizen morale",
            cost=min(500, treasury // 10),
            proposed_by="mayor",
        )

    def _generate_argument(
        self,
        member: CouncilMember,
        proposal: Proposal,
        town_state: dict,
    ) -> str:
        """Generate a rule-based debate argument for a council member."""
        style_base = _STYLE_ARGUMENTS.get(
            member.governing_style,
            _STYLE_ARGUMENTS["pragmatic"],
        )
        treasury = town_state.get("treasury", 0)

        if proposal.action_type == "build":
            if member.role == "treasurer":
                if proposal.cost > treasury * 0.5:
                    return f"This will consume over half our treasury. {style_base}"
                return f"The cost of {proposal.cost} coins is within budget. {style_base}"
            if member.domain in ("security",) and proposal.building_key in (
                "constabulary", "jail", "fire_station",
            ):
                return f"As {member.role}, I strongly support this. Safety is paramount."
            if member.domain in ("education",) and proposal.building_key in (
                "school", "library", "university",
            ):
                return f"Education is the foundation of progress. I support this proposal."
            if member.domain in ("health",) and proposal.building_key in (
                "clinic", "hospital", "well", "water_tower",
            ):
                return f"The health of our citizens must come first. I endorse this."
            return style_base

        if proposal.action_type == "tax_change":
            if member.role == "treasurer":
                return "Our coffers need replenishing. A modest tax increase is prudent."
            if member.governing_style == "conservative":
                return "Higher taxes burden our citizens. We should cut spending instead."
            return style_base

        # Event or policy.
        return style_base

    # ── resolution ──────────────────────────────────────────────────────

    def approve(self, meeting_id: str) -> dict:
        """Approve a pending meeting's proposal.  Returns the proposal as dict."""
        meeting = self._find_meeting(meeting_id)
        if not meeting or meeting.status != "pending":
            return {}
        meeting.status = "approved"
        return {
            "action_type": meeting.proposal.action_type,
            "building_key": meeting.proposal.building_key,
            "description": meeting.proposal.description,
            "cost": meeting.proposal.cost,
        }

    def reject(self, meeting_id: str) -> None:
        meeting = self._find_meeting(meeting_id)
        if meeting and meeting.status == "pending":
            meeting.status = "rejected"

    def counter(self, meeting_id: str, building_key: str) -> dict:
        """Counter-propose with a different building.  Returns new proposal dict."""
        meeting = self._find_meeting(meeting_id)
        if not meeting or meeting.status != "pending":
            return {}
        bdef = BUILDING_CATALOG.get(building_key)
        if not bdef:
            return {}
        meeting.proposal = Proposal(
            action_type="build",
            building_key=building_key,
            description=f"Counter-proposal: build {building_key.replace('_', ' ')}",
            cost=bdef.cost,
            proposed_by="player",
            arguments=meeting.proposal.arguments,
        )
        meeting.status = "countered"
        return {
            "action_type": "build",
            "building_key": building_key,
            "description": meeting.proposal.description,
            "cost": bdef.cost,
        }

    # ── queries ─────────────────────────────────────────────────────────

    def get_pending_meeting(self) -> CouncilMeeting | None:
        for m in reversed(self.meetings):
            if m.status == "pending":
                return m
        return None

    def _find_meeting(self, meeting_id: str) -> CouncilMeeting | None:
        for m in self.meetings:
            if m.meeting_id == meeting_id:
                return m
        return None

    # ── serialization ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "members": [
                {
                    "name": m.name,
                    "role": m.role,
                    "domain": m.domain,
                    "personality": m.personality,
                    "governing_style": m.governing_style,
                }
                for m in self.members
            ],
            "meetings": [
                {
                    "meeting_id": mt.meeting_id,
                    "tick_number": mt.tick_number,
                    "debate_transcript": mt.debate_transcript,
                    "proposal": {
                        "action_type": mt.proposal.action_type,
                        "building_key": mt.proposal.building_key,
                        "description": mt.proposal.description,
                        "cost": mt.proposal.cost,
                        "proposed_by": mt.proposal.proposed_by,
                        "arguments": mt.proposal.arguments,
                    },
                    "status": mt.status,
                }
                for mt in self.meetings
            ],
            "last_meeting_tick": self.last_meeting_tick,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Council:
        members = [
            CouncilMember(
                name=m["name"],
                role=m["role"],
                domain=m["domain"],
                personality=m["personality"],
                governing_style=m["governing_style"],
            )
            for m in data.get("members", [])
        ]
        council = cls(members=members)
        council.last_meeting_tick = data.get("last_meeting_tick", 0)
        for mt_data in data.get("meetings", []):
            prop_data = mt_data.get("proposal", {})
            proposal = Proposal(
                action_type=prop_data.get("action_type", "event"),
                building_key=prop_data.get("building_key"),
                description=prop_data.get("description", ""),
                cost=prop_data.get("cost", 0),
                proposed_by=prop_data.get("proposed_by", "mayor"),
                arguments=prop_data.get("arguments", {}),
            )
            meeting = CouncilMeeting(
                meeting_id=mt_data.get("meeting_id", uuid.uuid4().hex[:12]),
                tick_number=mt_data.get("tick_number", 0),
                debate_transcript=mt_data.get("debate_transcript", []),
                proposal=proposal,
                status=mt_data.get("status", "pending"),
            )
            council.meetings.append(meeting)
        return council
