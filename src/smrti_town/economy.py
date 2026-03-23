"""EconomyManager: wallets, transactions, prices, workplace assignment."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("smrti_town.economy")

# ── Base costs ──────────────────────────────────────────────────────
FOOD_COST = 10
RENT_COST = 5
GOODS_COST = 15
INCOME_EMPLOYED = 8     # per tick-hour
INCOME_ODD_JOBS = 2     # per tick-hour (unemployed)
INCOME_ELDER = 3        # per tick-hour (retired)
STARTING_BALANCE = 100

# Place types that qualify for economic actions
FOOD_PLACE_TYPES = frozenset({"market", "farm", "public"})
SHOP_PLACE_TYPES = frozenset({"shop", "market", "public"})
WORK_PLACE_TYPES = frozenset({"public", "other", "shop", "market", "farm"})

# ── Price dynamics ──────────────────────────────────────────────────
# More shops of same type -> modifier decreases; many buyers -> increases.
COMPETITION_DECAY = 0.1      # per additional shop of same type
DEMAND_GROWTH = 0.05         # per buyer in last cycle
PRICE_MODIFIER_MIN = 0.5
PRICE_MODIFIER_MAX = 2.0


def _clamp_price(value: float) -> float:
    return max(PRICE_MODIFIER_MIN, min(PRICE_MODIFIER_MAX, value))


@dataclass
class Transaction:
    """A single economic transaction."""

    tick: int
    agent: str
    action: str        # "work", "buy_food", "buy_goods", "pay_rent", "study"
    amount: int        # positive = earned, negative = spent
    place: str = ""
    balance_after: int = 0

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "agent": self.agent,
            "action": self.action,
            "amount": self.amount,
            "place": self.place,
            "balance_after": self.balance_after,
        }


class EconomyManager:
    """Manages town economy: wallets, transactions, prices."""

    def __init__(self) -> None:
        self.wallets: dict[str, int] = {}           # agent_name -> balance
        self.workplaces: dict[str, str] = {}         # agent_name -> place_name
        self.incomes: dict[str, float] = {}          # agent_name -> income_per_tick_hour
        self.price_modifiers: dict[str, float] = {}  # place_name -> modifier
        self.transaction_log: list[Transaction] = []
        self._buyer_counts: dict[str, int] = {}      # place_name -> buyers this cycle
        self._tick: int = 0

    # ── Initialisation ──────────────────────────────────────────────

    def init_agent(
        self,
        agent_name: str,
        starting_balance: int = STARTING_BALANCE,
    ) -> None:
        """Register an agent with a starting wallet balance."""
        if agent_name not in self.wallets:
            self.wallets[agent_name] = starting_balance

    def remove_agent(self, agent_name: str) -> None:
        """Unregister an agent (death / departure)."""
        self.wallets.pop(agent_name, None)
        self.workplaces.pop(agent_name, None)
        self.incomes.pop(agent_name, None)

    # ── Workplace ───────────────────────────────────────────────────

    def assign_workplace(
        self,
        agent_name: str,
        place_name: str,
        income: float = INCOME_EMPLOYED,
    ) -> None:
        """Assign an agent to a workplace with a given income rate."""
        self.workplaces[agent_name] = place_name
        self.incomes[agent_name] = income

    def unassign_workplace(self, agent_name: str) -> None:
        """Remove workplace assignment (layoff / retirement)."""
        self.workplaces.pop(agent_name, None)
        self.incomes.pop(agent_name, None)

    def get_income_rate(self, agent_name: str, life_stage: str) -> float:
        """Return the effective income rate for an agent.

        Children earn nothing.  Elders earn INCOME_ELDER.
        Employed adults earn their assigned rate; unemployed adults
        earn INCOME_ODD_JOBS.
        """
        if life_stage in ("infant", "child"):
            return 0.0
        if life_stage == "elder":
            return INCOME_ELDER
        # Adult
        if agent_name in self.incomes:
            return self.incomes[agent_name]
        return INCOME_ODD_JOBS

    # ── Tick processing ─────────────────────────────────────────────

    def set_tick(self, tick: int) -> None:
        self._tick = tick

    def process_work_tick(
        self,
        agent_name: str,
        delta_hours: float,
        life_stage: str = "adult",
    ) -> int:
        """Credit an agent for hours worked.  Returns amount earned."""
        rate = self.get_income_rate(agent_name, life_stage)
        if rate <= 0:
            return 0
        earned = int(rate * delta_hours)
        if earned <= 0:
            return 0
        self.wallets[agent_name] = self.wallets.get(agent_name, 0) + earned
        self._log(agent_name, "work", earned, self.workplaces.get(agent_name, ""))
        return earned

    # ── Purchases ───────────────────────────────────────────────────

    def buy_food(
        self,
        agent_name: str,
        place_name: str,
        base_cost: int = FOOD_COST,
    ) -> bool:
        """Attempt to buy food at *place_name*.  Returns True on success."""
        cost = self._effective_cost(place_name, base_cost)
        if not self.can_afford(agent_name, cost):
            return False
        self.wallets[agent_name] -= cost
        self._buyer_counts[place_name] = self._buyer_counts.get(place_name, 0) + 1
        self._log(agent_name, "buy_food", -cost, place_name)
        return True

    def buy_goods(
        self,
        agent_name: str,
        place_name: str,
        base_cost: int = GOODS_COST,
    ) -> bool:
        """Attempt to buy goods at *place_name*.  Returns True on success."""
        cost = self._effective_cost(place_name, base_cost)
        if not self.can_afford(agent_name, cost):
            return False
        self.wallets[agent_name] -= cost
        self._buyer_counts[place_name] = self._buyer_counts.get(place_name, 0) + 1
        self._log(agent_name, "buy_goods", -cost, place_name)
        return True

    def pay_study(
        self,
        agent_name: str,
        place_name: str,
        tuition: int = 0,
    ) -> bool:
        """Pay tuition (0 for public school).  Returns True on success."""
        if tuition <= 0:
            return True
        if not self.can_afford(agent_name, tuition):
            return False
        self.wallets[agent_name] -= tuition
        self._log(agent_name, "study", -tuition, place_name)
        return True

    # ── Daily expenses ──────────────────────────────────────────────

    def pay_daily_expenses(
        self,
        agent_name: str,
        has_home: bool = True,
    ) -> bool:
        """Deduct daily rent/living costs.  Returns True if fully paid.

        If the agent cannot afford the full cost, they pay what they can
        (wallet goes to 0) and the method returns False.
        """
        cost = RENT_COST if has_home else 0
        if cost <= 0:
            return True
        balance = self.wallets.get(agent_name, 0)
        paid = min(cost, balance)
        self.wallets[agent_name] = balance - paid
        if paid > 0:
            self._log(agent_name, "pay_rent", -paid, "")
        return paid >= cost

    # ── Price dynamics ──────────────────────────────────────────────

    def update_prices(self, building_counts: dict[str, int]) -> None:
        """Recalculate price modifiers based on competition and demand.

        *building_counts* maps a place_type (or place_name) to the number
        of buildings of that type in the town.
        """
        for place_name, mod in list(self.price_modifiers.items()):
            # Competition: more buildings of same type lowers prices
            count = building_counts.get(place_name, 1)
            competition_factor = 1.0 - COMPETITION_DECAY * max(0, count - 1)

            # Demand: more buyers raises prices
            buyers = self._buyer_counts.get(place_name, 0)
            demand_factor = 1.0 + DEMAND_GROWTH * buyers

            new_mod = _clamp_price(competition_factor * demand_factor)
            self.price_modifiers[place_name] = new_mod

        # Reset buyer counts for next cycle
        self._buyer_counts.clear()

    def set_price_modifier(self, place_name: str, modifier: float) -> None:
        self.price_modifiers[place_name] = _clamp_price(modifier)

    # ── Queries ─────────────────────────────────────────────────────

    def get_wallet(self, agent_name: str) -> int:
        return self.wallets.get(agent_name, 0)

    def can_afford(self, agent_name: str, cost: int) -> bool:
        return self.wallets.get(agent_name, 0) >= cost

    def get_workplace(self, agent_name: str) -> str | None:
        return self.workplaces.get(agent_name)

    def recent_transactions(self, limit: int = 50) -> list[dict]:
        """Return the most recent transactions as dicts."""
        return [t.to_dict() for t in self.transaction_log[-limit:]]

    # ── Serialisation ───────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "wallets": dict(self.wallets),
            "workplaces": dict(self.workplaces),
            "incomes": {k: round(v, 2) for k, v in self.incomes.items()},
            "price_modifiers": {
                k: round(v, 2) for k, v in self.price_modifiers.items()
            },
            "recent_transactions": self.recent_transactions(20),
        }

    # ── Internals ───────────────────────────────────────────────────

    def _effective_cost(self, place_name: str, base_cost: int) -> int:
        mod = self.price_modifiers.get(place_name, 1.0)
        return max(1, int(base_cost * mod))

    def _log(self, agent: str, action: str, amount: int, place: str) -> None:
        tx = Transaction(
            tick=self._tick,
            agent=agent,
            action=action,
            amount=amount,
            place=place,
            balance_after=self.wallets.get(agent, 0),
        )
        self.transaction_log.append(tx)
        # Cap log size to prevent unbounded growth
        if len(self.transaction_log) > 5000:
            self.transaction_log = self.transaction_log[-2500:]
