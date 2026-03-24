"""EconomyManager — treasury, wallets, taxation, commerce, and building costs."""

from __future__ import annotations

import math

from smrti_town.config import (
    BUILDING_CATALOG,
    COUNCIL_SALARY,
    ENTREPRENEURSHIP_COMMERCE_SKILL,
    ENTREPRENEURSHIP_SAVINGS_THRESHOLD,
    FOOD_COST,
    GOODS_COST,
    HOURS_PER_DAY,
    HOURS_PER_YEAR,
    INCOME_ELDER,
    INCOME_EMPLOYED,
    INCOME_ODD_JOBS,
    RENT_COST,
    STARTING_TREASURY,
    STARTING_WALLET,
    TAX_RATES_DEFAULT,
    BuildingDef,
)

# Citizen purchase costs by item type.
_ITEM_COSTS: dict[str, int] = {
    "food": FOOD_COST,
    "rent": RENT_COST,
    "goods": GOODS_COST,
}


class EconomyManager:
    """Central economy: treasury, citizen wallets, taxation, and commerce."""

    def __init__(
        self,
        treasury: int = STARTING_TREASURY,
        tax_rates: dict[str, float] | None = None,
    ) -> None:
        self.treasury: int = treasury
        self.tax_rates: dict[str, float] = dict(tax_rates or TAX_RATES_DEFAULT)
        self.wallets: dict[str, int] = {}
        self.maintenance_ledger: dict[str, int] = {}
        # Running totals for the current reporting period.
        self._tax_collected: int = 0
        self._maintenance_paid: int = 0
        self._salaries_paid: int = 0
        self._commerce_revenue: int = 0

    # ── citizen registration ────────────────────────────────────────────

    def register_citizen(self, name: str, starting_wallet: int = STARTING_WALLET) -> None:
        if name not in self.wallets:
            self.wallets[name] = starting_wallet

    def remove_citizen(self, name: str) -> None:
        self.wallets.pop(name, None)

    # ── building registration ───────────────────────────────────────────

    def register_building(self, place_name: str, building_def: BuildingDef) -> None:
        self.maintenance_ledger[place_name] = building_def.maintenance

    def remove_building(self, place_name: str) -> None:
        self.maintenance_ledger.pop(place_name, None)

    # ── taxation ────────────────────────────────────────────────────────

    def collect_taxes(
        self,
        citizens: list,
        buildings: list,
        delta_hours: float,
    ) -> int:
        """Collect property, business, and income taxes proportional to *delta_hours*.

        *citizens* — objects with ``.name``, ``.wallet``, ``.home``, ``.workplace``.
        *buildings* — Place objects with ``.building_key``.
        Returns total taxes collected this tick.
        """
        year_fraction = delta_hours / HOURS_PER_YEAR
        total = 0

        # Property tax: per housing building, scaled to year fraction.
        prop_rate = self.tax_rates.get("property", 0.0)
        for b in buildings:
            bdef = BUILDING_CATALOG.get(getattr(b, "building_key", None) or "", None)
            if bdef and bdef.provides_housing:
                tax = int(bdef.cost * prop_rate * year_fraction)
                total += tax

        # Business tax: per commercial/industrial building with revenue.
        biz_rate = self.tax_rates.get("business", 0.0)
        for b in buildings:
            bdef = BUILDING_CATALOG.get(getattr(b, "building_key", None) or "", None)
            if bdef and bdef.revenue_per_hour > 0:
                annual_revenue = bdef.revenue_per_hour * HOURS_PER_YEAR
                tax = int(annual_revenue * biz_rate * year_fraction)
                total += tax

        # Income tax: percentage of each citizen's wallet growth is handled
        # at earn-time via citizen_earn.  Here we collect a flat per-employed
        # citizen contribution so the treasury grows proportionally.
        income_rate = self.tax_rates.get("income", 0.0)
        for c in citizens:
            if not getattr(c, "alive", True):
                continue
            stage = getattr(c, "life_stage", "adult")
            if stage not in ("adult", "elder"):
                continue
            has_job = getattr(c, "workplace", None) is not None
            base = INCOME_EMPLOYED if has_job else INCOME_ODD_JOBS
            if stage == "elder":
                base = INCOME_ELDER
            tax = int(base * delta_hours * income_rate)
            name = getattr(c, "name", "")
            if name in self.wallets:
                deducted = min(tax, self.wallets[name])
                self.wallets[name] -= deducted
                total += deducted

        self.treasury += total
        self._tax_collected += total
        return total

    # ── maintenance ─────────────────────────────────────────────────────

    def pay_maintenance(self, delta_hours: float) -> int:
        """Deduct building maintenance from treasury.  Returns total paid."""
        year_fraction = delta_hours / HOURS_PER_YEAR
        total = 0
        for _place, annual_cost in self.maintenance_ledger.items():
            cost = int(annual_cost * year_fraction)
            total += cost
        self.treasury -= total
        self._maintenance_paid += total
        return total

    # ── salaries ────────────────────────────────────────────────────────

    def pay_salaries(self, council_members: list, delta_hours: float) -> int:
        """Pay council member salaries from treasury.

        *council_members* — objects with ``.name`` attribute.
        Returns total paid.
        """
        day_fraction = delta_hours / HOURS_PER_DAY
        total = 0
        for member in council_members:
            salary = int(COUNCIL_SALARY * day_fraction)
            name = getattr(member, "name", "")
            if name in self.wallets:
                self.wallets[name] += salary
            total += salary
        self.treasury -= total
        self._salaries_paid += total
        return total

    # ── commerce ────────────────────────────────────────────────────────

    def process_commerce(
        self,
        buildings: list,
        citizens: list,
        delta_hours: float,
    ) -> int:
        """Businesses generate revenue.  A portion goes to citizen workers,
        the rest flows to treasury via business tax (already counted in
        collect_taxes).  Returns total gross revenue generated.
        """
        total = 0
        pop = max(1, len([c for c in citizens if getattr(c, "alive", True)]))
        # Revenue scales with sqrt(population) — diminishing returns.
        pop_factor = math.sqrt(pop) / math.sqrt(10)
        pop_factor = max(0.5, min(pop_factor, 3.0))

        for b in buildings:
            bdef = BUILDING_CATALOG.get(getattr(b, "building_key", None) or "", None)
            if not bdef or bdef.revenue_per_hour <= 0:
                continue
            revenue = int(bdef.revenue_per_hour * delta_hours * pop_factor)
            total += revenue

        # 60% goes to treasury as business income, 40% distributed to workers.
        treasury_share = int(total * 0.6)
        worker_share = total - treasury_share
        self.treasury += treasury_share
        self._commerce_revenue += total

        # Distribute worker share among employed citizens.
        employed = [
            c for c in citizens
            if getattr(c, "alive", True)
            and getattr(c, "workplace", None) is not None
        ]
        if employed and worker_share > 0:
            per_worker = max(1, worker_share // len(employed))
            for c in employed:
                name = getattr(c, "name", "")
                if name in self.wallets:
                    self.wallets[name] += per_worker

        return total

    # ── citizen income ──────────────────────────────────────────────────

    def citizen_earn(
        self,
        name: str,
        delta_hours: float,
        employed: bool = True,
        skill_level: float = 0.0,
    ) -> int:
        """Add income to citizen wallet.  Skill level gives a bonus up to +50%.
        Returns amount earned (before tax — tax is collected separately).
        """
        base = INCOME_EMPLOYED if employed else INCOME_ODD_JOBS
        bonus = 1.0 + min(skill_level, 1.0) * 0.5
        earned = int(base * delta_hours * bonus)
        if name in self.wallets:
            self.wallets[name] += earned
        return earned

    # ── citizen spending ────────────────────────────────────────────────

    def citizen_buy(self, name: str, item_type: str) -> bool:
        """Deduct cost from citizen wallet.  Returns True if affordable."""
        cost = _ITEM_COSTS.get(item_type, 0)
        if cost <= 0:
            return True
        if name not in self.wallets:
            return False
        if self.wallets[name] < cost:
            return False
        self.wallets[name] -= cost
        # Purchase revenue goes partly back to treasury (simulating merchant tax).
        self.treasury += int(cost * 0.1)
        return True

    # ── building affordability ──────────────────────────────────────────

    def can_afford_building(self, building_key: str) -> bool:
        bdef = BUILDING_CATALOG.get(building_key)
        if not bdef:
            return False
        return self.treasury >= bdef.cost

    def deduct_building_cost(self, building_key: str) -> bool:
        bdef = BUILDING_CATALOG.get(building_key)
        if not bdef:
            return False
        if self.treasury < bdef.cost:
            return False
        self.treasury -= bdef.cost
        return True

    # ── checks ──────────────────────────────────────────────────────────

    def check_bankruptcy(self) -> bool:
        """Returns True if treasury is at or below zero."""
        return self.treasury <= 0

    def check_entrepreneurship(
        self,
        citizen_name: str,
        citizen_skills: dict[str, float],
    ) -> bool:
        """Returns True if citizen has enough savings and commerce skill to
        open a business."""
        wallet = self.wallets.get(citizen_name, 0)
        commerce = citizen_skills.get("commerce", 0.0)
        return (
            wallet >= ENTREPRENEURSHIP_SAVINGS_THRESHOLD
            and commerce >= ENTREPRENEURSHIP_COMMERCE_SKILL
        )

    # ── reporting ───────────────────────────────────────────────────────

    def reset_period_stats(self) -> dict[str, int]:
        """Reset and return period stats."""
        stats = {
            "tax_collected": self._tax_collected,
            "maintenance_paid": self._maintenance_paid,
            "salaries_paid": self._salaries_paid,
            "commerce_revenue": self._commerce_revenue,
        }
        self._tax_collected = 0
        self._maintenance_paid = 0
        self._salaries_paid = 0
        self._commerce_revenue = 0
        return stats

    # ── serialization ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "treasury": self.treasury,
            "tax_rates": dict(self.tax_rates),
            "wallets": dict(self.wallets),
            "maintenance_ledger": dict(self.maintenance_ledger),
            "stats": {
                "tax_collected": self._tax_collected,
                "maintenance_paid": self._maintenance_paid,
                "salaries_paid": self._salaries_paid,
                "commerce_revenue": self._commerce_revenue,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> EconomyManager:
        em = cls(
            treasury=data.get("treasury", STARTING_TREASURY),
            tax_rates=data.get("tax_rates"),
        )
        em.wallets = dict(data.get("wallets", {}))
        em.maintenance_ledger = dict(data.get("maintenance_ledger", {}))
        stats = data.get("stats", {})
        em._tax_collected = stats.get("tax_collected", 0)
        em._maintenance_paid = stats.get("maintenance_paid", 0)
        em._salaries_paid = stats.get("salaries_paid", 0)
        em._commerce_revenue = stats.get("commerce_revenue", 0)
        return em
