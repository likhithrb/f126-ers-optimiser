"""Planning across laps, not just within one.

Everything else in this tool solves a single lap with the battery ending where
it started. That constraint is a proxy, and a good one: if every lap were
identical it would be exactly optimal, because the way to spend a fixed budget
over identical opportunities is to spend the same amount on each.

Laps are not identical. The race ends, and charge you cross the final line with
is worth nothing. You pit, and a lap spent partly at pit-lane speed is a poor
place for energy. So the real problem is:

    choose e_1 .. e_N        charge at the end of each remaining lap
    minimise sum of T_k      total time over the rest of the race
    subject to 0 <= e_k <= capacity,  and e_N = 0 at the flag

The trick that makes this cheap is the **energy curve**: for one lap, how does
lap time fall as you spend more energy? Sample it with a handful of solves and
you can price any schedule without solving anything again. The curve is convex
-- diminishing returns -- so the optimum equalises its slope across laps, and
that slope is lambda. Multi-lap planning is the same "equalise the marginal
value" rule as within a lap, one level up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .optimiser import solve
from .track import TrackModel

N_LEVELS = 61  # charge levels for the across-laps dynamic program
PIT_VALUE = 0.45  # a lap you pit on is worth this much of a normal lap's energy


@dataclass
class EnergyCurve:
    """Lap time as a function of energy spent, for the current conditions."""

    spend: np.ndarray  # J
    time: np.ndarray  # s
    lam: np.ndarray  # s/J, the slope at each point

    def time_at(self, e: float) -> float:
        return float(np.interp(e, self.spend, self.time))

    def price_at(self, e: float) -> float:
        return float(np.interp(e, self.spend, self.lam))


@dataclass
class StintPlan:
    targets: list  # charge to finish each remaining lap on, J
    curve: EnergyCurve
    note: str = ""

    @property
    def this_lap(self) -> float:
        return self.targets[0] if self.targets else 0.0


def energy_curve(track: TrackModel, harvest: np.ndarray, v0: float,
                 soc: float, points: int = 5) -> EnergyCurve:
    """Samples lap time against energy spent, by solving at several budgets.

    A handful of solves, then interpolation. Solving once per candidate schedule
    would be hundreds of solves a lap; this is five, and the curve is smooth
    enough that interpolating between them costs nothing real.
    """
    cap = track.capacity
    budgets = np.linspace(0.0, min(soc + float(harvest.sum()), cap * 1.5), points)
    spends, times = [], []
    for b in budgets:
        # e_target = start + harvest - budget, i.e. "spend exactly b this lap".
        target = float(np.clip(soc + harvest.sum() - b, 0.0, cap))
        plan = solve(track, soc, harvest, v0, e_target=target)
        spends.append(plan.energy_used)
        times.append(plan.lap_time)
    spends = np.asarray(spends)
    times = np.asarray(times)
    order = np.argsort(spends)
    spends, times = spends[order], times[order]
    # Slope of the curve = seconds per joule = lambda. Negative of the gradient
    # because time falls as spending rises.
    # Duplicate spends (two budgets the solver could not tell apart) make
    # np.gradient divide by a zero step. Keep one point per distinct spend.
    keep = np.concatenate(([True], np.diff(spends) > 1.0))
    spends, times = spends[keep], times[keep]
    lam = np.zeros_like(spends)
    if spends.size > 1:
        lam = -np.gradient(times, spends)
    return EnergyCurve(spends, times, np.maximum(lam, 0.0))


def plan_stint(track: TrackModel, curve: EnergyCurve, soc: float,
               harvest_per_lap: float, laps_left: int,
               pit_in: int | None = None) -> StintPlan:
    """Charge to finish each remaining lap on, by dynamic programming over laps.

    State is the charge at a lap boundary, discretised. Small enough to solve
    exactly: 26 levels x however many laps are left, and each step is a lookup
    on the energy curve rather than a fresh optimisation.
    """
    cap = track.capacity
    laps_left = max(int(laps_left), 1)
    levels = np.linspace(0.0, cap, N_LEVELS)

    # value[j] = best total time from here to the flag, starting with levels[j].
    # At the flag, charge is worthless -- so no bonus for arriving with any, and
    # that alone is what makes the plan run the battery down at the end.
    value = np.zeros(N_LEVELS)
    choice = np.zeros((laps_left, N_LEVELS), dtype=np.int16)

    for k in range(laps_left - 1, -1, -1):
        # Stage k is the k-th lap from now: the backward induction starts at
        # the last lap (k = laps_left-1, no future beyond it) and works back to
        # k = 0, which the forward walk then drives first. Getting this the
        # wrong way round applies the pit weighting to the wrong lap and is
        # invisible on a coarse grid.
        weight = PIT_VALUE if (pit_in is not None and k == pit_in) else 1.0
        nxt = np.empty((N_LEVELS, N_LEVELS))
        # spend[j, m] = energy used going from level j to level m this lap.
        spend = levels[:, None] + harvest_per_lap - levels[None, :]
        # Multiply, do not divide: on a lap spent partly in the pit lane a joule
        # buys only `weight` of what it buys on a green lap, so the lookup is the
        # *reduced* effective spend. Dividing made the pit lap look like the most
        # productive place on the whole stint.
        t = np.interp(np.maximum(spend, 0.0) * weight, curve.spend, curve.time)
        nxt = np.where(spend >= 0, t, np.inf) + value[None, :]
        best = np.argmin(nxt, axis=1)
        choice[k] = best
        value = np.take_along_axis(nxt, best[:, None], axis=1)[:, 0]

    # Walk the policy forward from where the battery actually is.
    j = int(np.clip(np.searchsorted(levels, soc), 0, N_LEVELS - 1))
    targets = []
    for k in range(laps_left):
        j = int(choice[k, j])
        targets.append(float(levels[j]))

    note = ""
    if laps_left <= 3:
        note = f"running the battery down over the last {laps_left} laps"
    elif pit_in is not None:
        note = f"holding energy back from the pit lap ({pit_in} laps away)"
    return StintPlan(targets, curve, note)


@dataclass
class StintState:
    """Keeps the curve between laps so it is not rebuilt from scratch."""

    curve: EnergyCurve | None = None
    plan: StintPlan | None = None
    _laps: int = field(default=0)

    def update(self, track: TrackModel, harvest: np.ndarray, v0: float,
               soc: float, laps_left: int, pit_in: int | None = None
               ) -> StintPlan | None:
        """Rebuilds the schedule. Returns None outside a race of known length."""
        if laps_left <= 0:
            return None
        self.curve = energy_curve(track, harvest, v0, soc)
        self.plan = plan_stint(track, self.curve, soc, float(harvest.sum()),
                               laps_left, pit_in)
        self._laps = laps_left
        return self.plan
