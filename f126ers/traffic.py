"""Is this pass worth the battery? Derived, not guessed.

The lap-time optimiser prices energy against one lap. That is the wrong
objective when you are stuck behind someone: the cost of staying there is not
the tenth you lose this lap, it is *their pace instead of yours, every lap until
you get past*. Joules that break the deadlock are worth far more than lambda
says, and the single-lap solver will happily tell you to save them.

Four quantities decide it. Three are measurable:

    deficit    seconds per lap you are losing to the car ahead, from your own
               clean-air envelope minus this lap's speed in the bins where you
               were within following range. No aerodynamic model needed -- it
               is a speed difference you already recorded.
    laps_left  from totalLaps in the session packet.
    energy     what an attack costs, from the forward simulator: deploy flat out
               to the braking zone instead of to plan, and difference the two.

The fourth -- whether the move actually sticks -- depends on the other driver
and is not derivable from your telemetry. So it is inverted rather than guessed:

    p* = (energy * lambda) / (laps_left * deficit)

the success probability the attempt has to beat to pay for itself. That is exact
given the three measured quantities, and it is a more honest output than a
made-up probability: a break-even of 8% means attack, 140% means the pass cannot
repay its own energy even if it works every time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .optimiser import Plan, simulate
from .track import DS, Lap, TrackModel

FOLLOW_RANGE = 1.2  # s; inside this the car ahead is measurably costing you time
MIN_DEFICIT = 0.03  # s/lap; below this you are not actually being held up
ATTACK_BELOW = 0.35  # break-even probability under which attacking is clearly right
HOLD_ABOVE = 0.80  # ... and over which it clearly is not



@dataclass
class PassCall:
    """The economics of attacking the car ahead, all figures derived."""

    deficit: float  # s/lap lost while following
    laps_left: int
    energy: float  # J an attack would cost above plan
    cost: float  # s of lap time that energy is worth (energy * lambda)
    prize: float  # s saved over the stint if the pass sticks
    breakeven: float  # success probability needed to pay for itself
    verdict: str  # "attack" | "marginal" | "hold" | "none"
    detail: str
    advice: str

    @property
    def worth_it(self) -> bool:
        return self.verdict == "attack"


def following_deficit(track: TrackModel, lap: Lap) -> tuple[float, np.ndarray]:
    """Seconds lost this lap to the car ahead, and where.

    Compares this lap against the driver's own best speed in each bin -- their
    demonstrated clean-air pace on this circuit -- but only in bins where they
    were within following range. Somewhere else on the lap being slow is a
    driving error, not traffic, and must not be charged to the car ahead.
    """
    close = lap.delta_front > 0.0
    close &= lap.delta_front < FOLLOW_RANGE
    if not close.any() or track.v_obs.size != lap.v.size:
        return 0.0, close

    clean = np.maximum(track.v_obs, lap.v)  # best seen here, never below today
    lost = np.zeros_like(lap.v)
    # Time through a bin is ds/v, so the loss is the difference of reciprocals.
    lost[close] = DS * (1.0 / np.maximum(lap.v[close], 1.0)
                        - 1.0 / np.maximum(clean[close], 1.0))
    return float(max(lost.sum(), 0.0)), close


def attack_energy(track: TrackModel, plan: Plan, v_now: float, bin_now: int,
                  charge: float) -> tuple[float, float]:
    """Energy and time gain from deploying flat out to the next braking zone.

    Returns (extra joules over plan, seconds gained). The time gain is what
    closes the gap, so comparing it against delta_front says whether the move is
    geometrically on at all -- before any question of whether it is worth it.
    """
    end = _next_braking_bin(track, bin_now)
    if end <= bin_now:
        return 0.0, 0.0

    t_plan, e_plan = _segment(track, plan.u, bin_now, end, v_now)
    t_hard, e_hard = _segment(track, None, bin_now, end, v_now)  # None = flat out
    extra = min(max(e_hard - e_plan, 0.0), max(charge, 0.0))
    return extra, max(t_plan - t_hard, 0.0)


def _segment(track: TrackModel, u: np.ndarray | None, lo: int, hi: int,
             v0: float) -> tuple[float, float]:
    """Time and energy over bins [lo, hi) starting at v0.

    simulate() always runs a whole lap from the start line, so its bin times do
    not describe a stretch entered mid-lap at a given speed. This integrates the
    same dynamics over just that stretch. u=None deploys the taper-limited
    maximum, which is the attack case.
    """
    n = track.n
    a, c = track.a, track.c
    v = max(min(v0, float(track.v_env[lo % n])), 1.0)
    t = e = 0.0
    for k in range(lo, hi):
        i = k % n
        p_k = float(track.p_max(v)) if u is None else float(u[i])
        e_kin = 0.5 * v ** 2 + DS * (
            a * (track.p_ice[i] + p_k) / v - track.drag[i] * v ** 2 - c)
        v_next = min(np.sqrt(2.0 * max(e_kin, 0.5)), float(track.v_env[(i + 1) % n]))
        dt = 2.0 * DS / (v + v_next)
        t += dt
        e += p_k * dt
        v = v_next
    return t, e


def _next_braking_bin(track: TrackModel, start: int) -> int:
    """First bin ahead where the envelope drops sharply: the next braking zone."""
    n = track.n
    v = track.v_env
    for k in range(1, n):
        i = (start + k) % n
        if v[i] < v[(i - 1) % n] - 1.5:
            return start + k
    return start + n // 4


def pass_call(track: TrackModel, plan: Plan, lam: float, deficit: float,
              delta_front: float, laps_left: int, v_now: float, bin_now: int,
              charge: float) -> PassCall:
    """Should you spend the battery attacking, or bank it?"""
    energy, gain = attack_energy(track, plan, v_now, bin_now, charge)
    cost = energy * lam
    prize = max(laps_left, 0) * max(deficit, 0.0)

    if deficit < MIN_DEFICIT or laps_left <= 0 or energy <= 0:
        return PassCall(deficit, laps_left, energy, cost, prize, float("inf"),
                        "none", "not being held up", "")

    breakeven = cost / prize if prize > 0 else float("inf")
    # Can the move physically get you alongside? Your own deployment buys `gain`
    # seconds; the tow adds more. Going dv faster over distance d at speed v
    # saves about d*dv/v^2 -- so the tow is converted into seconds here rather
    # than left as a dimensionless fudge factor, and it is measured from your
    # own laps when there is data for it.
    tow_dv = track.tow_gain
    run = _next_braking_bin(track, bin_now) - bin_now
    tow_gain = (run * DS) * tow_dv / max(v_now, 1.0) ** 2 if run > 0 else 0.0
    reachable = (gain + tow_gain) >= delta_front

    if not reachable:
        verdict = "hold"
        detail = (f"attacking gains {gain:.2f}s and the tow another "
                  f"{tow_gain:.2f}s, but you are {delta_front:.2f}s back — "
                  f"not enough to get alongside")
        advice = ("Bank it and set up the next lap: use the exit before this "
                  "straight instead, so you arrive with more speed.")
    elif breakeven <= ATTACK_BELOW:
        verdict = "attack"
        detail = (f"{energy/1e6:.2f} MJ costs {cost:.2f}s of lap time; "
                  f"clearing them saves {prize:.1f}s over {laps_left} laps")
        advice = (f"Attack — this only has to work {breakeven*100:.0f}% of the "
                  f"time to pay for itself. Override out of the last corner.")
    elif breakeven >= HOLD_ABOVE:
        verdict = "hold"
        detail = (f"{energy/1e6:.2f} MJ costs {cost:.2f}s but clearing them only "
                  f"saves {prize:.1f}s over {laps_left} laps")
        advice = ("Hold station and save it — with this little of the race left "
                  "the energy is worth more as lap time than as a pass.")
    else:
        verdict = "marginal"
        detail = (f"break-even {breakeven*100:.0f}% — {energy/1e6:.2f} MJ for "
                  f"{prize:.1f}s if it sticks")
        advice = ("Marginal: worth it if they are defending badly or you have "
                  "the tyre advantage, otherwise wait for a better run.")

    return PassCall(deficit, laps_left, energy, cost, prize, breakeven,
                    verdict, detail, advice)
