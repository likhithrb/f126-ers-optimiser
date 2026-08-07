"""Time-loss attribution: turn a lap into one ranked, quantified verdict.

Three separate quantities, deliberately not mixed:

1. Allocation loss (this lap) = simulated actual lap - optimal lap, both on the
   energy the driver actually had. This is measured, not estimated, by running
   both deployment profiles through the same simulator with the same speed
   envelope, so the driving line and braking points cancel out.

2. Harvest loss (this lap) = optimal lap on the energy harvested - optimal lap
   on the energy the driver's own best braking has recovered in those zones.
   Also measured, by re-solving.

3. Sustainability cost (next lap) = lambda x the charge deficit carried over.
   A different lap's seconds, so it is reported in its own pool.

Within (1), the shadow price lambda and the local marginal value g(s) rank the
symptoms -- deploying where g < lambda, failing to deploy where g > lambda --
and the measured loss is apportioned across them by those weights. The ranking
comes from the economics; the total comes from the simulator. Pricing a whole
block of energy at the marginal price would overstate it, because lambda is by
definition the value of the *last* joule, and energy has diminishing returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .optimiser import (Plan, cap_harvest, simulate, solve,
                        time_gain_per_joule)
from .track import DS, Lap, TrackModel
from .traffic import FOLLOW_RANGE

LOW_SOC_FRAC = 0.03  # charge below this counts as depleted
DEPLOY_EPS = 1.0e4  # W, ignore deployment differences smaller than this
MIN_REPORT = 0.01  # s, don't bother the driver below this


@dataclass
class Issue:
    name: str
    cost: float  # seconds
    where: tuple[float, float]  # start/end distance in metres
    energy: float  # joules involved
    detail: str
    advice: str
    pool: str = "lap"  # "lap" = seconds lost this lap, "next" = next lap

    def __str__(self) -> str:
        tag = "" if self.pool == "lap" else " (next lap)"
        return f"{self.cost:+.2f}s{tag}  {self.name}: {self.detail}"


@dataclass
class LapReport:
    lap_num: int
    lap_time: float
    sim_time: float  # the same lap reproduced by the model
    optimal_time: float
    ers_loss: float  # seconds recoverable by re-allocating the energy you had
    harvest_loss: float  # extra seconds available from harvesting as well as you can
    fidelity: float  # |sim - actual| / actual
    lam: float  # shadow price of energy, s/J
    energy_used: float
    energy_harvested: float
    soc_start: float
    soc_end: float
    issues: list[Issue] = field(default_factory=list)
    plan: Plan | None = None
    actual: Plan | None = None
    same_e: Plan | None = None  # best lap on exactly the energy you did spend

    @property
    def verdict(self) -> Issue | None:
        return self.issues[0] if self.issues else None

    @property
    def explained(self) -> float:
        """Seconds available this lap (next-lap costs excluded)."""
        return sum(i.cost for i in self.issues if i.pool == "lap")


def _clusters(mask: np.ndarray, min_len: int = 2) -> list[tuple[int, int]]:
    """Contiguous runs of True, as (start, end) bin indices."""
    if not mask.any():
        return []
    edges = np.diff(mask.astype(int))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]


def _biggest(mask: np.ndarray, weight: np.ndarray) -> tuple[int, int] | None:
    """The run of `mask` carrying the most `weight`."""
    runs = _clusters(mask)
    if not runs:
        return None
    return max(runs, key=lambda r: weight[r[0]:r[1]].sum())


def analyse(track: TrackModel, lap: Lap, e_target: float | None = None) -> LapReport:
    """Compare one driven lap against the optimum for the same conditions."""
    n = track.n
    v0 = float(lap.v[0])
    e_start = float(lap.soc[0])
    if e_target is None:
        e_target = e_start  # race default: the lap must be repeatable

    actual = simulate(track, lap.p_mguk, v0, e_start, lap.harvest)

    # Two references, because "you put the energy in the wrong place" and "you
    # spent more energy than the lap can afford" are different mistakes:
    #
    #   plan     - the sustainable lap. What to do next lap; its lambda is the
    #              honest price of energy over a stint.
    #   same_e   - the best possible lap on exactly the energy you did spend.
    #              Comparing against this isolates allocation from overspending,
    #              which is then charged once, to the next lap.
    plan = solve(track, e_start, lap.harvest, v0, e_target=e_target)
    harvest_total = float(lap.harvest.sum())
    e_target_same = float(np.clip(
        e_start + harvest_total - actual.energy_used, 0.0, track.capacity))
    same_e = solve(track, e_start, lap.harvest, v0, e_target=e_target_same)

    lam = float(np.median(plan.lam[np.isfinite(plan.lam)])) if plan.lam.size else 0.0
    gain = time_gain_per_joule(track, actual)

    e_act = lap.p_mguk * actual.dt  # joules deployed per bin
    e_opt = same_e.u * same_e.dt
    depleted = actual.soc < LOW_SOC_FRAC * track.capacity
    over = lap.p_mguk > same_e.u + DEPLOY_EPS
    under = lap.p_mguk < same_e.u - DEPLOY_EPS

    issues: list[Issue] = []
    # Allocation symptoms: (name, mask, lambda-priced weight, energy, text...).
    # The weights only rank and apportion; the total comes from the simulator.
    candidates = []

    def consider(name, mask, weight, energy, detail, advice):
        if not mask.any() or weight <= 0 or energy <= 0:
            return
        candidates.append((name, mask, float(weight), float(energy), detail, advice))

    def emit(name, mask, cost, energy, detail, advice, pool="lap"):
        if cost < MIN_REPORT:
            return
        run = _biggest(mask, np.abs(e_act - e_opt)) if mask.any() else None
        where = (run[0] * DS, run[1] * DS) if run else (0.0, 0.0)
        issues.append(
            Issue(name, float(cost), where, float(energy), detail, advice, pool))

    # 1. Energy spent where it buys almost nothing: above the taper knee, or
    #    where the car is already pinned to the speed envelope.
    knee = track.taper_knee
    waste = over & (gain < 0.5 * lam)
    e_waste = float(e_act[waste].sum())
    if e_waste > 0:
        fast = waste & (actual.v > knee)
        detail = (f"{e_waste/1e6:.2f} MJ deployed where it buys "
                  f"{gain[waste].mean()*1e6:.3f} s/MJ against a lap value of "
                  f"{lam*1e6:.3f} s/MJ")
        if fast.any():
            detail += (f"; {float(e_act[fast].sum())/1e6:.2f} MJ of it above "
                       f"{knee*3.6:.0f} km/h where the MGU-K is tapering out")
        consider("Wasted deployment", waste,
                 np.sum((lam - gain[waste]) * e_act[waste]), e_waste, detail,
                 "Come off the ERS before the end of the straight — up there "
                 "it is barely adding speed. The same battery out of the next "
                 "corner is worth several times more.")

    # 2. Ran out of energy where the optimum still wanted to deploy.
    starved = depleted & under
    if starved.any():
        short = np.maximum(e_opt - e_act, 0.0)[starved]
        sectors = sorted(set(lap.sector[starved].tolist()))
        consider("Battery depleted", starved,
                 np.sum(np.maximum(gain[starved] - lam, 0.0) * short),
                 short.sum(),
                 f"empty through sector {'/'.join(str(s + 1) for s in sectors)}; "
                 f"{short.sum()/1e6:.2f} MJ short of the optimal deployment there",
                 "You are spending sector 1 energy that sector 3 needs. Hold a "
                 "reserve through the early part of the lap.")

    # 3. Energy available but not used where it was worth most.
    lazy = under & ~depleted & (gain > lam)
    if lazy.any():
        short = np.maximum(e_opt - e_act, 0.0)[lazy]
        consider("Under-deployed", lazy, np.sum((gain[lazy] - lam) * short),
                 short.sum(),
                 f"{short.sum()/1e6:.2f} MJ belonged in these corners and went "
                 f"elsewhere on the lap — it is worth "
                 f"{gain[lazy].mean()*1e6:.3f} s/MJ here",
                 "Get on the ERS as you pick up the throttle at the apex, not "
                 "once the car is already straight. This is about *where* the "
                 "energy goes, not about having spare — you can be flat out on "
                 "battery all lap and still be losing this.")

    # 4. Override used where it adds nothing. Under the 2026 rules the override
    #    does not raise peak power -- it holds full power to 337 km/h instead of
    #    tapering from 290 -- so its whole value sits in that high-speed band.
    #    Below it the car already has everything, and pressing the button just
    #    spends the extra allowance for no gain.
    if lap.overtake.any():
        # Using the override to pass someone looks identical to wasting it: both
        # spend energy where it buys little lap time. The difference is whether
        # there was a car there. Judging a pass by lap time alone would tell the
        # driver to stop overtaking, so traffic is excluded and priced instead
        # by traffic.pass_call, which values position rather than tenths.
        in_traffic = (lap.delta_front > 0.0) & (lap.delta_front < FOLLOW_RANGE)
        adds_nothing = track.boost_gain(lap.v) < 1.0e4  # W
        bad = lap.overtake & adds_nothing & ~in_traffic
        if bad.any() and e_act[bad].sum() > 0:
            consider("Override misused", bad,
                     np.sum((lam - gain[bad]) * e_act[bad]), e_act[bad].sum(),
                     f"override active for {int(bad.sum()) * DS:.0f} m below "
                     f"{track.boost_knee*3.6:.0f} km/h, where normal deployment "
                     f"already gives you full power",
                     "Save the override for the fast part of the straight. It "
                     "does not add power low down — it keeps full power on past "
                     "the speed where normal deployment fades out.")

    # Apportion the measured allocation loss across the symptoms found.
    ers_loss = float(actual.lap_time - same_e.lap_time)
    total_w = sum(c[2] for c in candidates)
    if ers_loss > MIN_REPORT and total_w > 0:
        for name, mask, w, energy, detail, advice in candidates:
            emit(name, mask, ers_loss * w / total_w, energy, detail, advice)

    # 5. Harvest left on the table. Measured by re-solving on the energy the
    #    driver's own best braking has recovered in these same zones.
    missed = np.maximum(track.harvest_best - lap.harvest, 0.0)
    missed[lap.brake < 0.05] = 0.0
    # harvest_best is the best each bin has ever produced, possibly on different
    # laps, so the sum can exceed what the rules allow in any single lap. Cap it
    # to what is actually achievable before calling the difference "missed".
    achievable = cap_harvest(track, lap.harvest + missed)
    missed = np.maximum(achievable - lap.harvest, 0.0)
    e_missed = float(missed.sum())
    harvest_loss = 0.0
    if e_missed > 0.02 * track.capacity:
        better = solve(track, e_start, achievable, v0, e_target=e_target)
        harvest_loss = float(plan.lap_time - better.lap_time)
        emit("Missed harvest", missed > 0, harvest_loss, e_missed,
             f"{e_missed/1e6:.2f} MJ less recovered than your own best braking "
             f"in the same zones",
             "Brake a little earlier and longer, or lift off before the corner. "
             "The battery you get back is worth more than the entry speed.")

    # 6. Arrived at a braking zone already full, so the recovery had nowhere to
    #    go. Free energy binned, and unlike the others this one is entirely the
    #    driver's to fix: spend a little before the zone and it all lands.
    headroom = track.capacity - actual.soc
    clipped = np.minimum(lap.harvest, np.maximum(lap.harvest - headroom, 0.0))
    e_clip = float(clipped.sum())
    if e_clip > 0.02 * track.capacity:
        run = _biggest(clipped > 0, clipped)
        where = f" into {run[0]*DS:.0f} m" if run else ""
        # Worth what that energy would have bought at this lap's price.
        emit("Harvest clipped", clipped > 0, e_clip * lam, e_clip,
             f"battery was full through {int((clipped > 0).sum()) * DS:.0f} m of "
             f"braking{where}, so {e_clip/1e6:.2f} MJ of recovery was binned",
             "Use some ERS on the way in. Arrive with room in the battery and "
             "the same braking hands you that energy for free.")

    # 7. The race-only trap: a quick lap paid for with next lap's battery.
    soc_end = float(actual.soc[-1] + lap.harvest[-1] - e_act[-1])
    deficit = e_start - soc_end
    if deficit > 0.05 * track.capacity:
        emit("Unsustainable", np.zeros(n, bool), lam * deficit, deficit,
             f"finished {deficit/1e6:.2f} MJ down on where you started, so next "
             f"lap begins short",
             "This lap was borrowed. Repay it or the deficit compounds every "
             "lap of the stint.", pool="next")

    issues.sort(key=lambda i: -i.cost)
    return LapReport(
        lap_num=lap.lap_num,
        lap_time=lap.lap_time,
        sim_time=actual.lap_time,
        optimal_time=plan.lap_time,
        ers_loss=ers_loss,
        harvest_loss=harvest_loss,
        fidelity=abs(actual.lap_time - lap.lap_time) / max(lap.lap_time, 1e-6),
        lam=lam,
        energy_used=actual.energy_used,
        energy_harvested=float(lap.harvest.sum()),
        soc_start=e_start,
        soc_end=float(actual.soc[-1]),
        issues=issues,
        plan=plan,
        actual=actual,
    )


# -- live, in-lap cues ---------------------------------------------------

class Cues:
    """Compares the lap in progress against the plan and speaks up.

    Deliberately quiet: at most one message every `gap` seconds, and only for
    things the driver can still act on this lap.
    """

    def __init__(self, gap: float = 6.0) -> None:
        self.gap = gap
        self.last_t = -1e9
        self.last_msg = ""

    def check(self, track: TrackModel, plan: Plan, sample, now: float) -> str | None:
        if plan is None or sample.lap_dist <= 0:
            return None
        i = min(int(sample.lap_dist / DS), track.n - 1)
        msg = None

        planned = float(plan.cum_energy[i])
        spent = float(sample.ers_deployed)
        overspend = spent - planned
        soc = sample.ers_store
        remaining_plan = float(plan.cum_energy[-1] - plan.cum_energy[i])

        if soc < LOW_SOC_FRAC * track.capacity and remaining_plan > 0.2e6:
            msg = "BATTERY EMPTY — nothing left for the rest of the lap"
        elif overspend > 0.5e6 and soc < remaining_plan:
            msg = (f"EASE OFF THE ERS — {overspend/1e6:.1f} MJ ahead of plan, "
                   f"you will run out before the lap ends")
        elif (sample.p_mguk > DEPLOY_EPS and sample.speed > track.taper_knee
              and plan.u[i] < DEPLOY_EPS):
            msg = (f"COME OFF THE ERS — above {track.taper_knee*3.6:.0f} km/h "
                   f"it is barely adding speed")
        elif overspend < -0.5e6 and soc > 0.7 * track.capacity:
            msg = (f"USE MORE ERS — battery is full and you are "
                   f"{-overspend/1e6:.1f} MJ behind plan")

        if msg is None or now - self.last_t < self.gap or msg == self.last_msg:
            return None
        self.last_t, self.last_msg = now, msg
        return msg


# -- session-level metrics ----------------------------------------------

@dataclass
class SessionMetrics:
    """The quantifiable outcome: predicted gain vs the gain actually realised."""

    reports: list[LapReport] = field(default_factory=list)
    baseline_laps: int = 0

    def add(self, report: LapReport) -> None:
        self.reports.append(report)

    def _times(self, rs):
        """Green-lap times only.

        A lap with a pit stop in it is ~20 s longer than a racing lap. Averaged
        in, one of them moved the session's "improvement" by three and a half
        seconds -- which is a statement about the pit lane, not about ERS.
        """
        t = np.array([r.lap_time for r in rs if r.lap_time > 0])
        if t.size < 3:
            return t
        return t[t < 1.15 * np.median(t)]

    def summary(self) -> dict:
        rs = self.reports
        if not rs:
            return {}
        base = rs[:self.baseline_laps] or rs[:max(len(rs) // 2, 1)]
        rest = rs[len(base):]
        bt, ct = self._times(base), self._times(rest)
        losses = np.array([r.ers_loss + r.harvest_loss for r in rs])
        predicted = float(np.mean(losses[:len(base)])) if base else 0.0
        # Only a deliberate baseline makes this a controlled comparison. Without
        # --baseline, first-half vs second-half just measures whether the driver
        # warmed up, and reporting it as "captured %" is a confident lie.
        controlled = self.baseline_laps > 0
        realised = (float(bt.mean() - ct.mean())
                    if controlled and bt.size and ct.size else 0.0)
        return {
            "laps": len(rs),
            "baseline_laps": len(base),
            "ers_loss_first": float(np.mean(losses[:len(base)])),
            "ers_loss_last": float(np.mean(losses[len(base):])) if rest else 0.0,
            "predicted_gain": predicted,
            "realised_gain": realised,
            "controlled": controlled,
            "capture": (realised / predicted
                        if controlled and predicted > 1e-6 else 0.0),
            "fidelity": float(np.mean([r.fidelity for r in rs])),
            "best_lap": float(self._times(rs).min()) if self._times(rs).size else 0.0,
        }
