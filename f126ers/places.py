"""Turns distances into places, and losses into instructions about places.

"0.19 s of wasted deployment between 690 and 820 m" is true and nearly useless
at 300 km/h. "Turn 4: stop deploying on the way in" is the same fact in a form
you can act on before you get there.

Corners are found from the learned speed envelope rather than a track database:
a corner is a local minimum of v_env, so the numbering is the driver's own
braking points, on the circuit as they actually drive it. That also means it
works on any track, including ones that did not exist when this was written.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .track import DS, Lap, TrackModel

MIN_DROP = 8.0  # m/s of deceleration before a dip counts as a corner
MIN_TIP = 0.04  # s; below this it is not worth saying out loud
LIFT_BINS = 15  # bins before a braking zone that count as the run-up (~150 m)
MIN_ZONE_E = 0.01  # fraction of battery capacity worth mentioning in one zone
# Tips that describe the whole lap rather than one place on it.
LAP_WIDE = ("budget", "cutoff", "mode")


@dataclass
class Corner:
    number: int
    apex: int  # bin of minimum speed
    entry: int  # bin where braking starts
    exit: int  # bin where the car is back up to speed
    v_apex: float

    n: int = 0  # bins in the lap, so the zones can wrap the start line

    def _span(self, start: int, end: int) -> np.ndarray:
        """Bin indices from start to end, going forwards around the lap.

        Index arrays, not slices: the last corner of a lap starts before the
        start line and ends after it, and slice(420, 3) is silently empty --
        which would drop every tip about the most important corner on the
        circuit without any error to notice.
        """
        n = self.n or 1
        length = (end - start) % n
        if length == 0:
            return np.empty(0, dtype=int)
        return (start + np.arange(length)) % n

    @property
    def brake_zone(self) -> np.ndarray:
        return self._span(self.entry, self.apex)

    @property
    def exit_zone(self) -> np.ndarray:
        return self._span(self.apex, self.exit)

    def contains(self, i: int) -> bool:
        n = self.n or 1
        return ((i - self.entry) % n) < ((self.exit - self.entry) % n)


@dataclass
class Tip:
    """One actionable instruction tied to a place on the circuit."""

    where: str  # "Turn 4" / "sector 2"
    action: str  # what to do, in plain words, readable at 300 km/h
    why: str  # one clause of plain English; no jargon, no numbers
    detail: str  # the numbers, for the panel rather than the live line
    gain: float  # seconds
    bin_start: int
    bin_end: int
    kind: str  # deploy_less | deploy_more | harvest | clip | save

    def __str__(self) -> str:
        return f"{self.where}: {self.action}  (+{self.gain:.2f}s)"

    @property
    def headline(self) -> str:
        return f"{self.where} — {self.action}"


def find_corners(track: TrackModel) -> list[Corner]:
    """Numbered corners, from local minima of the learned speed envelope."""
    v = track.v_env
    n = len(v)
    if n < 12 or v.max() <= 0:
        return []
    corners: list[Corner] = []
    # A corner is a bin lower than everything within a window either side.
    w = max(int(60.0 / DS), 3)
    for i in range(n):
        window = [v[(i + k) % n] for k in range(-w, w + 1)]
        if v[i] > min(window) + 1e-9:
            continue
        if corners and (i - corners[-1].apex) < w:
            continue  # same corner, flat-bottomed
        # Walk out to where the car was last at speed, and back up to speed.
        # Every index is taken modulo n: a corner near the start line walks off
        # both ends of the array, and the last corner onto the pit straight is
        # exactly the one this has to get right.
        entry = i
        while (v[(entry - 1) % n] > v[entry % n] + 0.05
               and (i - entry) < n // 4):
            entry -= 1
        ex = i
        while (v[(ex + 1) % n] > v[ex % n] + 0.05 and (ex - i) < n // 4):
            ex += 1
        if v[entry % n] - v[i] < MIN_DROP:
            continue  # a kink, not a corner
        corners.append(Corner(0, i, entry % n, ex % n, float(v[i]), n=n))
    corners.sort(key=lambda c: c.apex)
    for k, c in enumerate(corners, 1):
        c.number = k
    return corners


def _label(corners: list[Corner], i: int, sector: np.ndarray) -> str:
    for c in corners:
        if c.contains(i):
            return f"Turn {c.number}"
    # Between corners: name the corner it leads to, which is what the driver
    # is about to arrive at and can still do something about.
    if corners:
        n = corners[0].n or 1
        nxt = min(corners, key=lambda c: (c.entry - i) % n)
        return f"the run to Turn {nxt.number}"
    return f"sector {int(sector[i]) + 1}"


def tips(track: TrackModel, lap: Lap, report, corners: list[Corner] | None = None,
         limit: int = 3) -> list[Tip]:
    """Location-specific instructions, ranked by seconds available.

    Works from the same per-bin quantities the verdict uses, but groups them by
    place instead of by symptom -- the driver cannot act on "wasted deployment",
    only on "wasted deployment into Turn 7".
    """
    if report is None or report.plan is None or report.actual is None:
        return []
    corners = find_corners(track) if corners is None else corners
    actual, plan = report.actual, report.plan
    lam = report.lam
    n = track.n

    # Placement is judged against the best lap achievable on *the energy you
    # actually spent*, not against the energy-neutral plan. Judged against the
    # plan, a driver who is simply conservative comes out under-deployed
    # everywhere and every placement mistake inside the lap is drowned out --
    # which is exactly what happens on a track where you are saving battery.
    # How much to spend in total is a separate question, answered below.
    ref = report.same_e if report.same_e is not None else plan
    e_act = lap.p_mguk * actual.dt
    e_opt = ref.u * ref.dt
    diff = e_act - e_opt  # positive: spent more here than the reference wants
    from .optimiser import time_gain_per_joule
    gain = time_gain_per_joule(track, actual)

    # Joules recovered per joule of kinetic energy shed, at the driver's best
    # braking zone this lap. Zones below it are leaving recovery on the table.
    # Reference is the *median* corner, and only clear outliers are flagged.
    # Taking the best corner as the target means the second-best of two is
    # always "deficient" -- on a symmetric lap that fires on a flawless drive.
    # Needs a handful of corners before the median means anything at all.
    rates = []
    for c in corners:
        shed = _shed(lap, c.brake_zone)
        if shed > 1.0e4:
            rates.append(float(lap.harvest[c.brake_zone].sum()) / shed)
    rate_ref = float(np.median(rates)) if len(rates) >= 4 else 0.0
    OUTLIER = 0.8  # recover less than this share of a typical corner to count

    out: list[Tip] = []

    def add(mask, kind, action, why, detail_fmt, value):
        if not mask.any() or value < MIN_TIP:
            return
        idx = np.flatnonzero(mask)
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        where = _label(corners, int(idx[np.argmax(np.abs(diff[idx]))]), lap.sector)
        out.append(Tip(where, action, why, detail_fmt, float(value), lo, hi, kind))

    # Group the per-bin excess and shortfall into contiguous stretches, so each
    # tip is one place rather than a scatter of bins.
    for lo, hi in _runs(diff > 1.0e4):
        e = float(diff[lo:hi].sum())
        val = float(np.sum((lam - gain[lo:hi]) * np.maximum(diff[lo:hi], 0)))
        if val < MIN_TIP or e <= 0:
            continue
        seg = np.zeros(n, bool)
        seg[lo:hi] = True
        # Overspending in the run-up to a braking zone is not generic waste: the
        # speed it buys is braked straight off again, so the fix has a name.
        if _in_run_up(corners, lo, hi, n):
            add(seg, "coast", "LIFT AND COAST INTO HERE",
                "you are still on the ERS right up to the braking point — that "
                "speed gets braked straight off, so the battery is thrown away",
                f"{e/1e6:.2f} MJ spent in the last {(hi-lo)*DS:.0f} m before the "
                f"brakes, worth {gain[lo:hi].mean()*1e6:.2f} s/MJ", val)
        else:
            add(seg, "deploy_less", "USE LESS ERS",
                "you are burning battery where it barely adds any speed — "
                "it is worth far more later in the lap",
                f"{e/1e6:.2f} MJ more than the plan wants here, worth "
                f"{gain[lo:hi].mean()*1e6:.2f} s/MJ against {lam*1e6:.2f} "
                f"elsewhere", val)

    for lo, hi in _runs(diff < -1.0e4):
        e = float(-diff[lo:hi].sum())
        val = float(np.sum(np.maximum(gain[lo:hi] - lam, 0)
                           * np.maximum(-diff[lo:hi], 0)))
        if val < MIN_TIP or e <= 0:
            continue
        seg = np.zeros(n, bool)
        seg[lo:hi] = True
        add(seg, "deploy_more", "USE MORE ERS, EARLIER",
            "you are saving battery here, but this is the spot on the lap "
            "where it buys you the most lap time",
            f"{e/1e6:.2f} MJ left unused where it is worth "
            f"{gain[lo:hi].mean()*1e6:.2f} s/MJ", val)

    # Deployment into a corner exit the tyres cannot take. The envelope fit
    # already separates grip-limited bins from power-limited ones; here the car
    # is using every newton the tyres will accept, so extra power is wheelspin,
    # not lap time. Without this the same bins read as "use more ERS" -- advice
    # that would make the driver slower.
    if track.uncapped.size == n and track.uncapped.any():
        gripped = (lap.throttle > 0.9) & (lap.brake < 0.05) & ~track.uncapped
        for c in corners:
            z = c.exit_zone
            if z.size == 0:
                continue
            bad = z[gripped[z] & (e_act[z] > 5.0e3)]
            if bad.size < 3:
                continue
            # How far past the apex before the car can actually take the power.
            free = z[~gripped[z]]
            if free.size == 0:
                continue  # never gets on top of the grip: not a deployment call
            wait = int((free[0] - c.apex) % n) * DS
            if not 10.0 <= wait <= 200.0:
                continue  # implausible as an instruction; say nothing
            e = float(e_act[bad].sum())
            add(_mask(n, bad), "traction", f"WAIT {wait:.0f} m BEFORE FULL ERS",
                "the tyres are already at their limit on this exit — more power "
                "here is wheelspin, not speed",
                f"{e/1e6:.2f} MJ deployed across {bad.size * DS:.0f} m of "
                f"grip-limited exit", e * lam * 0.5)

    # Braking zones that gave up recoverable energy, named by the corner.
    for c in corners:
        z = c.brake_zone
        if z.size == 0:
            continue
        prev = (track.harvest_prev if track.harvest_prev.size == n
                else track.harvest_best)
        # Two references, whichever is higher: what you have recovered here
        # before, and what your best braking zone *on this lap* recovers per
        # unit of speed shed. The second works on the very first lap, when
        # there is no previous best to compare against at all.
        got = float(lap.harvest[z].sum())
        shed = _shed(lap, z)
        typical = rate_ref * shed
        # Only if this corner is a clear outlier against the rest of the lap.
        from_rate = typical if (rate_ref > 0 and got < OUTLIER * typical) else 0.0
        best = max(float(prev[z].sum()), from_rate)
        if best - got > MIN_ZONE_E * track.capacity:
            seg = np.zeros(n, bool)
            seg[z] = True
            add(seg, "harvest", "BRAKE EARLIER TO RECHARGE",
                "you have braked harder into here before and recovered more "
                "battery for it — lift a touch sooner and it comes back",
                f"{(best-got)/1e6:.2f} MJ less recovered than your own best "
                f"braking into here", (best - got) * lam)
        # Full on the way in: the recovery had nowhere to go.
        head = track.capacity - actual.soc[z]
        clip = float(np.minimum(lap.harvest[z],
                                np.maximum(lap.harvest[z] - head, 0.0)).sum())
        if clip > MIN_ZONE_E * track.capacity:
            seg = np.zeros(n, bool)
            seg[z] = True
            add(seg, "clip", "USE ERS BEFORE HERE — BATTERY FULL",
                "you arrive with a full battery, so the energy braking would "
                "recover has nowhere to go and is thrown away",
                f"battery full on entry, {clip/1e6:.2f} MJ of recovery binned",
                clip * lam)

    # Which third of the lap the battery goes into. The original problem this
    # tool exists for: burn it all in sector 1 and there is nothing left for
    # sector 3, whatever the per-corner placement looks like. Compared as
    # shares, so it is a statement about balance, not about total spend.
    tot_act, tot_ref = float(e_act.sum()), float(e_opt.sum())
    if tot_act > 1.0e5 and tot_ref > 1.0e5:
        for sec in (0, 1, 2):
            m = lap.sector == sec
            if m.sum() < 5:
                continue
            share_a = float(e_act[m].sum()) / tot_act
            share_r = float(e_opt[m].sum()) / tot_ref
            excess = (share_a - share_r) * tot_act
            val = float(np.sum(np.maximum(lam - gain[m], 0)
                               * np.maximum(diff[m], 0)))
            if share_a - share_r > 0.12 and val > MIN_TIP:
                out.append(Tip(
                    f"sector {sec + 1}", f"SPREAD IT OUT OF SECTOR {sec + 1}",
                    f"you put {share_a*100:.0f}% of the battery into this third "
                    f"of the lap and leave yourself short later",
                    f"{share_a*100:.0f}% of the lap's energy here against "
                    f"{share_r*100:.0f}% in the plan ({excess/1e6:+.2f} MJ)",
                    val, int(np.flatnonzero(m)[0]), int(np.flatnonzero(m)[-1]),
                    "sector"))

    # One speed to drive to, instead of a list of places: above this the car is
    # deep enough into the taper that deployment stops paying for itself.
    fast = e_act > 5.0e3
    if fast.any() and lam > 0:
        poor = fast & (gain < 0.3 * lam)
        if poor.sum() >= 4:
            cut = float(np.percentile(actual.v[poor], 25))
            e = float(e_act[poor].sum())
            if e * lam > MIN_TIP and cut * 3.6 > 150:
                out.append(Tip(
                    "on the straights", f"COME OFF THE ERS ABOVE {cut*3.6:.0f} KM/H",
                    "past this speed the motor has so little left to give that "
                    "the battery is better spent anywhere else on the lap",
                    f"{e/1e6:.2f} MJ spent above {cut*3.6:.0f} km/h, worth "
                    f"{gain[poor].mean()*1e6:.2f} s/MJ against {lam*1e6:.2f}",
                    e * lam * 0.5, 0, n, "cutoff"))

    # Where the override is worth pressing. Under the 2026 rules it does not
    # raise peak power, it holds full power past the speed where normal
    # deployment fades -- so its value is the *extra* power it unlocks, which is
    # zero low down and largest near the end of a long straight.
    extra = track.boost_gain(actual.v)
    if extra.max() > 1.0e4:
        # Net, not gross. The override spends energy that could have gone to a
        # corner exit instead, so the gain per joule has to beat the lap's price
        # to be worth it: (g - lambda), never just g. On lap time alone this is
        # usually negative -- the override unlocks power exactly where a joule
        # is worth least -- which is the honest answer. It earns its keep when
        # there is a car in front, and that case is priced in traffic.pass_call.
        worth = extra * actual.dt * (gain - lam)
        best = _best_window(worth, max(int(300.0 / DS), 3))
        if best is not None and worth[best[0]:best[1]].sum() > MIN_TIP:
            lo, hi = best
            used = bool(lap.overtake[lo:hi].any())
            where = _label(corners, int(lo + np.argmax(worth[lo:hi])), lap.sector)
            if not used:
                out.append(Tip(
                    where, "USE THE OVERRIDE HERE",
                    "this is the fastest part of the lap, where normal power "
                    "has faded and the override is the only way to get it back",
                    f"{worth[lo:hi].sum():.2f}s available across "
                    f"{(hi-lo)*DS:.0f} m at up to {actual.v[lo:hi].max()*3.6:.0f} "
                    f"km/h", float(worth[lo:hi].sum()), lo, hi, "override"))

    # How much to spend in total is a different question from where to spend
    # it, and mixing them is what made every tip read "use more". Priced against
    # the sustainable plan rather than the same-energy reference.
    spare = float(plan.energy_used - actual.energy_used)
    if abs(spare) * lam > MIN_TIP:
        if spare > 0:
            out.append(Tip(
                "over the whole lap", "YOU ARE SAVING TOO MUCH BATTERY",
                "you finish the lap with charge to spare — that is lap time you "
                "are carrying around and never using",
                f"{spare/1e6:.2f} MJ left over against a lap that stays "
                f"energy-neutral", spare * lam, 0, n, "budget"))
        else:
            out.append(Tip(
                "over the whole lap", "YOU ARE USING TOO MUCH BATTERY",
                "you are spending faster than the lap recovers, so you will "
                "start the next lap short and lose the time back",
                f"{-spare/1e6:.2f} MJ more than the lap can sustain",
                -spare * lam, 0, n, "budget"))

    out.sort(key=lambda t: -t.gain)
    return _dedupe(out)[:limit]


def _mask(n: int, idx: np.ndarray) -> np.ndarray:
    m = np.zeros(n, bool)
    m[idx] = True
    return m


def _shed(lap: Lap, zone: np.ndarray) -> float:
    """Kinetic energy given up across a braking zone, in joules."""
    if zone.size < 2:
        return 0.0
    v_in, v_out = float(lap.v[zone[0]]), float(lap.v[zone[-1]])
    return max(0.5 * lap.mass * (v_in ** 2 - v_out ** 2), 0.0)


def _in_run_up(corners: list[Corner], lo: int, hi: int, n: int) -> bool:
    """Does this stretch sit in the braking run-up to a corner?"""
    for c in corners:
        for i in (lo, hi - 1):
            if 0 < (c.entry - i) % n <= LIFT_BINS:
                return True
    return False


def _best_window(w: np.ndarray, width: int) -> tuple[int, int] | None:
    """Contiguous window of `width` bins carrying the most weight."""
    if w.size <= width:
        return None
    cs = np.concatenate([[0.0], np.cumsum(w)])
    totals = cs[width:] - cs[:-width]
    i = int(np.argmax(totals))
    return (i, i + width) if totals[i] > 0 else None


def _runs(mask: np.ndarray, min_len: int = 3) -> list[tuple[int, int]]:
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


def _dedupe(items: list[Tip]) -> list[Tip]:
    """One tip per place: the biggest. Three notes about Turn 4 is noise."""
    seen: dict[tuple[str, str], Tip] = {}
    for t in items:
        key = (t.where, t.kind)
        if key not in seen:
            seen[key] = t
    return sorted(seen.values(), key=lambda t: -t.gain)


def upcoming(corners: list[Corner], tips_list: list[Tip], bin_now: int,
             n: int, lookahead: int = 40) -> Tip | None:
    """The tip for the place the car is about to reach, if there is one.

    Live coaching has to arrive *before* the mistake. A tip about Turn 7
    delivered at the exit of Turn 7 is a lap-time report, not coaching.
    """
    best = None
    for t in tips_list:
        ahead = (t.bin_start - bin_now) % n
        if ahead <= lookahead and (best is None or ahead < best[0]):
            best = (ahead, t)
    return best[1] if best else None
