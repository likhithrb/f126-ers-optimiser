"""Advice that depends on the race, not on the corner.

places.py answers "where on this lap"; this answers "what is going on around
you right now". Defending, blue flags, the last lap, a restart, tyres past
their best -- all of them change what the battery is for, and none of them are
visible in the lap trace alone.

Everything here is priced in the same currency as the rest of the tool: seconds,
via the shadow price lambda. Nothing fires without a reason expressed that way.
"""

from __future__ import annotations

import numpy as np

from .places import Tip
from .track import TrackModel

DEFEND_RANGE = 1.0  # s; inside this the car behind is a real threat
DRS_RANGE = 1.0  # s; the window in which they get a tow and the override
OLD_TYRES = 15  # laps before wear starts changing the exit advice
DEFEND_RESERVE = 5.0e5  # J worth holding back for the next straight


def race_tips(track: TrackModel, sample, plan, lam: float, lap_num: int,
              total_laps: int, safety_car: int, deficit: float = 0.0
              ) -> list[Tip]:
    """Situational advice for right now. Empty when nothing applies."""
    out: list[Tip] = []
    if sample is None or track is None:
        return out
    n = track.n
    cap = track.capacity
    soc = float(getattr(sample, "ers_store", 0.0))
    laps_left = max(total_laps - lap_num, 0) if total_laps else 0

    # -- the last lap: energy you finish with is energy you wasted ----------
    if total_laps and lap_num >= total_laps:
        if soc > 0.15 * cap:
            out.append(Tip(
                "last lap", "EMPTY THE BATTERY",
                "there is no next lap to save it for — anything you cross the "
                "line with is time you gave away",
                f"{soc/1e6:.2f} MJ still in the battery on the final lap",
                soc * lam, 0, n, "lastlap"))

    # -- safety car: the restart is worth more than the crawl ---------------
    if safety_car in (1, 2):
        room = cap - soc
        if room > 0.2 * cap:
            out.append(Tip(
                "under the safety car", "CHARGE IT UP NOW",
                "you are going slowly anyway, so recovering costs you nothing "
                "here and the restart is where the race is won",
                f"{room/1e6:.2f} MJ of room left in the battery",
                room * lam, 0, n, "restart"))
        else:
            out.append(Tip(
                "under the safety car", "BATTERY FULL — READY FOR THE RESTART",
                "hold it there and use it the moment the green flag drops",
                f"{soc/1e6:.2f} MJ banked", 0.0, 0, n, "restart"))

    # -- someone behind ------------------------------------------------------
    behind = float(getattr(sample, "gap_behind", 0.0) or 0.0)
    lapped = bool(getattr(sample, "lapped_behind", False))
    if 0.0 < behind < DEFEND_RANGE and laps_left > 0:
        if lapped:
            out.append(Tip(
                "car behind", "IGNORE THEM — THEY ARE A LAP DOWN",
                "blue flags are their problem, not yours; spending battery to "
                "hold them up costs you and gains you nothing",
                f"car behind is {behind:.1f}s back and a lap down",
                0.0, 0, n, "blueflag"))
        else:
            # Defending is worth the position you would otherwise lose. Priced
            # the same way as attacking: what it costs against what it saves.
            # Value the reserve as energy, not power: holding back a chunk of
            # battery for the next straight is worth what that energy buys.
            reserve = min(DEFEND_RESERVE, cap)
            cost = reserve * lam
            out.append(Tip(
                "car behind", "KEEP SOMETHING FOR THE STRAIGHT",
                "they are close enough to get a run on you — arrive at the "
                "next straight with battery or you will be defending with none",
                f"{behind:.1f}s behind, {laps_left} laps to hold them off",
                max(cost, 0.05), 0, n, "defend"))

    # -- tyres past their best ----------------------------------------------
    age = int(getattr(sample, "tyre_age", 0) or 0)
    if age >= OLD_TYRES and track.grip_at(age) < 0.985:
        lost = (1.0 - track.grip_at(age)) * 100
        out.append(Tip(
            "corner exits", "GET ON THE POWER LATER OUT OF THE SLOW CORNERS",
            "the tyres have gone off, so the same power that worked on a fresh "
            "set will just spin the wheels now",
            f"{age} laps on this set, about {lost:.0f}% of the grip gone",
            0.05, 0, n, "tyres"))

    return out


def consistency_tips(history: list, corners, lam: float,
                     limit: int = 1) -> list[Tip]:
    """Corners where the driver's own energy use swings most, lap to lap.

    Nothing to do with the optimum: this measures repeatability against
    yourself. A corner you get right half the time is worth more than one you
    get wrong consistently, because the fix is already in your hands.
    """
    out: list[Tip] = []
    if len(history) < 4 or not corners:
        return out
    n = corners[0].n or 1
    for c in corners:
        z = c.exit_zone
        if z.size == 0:
            continue
        per_lap = [float(np.sum(h[z])) for h in history if h.size == n]
        if len(per_lap) < 4:
            continue
        # Scatter about a *trend*, not about the mean. A driver who uses a
        # little more energy here every lap as they learn the corner is
        # improving, not being inconsistent, and plain standard deviation
        # cannot tell the two apart -- it would scold them for getting better.
        y = np.asarray(per_lap, float)
        x = np.arange(y.size, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        spread = float(np.std(y - (slope * x + intercept)))
        if spread < 5.0e4:  # under 50 kJ of scatter is just noise
            continue
        out.append(Tip(
            f"Turn {c.number}", "BE MORE CONSISTENT HERE",
            "your energy use out of this corner swings a long way lap to lap — "
            "doing the same thing every time is free lap time",
            f"{spread/1e6:.2f} MJ of lap-to-lap scatter across "
            f"{len(per_lap)} laps", spread * lam, int(z[0]), int(z[-1]) + 1,
            "consistency"))
    out.sort(key=lambda t: -t.gain)
    return out[:limit]


def mode_tip(track: TrackModel, lap, plan, lam: float) -> Tip | None:
    """Whether the ERS mode is set where the plan needs it.

    The mode is a button, so this is the most directly executable advice the
    tool can give -- but only worth saying when the gap is large, since the
    driver may be managing something the model cannot see.
    """
    if plan is None or lap is None:
        return None
    n = len(lap.p_mguk)
    if plan.u.size != n:
        return None
    want = float(np.percentile(plan.u, 90))
    got = float(np.percentile(lap.p_mguk, 90))
    if want <= 1.0e4:
        return None
    short = (want - got) / want
    # Seconds, via energy: watts alone are not a quantity lambda can price.
    delta_e = abs(float(np.sum((plan.u - lap.p_mguk) * plan.dt)))
    value = delta_e * lam
    if short > 0.35:
        return Tip(
            "ERS mode", "TURN THE ERS MODE UP",
            "the car is holding back far more than the plan wants, which "
            "usually means the mode is set low rather than a driving mistake",
            f"peak deployment {got/1e3:.0f} kW against {want/1e3:.0f} kW planned",
            value, 0, n, "mode")
    if short < -0.35:
        return Tip(
            "ERS mode", "TURN THE ERS MODE DOWN",
            "you are deploying harder than the lap can sustain, so you will "
            "run the battery down and lose it back later",
            f"peak deployment {got/1e3:.0f} kW against {want/1e3:.0f} kW planned",
            value, 0, n, "mode")
    return None
