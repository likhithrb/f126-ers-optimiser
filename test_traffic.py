"""Checks the overtaking economics on a synthetic stuck-behind-a-car lap.

The pace penalty itself is measured from real telemetry at runtime; the numbers
here are constructed so the right answer is known by hand, which is what lets
these assertions mean anything.
"""

import numpy as np

from f126ers.optimiser import solve
from f126ers.track import DS
from f126ers.traffic import (FOLLOW_RANGE, following_deficit, attack_energy,
                             pass_call)
from make_fake_session import toy_track, N, V_STRAIGHT

HOLDUP = slice(120, 175)  # bins where a car ahead caps our speed


def stuck_lap(tm, harvest, cap_frac=0.90, gap=0.6):
    """A lap where a car ahead holds us to `cap_frac` of our own pace."""
    from f126ers.track import Lap
    v = tm.v_env.copy()
    v[HOLDUP] *= cap_frac
    delta = np.zeros(N)
    delta[HOLDUP] = gap
    z = np.zeros(N)
    return Lap(lap_num=5, n=N, lap_time=float((DS / v).sum()), valid=True,
               v=v, throttle=np.ones(N), brake=z, p_ice=tm.p_ice.copy(),
               p_mguk=z.copy(), soc=np.full(N, 2.0e6), harvest=harvest.copy(),
               delta_front=delta, overtake=np.zeros(N, bool),
               sector=(np.arange(N) * 3 // N).astype(int), aero=tm.aero.copy())


def test_deficit_measures_only_traffic():
    tm, harvest = toy_track()
    tm.v_obs = tm.v_env.copy()  # clean-air best, recorded on earlier laps
    lap = stuck_lap(tm, harvest)
    deficit, close = following_deficit(tm, lap)

    # Expected: time lost only in the held-up bins.
    v_clean, v_stuck = tm.v_env[HOLDUP], lap.v[HOLDUP]
    want = float((DS / v_stuck - DS / v_clean).sum())
    assert abs(deficit - want) < 1e-6, (deficit, want)
    assert close.sum() == (HOLDUP.stop - HOLDUP.start)
    assert deficit > 0.1, deficit
    print(f"deficit ok            {deficit:.3f}s/lap lost to the car ahead")

    # Slow in a bin with nobody ahead is a driving error, not traffic.
    lap2 = stuck_lap(tm, harvest)
    lap2.v[20:40] *= 0.5  # badly slow, but delta_front is 0 there
    d2, _ = following_deficit(tm, lap2)
    assert abs(d2 - deficit) < 1e-6, (d2, deficit)
    print("attribution ok        slow in clear air not charged to traffic")
    return deficit


def test_attack_is_bounded_by_charge():
    tm, harvest = toy_track()
    plan = solve(tm, 2.0e6, harvest, v0=V_STRAIGHT)
    e, gain = attack_energy(tm, plan, v_now=70.0, bin_now=100, charge=5.0e6)
    assert e > 0 and gain > 0, (e, gain)
    e_low, _ = attack_energy(tm, plan, v_now=70.0, bin_now=100, charge=1.0e5)
    assert e_low <= 1.0e5 + 1, e_low
    print(f"attack energy ok      {e/1e6:.2f} MJ buys {gain:.2f}s, "
          f"capped to charge when short")
    return e, gain


def test_breakeven_falls_with_laps_remaining():
    tm, harvest = toy_track()
    tm.v_obs = tm.v_env.copy()
    plan = solve(tm, 2.0e6, harvest, v0=V_STRAIGHT)
    lam = float(np.median(plan.lam))
    deficit, _ = following_deficit(tm, stuck_lap(tm, harvest))

    calls = {}
    for laps in (1, 5, 30):
        calls[laps] = pass_call(tm, plan, lam, deficit, delta_front=0.12,
                                laps_left=laps, v_now=70.0, bin_now=100,
                                charge=2.0e6)
    b = [calls[k].breakeven for k in (1, 5, 30)]
    assert b[0] > b[1] > b[2], b  # more laps to gain -> easier to justify
    assert calls[30].verdict == "attack", calls[30].verdict
    print(f"break-even ok         1 lap {b[0]*100:4.1f}%   5 laps {b[1]*100:4.1f}%"
          f"   30 laps {b[2]*100:4.1f}%")
    print(f"  30 laps left -> {calls[30].verdict}: {calls[30].detail}")

    # Held up barely, and almost no race left: the energy is worth more as pace.
    thin = pass_call(tm, plan, lam, deficit=0.04, delta_front=0.12, laps_left=2,
                     v_now=70.0, bin_now=100, charge=2.0e6)
    assert thin.verdict == "hold", (thin.verdict, thin.breakeven)
    print(f"  small prize  -> {thin.verdict}: break-even "
          f"{thin.breakeven*100:.0f}% > 100%, cannot repay")
    return calls


def test_out_of_range_says_hold():
    tm, harvest = toy_track()
    tm.v_obs = tm.v_env.copy()
    plan = solve(tm, 2.0e6, harvest, v0=V_STRAIGHT)
    lam = float(np.median(plan.lam))
    deficit, _ = following_deficit(tm, stuck_lap(tm, harvest))
    far = pass_call(tm, plan, lam, deficit, delta_front=3.0, laps_left=30,
                    v_now=70.0, bin_now=100, charge=2.0e6)
    assert far.verdict == "hold", far.verdict
    assert "alongside" in far.detail, far.detail
    print(f"range check ok        3.0s back -> {far.verdict}")


def test_clear_air_is_silent():
    tm, harvest = toy_track()
    tm.v_obs = tm.v_env.copy()
    plan = solve(tm, 2.0e6, harvest, v0=V_STRAIGHT)
    lam = float(np.median(plan.lam))
    clear = stuck_lap(tm, harvest, cap_frac=1.0, gap=0.0)
    deficit, _ = following_deficit(tm, clear)
    assert deficit == 0.0, deficit
    call = pass_call(tm, plan, lam, deficit, delta_front=0.0, laps_left=20,
                     v_now=70.0, bin_now=100, charge=2.0e6)
    assert call.verdict == "none", call.verdict
    print("clear air ok          no car ahead -> no recommendation")


if __name__ == "__main__":
    import time
    t0 = time.monotonic()
    test_deficit_measures_only_traffic()
    test_attack_is_bounded_by_charge()
    test_breakeven_falls_with_laps_remaining()
    test_out_of_range_says_hold()
    test_clear_air_is_silent()
    print(f"\nall traffic checks passed in {time.monotonic()-t0:.1f}s")
