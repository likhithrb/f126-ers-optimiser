"""Race-situation advice, and the 2026 power-unit figures it rests on.

The regulation numbers here are the published 2026 ones (350 kW MGU-K,
deployment tapering from 290 km/h to nothing at 355, Manual Override holding
full power to 337 before tapering to 350, 4 MJ store). They are only the
starting shape -- the model fits the real curve from telemetry -- but if the
fallback is wrong the first laps are coached against a car that does not exist.
"""

import numpy as np

from f126ers.optimiser import solve
from f126ers.places import find_corners
from f126ers.situations import consistency_tips, mode_tip, race_tips
from f126ers.telemetry import Sample
from f126ers.track import (BOOST_FROM, BOOST_ZERO, P_MGUK_MAX, TAPER_FROM,
                           TAPER_ZERO, TrackModel)
from make_fake_session import N, V_STRAIGHT, toy_track


def test_taper_matches_the_2026_regulations():
    tm = TrackModel(length=5793.0)
    # The rulebook says 350 kW; the game peaks the MGU-K at 315 kW, measured
    # over a real Bahrain race. The prior follows the game, not the rulebook,
    # because it is the game the driver is racing in.
    assert abs(P_MGUK_MAX - 3.15e5) < 1.0, P_MGUK_MAX
    assert abs(TAPER_FROM * 3.6 - 290) < 0.5
    assert abs(TAPER_ZERO * 3.6 - 355) < 0.5
    assert abs(BOOST_FROM * 3.6 - 337) < 0.5
    assert abs(BOOST_ZERO * 3.6 - 350) < 0.5

    # Full power right up to the taper, nothing past the end of it.
    assert tm.p_max(280 / 3.6) > 0.99 * P_MGUK_MAX
    assert tm.p_max(360 / 3.6) < 1.0
    # The override does not raise peak power, it holds it on for longer.
    assert tm.p_max(200 / 3.6, boost=True) <= P_MGUK_MAX + 1
    assert abs(tm.boost_gain(200 / 3.6)) < 1.0, "override must add nothing low down"
    assert tm.boost_gain(330 / 3.6) > 1.0e5, "override must pay at high speed"
    print(f"2026 figures ok       {P_MGUK_MAX/1e3:.0f} kW, taper {TAPER_FROM*3.6:.0f}"
          f"–{TAPER_ZERO*3.6:.0f} km/h, override {BOOST_FROM*3.6:.0f}"
          f"–{BOOST_ZERO*3.6:.0f} km/h")
    print(f"  override adds {tm.boost_gain(330/3.6)/1e3:.0f} kW at 330 km/h, "
          f"{tm.boost_gain(200/3.6)/1e3:.0f} kW at 200 km/h")


def test_override_taper_learned_separately():
    """A boosted lap must not teach the model that normal power reaches up there."""
    from f126ers.track import Lap
    tm, harvest = toy_track()
    n = N
    v = np.full(n, 95.0)  # 342 km/h, deep in the taper
    boost = np.zeros(n, bool)
    boost[:50] = True
    p = np.where(boost, 3.0e5, 4.0e4)  # full power only while boosting
    lap = Lap(lap_num=1, n=n, lap_time=60.0, valid=True, v=v,
              throttle=np.ones(n), brake=np.zeros(n), p_ice=np.full(n, 5e5),
              p_mguk=p, soc=np.full(n, 2e6), harvest=harvest,
              delta_front=np.zeros(n), overtake=boost,
              sector=np.zeros(n, int), aero=np.ones(n))
    tm._fit_taper(lap)
    assert tm.p_max(95.0, boost=True) >= 2.9e5, tm.p_max(95.0, boost=True)
    assert tm.p_max(95.0) < 2.0e5, (
        "override laps leaked into the normal taper: the optimiser would plan "
        f"a lap needing the button, {tm.p_max(95.0)/1e3:.0f} kW")
    print(f"taper separation ok   normal {tm.p_max(95.0)/1e3:.0f} kW vs "
          f"override {tm.p_max(95.0, boost=True)/1e3:.0f} kW at 342 km/h")


def _setup():
    tm, harvest = toy_track()
    tm.harvest_limit = 4.0e6
    plan = solve(tm, 2.0e6, harvest, v0=V_STRAIGHT)
    return tm, plan, float(np.median(plan.lam))


def test_last_lap_and_restart():
    tm, plan, lam = _setup()
    got = race_tips(tm, Sample(lap=12, ers_store=2.4e6), plan, lam, 12, 12, 0)
    assert any(t.kind == "lastlap" for t in got), got
    # Not the last lap: silent.
    assert not any(t.kind == "lastlap" for t in
                   race_tips(tm, Sample(lap=6, ers_store=2.4e6), plan, lam,
                             6, 12, 0))
    sc = race_tips(tm, Sample(lap=5, ers_store=1.6e6), plan, lam, 5, 30, 1)
    assert any(t.kind == "restart" for t in sc), sc
    print("last lap / restart ok fire only in their own situation")


def test_defending_and_blue_flags():
    tm, plan, lam = _setup()
    close = Sample(lap=10, ers_store=2e6, gap_behind=0.6)
    assert any(t.kind == "defend" for t in
               race_tips(tm, close, plan, lam, 10, 30, 0))
    lapped = Sample(lap=10, ers_store=2e6, gap_behind=0.6, lapped_behind=True)
    tips = race_tips(tm, lapped, plan, lam, 10, 30, 0)
    assert any(t.kind == "blueflag" for t in tips)
    assert not any(t.kind == "defend" for t in tips), (
        "should not tell you to defend from a car being shown blue flags")
    far = Sample(lap=10, ers_store=2e6, gap_behind=4.0)
    assert not any(t.kind in ("defend", "blueflag") for t in
                   race_tips(tm, far, plan, lam, 10, 30, 0))
    print("defending ok          fires inside 1s, silent at 4s, blue flags win")


def test_clean_air_is_silent():
    tm, plan, lam = _setup()
    got = race_tips(tm, Sample(lap=8, ers_store=2e6, tyre_age=4), plan, lam,
                    8, 40, 0)
    assert not got, [str(t) for t in got]
    print("no false alarms ok    nothing to say mid-race in clear air")


def test_values_are_seconds_not_watts():
    """Every gain must be a time, priced through lambda from an energy."""
    tm, plan, lam = _setup()
    s = Sample(lap=12, ers_store=2.4e6)
    tip = [t for t in race_tips(tm, s, plan, lam, 12, 12, 0)
           if t.kind == "lastlap"][0]
    assert abs(tip.gain - 2.4e6 * lam) < 1e-6, (tip.gain, 2.4e6 * lam)
    assert 0.0 < tip.gain < 100.0, tip.gain
    print(f"units ok              {2.4e6/1e6:.1f} MJ x {lam*1e6:.2f} s/MJ = "
          f"{tip.gain:.2f}s")


if __name__ == "__main__":
    import time
    t0 = time.monotonic()
    test_taper_matches_the_2026_regulations()
    test_override_taper_learned_separately()
    test_last_lap_and_restart()
    test_defending_and_blue_flags()
    test_clean_air_is_silent()
    test_values_are_seconds_not_watts()
    print(f"\nall situation checks passed in {time.monotonic()-t0:.1f}s")
