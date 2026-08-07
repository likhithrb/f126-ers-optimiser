"""Corner detection and location-specific tips."""

import numpy as np

from f126ers.analysis import analyse
from f126ers.places import LAP_WIDE, Tip, find_corners, tips, upcoming
from f126ers.track import DS, Lap
from make_fake_session import APEXES, N, V_STRAIGHT, toy_track, charge_limited
from f126ers.optimiser import simulate, solve


def build(tm, harvest, u, soc=2.0e6):
    sim = simulate(tm, u, V_STRAIGHT, soc, harvest)
    dv = np.diff(sim.v, append=sim.v[0])
    brake = (dv < -0.3).astype(float)
    return Lap(lap_num=4, n=N, lap_time=sim.lap_time, valid=True, v=sim.v,
               throttle=1.0 - brake, brake=brake, p_ice=tm.p_ice.copy(),
               p_mguk=u, soc=sim.soc, harvest=harvest.copy(),
               delta_front=np.zeros(N), overtake=np.zeros(N, bool),
               sector=(np.arange(N) * 3 // N).astype(int), aero=tm.aero.copy(),
               harvest_limit=4.0e6)


def test_corners_match_the_track():
    tm, _ = toy_track()
    cs = find_corners(tm)
    assert len(cs) == len(APEXES), [c.apex for c in cs]
    for c, want in zip(cs, APEXES):
        assert abs(c.apex - want) <= 2
        assert c.entry < c.apex < c.exit
    print(f"corners ok            {len(cs)} found at bins "
          f"{[c.apex for c in cs]}, numbered in track order")
    return cs


def test_corners_wrap_around_the_start_line():
    """A corner near the start line must not walk off the end of the array.

    Real circuits put a corner just before and just after the line; the toy
    track's corners both sit mid-lap, which hid an unwrapped index until a
    4.3 km track crashed on it.
    """
    from f126ers.track import TrackModel
    for n in (433, 200, 517):  # odd sizes, as real track lengths give
        tm = TrackModel(length=n * DS, n=n)
        d = np.arange(n)
        v = np.full(n, 90.0)
        for apex in (3, n // 2, n - 4):  # hard against both ends
            dist = np.minimum(np.abs(d - apex), n - np.abs(d - apex))
            v = np.minimum(v, 25.0 + 2.2 * dist)
        tm.v_env = v
        cs = find_corners(tm)  # must not raise
        assert cs, f"no corners found on an {n}-bin track"
        for c in cs:
            assert 0 <= c.entry < n and 0 <= c.exit < n and 0 <= c.apex < n
    print(f"wrap-around ok        corners at the start line handled on "
          f"433/200/517-bin tracks")

    # And a corner whose braking zone genuinely crosses the line: under the old
    # slice-based zones this was empty, so it produced no tips and no error.
    n = 433
    tm = TrackModel(length=n * DS, n=n)
    d = np.arange(n)
    v = np.full(n, 90.0)
    for apex in (5, 200):
        dist = np.minimum(np.abs(d - apex), n - np.abs(d - apex))
        v = np.minimum(v, 25.0 + 2.2 * dist)
    tm.v_env = v
    cs = find_corners(tm)
    wrapped = [c for c in cs if c.entry > c.apex]
    assert wrapped, "expected a corner braking before the start line"
    c = wrapped[0]
    assert c.brake_zone.size > 0 and c.exit_zone.size > 0
    assert c.brake_zone[0] > c.brake_zone[-1], "zone should cross zero"
    assert c.contains(c.apex) and c.contains(int(c.brake_zone[1]))
    print(f"wrapped corner ok     Turn {c.number} brakes at {c.entry}, "
          f"apexes at {c.apex}: {c.brake_zone.size} bins across the line")


def test_tip_names_the_place_and_the_action():
    tm, harvest = toy_track()
    tm.harvest_limit = 4.0e6
    tm.v_obs = tm.v_env.copy()
    tm.harvest_best = harvest.copy()
    # Dump the battery down the first straight: wrong place, wrong time.
    bad = np.zeros(N)
    bad[:45] = tm.p_max(np.minimum(84.0, tm.v_env[:45]))
    bad = charge_limited(tm, bad, V_STRAIGHT, 2.0e6, harvest)
    lap = build(tm, harvest, bad)
    rep = analyse(tm, lap)
    got = tips(tm, lap, rep)
    assert got, "a battery dump should produce at least one tip"
    for t in got:
        assert t.where and t.action and t.why and t.gain > 0
        assert ("Turn" in t.where or "sector" in t.where
                or t.kind in LAP_WIDE), t.where
        # Driver-facing wording must not need translating at 300 km/h.
        for jargon in ("deploy", "MJ", "s/MJ", "lambda", "harvest"):
            assert jargon not in t.action.lower(), (jargon, t.action)
            assert jargon not in t.why.lower(), (jargon, t.why)
    print(f"tips ok               {len(got)} tips, top: {got[0]}")
    for t in got:
        print(f"    {t.where:<22} {t.action:<32} +{t.gain:.2f}s")
    return got


def test_tips_are_ranked_and_deduped():
    tm, harvest = toy_track()
    tm.harvest_limit = 4.0e6
    tm.v_obs = tm.v_env.copy()
    tm.harvest_best = harvest.copy()
    bad = np.zeros(N)
    bad[:45] = tm.p_max(np.minimum(84.0, tm.v_env[:45]))
    bad = charge_limited(tm, bad, V_STRAIGHT, 2.0e6, harvest)
    lap = build(tm, harvest, bad)
    got = tips(tm, lap, analyse(tm, lap), limit=5)
    gains = [t.gain for t in got]
    assert gains == sorted(gains, reverse=True), gains
    keys = [(t.where, t.kind) for t in got]
    assert len(keys) == len(set(keys)), keys
    print(f"ranking ok            {gains} descending, one tip per place")


def test_upcoming_fires_before_the_corner_not_after():
    tm, _ = toy_track()
    cs = find_corners(tm)
    t = Tip("Turn 2", "USE LESS ERS", "why", "", 0.3, cs[1].entry,
            cs[1].apex, "deploy_less")
    # 20 bins before the entry: should fire.
    early = upcoming(cs, [t], (cs[1].entry - 20) % N, N)
    assert early is not None, "tip should arrive before the corner"
    # Just past it: should not.
    late = upcoming(cs, [t], (cs[1].entry + 60) % N, N)
    assert late is None, "tip should not fire once the corner is behind you"
    print("timing ok             tip arrives before the corner, silent after")


def test_conservative_driver_still_gets_placement_advice():
    """A driver who under-uses the battery overall must still be told *where*.

    Judged against the energy-neutral plan, someone saving battery reads as
    under-deployed in every bin and every placement mistake inside the lap is
    drowned out -- a real Monza run produced nothing but "use more ERS on the
    corners". Placement is judged on the energy actually spent; how much to
    spend in total is reported separately.
    """
    tm, harvest = toy_track()
    tm.harvest_limit = 4.0e6
    tm.v_obs = tm.v_env.copy()
    tm.harvest_best = harvest.copy()
    cs = find_corners(tm)
    opt = solve(tm, 2.0e6, harvest, v0=V_STRAIGHT)

    # Conservative overall, but still on the ERS into every braking zone.
    u = opt.u * 0.6
    for c in cs:
        z = c._span((c.entry - 14) % N, c.entry)
        u[z] = tm.p_max(np.minimum(opt.v[z], tm.v_env[z]))
    lap = build(tm, harvest, charge_limited(tm, u, V_STRAIGHT, 2.0e6, harvest))
    got = tips(tm, lap, analyse(tm, lap), cs, limit=6)
    kinds = {t.kind for t in got}
    assert "budget" in kinds, kinds  # told they are saving too much
    assert kinds - {"budget"}, "no placement advice at all: " + str(kinds)
    assert "coast" in kinds, ("should spot ERS held on into the braking zones",
                              kinds)
    print(f"conservative ok       {len(got)} tips covering {sorted(kinds)}")


def test_clean_lap_says_nothing():
    tm, harvest = toy_track()
    tm.harvest_limit = 4.0e6
    tm.v_obs = tm.v_env.copy()
    tm.harvest_best = harvest.copy()
    opt = solve(tm, 2.0e6, harvest, v0=V_STRAIGHT)
    lap = build(tm, harvest, opt.u)
    got = tips(tm, lap, analyse(tm, lap))
    assert not got, [str(t) for t in got]
    print("no false alarms ok    optimal lap produces no tips")


if __name__ == "__main__":
    import time
    t0 = time.monotonic()
    test_corners_match_the_track()
    test_corners_wrap_around_the_start_line()
    test_tip_names_the_place_and_the_action()
    test_tips_are_ranked_and_deduped()
    test_conservative_driver_still_gets_placement_advice()
    test_upcoming_fires_before_the_corner_not_after()
    test_clean_lap_says_nothing()
    print(f"\nall places checks passed in {time.monotonic()-t0:.1f}s")
