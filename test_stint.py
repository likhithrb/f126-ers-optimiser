"""Planning energy across laps rather than within one."""

import numpy as np

from f126ers.stint import PIT_VALUE, energy_curve, plan_stint
from make_fake_session import V_STRAIGHT, toy_track


def setup():
    tm, h = toy_track()
    tm.harvest_limit = 4.0e6
    return tm, h, energy_curve(tm, h, V_STRAIGHT, 2.0e6)


def spends(targets, soc, hpl):
    out = []
    for t in targets:
        out.append(soc + hpl - t)
        soc = t
    return out


def test_curve_is_decreasing_and_convex():
    tm, h, c = setup()
    assert np.all(np.diff(c.time) <= 1e-6), "more energy must never be slower"
    # Convex: each extra megajoule buys less than the one before it.
    slopes = -np.diff(c.time) / np.maximum(np.diff(c.spend), 1.0)
    assert np.all(np.diff(slopes) <= 1e-9), slopes
    print(f"curve ok              {c.time[0]:.2f}s at 0 MJ down to "
          f"{c.time[-1]:.2f}s at {c.spend[-1]/1e6:.2f} MJ, diminishing returns")
    return c


def test_equal_spending_beats_uneven():
    """The reason energy-neutral is a good proxy: convexity."""
    tm, h, c = setup()
    even = 2 * c.time_at(2.0e6)
    uneven = c.time_at(1.0e6) + c.time_at(3.0e6)
    assert even < uneven, (even, uneven)
    print(f"convexity ok          2+2 MJ = {even:.3f}s beats 1+3 MJ = {uneven:.3f}s")


def test_long_stint_spends_evenly():
    tm, h, c = setup()
    hpl = float(h.sum())
    p = plan_stint(tm, c, 2.0e6, hpl, 20)
    sp = np.array(spends(p.targets, 2.0e6, hpl))
    # Every lap spends within a discretisation step of the same amount.
    assert sp.std() < 0.2e6, sp / 1e6
    assert np.all(sp > 0.5 * hpl), sp / 1e6
    print(f"long stint ok         spends {sp.mean()/1e6:.2f} MJ/lap "
          f"(spread {sp.std()/1e6:.2f}), not front-loaded")


def test_finishes_the_race_empty():
    tm, h, c = setup()
    hpl = float(h.sum())
    for laps in (1, 3, 6):
        p = plan_stint(tm, c, 2.0e6, hpl, laps)
        assert p.targets[-1] <= 1e-6, (laps, p.targets)
    # ... and it does not dump everything on the very first of them.
    p = plan_stint(tm, c, 2.0e6, hpl, 6)
    assert p.targets[0] > 0.2e6, p.targets
    print(f"endgame ok            finishes on empty, drawn down over the last "
          f"laps: {' '.join(f'{t/1e6:.2f}' for t in p.targets)}")


def test_more_energy_never_hurts():
    tm, h, c = setup()
    hpl = float(h.sum())
    prev = None
    for soc in (0.5e6, 1.5e6, 2.5e6, 3.5e6):
        p = plan_stint(tm, c, soc, hpl, 10)
        total = sum(c.time_at(s) for s in spends(p.targets, soc, hpl))
        if prev is not None:
            assert total <= prev + 1e-6, (soc, total, prev)
        prev = total
    print("monotone ok           starting with more charge is never slower")


def test_pit_lap_gets_less_energy():
    """A lap spent partly in the pit lane is a poor place to spend battery."""
    tm, h, c = setup()
    hpl = float(h.sum())
    # Needs a regime with freedom to move energy: harvest so high that the
    # battery would overflow, or so low that every lap spends exactly what it
    # recovers, both leave nowhere to shift it to. That is physics, not a bug.
    hpl = 1.2e6
    normal = plan_stint(tm, c, 2.0e6, hpl, 12)
    pitting = plan_stint(tm, c, 2.0e6, hpl, 12, pit_in=3)
    # Check the spend on the pit lap itself, not the lap before it. pit_in=3
    # means the stop happens on the fourth lap from now, index 3.
    sn = spends(normal.targets, 2.0e6, hpl)
    sp = spends(pitting.targets, 2.0e6, hpl)
    assert sp[3] < sn[3] - 0.02e6, (
        f"a lap spent partly in the pit lane is a poor place for energy, but "
        f"the plan puts {sp[3]/1e6:.2f} MJ there against {sn[3]/1e6:.2f} MJ "
        f"on a normal lap")
    assert "pit" in pitting.note
    print(f"pit lap ok            spends {sp[3]/1e6:.2f} MJ on the pit lap vs "
          f"{sn[3]/1e6:.2f} MJ normally (worth {PIT_VALUE:.0%} of a green lap)")


if __name__ == "__main__":
    import time
    t0 = time.monotonic()
    test_curve_is_decreasing_and_convex()
    test_equal_spending_beats_uneven()
    test_long_stint_spends_evenly()
    test_finishes_the_race_empty()
    test_more_energy_never_hurts()
    test_pit_lap_gets_less_energy()
    print(f"\nall stint checks passed in {time.monotonic()-t0:.1f}s")
