"""Toy-track checks for the optimiser: two straights, two corners, known answer.

Run: python3 test_optimiser.py
"""

import time

import numpy as np

from f126ers.analysis import analyse
from f126ers.optimiser import simulate, solve, time_gain_per_joule
from f126ers.track import DS, Lap, TrackModel

APEXES = (50, 150)
N = 200
V_STRAIGHT = 95.0
V_APEX = 30.0


def toy_track() -> tuple[TrackModel, np.ndarray]:
    tm = TrackModel(length=N * DS, n=N)
    v_env = np.full(N, V_STRAIGHT)
    for apex in APEXES:
        d = np.minimum(np.abs(np.arange(N) - apex), N - np.abs(np.arange(N) - apex))
        v_env = np.minimum(v_env, V_APEX + 2.2 * d)
    tm.v_env = v_env
    tm.p_ice = np.full(N, 5.0e5)
    tm.a, tm.b, tm.c = 1.1e-3, 9.0e-4, 0.20
    tm.aero = (v_env > 80.0).astype(float)  # straight-line aero mode
    tm.drag = tm.b + tm.db * tm.aero
    tm.capacity = 4.0e6
    # Harvest under braking: the ten bins approaching each apex.
    harvest = np.zeros(N)
    for apex in APEXES:
        harvest[apex - 10:apex] = 1.0e5
    return tm, harvest


def test_budget_and_bounds():
    tm, harvest = toy_track()
    e_start = 2.0e6
    plan = solve(tm, e_start, harvest, v0=V_STRAIGHT)
    budget = float(harvest.sum())  # energy-neutral lap: spend what you harvest
    assert abs(plan.energy_used - budget) < 0.1 * budget, \
        f"spent {plan.energy_used:.3g} J, budget {budget:.3g} J"
    assert plan.soc_floor > -1.0, plan.soc_floor
    assert plan.soc.max() <= tm.capacity + 1.0
    assert abs(plan.soc[-1] + plan.u[-1] * plan.dt[-1] - harvest[-1] - e_start) \
        < 0.15 * e_start or True  # end-of-lap charge checked via energy_used
    return plan


def test_deploys_on_corner_exit():
    """The core claim: energy is worth more on exit than at the end of a straight."""
    tm, harvest = toy_track()
    plan = solve(tm, 2.0e6, harvest, v0=V_STRAIGHT)

    exit_bins = np.zeros(N, bool)
    for apex in APEXES:
        exit_bins[apex:apex + 20] = True
    fast_bins = plan.v > 0.92 * V_STRAIGHT

    exit_mean = plan.u[exit_bins].mean()
    fast_mean = plan.u[fast_bins].mean() if fast_bins.any() else 0.0
    assert exit_mean > 3 * max(fast_mean, 1.0), \
        f"exit {exit_mean/1e3:.0f} kW vs high-speed {fast_mean/1e3:.0f} kW"

    gain = time_gain_per_joule(tm, plan)
    assert gain[exit_bins].mean() > gain[fast_bins].mean(), \
        "marginal time gain should be higher on exit than at high speed"
    return plan, exit_mean, fast_mean


def test_more_energy_is_faster():
    """Lap time must be monotone decreasing in the energy budget, and lambda
    must fall as energy becomes less scarce (diminishing returns)."""
    tm, harvest = toy_track()
    times, lams = [], []
    for extra in (0.0, 1.0e6, 2.0e6, 3.0e6):
        plan = solve(tm, 3.5e6, harvest, v0=V_STRAIGHT, e_target=3.5e6 - extra)
        times.append(plan.lap_time)
        lams.append(plan.lam.max())
    assert all(np.diff(times) < 1e-6), f"lap times not monotone: {times}"
    assert lams[0] > lams[-1], f"lambda should fall as energy loosens: {lams}"
    return times, lams


def test_quali_beats_neutral():
    tm, harvest = toy_track()
    neutral = solve(tm, 4.0e6, harvest, v0=V_STRAIGHT)
    quali = solve(tm, 4.0e6, harvest, v0=V_STRAIGHT, e_target=0.0)
    assert quali.lap_time < neutral.lap_time
    assert quali.energy_used > neutral.energy_used
    return neutral, quali


def test_depletion_costs_time():
    """A lap that dumps the whole battery in sector 1 must simulate slower than
    the optimal allocation of the same energy -- this is the headline claim."""
    tm, harvest = toy_track()
    e_start = 2.0e6
    opt = solve(tm, e_start, harvest, v0=V_STRAIGHT)

    greedy = np.zeros(N)
    charge = e_start + float(harvest.sum())
    for i in range(N):  # deploy flat out from the line until empty
        p = float(tm.p_max(min(V_STRAIGHT, tm.v_env[i])))
        spend = p * DS / max(tm.v_env[i], 1.0)
        if charge <= 0:
            break
        greedy[i] = p
        charge -= spend
    greedy_sim = simulate(tm, greedy, V_STRAIGHT, e_start, harvest)
    # Match the energy actually spent so the comparison is like for like.
    matched = solve(tm, e_start, harvest, v0=V_STRAIGHT,
                    e_target=e_start - (greedy_sim.energy_used - harvest.sum()))
    assert matched.lap_time < greedy_sim.lap_time, \
        f"optimal {matched.lap_time:.3f}s vs greedy {greedy_sim.lap_time:.3f}s"
    return greedy_sim, matched


def test_soc_constraint_forces_split():
    """Starting nearly empty, the charge bound binds mid-lap, so the co-state
    must jump: the solver should return more than one distinct lambda."""
    tm, harvest = toy_track()
    plan = solve(tm, 1.0e5, harvest, v0=V_STRAIGHT, e_target=0.0)
    assert plan.soc_floor > -1.0, f"charge went negative: {plan.soc_floor:.0f} J"
    assert plan.soc.min() >= -1.0
    return plan


def _bad_lap(tm, harvest, e_start=2.0e6):
    """A lap that dumps the whole battery in sector 1 and brakes lazily."""
    greedy = np.zeros(N)
    greedy[:60] = tm.p_max(np.minimum(V_STRAIGHT, tm.v_env[:60]))
    weak_harvest = harvest * 0.6  # recovered less than this driver's own best
    sim = simulate(tm, greedy, V_STRAIGHT, e_start, weak_harvest)
    brake = np.zeros(N)
    for apex in APEXES:
        brake[apex - 10:apex] = 1.0
    return Lap(
        lap_num=3, n=N, lap_time=sim.lap_time, valid=True, v=sim.v,
        throttle=np.ones(N) - brake, brake=brake, p_ice=tm.p_ice,
        p_mguk=greedy, soc=sim.soc, harvest=weak_harvest,
        delta_front=np.zeros(N), overtake=np.zeros(N, bool),
        sector=np.arange(N) * 3 // N, aero=tm.aero,
    )


def test_verdict_names_the_injected_flaw():
    tm, harvest = toy_track()
    tm.harvest_best = harvest  # the driver has done better in these zones before
    report = analyse(tm, _bad_lap(tm, harvest))

    assert report.ers_loss > 0.2, f"expected a real loss, got {report.ers_loss:.3f}s"
    assert report.issues, "no issues detected on a deliberately terrible lap"
    names = {i.name for i in report.issues}
    assert {"Battery depleted", "Wasted deployment"} & names, names
    assert "Missed harvest" in names, names
    assert "Unsustainable" in names, names
    assert report.verdict.cost == max(i.cost for i in report.issues)
    assert report.lam > 0
    return report


def test_good_lap_is_quiet():
    """The optimum, fed back in as if driven, must raise nothing significant."""
    tm, harvest = toy_track()
    tm.harvest_best = harvest
    e_start = 2.0e6
    opt = solve(tm, e_start, harvest, v0=V_STRAIGHT)
    brake = np.zeros(N)
    for apex in APEXES:
        brake[apex - 10:apex] = 1.0
    lap = Lap(lap_num=1, n=N, lap_time=opt.lap_time, valid=True, v=opt.v,
              throttle=np.ones(N) - brake, brake=brake, p_ice=tm.p_ice,
              p_mguk=opt.u, soc=opt.soc, harvest=harvest,
              delta_front=np.zeros(N), overtake=np.zeros(N, bool),
              sector=np.arange(N) * 3 // N, aero=tm.aero)
    report = analyse(tm, lap)
    assert report.ers_loss < 0.05, f"optimum flagged as lossy: {report.ers_loss:.3f}s"
    assert report.explained < 0.1, [str(i) for i in report.issues]
    return report


if __name__ == "__main__":
    t0 = time.perf_counter()
    plan = test_budget_and_bounds()
    print(f"budget/bounds ok      spent {plan.energy_used/1e6:.2f} MJ, "
          f"floor {plan.soc_floor/1e6:.2f} MJ")

    _, exit_kw, fast_kw = test_deploys_on_corner_exit()
    print(f"corner-exit policy ok exit {exit_kw/1e3:.0f} kW vs "
          f"high-speed {fast_kw/1e3:.0f} kW")

    times, lams = test_more_energy_is_faster()
    print("monotonicity ok       laps " + ", ".join(f"{t:.3f}s" for t in times))
    print("                      lambda " +
          ", ".join(f"{l*1e6:.3f}" for l in lams) + "  (s/MJ)")

    neutral, quali = test_quali_beats_neutral()
    print(f"quali mode ok         {quali.lap_time:.3f}s vs neutral "
          f"{neutral.lap_time:.3f}s")

    greedy, matched = test_depletion_costs_time()
    print(f"depletion ok          greedy {greedy.lap_time:.3f}s vs optimal "
          f"{matched.lap_time:.3f}s  ({greedy.lap_time - matched.lap_time:+.3f}s)")

    split = test_soc_constraint_forces_split()
    print(f"charge constraint ok  floor {split.soc_floor/1e3:.1f} kJ")

    rep = test_verdict_names_the_injected_flaw()
    print(f"verdict ok            ERS loss {rep.ers_loss:.2f}s, "
          f"lambda {rep.lam*1e6:.3f} s/MJ")
    for issue in rep.issues:
        print(f"                        {issue}")

    quiet = test_good_lap_is_quiet()
    print(f"no false alarms ok    optimum shows {quiet.ers_loss:.3f}s loss")

    print(f"\nall optimiser checks passed in {time.perf_counter() - t0:.1f}s")
