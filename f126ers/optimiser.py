"""Optimal ERS deployment as a constrained optimal-control problem.

Problem:  minimise lap time  T = sum_i ds / v_i
          over deployment power u(s) in [0, P_max(v)]
          subject to  dE/ds = -u/v + h(s),  0 <= E <= E_cap,  E(L) >= E_target.

Solved by Pontryagin / Lagrangian relaxation rather than a 2-D dynamic program
over (distance, charge). Relaxing the energy budget with a multiplier lambda
gives, for each fixed lambda, a 1-D dynamic program over speed alone:

          min_u  sum_i [ dt_i + lambda * u_i * dt_i ]

lambda is the co-state dT/dE: the shadow price of energy, in seconds per joule.
Bisecting lambda until the solution spends exactly the available budget recovers
the constrained optimum, and hands back the co-state as a first-class object
instead of a finite difference of a value surface. When the charge hits a bound
mid-lap the co-state is no longer constant: it jumps at the contact point, so
the lap is split there and each segment gets its own lambda (the standard
state-constrained maximum-principle construction, `_solve_constrained`).

The optimal policy is bang-bang in the local time-gain-per-joule: deploy where
the marginal seconds bought per joule exceed lambda. That is why low-speed
corner exits beat the end of a straight -- not a rule of thumb, a consequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .track import DS, TrackModel

N_SPEED = 56  # speed grid levels for the inner DP
N_CTRL = 9  # deployment levels, as fractions of the taper-limited maximum
V_MIN = 5.0


@dataclass
class Plan:
    """An optimal (or simulated) lap."""

    u: np.ndarray  # deployment power per bin, W
    v: np.ndarray  # speed per bin, m/s
    dt: np.ndarray  # time spent in each bin, s
    soc: np.ndarray  # state of charge entering each bin, J
    lam: np.ndarray  # co-state per bin, s/J
    lap_time: float
    energy_used: float
    soc_floor: float  # lowest charge reached

    @property
    def cum_energy(self) -> np.ndarray:
        return np.cumsum(self.u * self.dt)


def simulate(track: TrackModel, u: np.ndarray, v0: float,
             e_start: float = 0.0, harvest: np.ndarray | None = None,
             lam: np.ndarray | None = None) -> Plan:
    """Forward lap simulation under a given deployment profile.

    Speed is capped by the learned envelope v_env, which already encodes the
    driver's braking points and cornering grip. So this isolates the effect of
    ERS: identical line and braking, different energy.
    """
    n = track.n
    a, c = track.a, track.c
    drag = track.drag
    v = np.empty(n)
    dt = np.empty(n)
    v[0] = min(v0, track.v_env[0])
    for i in range(n):
        e_kin = 0.5 * v[i] ** 2 + DS * (
            a * (track.p_ice[i] + u[i]) / v[i] - drag[i] * v[i] ** 2 - c)
        v_next = np.sqrt(2.0 * max(e_kin, 0.5 * V_MIN ** 2))
        nxt = (i + 1) % n
        v_next = min(v_next, track.v_env[nxt])
        dt[i] = 2.0 * DS / (v[i] + v_next)
        if i + 1 < n:
            v[i + 1] = v_next

    if harvest is None:
        harvest = np.zeros(n)
    soc = np.empty(n)
    charge = e_start
    for i in range(n):
        soc[i] = charge
        charge = min(charge - u[i] * dt[i] + harvest[i], track.capacity)
    return Plan(
        u=u, v=v, dt=dt, soc=soc,
        lam=np.zeros(n) if lam is None else lam,
        lap_time=float(dt.sum()),
        energy_used=float(np.sum(u * dt)),
        soc_floor=float(soc.min()),
    )


def _dp_segment(track: TrackModel, lo: int, hi: int, lam: float,
                v_grid: np.ndarray, v_end_target: float,
                terminal_weight: float) -> np.ndarray:
    """Backward value iteration over speed for a fixed co-state lambda.

    Returns the optimal deployment fraction for each (bin, speed-level).
    """
    a, c = track.a, track.c
    drag = track.drag
    fracs = np.linspace(0.0, 1.0, N_CTRL)
    # Terminal cost: giving up speed at the segment end is paid for later.
    value = terminal_weight * np.maximum(v_end_target - v_grid, 0.0)
    policy = np.zeros((hi - lo, len(v_grid)), dtype=np.int8)

    p_cap = track.p_max(v_grid)[:, None] * fracs[None, :]  # (n_v, n_u)
    for i in range(hi - 1, lo - 1, -1):
        u = p_cap
        e_kin = 0.5 * v_grid[:, None] ** 2 + DS * (
            a * (track.p_ice[i] + u) / v_grid[:, None]
            - drag[i] * v_grid[:, None] ** 2 - c)
        v_next = np.sqrt(2.0 * np.maximum(e_kin, 0.5 * V_MIN ** 2))
        v_next = np.minimum(v_next, track.v_env[(i + 1) % track.n])
        dt = 2.0 * DS / (v_grid[:, None] + v_next)
        cost = dt * (1.0 + lam * u)
        nxt = np.interp(v_next.ravel(), v_grid, value).reshape(v_next.shape)
        total = cost + nxt
        best = np.argmin(total, axis=1)  # ties -> lowest deployment
        policy[i - lo] = best
        value = np.take_along_axis(total, best[:, None], axis=1)[:, 0]
    return policy


def _roll_out(track: TrackModel, lo: int, hi: int, policy: np.ndarray,
              v_grid: np.ndarray, v0: float) -> tuple[np.ndarray, np.ndarray]:
    """Plays the DP policy forward, returning (deployment in watts, bin times)."""
    fracs = np.linspace(0.0, 1.0, N_CTRL)
    a, c = track.a, track.c
    drag = track.drag
    u = np.zeros(hi - lo)
    dt = np.zeros(hi - lo)
    v = min(v0, track.v_env[lo])
    for i in range(lo, hi):
        j = i - lo
        frac = np.interp(v, v_grid, fracs[policy[j]])
        u[j] = frac * float(track.p_max(v))
        e_kin = 0.5 * v ** 2 + DS * (
            a * (track.p_ice[i] + u[j]) / v - drag[i] * v ** 2 - c)
        v_free = np.sqrt(2.0 * max(e_kin, 0.5 * V_MIN ** 2))
        cap = track.v_env[(i + 1) % track.n]
        if v_free > cap:
            # The envelope, not power, sets the next speed. Spend only what it
            # takes to reach it: watts pushed against a grip limit buy nothing,
            # and the DP's interpolated policy cannot see the clamp coming.
            need = ((0.5 * cap ** 2 - 0.5 * v ** 2) / DS
                    + drag[i] * v ** 2 + c) * v / a - track.p_ice[i]
            u[j] = float(np.clip(need, 0.0, u[j]))
        v_next = min(v_free, cap)
        dt[j] = 2.0 * DS / (v + v_next)
        v = v_next
    return u, dt


def _solve_segment(track: TrackModel, lo: int, hi: int, budget: float,
                   v0: float, v_end: float, terminal_weight: float,
                   tol: float = 5e3) -> tuple[np.ndarray, float]:
    """Bisects the co-state so the segment spends exactly `budget` joules.

    Energy spent is monotonically decreasing in lambda, so plain bisection on
    lambda converges; returns (deployment profile, lambda).
    """
    v_grid = np.linspace(V_MIN, max(track.v_env.max() * 1.02, V_MIN + 1), N_SPEED)

    def spend(lam: float) -> tuple[np.ndarray, float]:
        pol = _dp_segment(track, lo, hi, lam, v_grid, v_end, terminal_weight)
        u, dt = _roll_out(track, lo, hi, pol, v_grid, v0)
        return u, float(np.sum(u * dt))

    u_free, e_free = spend(0.0)
    if budget >= e_free:  # budget is not binding: deploy flat out
        return u_free, 0.0
    if budget <= 0:
        return np.zeros(hi - lo), float("inf")

    lo_lam, hi_lam = 0.0, 1e-7
    for _ in range(14):  # find an upper bracket where spending drops below budget
        _, e = spend(hi_lam)
        if e <= budget:
            break
        lo_lam, hi_lam = hi_lam, hi_lam * 5
    else:
        return np.zeros(hi - lo), hi_lam

    u = u_free
    lam = hi_lam
    for _ in range(18):
        lam = 0.5 * (lo_lam + hi_lam)
        u, e = spend(lam)
        if abs(e - budget) < tol:
            break
        if e > budget:
            lo_lam = lam
        else:
            hi_lam = lam
    return u, lam


def cap_harvest(track: TrackModel, harvest: np.ndarray) -> np.ndarray:
    """Scales a harvest map down to the per-lap regulatory limit.

    The limit is reported by the game in `ErsHarvestLimitPerLap`. Without it the
    optimiser will happily plan a lap that recovers more than the rules allow --
    an optimum the driver cannot reach at any skill level, against which every
    lap forever reports a shortfall in harvesting that was never available.
    """
    limit = track.harvest_limit
    total = float(harvest.sum())
    if limit <= 0 or total <= limit:
        return harvest
    return harvest * (limit / total)


def _embed(u_seg: np.ndarray, lo: int, hi: int, n: int) -> np.ndarray:
    u = np.zeros(n)
    u[lo:hi] = u_seg
    return u


def solve(track: TrackModel, e_start: float, harvest: np.ndarray,
          v0: float, e_target: float | None = None,
          max_splits: int = 3) -> Plan:
    """Optimal deployment for one lap.

    e_target defaults to e_start: the lap must be energy-neutral, so the plan is
    repeatable every lap of the stint. Pass 0.0 for a qualifying-style lap that
    is allowed to finish flat.
    """
    n = track.n
    if e_target is None:
        e_target = e_start
    harvest = cap_harvest(track, harvest)
    u = np.zeros(n)
    lam = np.zeros(n)
    bounds = [(0, n, e_start, e_target, v0)]

    for _ in range(max_splits):
        u = np.zeros(n)
        lam = np.zeros(n)
        for lo, hi, seg_start, seg_target, seg_v0 in bounds:
            budget = seg_start - seg_target + float(harvest[lo:hi].sum())
            u_seg, lam_seg = _solve_segment(
                track, lo, hi, budget, seg_v0,
                v_end=track.v_env[hi % n] if hi < n else track.v_env[0],
                terminal_weight=track.terminal_weight)
            u[lo:hi] = u_seg
            lam[lo:hi] = lam_seg
        u = enforce_charge(track, u, v0, e_start, harvest)
        plan = simulate(track, u, v0, e_start, harvest, lam)

        # Charge went negative somewhere: the constraint is active, so the
        # co-state jumps there. Split the lap at first contact and re-solve.
        breach = np.flatnonzero(plan.soc < -1.0)
        if breach.size == 0 or len(bounds) > max_splits:
            # Recalibrate the terminal cost from this solution, damped, so the
            # next solve uses a value measured on the circuit rather than a
            # guess. Converges within a couple of laps and costs one O(n) pass.
            track.terminal_weight = 0.5 * (
                track.terminal_weight + terminal_sensitivity(track, plan))
            return plan
        k = max(int(breach[0]), 1)
        # Target a small reserve at the contact point rather than exactly zero.
        # The bisection resolves the budget to a few kJ, and a plan that aims at
        # precisely empty lands just below it -- a plan that schedules a negative
        # battery is not a plan anyone can drive.
        reserve = max(1.0e4, 0.005 * track.capacity)
        bounds = _split_bounds(bounds, k, plan, e_target, reserve)
    u = enforce_charge(track, u, v0, e_start, harvest)
    return simulate(track, u, v0, e_start, harvest, lam)


def enforce_charge(track: TrackModel, u: np.ndarray, v0: float, e_start: float,
                   harvest: np.ndarray, passes: int = 2) -> np.ndarray:
    """Trims a deployment profile to charge the battery actually holds.

    The co-state split targets a small reserve at each contact point, which
    relies on the bisection landing close enough. That is a tolerance, not a
    guarantee, and it moves whenever the control grid or the taper changes. This
    enforces the bound outright, so the plan handed to a driver is feasible by
    construction rather than by luck. Deployment changes bin times, which change
    the energy drawn, so it iterates.
    """
    n = track.n
    u = u.copy()
    for _ in range(passes):
        sim = simulate(track, u, v0, e_start, harvest)
        charge = e_start
        for i in range(n):
            draw = u[i] * sim.dt[i]
            if draw > charge:
                u[i] = max(charge, 0.0) / sim.dt[i] if sim.dt[i] > 0 else 0.0
                draw = max(charge, 0.0)
            charge = min(charge - draw + harvest[i], track.capacity)
    return u


def _split_bounds(bounds, k, plan, e_target, reserve=0.0):
    """Splits the segment containing bin k: the charge bound is active there."""
    out = []
    for lo, hi, seg_start, seg_target, seg_v0 in bounds:
        if lo < k < hi:
            out.append((lo, k, seg_start, reserve, seg_v0))
            out.append((k, hi, reserve, seg_target, float(plan.v[k])))
        else:
            out.append((lo, hi, seg_start, seg_target, seg_v0))
    return out


def time_gain_per_joule(track: TrackModel, plan: Plan) -> np.ndarray:
    """Marginal seconds saved per joule deployed in each bin, g(s).

    This is the quantity the optimal policy compares against lambda, so it has
    to be the *total* derivative, not the local one. A joule spent here does not
    only shorten this bin: the car leaves faster and stays faster until the next
    braking point, and most of the benefit is in that tail. Measuring only the
    time saved inside the bin understates the value by a factor of ten or more,
    and would make the comparison against lambda meaningless.

    Computed by the adjoint (backward sensitivity) recursion, which gets every
    downstream effect in one O(n) pass rather than n re-simulations:

        phi_i = d(remaining time)/d(v_i)
        phi_i = ddt_i/dv_i + (ddt_i/dv_next + phi_{i+1}) * dv_next/dv_i

    Where the next speed is clamped by the envelope -- a braking zone, or a
    corner the car is already taking at the limit -- the derivative is zero and
    the chain terminates: extra speed there is simply thrown away, which is
    exactly why deployment into a braking zone or a traction-limited exit buys
    nothing.
    """
    return _adjoint(track, plan)[0]


def terminal_sensitivity(track: TrackModel, plan: Plan) -> float:
    """Seconds of lap time bought by 1 m/s more speed at the start line.

    A lap is a loop, so this is exactly what a m/s at the *finish* line is worth,
    which is the terminal cost the solver needs. Measuring it from the lap beats
    picking a constant: too high and the solver buys speed over the line at a
    terrible price, too low and it coasts across to bank energy.
    """
    return float(np.clip(_adjoint(track, plan)[1], 0.0, 0.5))


def _adjoint(track: TrackModel, plan: Plan) -> tuple[np.ndarray, float]:
    n = track.n
    a, c = track.a, track.c
    drag = track.drag
    v, u, dt = plan.v, plan.u, plan.dt

    gain = np.zeros(n)
    phi_next = 0.0  # sensitivity of the remaining lap to speed at the finish
    for i in range(n - 1, -1, -1):
        vi = v[i]
        v_next = v[i + 1] if i + 1 < n else v[0]
        # Unclamped next speed, and whether the envelope is what set it.
        e_kin = 0.5 * vi ** 2 + DS * (
            a * (track.p_ice[i] + u[i]) / vi - drag[i] * vi ** 2 - c)
        v_free = np.sqrt(2.0 * max(e_kin, 0.5 * V_MIN ** 2))
        clamped = v_free > track.v_env[(i + 1) % n] + 1e-9

        ddt_dv = -2.0 * DS / (vi + v_next) ** 2  # same for v_i and v_next
        if clamped or v_free <= V_MIN:
            dvn_dv = dvn_du = 0.0
        else:
            de_dv = vi + DS * (-a * (track.p_ice[i] + u[i]) / vi ** 2
                               - 2.0 * drag[i] * vi)
            de_du = DS * a / vi
            dvn_dv = de_dv / v_free
            dvn_du = de_du / v_free

        downstream = ddt_dv + phi_next
        dT_du = downstream * dvn_du  # negative: more power, less time
        if dt[i] > 0:
            gain[i] = max(-dT_du / dt[i], 0.0)  # per joule, since dE = u * dt
        phi_next = ddt_dv + downstream * dvn_dv
    return gain, -phi_next  # phi_next is now the sensitivity at the start line
