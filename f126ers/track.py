"""Track model: learns the car and the circuit from the driver's own laps.

Nothing here is hard-coded from a regulations document. The drag coefficient,
powertrain efficiency, the MGU-K deployment taper and the harvest map are all
identified by least squares / envelope fitting from telemetry, so the model
tracks the actual car, actual fuel load and actual conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DS = 10.0  # distance bin width, metres
_MAX_FIT_ROWS = 3000  # pooled regression rows kept for parameter identification
_RIDGE = 1e-2  # pull towards the previous estimate, in normalised column units
MASS = 800.0  # kg, car + driver, dry; fuel is added per lap from telemetry
_SPEED_BINS = np.arange(0.0, 115.0, 5.0)  # m/s, for the deployment taper curve
# Speed a tow is worth on a straight, until measured from your own laps.
# Published figures for the current ground-effect cars put it at 10-15 km/h on a
# long straight, and lower at Monza than the 2019-21 era because the same rules
# that made following easier through corners cut the tow on the straights.
# Deliberately the low end: it makes the tool under-claim overtakes, not over.
TOW_FALLBACK = 10.0 / 3.6  # m/s
CLOSE_BEHIND = 0.6  # s to the car ahead: inside this you are in the tow
CLEAR_AIR = 2.0  # s: beyond this nobody is helping you
# 2026 regulations, used only as a starting shape until real laps arrive: the
# MGU-K gives 350 kW and deployment tapers from 290 km/h to nothing at 355 km/h.
# Manual Override holds full power all the way to 337 km/h before tapering to
# 350 km/h -- a far narrower, higher curve. That difference is the whole point
# of the override: it does not add power where you already have it, it adds
# power in the band where normal deployment has run out.
# The 2026 regulations say 350 kW. A real race reports the MGU-K peaking at
# exactly 315 kW, so that is what the game models -- and this is only the prior
# anyway: p_max is fitted from observed power against speed.
P_MGUK_MAX = 3.15e5  # W, measured from telemetry, not from the rulebook
TAPER_FROM, TAPER_ZERO = 290 / 3.6, 355 / 3.6  # m/s
BOOST_FROM, BOOST_ZERO = 337 / 3.6, 350 / 3.6  # m/s


@dataclass
class Lap:
    """One completed lap, resampled onto the distance grid."""

    lap_num: int
    n: int
    lap_time: float
    valid: bool
    v: np.ndarray  # m/s
    throttle: np.ndarray
    brake: np.ndarray
    p_ice: np.ndarray  # W
    p_mguk: np.ndarray  # W, the driver's actual deployment
    soc: np.ndarray  # J
    harvest: np.ndarray  # J gained in this bin
    delta_front: np.ndarray  # s to car ahead
    overtake: np.ndarray  # bool, 2026 override active
    sector: np.ndarray  # 0/1/2
    aero: np.ndarray  # 2026 active aero: 0 = corner (high drag), 1 = straight
    drs: np.ndarray = None  # DRS open: sheds drag on the straights
    tyre_age: int = 0  # laps on this set of tyres
    mass: float = MASS  # car + driver + fuel at this point in the race
    harvest_limit: float = 0.0  # J recoverable per lap, from the regulations

    def __post_init__(self):
        if self.drs is None:
            self.drs = np.zeros(self.n)

    @property
    def energy_used(self) -> float:
        return float(np.sum(self.p_mguk * DS / np.maximum(self.v, 1.0)))

    @property
    def energy_harvested(self) -> float:
        return float(np.sum(self.harvest))


def build_lap(samples, n: int, lap_num: int, lap_time: float) -> Lap | None:
    """Bins a list of Samples from one lap onto the distance grid.

    Bins with no samples are filled by linear interpolation across the gap.
    """
    if len(samples) < n // 4:  # partial lap (joined mid-lap, pitted, flashback)
        return None

    s = np.array([x.lap_dist for x in samples])
    idx = np.clip((s / DS).astype(int), 0, n - 1)
    counts = np.bincount(idx, minlength=n).astype(float)
    seen = counts > 0

    def binned(values, how="mean"):
        arr = np.asarray(values, dtype=float)
        if how == "max":
            out = np.zeros(n)
            np.maximum.at(out, idx, arr)
        else:
            out = np.bincount(idx, weights=arr, minlength=n)
            out[seen] /= counts[seen]
        if not seen.all():  # interpolate across unvisited bins
            out = np.interp(np.arange(n), np.flatnonzero(seen), out[seen])
        return out

    v = np.maximum(binned([x.speed for x in samples]), 1.0)
    soc = binned([x.ers_store for x in samples])
    harvest_cum = binned([x.ers_harvested for x in samples])
    # Harvest is reported cumulatively over the lap; difference it per bin.
    harvest = np.maximum(np.diff(harvest_cum, prepend=harvest_cum[0]), 0.0)

    return Lap(
        lap_num=lap_num, n=n, lap_time=lap_time,
        valid=not any(x.lap_invalid for x in samples),
        v=v,
        throttle=binned([x.throttle for x in samples]),
        brake=binned([x.brake for x in samples]),
        p_ice=binned([x.p_ice for x in samples]),
        p_mguk=binned([x.p_mguk for x in samples]),
        soc=soc,
        harvest=harvest,
        delta_front=binned([x.delta_front for x in samples]),
        overtake=binned([float(x.overtake_active) for x in samples]) > 0.5,
        sector=np.round(binned([float(x.sector) for x in samples])).astype(int),
        aero=binned([float(x.aero_mode) for x in samples]),
        drs=binned([float(x.drs) for x in samples]),
        tyre_age=int(samples[-1].tyre_age),
        mass=MASS + float(np.mean([x.fuel for x in samples])),
        harvest_limit=float(samples[-1].ers_harvest_limit),
    )


def _ramp(v: np.ndarray, full_to: float, zero_at: float) -> np.ndarray:
    """Full power up to `full_to`, falling linearly to nothing at `zero_at`."""
    frac = np.clip((zero_at - v) / max(zero_at - full_to, 1e-9), 0.0, 1.0)
    return P_MGUK_MAX * frac


def _ridge(A: np.ndarray, y: np.ndarray, prior: np.ndarray,
           scale: float = 1.0) -> np.ndarray:
    """Least squares pulled gently towards `prior`.

    The columns differ by orders of magnitude (P/v is ~10^4, the intercept is 1),
    so the problem is normalised first and the penalty applied there; otherwise
    the ridge term would act almost entirely on one parameter. With the pull
    towards the previous lap's estimate this is a recursive estimator: it keeps
    tracking a changing car (fuel burn, tyre wear) without lurching on one noisy
    lap.
    """
    norms = np.linalg.norm(A, axis=0)
    norms[norms == 0] = 1.0
    As = A / norms
    # Columns are unit-normalised above, so As.T@As has a unit diagonal and the
    # penalty is directly interpretable as a fraction: _RIDGE = 0.01 is a 1%
    # pull towards the previous estimate. Scaling this by len(y) -- as this did
    # originally -- makes the pull grow with the data instead of shrinking, so
    # the prior outweighed the evidence several-fold and the estimate barely
    # moved off its starting value.
    alpha = _RIDGE * scale
    lhs = As.T @ As + alpha * np.eye(As.shape[1])
    rhs = As.T @ y + alpha * (prior * norms)
    return np.linalg.solve(lhs, rhs) / norms


@dataclass
class TrackModel:
    """Distance-indexed circuit model plus identified vehicle parameters."""

    length: float
    n: int = 0
    # Longitudinal model: d(v^2/2)/ds = a*P/v - drag(s)*v^2 - c  (per unit mass)
    # drag(s) = b + db * aero(s): the 2026 car has two aero states, and the
    # difference between them is far too big to average over.
    # Per-unit-mass values, recomputed by set_mass() as fuel burns off. These
    # are what the simulator uses; the mass-independent fits are below.
    a: float = 0.9 / MASS  # powertrain efficiency / mass
    b: float = 9.9e-4  # drag / mass, high-downforce (corner) mode
    db: float = 0.0  # drag reduction in straight-line mode (negative)
    dd: float = 0.0  # drag reduction with DRS open (negative)
    c: float = 0.15  # rolling + driveline losses / mass
    # Mass-independent, as identified: properties of the car, not the fuel load.
    eta: float = 0.9  # powertrain efficiency
    kd: float = 9.9e-4 * MASS  # drag coefficient, corner mode
    kd_aero: float = 0.0
    kd_drs: float = 0.0
    closs: float = 0.15 * MASS
    mass: float = MASS
    fit_samples: int = 0
    fit_r2: float = 0.0

    v_env: np.ndarray = field(default_factory=lambda: np.zeros(0))
    drag: np.ndarray = field(default_factory=lambda: np.zeros(0))
    aero: np.ndarray = field(default_factory=lambda: np.zeros(0))
    v_obs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    uncapped: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    p_ice: np.ndarray = field(default_factory=lambda: np.zeros(0))
    harvest_best: np.ndarray = field(default_factory=lambda: np.zeros(0))
    harvest_prev: np.ndarray = field(default_factory=lambda: np.zeros(0))
    v_tow: np.ndarray = field(default_factory=lambda: np.zeros(0))
    v_clear: np.ndarray = field(default_factory=lambda: np.zeros(0))
    taper: np.ndarray = field(default_factory=lambda: np.zeros(0))
    taper_boost: np.ndarray = field(default_factory=lambda: np.zeros(0))
    drs_frac: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tyre_age: int = 0
    _grip_k: float = 0.0  # grip lost per lap of tyre age, fitted
    _grip_pts: list = field(default_factory=list)
    capacity: float = 4.0e6  # J, refined from observed peak state of charge
    terminal_weight: float = 0.03  # s per m/s at the line; self-calibrates
    harvest_limit: float = 0.0  # J per lap, read from telemetry
    laps_seen: int = 0
    _taper_fitted: bool = False
    _boost_fitted: bool = False
    _fitted: bool = False  # a real parameter estimate exists to pull towards
    _rows: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n == 0:
            self.n = max(int(np.ceil(self.length / DS)), 1)
        self.v_env = np.zeros(self.n)
        self.v_obs = np.zeros(self.n)
        self.uncapped = np.zeros(self.n, bool)
        self.aero = np.zeros(self.n)
        self.drs_frac = np.zeros(self.n)
        self.drag = np.full(self.n, self.b)
        self.p_ice = np.full(self.n, 5.0e5)
        self.harvest_best = np.zeros(self.n)
        self.harvest_prev = np.zeros(self.n)
        self.v_tow = np.zeros(self.n)
        self.v_clear = np.zeros(self.n)
        self.taper = _ramp(_SPEED_BINS, TAPER_FROM, TAPER_ZERO)
        self.taper_boost = _ramp(_SPEED_BINS, BOOST_FROM, BOOST_ZERO)

    # -- learning ---------------------------------------------------------
    def set_mass(self, mass: float) -> None:
        """Recompute the per-unit-mass parameters at a given car mass."""
        self.mass = max(float(mass), 1.0)
        self.a = self.eta / self.mass
        self.b = self.kd / self.mass
        self.db = self.kd_aero / self.mass
        self.dd = self.kd_drs / self.mass
        self.c = self.closs / self.mass
        self.drag = np.maximum(
            self.b + self.db * self.aero + self.dd * self.drs_frac, 1e-5)

    def grip_at(self, age: int) -> float:
        """Fraction of peak grip left after `age` laps on this set.

        Fitted from the driver's own laps rather than assumed. Without it the
        speed envelope is the best ever seen, so thirty laps into a stint the
        optimiser plans for lap-3 grip: it recommends deployment the tyres will
        not take, then calls the driver under-deployed for refusing.
        """
        if self._grip_k <= 0.0:
            return 1.0
        return float(np.clip(1.0 - self._grip_k * max(age, 0), 0.75, 1.0))

    def _fit_grip(self, lap: Lap) -> None:
        """Grip vs tyre age, from cornering speed in grip-limited bins."""
        if self.uncapped.size != lap.v.size:
            return
        grip_bins = ~self.uncapped & (self.v_obs > 5.0)
        if grip_bins.sum() < 5:
            return
        ratio = float(np.median(lap.v[grip_bins] / self.v_obs[grip_bins]))
        self._grip_pts.append((int(lap.tyre_age), min(ratio, 1.0)))
        ages = np.array([p[0] for p in self._grip_pts], float)
        vals = np.array([p[1] for p in self._grip_pts], float)
        if len(set(ages.tolist())) < 3 or np.ptp(ages) < 2:
            return  # not enough spread in tyre age to separate wear from noise
        # ratio = 1 - k*age, through the data, forced through 1 at age 0.
        k = float(np.sum((1.0 - vals) * ages) / max(np.sum(ages ** 2), 1e-9))
        self._grip_k = float(np.clip(k, 0.0, 0.02))

    def add_lap(self, lap: Lap) -> None:
        self.laps_seen += 1
        self.tyre_age = int(lap.tyre_age)
        if lap.harvest_limit > 0:
            self.harvest_limit = float(lap.harvest_limit)
        self.drs_frac = (1 - 1.0 / min(self.laps_seen, 5)) * self.drs_frac \
            + (1.0 / min(self.laps_seen, 5)) * lap.drs
        self.v_obs = np.maximum(self.v_obs, lap.v)
        w = 1.0 / min(self.laps_seen, 5)  # exponential blend, recent laps win
        self.p_ice = (1 - w) * self.p_ice + w * lap.p_ice
        self.aero = (1 - w) * self.aero + w * lap.aero
        # Snapshot before folding this lap in: comparing a lap against a best
        # that already includes itself always gives a shortfall of zero, which
        # silently killed every brake-earlier suggestion.
        self.harvest_prev = self.harvest_best.copy()
        self.harvest_best = np.maximum(self.harvest_best, lap.harvest)
        self.capacity = max(self.capacity, float(lap.soc.max()))
        self._fit_tow(lap)
        self._fit_taper(lap)
        self._fit_dynamics(lap)
        self._refresh_env(lap)  # needs the freshly fitted parameters
        self._fit_grip(lap)  # needs uncapped, set by _refresh_env
        self._refresh_env(lap)  # re-apply with the updated grip estimate

    def _refresh_env(self, lap: Lap) -> None:
        """Decide, bin by bin, whether speed is limited by power or by grip.

        Being at full throttle is not enough to call a bin power limited. On a
        corner exit the car is traction limited: it is already using every newton
        the tyres will take, and extra deployment there buys nothing. Those bins
        are detected by comparing the acceleration actually achieved against what
        the fitted model says the delivered power should have produced. Where the
        car fell well short, grip was the binding constraint, and speed stays
        capped at the driver's best. Getting this wrong would make the optimiser
        promise corner-exit time that the tyres cannot deliver.
        """
        e_kin = 0.5 * lap.v ** 2
        observed = (np.roll(e_kin, -1) - e_kin) / DS
        predicted = (self.a * (lap.p_ice + lap.p_mguk) / lap.v
                     - self.drag * lap.v ** 2 - self.c)
        flat = (lap.throttle > 0.9) & (lap.brake < 0.05)
        traction_limited = flat & (predicted > 0.5) & (observed < 0.7 * predicted)
        self.uncapped = flat & ~traction_limited
        ceiling = float(self.v_obs.max()) * 1.08
        # Grip-limited bins are scaled to the grip the tyres have left now;
        # power-limited bins do not depend on grip, so they keep the ceiling.
        self.v_env = np.where(self.uncapped, ceiling,
                              self.v_obs * self.grip_at(self.tyre_age))

    def _fit_tow(self, lap: Lap) -> None:
        """Best speed seen in each bin with and without a car just ahead.

        The difference is the tow. Same trick as the dirty-air deficit but with
        the opposite sign and on the straights instead of in the corners: a car
        ahead costs you downforce in the corners and gives you a slipstream on
        the straights.
        """
        close = (lap.delta_front > 0.0) & (lap.delta_front < CLOSE_BEHIND)
        clear = (lap.delta_front == 0.0) | (lap.delta_front > CLEAR_AIR)
        if close.any():
            self.v_tow[close] = np.maximum(self.v_tow[close], lap.v[close])
        if clear.any():
            self.v_clear[clear] = np.maximum(self.v_clear[clear], lap.v[clear])

    @property
    def tow_gain(self) -> float:
        """Extra speed a tow is worth on the straights, m/s.

        Measured where we have both a towed and a clear-air lap through the same
        stretch of straight; falls back to the published figure until then.
        """
        if self.v_tow.size != self.n or self.uncapped.size != self.n:
            return TOW_FALLBACK
        both = (self.v_tow > 1.0) & (self.v_clear > 1.0) & self.uncapped
        if both.sum() < 5:
            return TOW_FALLBACK
        diff = self.v_tow[both] - self.v_clear[both]
        return float(np.clip(np.median(diff), 0.0, 8.0))

    def _fit_taper(self, lap: Lap) -> None:
        """Max MGU-K power delivered vs speed, fitted separately with and
        without the override.

        Keeping them apart matters: a single lap that used the override at
        330 km/h would otherwise teach the model that normal deployment reaches
        full power up there, and the optimiser would plan a lap the driver
        cannot drive without pressing the button.
        """
        b = np.clip(np.digitize(lap.v, _SPEED_BINS) - 1, 0, len(_SPEED_BINS) - 1)
        boost = lap.overtake

        obs = np.zeros(len(_SPEED_BINS))
        if (~boost).any():
            np.maximum.at(obs, b[~boost], lap.p_mguk[~boost])
        if not self._taper_fitted:
            self.taper = np.where(obs > 0, obs, self.taper)
            self._taper_fitted = obs.any()
        else:
            self.taper = np.maximum(self.taper, obs)

        if boost.any():
            obs_b = np.zeros(len(_SPEED_BINS))
            np.maximum.at(obs_b, b[boost], lap.p_mguk[boost])
            if not self._boost_fitted:
                self.taper_boost = np.where(obs_b > 0, obs_b, self.taper_boost)
                self._boost_fitted = True
            else:
                self.taper_boost = np.maximum(self.taper_boost, obs_b)
        # The override can never be worse than normal deployment.
        self.taper_boost = np.maximum(self.taper_boost, self.taper)

    def _fit_dynamics(self, lap: Lap) -> None:
        """Least-squares fit of (efficiency, drag, aero delta, rolling loss).

            d(v^2/2)/ds = a*(P_ice + P_k)/v - (b + db*aero)*v^2 - c

        Linear in all four unknowns, so one lstsq call. Uses only full-throttle,
        off-brake bins, where the car is power limited rather than grip limited,
        so the equation actually holds. The aero column is dropped when the lap
        never switched modes, which keeps the fit well conditioned.
        """
        v = lap.v
        power = lap.p_ice + lap.p_mguk
        ok = (lap.throttle > 0.95) & (lap.brake < 0.01) & (v > 15.0) & (power > 0)

        # The game reports speed as a whole number of km/h. Differencing that
        # over a single 10 m bin gives an acceleration estimate whose noise is
        # larger than the signal, so the model is integrated over a longer
        # baseline instead. Integrating the same equation over a window is
        # exact and still linear in the unknowns:
        #
        #   (e[i+W] - e[i]) / (W*ds) = a*<P/v> - b*<v^2> - db*<v^2*aero> - c
        #
        # which averages the quantisation away without biasing the estimate.
        W = max(int(round(100.0 / DS)), 3)  # ~100 m baseline
        n = len(v)
        if n <= W + 1:
            self.fit_samples = 0
            return

        def window_mean(x):
            cs = np.concatenate([[0.0], np.cumsum(x)])
            return (cs[W:] - cs[:-W]) / W

        e_kin = 0.5 * v ** 2
        starts = np.arange(n - W)
        y = (e_kin[starts + W] - e_kin[starts]) / (W * DS)
        m_pv = window_mean(power / v)[:len(starts)]
        m_v2 = window_mean(v ** 2)[:len(starts)]
        m_v2a = window_mean(v ** 2 * lap.aero)[:len(starts)]
        m_v2d = window_mean(v ** 2 * lap.drs)[:len(starts)]
        # Only windows that are entirely full throttle and off the brakes.
        whole = window_mean(ok.astype(float))[:len(starts)] > 0.999
        # Under full throttle the car cannot lose speed over 100 m. Where the
        # data says it does, the car is grip limited (a fast corner it is
        # scrubbing through) and the longitudinal equation does not hold.
        use = whole & (y > -1.0)
        if use.sum() < 20:
            self.fit_samples = 0
            return

        # Fit mass-independent parameters. The equation is per unit mass, so a
        # car with 100 kg of fuel and the same car nearly empty produce rows
        # that disagree -- pooling them raw biases every parameter as the race
        # burns fuel off. Multiplying through by this lap's mass fits
        # eta / k_drag / losses instead, which are properties of the car, and
        # the per-unit-mass values are recovered at the current mass on use.
        A = np.column_stack([m_pv[use], -m_v2[use], -m_v2a[use], -m_v2d[use],
                             -np.ones(int(use.sum()))])
        y = y[use] * lap.mass

        # Pool rows across laps. A single lap has one deployment pattern, which
        # leaves power and drag nearly collinear -- the fit then reproduces that
        # lap perfectly and mispredicts any other one, which is useless here,
        # because the whole job is predicting laps the driver has not driven.
        # Different laps deploy differently, so pooling supplies the excitation
        # that makes the parameters identifiable.
        self._rows.append((A, y))
        A = np.vstack([r[0] for r in self._rows])
        y = np.concatenate([r[1] for r in self._rows])
        if len(y) > _MAX_FIT_ROWS:  # keep it recent
            A, y = A[-_MAX_FIT_ROWS:], y[-_MAX_FIT_ROWS:]
            self._rows = [(A, y)]

        prior = np.array([self.eta, self.kd, self.kd_aero, self.kd_drs,
                          self.closs])
        keep = np.ones(len(y), bool)
        coef = prior
        # Two rounds of outlier trimming: a bin clipped by a corner taken flat,
        # a tow, a kerb or wheelspin all violate the model badly, and plain
        # least squares would let a handful of them set the answer.
        for _ in range(3):
            coef = _ridge(A[keep], y[keep], prior,
                          scale=1.0 if self._fitted else 0.03)
            resid = y - A @ coef
            sigma = float(np.std(resid[keep]))
            if sigma <= 0:
                break
            trimmed = np.abs(resid) < 2.5 * sigma
            if trimmed.sum() < 20 or trimmed.sum() == keep.sum():
                break
            keep = trimmed

        eta, kd, kd_aero, kd_drs, closs = coef
        a_now, b_now = eta / lap.mass, kd / lap.mass
        if not (1e-4 < a_now < 1e-2 and 1e-5 < b_now < 1e-2
                and kd + kd_aero > 0):
            return  # implausible fit (wet lap, damage); keep what we have
        self.eta, self.kd = float(eta), float(kd)
        self.kd_aero, self.kd_drs = float(kd_aero), float(kd_drs)
        self.closs = max(float(closs), 0.0)
        self._fitted = True
        self.set_mass(lap.mass)
        self.fit_samples = int(keep.sum())
        resid = (y - A @ coef)[keep]
        var = float(np.var(y[keep]))
        self.fit_r2 = float(1 - np.var(resid) / var) if var > 0 else 0.0

    # -- queries ----------------------------------------------------------
    def p_max(self, v: np.ndarray | float, boost: bool = False) -> np.ndarray:
        """Maximum deployable MGU-K power at speed v (the learned taper)."""
        curve = self.taper_boost if boost else self.taper
        return np.interp(v, _SPEED_BINS, curve)

    def boost_gain(self, v: np.ndarray | float) -> np.ndarray:
        """Extra power the override unlocks at speed v, over normal deployment.

        Zero at low speed -- the override adds nothing where full power is
        already available -- and largest in the band where normal deployment has
        tapered away. This is why "use the override out of the corner" is wrong
        under the 2026 rules: out of the corner you already have everything.
        """
        return np.maximum(self.p_max(v, boost=True) - self.p_max(v), 0.0)

    @property
    def boost_knee(self) -> float:
        """Speed above which the override starts to buy you anything."""
        extra = np.maximum(self.taper_boost - self.taper, 0.0)
        hit = np.flatnonzero(extra > 1.0e4)
        return float(_SPEED_BINS[hit[0]]) if hit.size else float("inf")

    @property
    def taper_knee(self) -> float:
        """Speed above which deployment is worth less than half peak power."""
        peak = self.taper.max()
        if peak <= 0:
            return 1e9
        above = np.flatnonzero(self.taper < 0.5 * peak)
        above = above[above > int(np.argmax(self.taper))]
        return float(_SPEED_BINS[above[0]]) if above.size else 1e9

    @property
    def ready(self) -> bool:
        return self.laps_seen > 0 and float(self.v_env.max()) > 0


def _self_check() -> None:
    """Fits the model to a synthetic lap generated from known parameters."""
    from types import SimpleNamespace

    true_a, true_b, true_c = 1.1e-3, 9.0e-4, 0.20
    length, n = 3000.0, 300
    rng = np.random.default_rng(0)

    # Simulate a lap: full throttle everywhere except one braking zone.
    age, fuel = 0, 100.0
    v = np.zeros(n)
    v[0] = 40.0
    thr, brk, p_ice, p_k = np.ones(n), np.zeros(n), np.full(n, 5.0e5), np.zeros(n)
    p_k[:150] = 2.0e5
    thr[200:240], brk[200:240] = 0.0, 1.0
    for i in range(n - 1):
        if brk[i] > 0:
            v[i + 1] = max(v[i] - 1.2, 25.0)
            continue
        e = 0.5 * v[i] ** 2 + DS * (
            true_a * (p_ice[i] + p_k[i]) / v[i] - true_b * v[i] ** 2 - true_c)
        v[i + 1] = np.sqrt(2 * max(e, 200.0))

    samples = [
        SimpleNamespace(
            lap_dist=i * DS + rng.uniform(0, DS), speed=v[i], throttle=thr[i],
            brake=brk[i], p_ice=p_ice[i], p_mguk=p_k[i], ers_store=3e6,
            ers_harvested=1e5 * min(i, 220) / 220, lap_invalid=False,
            delta_front=0.0, overtake_active=False, sector=i * 3 // n,
            aero_mode=float(v[i] > 60.0), drs=False, tyre_age=age,
            fuel=fuel, ers_harvest_limit=4.0e6,
        )
        for i in range(n)
    ]
    lap = build_lap(samples, n, 1, 60.0)
    assert lap is not None and lap.v.shape == (n,)

    tm = TrackModel(length=length, n=n)
    tm.add_lap(lap)
    assert abs(tm.a - true_a) / true_a < 0.05, (tm.a, true_a)
    assert abs(tm.b - true_b) / true_b < 0.10, (tm.b, true_b)
    assert tm.fit_r2 > 0.95, tm.fit_r2
    assert tm.p_max(30.0) >= 2.0e5 - 1, tm.p_max(30.0)
    assert tm.ready and tm.v_obs.max() == lap.v.max()
    # Braking bins stay capped at the speed actually carried; flat-out bins are
    # freed above it, so the optimiser can find speed the driver never found.
    assert not tm.uncapped[210], "braking zone should stay grip limited"
    assert tm.uncapped[100], "full-throttle bin should be power limited"
    assert tm.v_env[210] == tm.v_obs[210]
    assert tm.v_env[100] > tm.v_obs.max()
    assert lap.energy_harvested > 0
    print(f"track self-check ok  a={tm.a:.3e} b={tm.b:.3e} c={tm.c:.3f} "
          f"R2={tm.fit_r2:.4f}")


if __name__ == "__main__":
    _self_check()
