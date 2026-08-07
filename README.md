# F1 26 ERS Race Optimiser

Live energy-deployment coaching for **EA F1 25: 2026 Season Pack**. It reads the
game's UDP telemetry at 60 Hz, identifies the car's physics from your own laps,
solves the optimal deployment problem as an optimal-control problem, and at the
end of every lap tells you the single biggest thing ERS is costing you, in
seconds.

It is not a lookup table of "deploy on corner exit". It re-derives what to do
from your data, every lap, under the conditions you are actually in — the
battery you have left, the traffic you are in, the harvest your braking is
actually producing.

```bash
python3 -m f126ers.app --web                 # browser dashboard (recording on)
python3 -m f126ers.app                       # terminal dashboard (recording on)
python3 -m f126ers.app --index               # what is in what I just drove?
python3 -m f126ers.app --replay              # re-analyse it in full
python3 -m f126ers.app --quali               # one-lap mode, may finish empty
python3 -m f126ers.app --baseline 5          # 5 uncoached laps, then coach
```

In game: **Settings → Telemetry → UDP On, Format 2026, Port 20777, Rate 60 Hz,
Your Telemetry: Public**. The last two matter: 20 Hz halves the resolution the
dynamics fit needs, and anything other than Public strips the other cars, which
kills the tow and gap-behind measurements.

## Recording

Recording is always on. Every run writes two files into `sessions/`:

| | |
|---|---|
| `<timestamp>.f1` | every UDP packet, raw and unfiltered — replay is byte-identical |
| `<timestamp>.log` | plain-text, one line per lap: time, energy, and what that lap contained |

Nothing has to be noted by hand. Pit stops, laps spent within a second of
another car, low-deployment laps, override use, assist settings and safety-car
periods are all *derived* from the packets, so when the session ends the tool
prints which laps are worth looking at and which of the model's fits this
session can feed:

```
WHAT THIS RECORDING CAN CALIBRATE
  [yes] pit stop                 lap 8
         prices a lap spent in the pit lane (PIT_VALUE, currently an estimate)
  [yes] following within 1s      lap 4, 5, 9
         fits the tow and the time you lose stuck behind someone
  [ no] clean air                none found
         the baseline every other lap is measured against
```

Same thing on demand: `--index` on its own reports the newest recording and
lists the older ones. `--replay` likewise defaults to the newest.

No dependencies beyond numpy. Try it without the game:

```bash
python3 make_fake_session.py session.f1 && python3 -m f126ers.app --replay session.f1
```

## The problem

Every ERS guide online is written for a qualifying lap: one lap, start full,
finish empty, deploy on the exits. A race is a different problem, and a harder
one:

- The lap has to be **repeatable**. Energy spent this lap that you don't harvest
  back is energy you don't have next lap. A quick lap that leaves you empty is a
  loan, and the tool prices it as one.
- **Harvest is not a constant.** With no car ahead you brake later and recover
  less. The energy budget therefore changes lap to lap, and so does the right
  deployment.
- Spending sector 1's energy leaves sector 3 dry, and **where** the shortfall
  bites depends on the circuit, not on a rule of thumb.

## The maths

### 1. System identification

The longitudinal model, per unit mass, over distance `s`:

```
d(v²/2)/ds  =  a·P(s)/v  −  drag(s)·v²  −  c
drag(s)     =  b + db·aero(s)
```

`P` is measured directly — the 2026 spec reports `EnginePowerICE` and
`EnginePowerMGUK` in watts, so deployment never has to be inferred. The unknowns
`(a, b, db, c)` are fitted by least squares, and three details matter:

- **Integral form.** The game reports speed as a whole number of km/h.
  Differencing that over a 10 m bin gives an acceleration estimate noisier than
  the signal, so the equation is integrated over a ~100 m baseline first. Still
  linear in the unknowns, no longer swamped by quantisation.
- **Pooled across laps.** One lap has one deployment pattern, which leaves power
  and drag nearly collinear: the fit reproduces that lap and mispredicts every
  other one. Since the entire job is predicting laps you *haven't* driven, rows
  are pooled across laps, where different deployment patterns supply the
  excitation that makes the parameters identifiable.
- **Ridge toward the previous estimate**, in normalised column units. That makes
  it a recursive estimator: it tracks a car whose fuel load and tyres are
  changing, without lurching on one noisy lap.

`db` captures the 2026 active-aero drag difference between corner and
straight-line mode — far too large to average over.

### 2. Power limited or grip limited?

For each bin the model asks whether speed was set by engine power or by grip, by
comparing the acceleration achieved against what the delivered power should have
produced. Where the car fell well short at full throttle, grip was binding.

This is the step that makes corner-exit advice honest. A car on exit is
**traction limited** — already using every newton the tyres will take — and extra
deployment there buys nothing. Treating "full throttle" as "power limited" would
have the optimiser promise exit time the tyres cannot deliver.

### 3. The optimiser

```
minimise    T = Σ ds / v
over        u(s) ∈ [0, P_max(v)]
subject to  dE/ds = −u/v + h(s),   0 ≤ E ≤ E_cap,   E(L) ≥ E_target
```

Solved by **Pontryagin / Lagrangian relaxation**, not a 2-D dynamic program over
(distance, charge). Relaxing the energy budget with a multiplier λ makes the
problem separable into a 1-D dynamic program over speed alone:

```
min_u  Σ [ dt + λ·u·dt ]
```

Bisecting λ until the solution spends exactly the available budget recovers the
constrained optimum — and hands back λ as a first-class object rather than a
finite difference of a value surface.

**λ is the shadow price of energy: seconds per joule.** It is the whole idea. The
optimal policy is bang-bang in the local time-gain-per-joule `g(s)`: deploy where
`g(s) > λ`, don't where it isn't. Corner exits beating the end of a straight
isn't a rule the tool was taught — it's what falls out, because `g` is large at
low speed and collapses to zero above the MGU-K taper knee.

When charge hits a bound mid-lap, λ is no longer constant: it jumps at the
contact point. The lap is split there and each segment solved with its own λ —
the standard state-constrained maximum-principle construction. That is what makes
"empty through sector 3" a solvable case rather than an error.

`E_target = E_start` by default, so the plan is repeatable every lap of a stint.
Qualifying mode is the same solver with `E_target = 0`.

### 4. Attribution

Three quantities, deliberately not mixed:

| | measured by |
|---|---|
| **Allocation loss** (this lap) | simulated actual lap − optimal lap *on the energy you actually spent* |
| **Harvest loss** (this lap) | re-solving on the energy your own best braking recovers in those zones |
| **Sustainability cost** (next lap) | λ × the charge deficit carried over |

Both laps run through the same simulator with the same speed envelope, so line
and braking cancel and what's left is ERS.

Comparing against *the same energy you spent* is what separates "you put it in
the wrong place" from "you spent more than the lap can afford" — otherwise the
two cancel and a battery-dump lap looks fine. Within the allocation loss, λ and
`g(s)` rank the symptoms (deploying where `g < λ`, failing to deploy where
`g > λ`) and the measured total is apportioned across them. **The ranking comes
from the economics; the total comes from the simulator** — pricing a whole block
of energy at λ would overstate it, because λ is by definition the value of the
*last* joule.

Symptoms detected: wasted deployment, battery depleted, under-deployed, missed
harvest, override misused, unsustainable.

## Validation

`python3 test_optimiser.py` — a toy circuit with a known answer:

- optimiser deploys **162 kW on corner exits, 0 kW at high speed**, derived, not told
- lap time monotone in the energy budget; λ falls as energy loosens (diminishing returns)
- charge bounds and the per-lap budget respected; qualifying mode beats neutral
- a battery-dump lap costs **+0.95 s** against the optimal use of the same energy
- feeding the optimum back in raises no issues (no false alarms)

`python3 f126ers/telemetry.py` and `f126ers/track.py` self-check packet layouts
and parameter recovery. End to end on a synthetic session where truth is known:

| metric | result |
|---|---|
| model fidelity (simulated vs actual lap time) | **0.13 %** |
| parameter recovery (`a`, `b` vs truth) | within **1.5 %** |

The fidelity number is the one that matters: the model predicts laps driven with
a deployment pattern it was never fitted to.

### On real telemetry

A 14-lap Bahrain race, 554 MB, 548,178 packets at 59 Hz:

| metric | result |
|---|---|
| model fidelity on a real car | **2.4 %** |
| laps analysed | 13 of 14 |
| pit stop, following laps, override use | all detected without being told |

Contact with real data found five bugs the synthetic fixture could not produce,
the worst of which was silent: `sessionType` is **15** in the 2026 format, not
10, so every race-only feature — the across-laps plan, the race tips, the
overtake economics — had been switching itself off in the one situation it was
written for.

## Layout

| file | |
|---|---|
| `telemetry.py` | UDP parsing (2026 layout), record/replay |
| `track.py` | distance binning, system identification, speed envelope, taper, harvest map |
| `optimiser.py` | forward simulator, co-state DP, λ, marginal time-gain |
| `analysis.py` | attribution, issue detectors, verdict, in-lap cues, session metrics |
| `stint.py` | energy curve, across-laps DP, pit-lap weighting |
| `traffic.py` | following deficit, attack economics, break-even probability |
| `places.py`, `situations.py` | where on track a tip applies, and race-context advice |
| `index.py` | what a recording contains, derived from the packets |
| `dashboard.py`, `web.py` | terminal and browser UIs |
| `app.py` | live / replay / index / baseline modes |

## Limits

- Whether an overtake actually sticks is not derivable from your telemetry, so
  it is inverted into a break-even probability rather than guessed at.
- `PIT_VALUE`, the worth of energy on a lap spent partly in the pit lane, is
  still an estimate. It becomes measurable from the first recording containing
  a stop.
- Point-mass longitudinal model: no fuel burn or corner-by-corner grip evolution
  beyond what the envelope and the tyre-age fit pick up.
- 2026 packet format only. Older formats are rejected with a clear message
  rather than silently misparsed.
