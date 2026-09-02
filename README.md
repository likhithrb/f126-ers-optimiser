# F1 26 ERS Race Optimiser

Live energy-deployment coaching for **EA F1 25: 2026 Season Pack**. It listens to
the game's UDP telemetry at 60 Hz, fits a physics model of your car from your own
laps, works out where the battery should have gone, and after every lap tells you
the biggest thing ERS cost you, in seconds.

Python 3.10+ and numpy. No other dependencies.

![Browser dashboard mid-race at Bahrain](docs/dashboard.png)

*Live during lap 6 of a Bahrain race. Real telemetry, replayed.*

## Running it

```bash
python3 -m f126ers.app --web          # browser dashboard (recording on)
python3 -m f126ers.app                # terminal dashboard (recording on)
python3 -m f126ers.app --index        # what is in the session I just drove?
python3 -m f126ers.app --replay       # re-analyse it in full
python3 -m f126ers.app --quali        # one-lap mode, finishes the lap empty
python3 -m f126ers.app --baseline 5   # 5 uncoached laps first, then coach
```

In game: Settings → Telemetry → **UDP On, Format 2026, Port 20777, Rate 60 Hz,
Your Telemetry: Public**. The last two settings matter. At 20 Hz the model fit
loses half its resolution, and anything other than Public strips the other cars
out of the packets, which kills the tow and gap measurements.

You can try it without the game at all:

```bash
python3 make_fake_session.py session.f1 && python3 -m f126ers.app --replay session.f1
```

## What it shows you

Deployment around the lap: your speed trace, where you put the energy, and where
the optimiser would have put the same energy.

![Deployment trace](docs/trace.png)

The verdict at the end of each lap, priced in seconds and pointed at a specific
stretch of track.

![Lap verdict](docs/verdict.png)

A ranked list of fixes, each with the time it is worth and the reason, so you can
pick one to work on rather than being told six things at once.

![Ranked tips](docs/tips.png)

The terminal dashboard shows the same thing over SSH or on a second monitor:

```
F1 26 ERS OPTIMISER   lap 6   sector 2
──────────────────────────────────────────────────────────────────────────────
  SOC ██████████░░░░░░░░░░░░░░  1.62 MJ   deployed 1.84  harvested 3.37
  237.0 km/h   gear 6   MGU-K 144.0 kW   lap  29.90s

  vs plan -0.39 MJ   energy price 0.125 s/MJ

  ◆ car behind:  KEEP SOMETHING FOR THE STRAIGHT
    they are close enough to get a run on you — arrive at the next straight
    with battery or you will be defending with none
──────────────────────────────────────────────────────────────────────────────
LAP 5  93.505s   optimal ERS would give 0.73s   model error 0.7%
  speed  █████████▇▄  ▁▃▄▅▆▆▆▇▇▆▂ ▁▃▄▅▄▃▃▃▄▁ ▁▃▄▅▄▂  ▃▄▅▆▇▇▇▇▅▂▂▃▄▅▅▅▅▅▅▂▁▃▄▅
  you                  ▄▇▅▃▁       ▃▆▄   ▃   ▂▃▃▃    ▆▇▇▆▅▃▃▁   ▃▃▃▃▃▃▃   ▄▇▆
  optimal▃            ▂▇▇▇▆▄      ▃▇▇  ▁     ▃▇▇▅    ▅▇▇▅▄▃    ▁▄▇   ▂    ▅▇▇
  charge            ▁▃▃▃▂▁▁▁▁▁▁▁▂▃▃▃▂▂▂▃▃▃▃▄▅▅▅▅▄▅▅▇▇▇▆▆▅▅▄▄▄▄▅▆▅▅▅▄▄▄▃▄▄▅▅▄

  BIGGEST LOSS 0.66s  Under-deployed
  1.69 MJ belonged in these corners and went elsewhere on the lap
  worst stretch 2330–2510 m
  → Get on the ERS as you pick up the throttle at the apex, not once the car
    is already straight.
```

## Recording

Recording is always on. Every run writes two files into `sessions/`: a `.f1` with
every UDP packet exactly as it arrived, so a replay is byte-identical, and a
`.log` with one plain-text line per lap.

Nothing has to be noted by hand. Pit stops, laps spent within a second of another
car, low-deployment laps, override use, assists and safety-car periods are all
worked out from the packets. `--index` reports what a recording contains and which
parts of the model it can calibrate:

```
RECORDING  2026-08-07-185828.f1
  554.3 MB, 548,178 packets, 22.0 minutes of session time
  59 frames per second
  race, 14 laps scheduled, 5408 m lap
  assists: traction control off, ABS off

LAPS
  lap    time   deploy  harvest   ahead  behind   flags
    1 100.155  10.12MJ   7.51MJ    87%    91%   safety car, 5x override
    2  91.789   9.05MJ   8.45MJ    19%    98%   6x override
    ...
    8 113.980   9.59MJ   9.00MJ     0%    81%   PIT STOP, 5x override
  11 of 14 laps usable (the rest are pit, safety car or invalidated)

WHAT THIS RECORDING CAN CALIBRATE
  [yes] pit stop                 lap 8
  [yes] low-deployment laps      lap 3, 10
  [ no] clean air                none found

MISSING — worth two minutes next session
  - clean air: the baseline every other lap is measured against
```

## What it does under the hood

- **Real-time telemetry pipeline.** Parses the 2026 UDP packet layout at 60 Hz,
  merges packet types into one sample stream, records and replays byte-identically.
- **System identification.** Fits a longitudinal vehicle model (power, drag, active
  aero, rolling resistance) from your own laps by regularised least squares,
  pooled across laps and updated recursively as fuel and tyres change.
- **Optimal control.** Solves optimal deployment as a constrained optimal-control
  problem via Lagrangian relaxation and dynamic programming, which prices energy
  in seconds per joule and drives every piece of advice the tool gives.
- **Attribution.** Splits each lap's ERS loss into where the energy went, how much
  was harvested, and what the deficit costs next lap, all measured through the
  same simulator so line and braking cancel out.
- **Race context.** Traffic, tow, overtake break-even probability, pit-lap
  weighting and an across-laps energy plan for the stint.

## Results

`python3 test_optimiser.py` runs a toy circuit with a known answer: the optimiser
deploys 162 kW on corner exits and 0 kW at high speed without being told to, lap
time is monotone in the energy budget, charge bounds hold, and a battery-dump lap
is correctly priced at +0.95 s against the optimal use of the same energy.

| | synthetic session | real 14-lap Bahrain race |
|---|---|---|
| model fidelity (simulated vs actual lap time) | 0.13 % | 2.4 % |
| parameter recovery vs truth | within 1.5 % | — |
| laps analysed | all | 13 of 14 |
| pit stop, traffic, override use | — | all detected without being told |

Fidelity is the number that matters, because it is measured on laps driven with a
deployment pattern the model was never fitted to. The real session is 554 MB and
548,178 packets at 59 Hz, and finding bugs in it that the synthetic fixture could
not produce is most of why the numbers above are trustworthy.

## Layout

| file | |
|---|---|
| `telemetry.py` | UDP parsing (2026 layout), record/replay |
| `track.py` | distance binning, system identification, speed envelope, taper, harvest map |
| `optimiser.py` | forward simulator, co-state DP, energy price, marginal time-gain |
| `analysis.py` | attribution, issue detectors, verdict, in-lap cues, session metrics |
| `stint.py` | energy curve, across-laps DP, pit-lap weighting |
| `traffic.py` | following deficit, attack economics, break-even probability |
| `places.py`, `situations.py` | where on track a tip applies, and race-context advice |
| `index.py` | what a recording contains, derived from the packets |
| `dashboard.py`, `web.py` | terminal and browser UIs |
| `app.py` | live / replay / index / baseline modes |

Tests sit at the top level (`test_optimiser.py`, `test_stint.py`, `test_traffic.py`,
`test_places.py`, `test_situations.py`); `python3 f126ers/telemetry.py` and
`python3 f126ers/track.py` self-check packet layouts and parameter recovery.

## Limits

- Whether an overtake sticks is not derivable from your telemetry, so it is
  inverted into a break-even probability instead of guessed at.
- `PIT_VALUE`, the worth of energy on a lap partly spent in the pit lane, is still
  an estimate. It becomes measurable from the first recording with a stop in it.
- Point-mass longitudinal model: no fuel burn, and no corner-by-corner grip
  evolution beyond what the envelope and the tyre-age fit pick up.
- 2026 packet format only. Older formats are rejected with a message rather than
  silently misparsed.
