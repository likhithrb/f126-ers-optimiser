"""What is actually in a recording, found by reading it rather than by asking.

Every fit in this project needs a particular kind of driving to identify itself:
the drag model needs laps with the motor off, the taper needs laps with it flat
out, the tow needs laps spent within a second of somebody, PIT_VALUE needs a
stop. Up to now the way to find those was to remember which lap you did what on.

The telemetry already says. `pit` marks the in-lap, out-lap and the stationary
stop; `delta_front` and `gap_behind` say who was near you and for how long;
`ers_deployed` says how hard you leaned on the battery; ERS mode Overtake says
when you pressed the override. So this walks a recording once and writes down
what each lap contains, then says which laps are worth pointing the fitters at.

    python3 -m f126ers.app --index sessions/race.f1
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from . import telemetry

CLOSE = 1.0  # s; inside this you are in someone's wake and it shows in the data
CLEAR_AIR = 2.0  # s; outside this nobody is affecting you
# The game has no true motor-off mode -- a lap driven entirely in ERS mode
# "None" still deployed 7.1 MJ. So what the drag fit needs is not an absolute
# floor but a *spread*: laps that differ enough in deployment to separate motor
# power from drag. This is the fraction of the highest-deployment lap below
# which a lap counts as a low one.
LOW_DEPLOY_FRAC = 0.85
DEPLOY_SPREAD = 0.15  # min (max-min)/max across laps for the fit to be identifiable
PIT_SPEED = 2.0  # m/s; slower than this in the pit lane is a stationary stop
# 2026 numbering. The sprint-shootout sessions sit at 10-14, which is where
# Race used to be in older formats -- a real Bahrain race reports 15. Getting
# this wrong does not error, it just quietly switches off everything that only
# applies to a race: the across-laps plan, the race tips, the overtake maths.
RACE_SESSIONS = (15, 16, 17)

SESSION_NAMES = {1: "practice 1", 2: "practice 2", 3: "practice 3",
                 4: "short practice", 5: "qualifying 1", 6: "qualifying 2",
                 7: "qualifying 3", 8: "short qualifying", 9: "one-shot qualifying",
                 10: "sprint shootout 1", 11: "sprint shootout 2",
                 12: "sprint shootout 3", 13: "short sprint shootout",
                 14: "one-shot sprint shootout",
                 15: "race", 16: "race 2", 17: "race 3", 18: "time trial"}


@dataclass
class LapFacts:
    """One lap, summarised. Everything here is measured, nothing is assumed."""

    lap: int = 0
    time: float = 0.0  # s, as the game reported it at the end of the lap
    valid: bool = True
    samples: int = 0
    pit: bool = False  # any part of the lap spent in the pit lane
    stopped: bool = False  # ... and stationary in it, i.e. an actual stop
    following: float = 0.0  # fraction of the lap within CLOSE of the car ahead
    defended: float = 0.0  # ... with someone that close behind
    clean: float = 0.0  # ... with nobody near at all
    deployed: float = 0.0  # J
    harvested: float = 0.0  # J
    soc_end: float = 0.0  # J
    overrides: int = 0  # manual override activations
    top_speed: float = 0.0  # m/s
    tyre_age: int = 0
    safety_car: bool = False
    position: int = 0

    @property
    def usable(self) -> bool:
        """Is this a lap the model can learn anything from?"""
        return self.valid and not self.pit and not self.safety_car and self.time > 0


@dataclass
class Session:
    """The recording as a whole."""

    path: str = ""
    bytes: int = 0
    packets: int = 0
    duration: float = 0.0  # s of session time covered
    track_length: float = 0.0
    session_type: int = 0
    total_laps: int = 0
    rate: float = 0.0  # samples per second, so a wrong UDP rate is obvious
    traction_control: int = 0
    abs_on: bool = False
    laps: list = field(default_factory=list)

    @property
    def is_race(self) -> bool:
        return self.session_type in RACE_SESSIONS

    @property
    def name(self) -> str:
        return SESSION_NAMES.get(self.session_type, f"session type {self.session_type}")


def scan(path: str) -> Session:
    """Reads a recording end to end and writes down what each lap contains."""
    sess = Session(path=path, bytes=os.path.getsize(path))
    merger = telemetry.Merger()
    cur: LapFacts | None = None
    t_first = t_last = None
    was_override = False
    tc_seen: dict = {}
    # ers_deployed and ers_harvested are counters that reset at the line, so the
    # lap total is the largest value seen -- not the last, which can already be
    # the next lap's zero if the packets interleave across the line.
    for _offset, data in telemetry.replay(path):
        sess.packets += 1
        s = merger.feed(data)
        if s is None or s.lap <= 0:
            continue
        if t_first is None:
            t_first = s.t
        t_last = s.t
        sess.track_length = s.track_length or sess.track_length
        sess.session_type = s.session_type or sess.session_type
        sess.total_laps = s.total_laps or sess.total_laps
        if s.speed > 5.0:
            # Only while actually moving. A handful of frames in the garage
            # report assists the driver never raced with, and the last frame of
            # a session is one of them.
            tc_seen[s.traction_control] = tc_seen.get(s.traction_control, 0) + 1
            sess.abs_on = s.abs_on

        if cur is None or s.lap != cur.lap:
            if cur is not None:
                _finish(cur)
                sess.laps.append(cur)
            cur = LapFacts(lap=s.lap)
            was_override = False

        cur.samples += 1
        # The largest reading, not the last: at the flag the lap number stops
        # advancing while the lap clock resets, so the final lap of a race
        # reported 0.017s.
        cur.time = max(cur.time, s.lap_time)
        cur.valid = cur.valid and not s.lap_invalid
        if s.pit:
            cur.pit = True
            if s.speed < PIT_SPEED:
                cur.stopped = True
        if 0.0 < s.delta_front <= CLOSE:
            cur.following += 1
        if 0.0 < s.gap_behind <= CLOSE:
            cur.defended += 1
        near = (0.0 < s.delta_front < CLEAR_AIR) or (0.0 < s.gap_behind < CLEAR_AIR)
        if not near:
            cur.clean += 1
        cur.deployed = max(cur.deployed, s.ers_deployed)
        cur.harvested = max(cur.harvested, s.ers_harvested)
        cur.soc_end = s.ers_store
        cur.top_speed = max(cur.top_speed, s.speed)
        cur.tyre_age = s.tyre_age
        cur.position = s.position or cur.position
        if s.safety_car:
            cur.safety_car = True
        if s.overtake_active and not was_override:
            cur.overrides += 1
        was_override = s.overtake_active

    if cur is not None:
        _finish(cur)
        sess.laps.append(cur)
    if tc_seen:
        sess.traction_control = max(tc_seen, key=tc_seen.get)
    if t_first is not None:
        sess.duration = max(t_last - t_first, 0.0)
        total = sum(l.samples for l in sess.laps)
        sess.rate = total / sess.duration if sess.duration > 0 else 0.0
    return sess


def _finish(lap: LapFacts) -> None:
    """Turns the per-sample counts into fractions of the lap."""
    n = max(lap.samples, 1)
    lap.following /= n
    lap.defended /= n
    lap.clean /= n


# -- what the recording is good for ---------------------------------------

@dataclass
class Coverage:
    """One thing the model needs, and whether this recording has it."""

    what: str  # the quantity it would calibrate
    got: bool
    where: str  # which laps to look at
    why: str  # what it is for, in one line


def coverage(sess: Session) -> list:
    """Which fits this recording can feed, and which laps feed them."""
    laps = [l for l in sess.laps if l.usable]
    out = []

    def add(what, laps_found, why, need=1):
        nums = [l.lap for l in laps_found]
        got = len(nums) >= need
        where = ("lap " + ", ".join(str(n) for n in nums[:8])
                 + ("…" if len(nums) > 8 else "")) if nums else "none found"
        out.append(Coverage(what, got, where, why))

    stops = [l for l in sess.laps if l.stopped]
    add("pit stop", stops,
        "prices a lap spent in the pit lane (PIT_VALUE, currently an estimate)")

    add("following within 1s", [l for l in laps if l.following > 0.2],
        "fits the tow and the time you lose stuck behind someone")

    add("clean air", [l for l in laps if l.clean > 0.8],
        "the baseline every other lap is measured against", need=2)

    if laps:
        top = max(l.deployed for l in laps)
        low = [l for l in laps if l.deployed < LOW_DEPLOY_FRAC * top]
        spread = (top - min(l.deployed for l in laps)) / max(top, 1.0)
    else:
        low, spread = [], 0.0
    add("low-deployment laps", low if spread >= DEPLOY_SPREAD else [],
        f"separates drag from motor power ({spread*100:.0f}% spread between "
        f"your heaviest and lightest laps, {DEPLOY_SPREAD*100:.0f}% needed)")

    hard = sorted(laps, key=lambda l: -l.deployed)[:3]
    add("flat-out laps", [l for l in hard if l.deployed > 2.5e6],
        "fits motor efficiency and the 290-355 km/h deployment taper")

    add("manual override", [l for l in laps if l.overrides > 0],
        "fits the override taper, which is fitted separately from the normal one")

    ages = {l.tyre_age for l in laps}
    add("tyre age spread", laps if len(ages) >= 5 else [],
        f"fits grip against tyre age ({len(ages)} distinct ages seen)")

    add("race with a known length",
        laps if (sess.is_race and sess.total_laps) else [],
        "gives the stint planner a horizon to plan energy against")

    return out


def report(sess: Session) -> str:
    """The whole thing as plain text: what was captured, and what it is good for."""
    lines = []
    mb = sess.bytes / 1e6
    lines.append(f"RECORDING  {os.path.basename(sess.path)}")
    lines.append(f"  {mb:.1f} MB, {sess.packets:,} packets, "
                 f"{sess.duration/60:.1f} minutes of session time")
    if sess.rate:
        note = "" if sess.rate > 40 else "  <- game is sending 20 Hz, prefer 60"
        lines.append(f"  {sess.rate:.0f} frames per second{note}")
    lines.append(f"  {sess.name}"
                 + (f", {sess.total_laps} laps scheduled" if sess.total_laps
                    else ", no scheduled distance")
                 + (f", {sess.track_length:.0f} m lap" if sess.track_length else ""))
    tc = {0: "off", 1: "medium", 2: "full"}.get(sess.traction_control, "?")
    lines.append(f"  assists: traction control {tc}, ABS "
                 + ("on" if sess.abs_on else "off")
                 + ("   <- exits are assist-limited, not tyre-limited"
                    if sess.traction_control else ""))
    lines.append("")

    if not sess.laps:
        lines.append("No laps in this file. Either the recording stopped before you")
        lines.append("crossed the line, or the game was not sending lap data.")
        return "\n".join(lines)

    lines.append("LAPS")
    lines.append("  lap    time   deploy  harvest   ahead  behind   flags")
    for l in sess.laps:
        flags = []
        if l.stopped:
            flags.append("PIT STOP")
        elif l.pit:
            flags.append("pit lane")
        if l.safety_car:
            flags.append("safety car")
        if not l.valid:
            flags.append("invalid")
        if l.overrides:
            flags.append(f"{l.overrides}x override")

        t = f"{l.time:7.3f}" if l.time else "      -"
        lines.append(
            f"  {l.lap:3d} {t} {l.deployed/1e6:6.2f}MJ {l.harvested/1e6:6.2f}MJ "
            f"{l.following*100:5.0f}% {l.defended*100:5.0f}%   "
            + ", ".join(flags))
    usable = sum(1 for l in sess.laps if l.usable)
    lines.append(f"  {usable} of {len(sess.laps)} laps usable "
                 f"(the rest are pit, safety car or invalidated)")
    lines.append("")
    lines.append("  'ahead'/'behind' are the share of the lap spent within one")
    lines.append("  second of the car in front / behind.")
    lines.append("")

    lines.append("WHAT THIS RECORDING CAN CALIBRATE")
    for c in coverage(sess):
        mark = "yes" if c.got else " no"
        lines.append(f"  [{mark}] {c.what:<24} {c.where}")
        lines.append(f"         {c.why}")
    missing = [c for c in coverage(sess) if not c.got]
    lines.append("")
    if missing:
        lines.append("MISSING — worth two minutes next session")
        for c in missing:
            lines.append(f"  - {c.what}: {c.why}")
    else:
        lines.append("Nothing missing: every fit in the model has data to work from.")
    return "\n".join(lines)


def _self_check() -> None:
    """Round-trips a tiny synthetic recording through scan()."""
    import struct
    import tempfile

    def header(ptype, t):
        return struct.Struct("<HBBBBBQfIIBB").pack(
            telemetry.PACKET_FORMAT, 26, 1, 0, 1, ptype, 1, t, 0, 0, 0, 255)

    path = os.path.join(tempfile.mkdtemp(), "t.f1")
    rec = telemetry.Recorder(path)

    # Session: race, 5 laps, 5000 m.
    body = bytearray(700)
    struct.pack_into("<BHB", body, 3, 5, 5000, 15)  # 15 = race
    rec.write(header(telemetry.SESSION, 0.0) + bytes(body))

    for lap, (dist, pit, df, spd, dep) in enumerate(
            [(100.0, 0, 0.5, 80.0, 3.0e6), (200.0, 1, 0.0, 1.0, 0.1e6)], start=1):
        lp = bytearray(24 * telemetry.LAP_ITEM)
        struct.pack_into("<IIHBHBHBHBfffBBBBBB", lp, 0,
                         0, 30000, 0, 0, 0, 0, int(df * 1000), 0, 0, 0,
                         dist, 0.0, 0.0, 1, lap, pit, 0, 1, 0)
        st = bytearray(24 * telemetry.STATUS_ITEM)
        struct.pack_into(
            "<BBBBBfffHHBBHBBBBfffBffff?", st, 0,
            0, 0, 0, 0, 0,          # traction control .. pit limiter
            100.0, 110.0, 20.0,     # fuel in tank, capacity, remaining laps
            0, 0, 0, 0, 0,          # rpm limits, drs allowed/distance
            0, 0, 3, 0,             # compounds, tyre age (index 15), flag
            0.0, 0.0, 4.0e6,        # p_ice, p_mguk, ers_store
            1,                      # ers mode
            1.0e6, 0.0, 8.5e6, dep,  # harvested, mgu-h, limit, deployed
            False)
        tl = bytearray(24 * telemetry.TEL_ITEM)
        struct.pack_into("<HfffBbHB", tl, 0, int(spd * 3.6), 1.0, 0.0, 0.0,
                         0, 6, 10000, 0)
        for _ in range(10):
            rec.write(header(telemetry.LAP_DATA, 0.0) + bytes(lp))
            rec.write(header(telemetry.CAR_STATUS, 0.0) + bytes(st))
            rec.write(header(telemetry.CAR_TELEMETRY, 0.0) + bytes(tl))
    rec.close()

    sess = scan(path)
    assert sess.total_laps == 5 and sess.is_race, (sess.total_laps, sess.session_type)
    assert len(sess.laps) == 2, sess.laps
    one, two = sess.laps
    assert one.following > 0.9, one.following  # 0.5 s behind the car ahead
    assert one.deployed == 3.0e6, one.deployed
    assert one.usable and not one.pit
    assert two.pit and two.stopped, (two.pit, two.stopped)  # in the pit, stationary
    assert not two.usable
    cov = {c.what: c for c in coverage(sess)}
    assert cov["pit stop"].got, "a stationary lap in the pit lane is a stop"
    assert cov["following within 1s"].got
    assert not cov["manual override"].got, "never pressed it"
    print(f"index ok              {len(sess.laps)} laps, pit stop on lap "
          f"{two.lap}, {sum(c.got for c in cov.values())}/{len(cov)} fits covered")
    os.remove(path)


if __name__ == "__main__":
    _self_check()
