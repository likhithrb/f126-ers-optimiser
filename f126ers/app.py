"""Entry point: live coaching, replay analysis, and session reporting.

    python3 -m f126ers.app --check              # is the game talking to us?
    python3 -m f126ers.app --web                # browser dashboard, recording on
    python3 -m f126ers.app                      # live, coaching on
    python3 -m f126ers.app --index              # what is in what I just drove?
    python3 -m f126ers.app --replay             # re-analyse it in full
    python3 -m f126ers.app --quali              # allow finishing the lap empty

Recording is always on and always goes to ./sessions: a race you did not record
is a race you cannot debug, and you only ever find that out afterwards. Each run
leaves two files -- the raw packet log (.f1) and a plain-text lap-by-lap account
of what went into it (.log) -- and prints, on the way out, which laps are worth
looking at and which of the model's fits this session can feed.
"""

from __future__ import annotations

import argparse
import os
import signal
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from . import (dashboard, index, places, situations, stint, telemetry,
               traffic, web)
from .analysis import Cues, LapReport, SessionMetrics, analyse
from .optimiser import Plan, solve
from .track import Lap, TrackModel, build_lap

REDRAW_INTERVAL = 0.25
# Recordings live with the code: they are this project's test data.
SESSION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sessions")
RACE_SESSIONS = index.RACE_SESSIONS
SAFETY_CAR_NAMES = {0: "", 1: "SAFETY CAR", 2: "VIRTUAL SAFETY CAR",
                    3: "FORMATION LAP"}
CUE_LINGER = 4.0  # seconds a cue stays on screen before it clears


@dataclass
class AppState:
    port: int = 20777
    track: TrackModel | None = None
    plan: Plan | None = None
    plan_lam: float = 0.0
    lap: int = 0
    sector: int = 0
    last_sample: object = None
    last_lap: Lap | None = None
    report: LapReport | None = None
    cue: str = ""
    safety_car: int = 0
    deficit: float = 0.0  # s/lap currently being lost to the car ahead
    total_laps: int = 0
    call: object = None  # latest traffic.PassCall
    corners: list = field(default_factory=list)
    tips: list = field(default_factory=list)
    next_tip: object = None  # the tip for the place we are about to reach
    race: list = field(default_factory=list)  # situational advice, right now
    stint_plan: object = None  # charge target per remaining lap
    pit_ideal: int = 0
    metrics: SessionMetrics = field(default_factory=SessionMetrics)

    @property
    def ers_mode_name(self) -> str:
        s = self.last_sample
        idx = getattr(s, "ers_mode", 0)
        return telemetry.ERS_MODES[idx] if idx < len(telemetry.ERS_MODES) else "?"


def speak(msg: str) -> None:
    """Non-blocking spoken cue via the macOS built-in voice."""
    if sys.platform != "darwin":
        return
    try:
        subprocess.Popen(["say", "-r", "260", msg],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


class Coach:
    """Accumulates samples, closes out laps, and keeps the plan current."""

    def __init__(self, quali: bool = False, baseline: int = 0,
                 verbose: bool = False) -> None:
        self.state = AppState()
        self.buffer: list = []
        self.cues = Cues()
        self._cue_at = 0.0
        self.quali = quali
        self.verbose = verbose
        self.web = False
        self.log = None  # open text file; every lap gets a line in it
        self.state.metrics.baseline_laps = baseline
        self.coaching = baseline == 0
        self._last_t = -1.0  # session time of the last distinct frame
        self._last_dist = 0.0
        self._changed_at = time.monotonic()  # wall clock when that frame arrived
        self._packet_at = time.monotonic()
        self._race = False
        self._call_at = 0.0  # rate-limits the pass-economics recompute
        self._under_sc = False
        self._sc_lap = False  # this lap was run behind the safety car
        self._history: list = []  # per-lap deployment, for consistency scoring
        self._stint = stint.StintState()
        self._max_deploy = 0.0  # heaviest lap so far, for the log's "light" flag

    def status(self, now: float | None = None) -> str:
        """live / paused / lost, from whether the session clock is advancing.

        A paused game may keep sending frozen packets or stop sending entirely,
        so neither packet arrival nor session time alone is enough: this watches
        wall time since the session clock last *changed*, which catches both.
        """
        now = time.monotonic() if now is None else now
        if now - self._packet_at > 2.0:
            return "lost"
        if now - self._changed_at > 0.5:
            return "paused"
        return "live"

    # -- lap lifecycle ----------------------------------------------------
    def _ensure_track(self, sample) -> bool:
        if self.state.track is not None:
            return True
        if sample.track_length <= 0:
            return False
        self.state.track = TrackModel(length=float(sample.track_length))
        self._note(
            f"session detected: {index.SESSION_NAMES.get(sample.session_type, '?')}"
            f", {sample.track_length:.0f} m lap"
            + (f", {sample.total_laps} laps" if sample.total_laps
               else ", no scheduled distance"))
        return True

    def _lap_target(self) -> float | None:
        """Charge to finish this lap on. None means energy-neutral.

        Where the number comes from, in order of authority:

        - qualifying: zero, there is no next lap to save for.
        - safety car: full, because the restart is worth far more than the crawl
          and recovering while crawling costs nothing.
        - a known race distance: whatever the across-laps plan says. That plan
          spends evenly while there is a lot of race left, runs the battery down
          into the flag, and holds energy back from a lap spent partly in the
          pit lane.
        - anything else (practice, unknown length): energy-neutral, so the lap
          is repeatable.
        """
        st = self.state
        if self.quali:
            return 0.0
        if self._under_sc and st.track is not None:
            return float(st.track.capacity)
        if st.stint_plan is not None:
            return float(st.stint_plan.this_lap)
        if st.total_laps and st.lap >= st.total_laps:
            return 0.0
        return None

    def _note(self, msg: str) -> None:
        """One line of plain English about what just went into the recording.

        Always to the log file. To the terminal too, unless the terminal is
        currently being redrawn over by the dashboard.
        """
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        if self.log is not None:
            print(line, file=self.log, flush=True)
        if self.verbose or self.web:
            print(line, flush=True)

    def _lap_note(self, lap_num: int, lap_time: float) -> None:
        """What this lap contained, said in the same terms as --index.

        Written before the lap is filtered for analysis, so pit laps and safety
        car laps -- the ones the model throws away but you still want to know
        were captured -- appear in the log too.
        """
        buf = self.buffer
        if not buf:
            return
        n = len(buf)
        deploy = max((s.ers_deployed for s in buf), default=0.0)
        harvest = max((s.ers_harvested for s in buf), default=0.0)
        follow = sum(1 for s in buf if 0.0 < s.delta_front <= index.CLOSE) / n
        flags = []
        if any(s.pit and s.speed < index.PIT_SPEED for s in buf):
            flags.append("PIT STOP — this is the lap that prices pit energy")
        elif any(s.pit for s in buf):
            flags.append("pit lane")
        if any(s.safety_car for s in buf):
            flags.append("safety car")
        if any(s.lap_invalid for s in buf):
            flags.append("invalidated")
        if deploy and deploy < index.LOW_DEPLOY_FRAC * self._max_deploy:
            flags.append("light on deploy — good for the drag fit")
        self._max_deploy = max(self._max_deploy, deploy)
        if follow > 0.2:
            flags.append(f"{follow*100:.0f}% of the lap within 1s of the car ahead")
        self._note(
            f"lap {lap_num} recorded: {lap_time:6.3f}s  "
            f"deployed {deploy/1e6:.2f} MJ  harvested {harvest/1e6:.2f} MJ"
            + ("  |  " + "; ".join(flags) if flags else ""))

    def _close_lap(self, lap_num: int, lap_time: float) -> None:
        track = self.state.track
        self._lap_note(lap_num, lap_time)
        samples = [s for s in self.buffer if s.pit == 0]
        self.buffer = []
        if track is None or lap_time <= 0:
            return
        lap = build_lap(samples, track.n, lap_num, lap_time)
        if lap is None or not lap.valid:
            return  # in-lap, out-lap, invalidated, or joined part way through
        if self._sc_lap:
            # Delta-time running: pace is dictated by the safety car, not by
            # the driver, so every ERS number from it would be meaningless.
            self._sc_lap = False
            return
        if lap_num <= 1 and self._race:
            # Standing start: the launch is traction limited in a way the model
            # does not describe, and the objective on lap 1 is positions, not
            # lap time. Analysing it would produce a confident wrong number.
            return

        track.add_lap(lap)
        self.state.deficit, _ = traffic.following_deficit(track, lap)
        report = analyse(track, lap, e_target=self._lap_target())
        self.state.last_lap = lap
        self.state.report = report
        self.state.metrics.add(report)
        if len(self.state.metrics.reports) >= self.state.metrics.baseline_laps:
            self.coaching = True

        # Plan the next lap from where this one finished.
        soc = float(lap.soc[-1])
        harvest = np.maximum(track.harvest_best, lap.harvest)
        if self._race and self.state.total_laps:
            laps_left = max(self.state.total_laps - lap_num, 0)
            pit_in = (self.state.pit_ideal - lap_num
                      if self.state.pit_ideal > lap_num else None)
            self.state.stint_plan = self._stint.update(
                track, harvest, float(lap.v[0]), soc, laps_left, pit_in)
        self.state.plan = solve(track, soc, harvest, float(lap.v[0]),
                                e_target=self._lap_target())
        self.state.plan_lam = float(np.median(self.state.plan.lam))
        if not self.state.corners:
            self.state.corners = places.find_corners(track)
        tips = places.tips(track, lap, report, self.state.corners, limit=4)
        mode = situations.mode_tip(track, lap, self.state.plan,
                                   self.state.plan_lam)
        if mode is not None:
            tips.append(mode)
        self._history.append(lap.p_mguk * report.actual.dt)
        self._history = self._history[-10:]
        tips += situations.consistency_tips(self._history, self.state.corners,
                                            self.state.plan_lam)
        tips.sort(key=lambda t: -t.gain)
        self.state.tips = tips[:5]

        if self.verbose:
            print(f"lap {lap_num}: {lap_time:.3f}s  ERS loss "
                  f"{report.ers_loss:.3f}s  " +
                  (str(report.verdict) if report.verdict else "clean"))

    def feed(self, sample, speak_cues: bool = False) -> None:
        st = self.state
        st.last_sample = sample
        if not self._ensure_track(sample):
            return

        # A paused game keeps sending the same frame. Binning hundreds of
        # identical samples would weight that one point on track absurdly, so
        # only distinct frames reach the lap buffer.
        if sample.t == self._last_t:
            return
        self._last_t = sample.t
        self._changed_at = time.monotonic()

        # Flashback: distance rewound inside the same lap. Everything after the
        # rewind point is a lap that no longer happened -- drop it, rather than
        # analyse a lap stitched from two attempts.
        # ... but the chequered flag is not a flashback. After the final lap the
        # lap number stops advancing while distance resets to zero, which looks
        # exactly like a rewind and used to delete the whole last lap.
        length = st.track.length if st.track is not None else 0.0
        finished = (length > 0 and self._last_dist > 0.9 * length
                    and sample.lap_dist < 0.1 * length)
        if (sample.lap == st.lap and self.buffer and not finished
                and sample.lap_dist < self._last_dist - 50.0):
            self.buffer = [s for s in self.buffer
                           if s.lap_dist <= sample.lap_dist]
        if finished:
            # Nothing after the line belongs to the lap that just ended.
            self._close_lap(st.lap, _lap_time(self.buffer))
            st.lap = 0
            return
        self._last_dist = sample.lap_dist

        if sample.lap != st.lap:
            if st.lap > 0 and self.buffer:
                self._close_lap(st.lap, _lap_time(self.buffer))
            st.lap = sample.lap
            st.cue = ""
        st.sector = sample.sector
        st.total_laps = sample.total_laps or st.total_laps
        st.pit_ideal = sample.pit_ideal_lap or st.pit_ideal
        self._race = sample.session_type in RACE_SESSIONS
        st.safety_car = sample.safety_car
        self._under_sc = sample.safety_car in (1, 2)
        if self._under_sc:
            self._sc_lap = True  # sticks until the lap is closed out
        self.buffer.append(sample)
        self._update_pass_call(sample)

        if self.coaching and st.track is not None:
            st.race = situations.race_tips(
                st.track, sample, st.plan, st.plan_lam, sample.lap,
                st.total_laps, sample.safety_car, st.deficit)

        if self.coaching and st.tips and st.track is not None and st.track.length:
            i = min(int(sample.lap_dist / st.track.length * st.track.n),
                    st.track.n - 1)
            st.next_tip = places.upcoming(st.corners, st.tips, i, st.track.n)

        if self.coaching and st.plan is not None:
            cue = self.cues.check(st.track, st.plan, sample, sample.t)
            if cue:
                st.cue, self._cue_at = cue, sample.t
                if speak_cues:
                    speak(cue)
            elif st.cue and sample.t - self._cue_at > CUE_LINGER:
                st.cue = ""

    def _update_pass_call(self, sample) -> None:
        """Re-price attacking the car ahead, at most once a second.

        Needs a plan (so lambda is known) and a measured deficit from a previous
        lap spent following, so it stays quiet until there is something real to
        say rather than guessing from the first frame in traffic.
        """
        st = self.state
        if st.plan is None or st.track is None or not self._race:
            return
        if sample.t - self._call_at < 1.0:
            return
        self._call_at = sample.t
        gap = sample.delta_front
        if not (0.0 < gap < traffic.FOLLOW_RANGE) or st.deficit <= 0.0:
            st.call = None
            return
        laps_left = max(st.total_laps - sample.lap, 0)
        i = min(int(sample.lap_dist / st.track.length * st.track.n),
                st.track.n - 1) if st.track.length else 0
        st.call = traffic.pass_call(
            st.track, st.plan, st.plan_lam, st.deficit, gap, laps_left,
            float(sample.speed), i, float(sample.ers_store))

    def snapshot(self, now: float) -> dict:
        """JSON-safe view of the current state for the browser dashboard."""
        st = self.state
        s = st.last_sample
        out: dict = {"status": self.status(now), "laps": len(st.metrics.reports),
                     "safety_car": SAFETY_CAR_NAMES.get(st.safety_car, "")}
        if s is None:
            return out

        cap = st.track.capacity if st.track else 4.0e6
        out.update(
            lap=s.lap, sector=s.sector, speed=s.speed * 3.6,
            mguk=s.p_mguk / 1e3, lap_time=s.lap_time,
            soc=s.ers_store / 1e6, soc_frac=s.ers_store / max(cap, 1.0),
            deployed=s.ers_deployed / 1e6, harvested=s.ers_harvested / 1e6,
            mode=st.ers_mode_name, cue=st.cue,
            gap=s.delta_front or None, laps_left=(
                max(st.total_laps - s.lap, 0) if st.total_laps else None),
        )
        if st.call is not None and st.call.verdict != "none":
            c = st.call
            out["pass"] = {
                "verdict": c.verdict, "detail": c.detail, "advice": c.advice,
                "breakeven": None if c.breakeven > 9 else c.breakeven,
                "deficit": c.deficit, "energy": c.energy / 1e6,
            }
        if st.plan is not None and st.track is not None and st.track.length:
            i = min(int(s.lap_dist / st.track.length * st.track.n),
                    st.track.n - 1)
            out["vs_plan"] = (s.ers_deployed - float(st.plan.cum_energy[i])) / 1e6
            out["lam"] = st.plan_lam * 1e6

        if st.next_tip is not None:
            t = st.next_tip
            out["next_tip"] = {"where": t.where, "action": t.action,
                               "why": t.why, "detail": t.detail,
                               "gain": t.gain, "kind": t.kind}
        def pack(ts):
            return [{"where": t.where, "action": t.action, "why": t.why,
                     "detail": t.detail, "gain": t.gain, "kind": t.kind}
                    for t in ts]

        if st.stint_plan is not None:
            out["stint"] = {"target": st.stint_plan.this_lap / 1e6,
                            "note": st.stint_plan.note,
                            "ahead": [round(t / 1e6, 2)
                                      for t in st.stint_plan.targets[:6]]}
        out["tips"] = pack(st.tips)
        out["race"] = pack(st.race)

        r, lap = st.report, st.last_lap
        if r is not None and lap is not None:
            step = max(len(lap.v) // 200, 1)  # keep the payload small
            out["trace_id"] = r.lap_num
            out["trace"] = {
                "v": [round(float(x) * 3.6, 1) for x in lap.v[::step]],
                "u_you": [round(float(x) / 1e3, 1) for x in lap.p_mguk[::step]],
                "u_opt": [round(float(x) / 1e3, 1) for x in r.plan.u[::step]],
                "soc": [round(float(x) / 1e6, 3) for x in lap.soc[::step]],
            }
            out["fidelity"] = r.fidelity * 100
            if r.verdict:
                v = r.verdict
                out["verdict"] = {
                    "name": v.name, "cost": v.cost, "detail": v.detail,
                    "advice": v.advice, "pool": v.pool,
                    "where": (f"worst stretch {v.where[0]:.0f}–{v.where[1]:.0f} m"
                              if v.where[1] > v.where[0] else ""),
                }
                out["also"] = [{"name": o.name, "cost": o.cost}
                               for o in r.issues[1:3]]
        m = st.metrics
        if m.reports:
            out["best"] = min(x.lap_time for x in m.reports)
            out["loss"] = sum(x.ers_loss for x in m.reports) / len(m.reports)
        return out

    def run(self, samples, live: bool = True, speak_cues: bool = False) -> None:
        last_draw = 0.0
        try:
            for sample in samples:
                if sample is None:  # socket timeout, just refresh the screen
                    pass
                else:
                    self._packet_at = time.monotonic()
                    self.feed(sample, speak_cues)
                now = time.monotonic()
                if live and now - last_draw > REDRAW_INTERVAL:
                    last_draw = now
                    if self.web:
                        web.SNAPSHOT = self.snapshot(now)
                    else:
                        print(dashboard.render(self.state), flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            if self.buffer and self.state.lap > 0:
                self._close_lap(self.state.lap, _lap_time(self.buffer))
            # No sample ever arrived: the run failed to start rather than
            # finished, so a session report would only bury the real error.
            if self.state.last_sample is not None:
                print(dashboard.session_report(self.state.metrics))


def _lap_time(buf) -> float:
    """The lap's time, from the largest clock reading it reached.

    Not the last frame's: at the chequered flag the lap number stops advancing
    while distance and the lap clock reset, so the final lap of a race read as
    0.017s -- and was then thrown away as too short to analyse.
    """
    return max((s.lap_time for s in buf), default=0.0)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def _default_recording() -> str:
    """Somewhere safe to put a session nobody asked to save.

    Recording is on by default: a session you did not record is a session you
    cannot debug, and you only find out you wanted it after it has gone.
    """
    # Next to the code, not in the home directory: the sessions are the test
    # data for this project, so they belong with it.
    os.makedirs(SESSION_DIR, exist_ok=True)
    return os.path.join(SESSION_DIR, time.strftime("%Y-%m-%d-%H%M%S") + ".f1")


def recordings() -> list:
    """Every .f1 in the session directory, newest first."""
    if not os.path.isdir(SESSION_DIR):
        return []
    return [os.path.join(SESSION_DIR, f)
            for f in sorted(os.listdir(SESSION_DIR), reverse=True)
            if f.endswith(".f1")]


def _newest() -> str | None:
    files = recordings()
    return files[0] if files else None


def show_index(path: str | None) -> int:
    """The report for one recording, defaulting to the one you just drove.

    No argument is the normal case: you have finished a session and want to
    know what is in it. Having to look the filename up first, then pass it
    back in, is a step that exists only to serve the program.
    """
    files = recordings()
    if path is None:
        if not files:
            print(f"no recordings yet — they will appear in {SESSION_DIR}")
            return 1
        path = files[0]
        print(f"newest recording ({len(files)} on disk)\n")
    print(index.report(index.scan(path)))
    others = [f for f in files if f != path]
    if others:
        print(f"\n{len(others)} older recording(s) in sessions/:")
        for f in others[:10]:
            print(f"  {os.path.basename(f):<28} "
                  f"{os.path.getsize(f)/1e6:6.1f} MB")
        print(f"  ...detail on one:  python3 -m f126ers.app --index "
              f"sessions/{os.path.basename(others[0])}")
    return 0


def _short(path: str) -> str:
    """Relative if that is actually shorter, absolute otherwise."""
    rel = os.path.relpath(path)
    return rel if not rel.startswith("..") else path


def _recording_banner(path: str, log_path: str, port: int) -> str:
    """Says up front where everything goes, so nobody has to go looking."""
    return (
        f"\nRECORDING to  {_short(path)}\n"
        f"  every UDP packet, raw — nothing is filtered, so anything the game\n"
        f"  sends can be re-analysed later\n"
        f"lap-by-lap log  {_short(log_path)}\n"
        f"listening on UDP port {port}\n"
        f"\nstop with ctrl-c. On stopping you get a summary of what was captured\n"
        f"and which laps are worth looking at.\n")


def _wrap_up(path: str, log_path: str) -> None:
    """After the session: what landed on disk, and what it is good for.

    Printed automatically rather than left as a command to remember, because the
    one moment you care whether the session captured what you needed is the
    moment you have just stopped driving and could still go back out.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return
    if size <= len(telemetry.Recorder.MAGIC):
        # Nothing arrived. Do not leave a stub file to be confused with a real
        # session later.
        for p in (path, log_path):
            try:
                os.unlink(p)
            except OSError:
                pass
        print("\nNo packets arrived, so nothing was saved. Check the game's UDP")
        print("settings, then:  python3 -m f126ers.app --check")
        return
    print(f"\nSAVED  {_short(path)}  ({size/1e6:.1f} MB)")
    print(f"       {_short(log_path)}  (lap-by-lap log)\n")
    try:
        print(index.report(index.scan(path)))
    except (OSError, ValueError) as exc:
        print(f"could not summarise the recording: {exc}", file=sys.stderr)
        return
    print(f"\nre-run the full analysis on it any time:\n"
          f"  python3 -m f126ers.app --replay {_short(path)}")


def check(port: int) -> int:
    """Prove the game is talking to us, and that every packet decodes.

    Runs until interrupted rather than reading a fixed number of packets: the
    useful signal is what happens while you drive, not what arrives in the first
    second. Reads the socket directly instead of going through listen() so a
    wrong UDP Format reports as a message rather than an exception.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.25)
    merger = telemetry.Merger()

    st = {"port": port, "total": 0, "types": {}, "src": "", "format": 0,
          "year": 0, "rate": 0.0, "sample": None, "error": "", "waited": 0.0,
          "track_len": 0.0, "mode": "?", "all_zero": True}
    t0 = last_draw = time.monotonic()
    window = 0

    try:
        while True:
            try:
                data, addr = sock.recvfrom(4096)
                st["src"] = addr[0]
                st["total"] += 1
                window += 1
                if len(data) >= telemetry.HEADER_SIZE:
                    fmt, yr, ptype = (
                        struct.unpack_from("<H", data, 0)[0],
                        data[2], data[6])
                    st["format"], st["year"] = fmt, yr
                    st["types"][ptype] = st["types"].get(ptype, 0) + 1
                    if fmt != telemetry.PACKET_FORMAT:
                        st["error"] = (
                            f"game is sending UDP Format {fmt}, this needs "
                            f"{telemetry.PACKET_FORMAT}.\n  Change it in "
                            f"Settings > Telemetry Settings > UDP Format.")
                    else:
                        st["error"] = ""
                        s = merger.feed(data)
                        if s is not None:
                            st["sample"] = s
                            st["track_len"] = s.track_length
                            st["mode"] = (
                                telemetry.ERS_MODES[s.ers_mode]
                                if s.ers_mode < len(telemetry.ERS_MODES) else "?")
                            st["all_zero"] = (s.speed == 0 and s.ers_store == 0)
            except socket.timeout:
                pass

            now = time.monotonic()
            if now - last_draw >= 0.4:
                st["rate"] = window / (now - last_draw)
                st["waited"] = now - t0
                window = 0
                last_draw = now
                print(dashboard.CLEAR + dashboard.HOME
                      + dashboard.check_screen(st), flush=True)
    except KeyboardInterrupt:
        print()
        if st["total"]:
            kinds = ", ".join(sorted(dashboard.PACKET_NAMES.get(k, str(k))
                                     for k in st["types"]))
            print(f"{st['total']} packets from {st['src']} ({kinds})")
        else:
            print("no packets received")
        return 0 if st["total"] and not st["error"] else 1
    finally:
        sock.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Live ERS optimiser for F1 26")
    p.add_argument("--port", type=int, default=20777)
    p.add_argument("--web", nargs="?", const=8765, type=int, metavar="PORT",
                   help="browser dashboard instead of the terminal (default 8765)")
    p.add_argument("--check", action="store_true",
                   help="verify the game is sending telemetry, then exit")
    p.add_argument("--replay", nargs="?", const="", metavar="FILE",
                   help="re-analyse a recording (no FILE = the newest one)")
    p.add_argument("--index", nargs="?", const="", metavar="FILE",
                   help="what is in a recording and what it can calibrate "
                        "(no FILE = the newest one)")
    p.add_argument("--record", metavar="FILE",
                   help="where to save raw packets "
                        "(default: ./sessions/<timestamp>.f1)")
    p.add_argument("--quali", action="store_true",
                   help="optimise for one lap, allowed to finish empty")
    p.add_argument("--baseline", type=int, default=0, metavar="N",
                   help="drive N laps with coaching muted, to measure against")
    p.add_argument("--speak", action="store_true", help="speak in-lap cues aloud")
    p.add_argument("--realtime", action="store_true",
                   help="replay at the original speed")
    args = p.parse_args(argv)

    # A plain `kill` sends SIGTERM, which by default tears the process down
    # without unwinding -- so the recording would never be closed. Turn it into
    # the same interrupt Ctrl-C raises, and the finally blocks run either way.
    signal.signal(signal.SIGTERM, _interrupt)

    if args.check:
        return check(args.port)

    if args.index is not None:
        return show_index(args.index or None)

    coach = Coach(quali=args.quali, baseline=args.baseline,
                  verbose=args.replay is not None and not args.realtime)
    coach.state.port = args.port
    if args.web:
        url = web.serve(args.web)
        coach.web = True
        print(f"dashboard: {url}   (ctrl-c here to stop)", flush=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", url], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)

    if args.replay is not None:
        path = args.replay or _newest()
        if path is None:
            print(f"no recordings yet — they will appear in {SESSION_DIR}")
            return 1
        print(f"replaying {_short(path)}\n")
        stream = telemetry.samples_from_log(path, realtime=args.realtime)
        coach.run(stream, live=args.realtime)
    else:
        path = args.record or _default_recording()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        log_path = os.path.splitext(path)[0] + ".log"
        rec = None
        try:
            rec = telemetry.Recorder(path)
            coach.log = open(log_path, "w")
        except OSError as exc:
            print(f"could not open {path} for recording: {exc}", file=sys.stderr)
        print(_recording_banner(path, log_path, args.port), flush=True)
        try:
            stream = telemetry.listen(args.port, recorder=rec)
            coach.run(stream, live=True, speak_cues=args.speak)
        except OSError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 1
        finally:
            # Runs on a clean exit, a Ctrl-C, or a crash. The recording is the
            # only way to reproduce whatever just went wrong, so it is the last
            # thing given up.
            if rec is not None:
                rec.close()
            if coach.log is not None:
                coach.log.close()
            _wrap_up(path, log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
