"""Live terminal dashboard. ANSI escapes only, no curses, no dependencies."""

from __future__ import annotations

import shutil

import numpy as np

BLOCKS = " ▁▂▃▄▅▆▇█"
CLEAR, HOME = "\033[2J", "\033[H"
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
RED, YELLOW, GREEN, CYAN, GREY = ("\033[31m", "\033[33m", "\033[32m",
                                  "\033[36m", "\033[90m")


def width() -> int:
    return max(shutil.get_terminal_size((100, 30)).columns, 60)


def _resample(values: np.ndarray, w: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(w)
    return np.interp(np.linspace(0, values.size - 1, w),
                     np.arange(values.size), values)


def sparkline(values: np.ndarray, w: int, lo: float | None = None,
              hi: float | None = None) -> str:
    v = _resample(np.asarray(values, dtype=float), w)
    lo = float(np.min(v)) if lo is None else lo
    hi = float(np.max(v)) if hi is None else hi
    if hi - lo < 1e-9:
        return BLOCKS[0] * w
    idx = np.clip(((v - lo) / (hi - lo) * (len(BLOCKS) - 1)).astype(int),
                  0, len(BLOCKS) - 1)
    return "".join(BLOCKS[i] for i in idx)


def bar(frac: float, w: int, colour: str = CYAN) -> str:
    frac = min(max(frac, 0.0), 1.0)
    filled = int(round(frac * w))
    return f"{colour}{'█' * filled}{GREY}{'░' * (w - filled)}{RESET}"


def _soc_colour(frac: float) -> str:
    return RED if frac < 0.15 else YELLOW if frac < 0.35 else GREEN


def render(state) -> str:
    """Builds the whole screen from an AppState. Called a few times a second."""
    w = width()
    inner = w - 2
    out = [f"{BOLD}F1 26 ERS OPTIMISER{RESET}{DIM}   lap {state.lap}   "
           f"sector {state.sector + 1}{RESET}"]
    out.append(GREY + "─" * inner + RESET)

    s = state.last_sample
    if s is None:
        out.append(f"{DIM}waiting for telemetry on UDP port {state.port}…{RESET}")
        out.append(f"{DIM}(game: Settings → Telemetry → UDP On, format 2026, "
                   f"port {state.port}){RESET}")
        return CLEAR + HOME + "\n".join(out)

    cap = state.track.capacity if state.track else 4.0e6
    frac = s.ers_store / max(cap, 1.0)
    out.append(
        f"  SOC {bar(frac, 24, _soc_colour(frac))} "
        f"{s.ers_store/1e6:5.2f} MJ   "
        f"{DIM}deployed{RESET} {s.ers_deployed/1e6:4.2f}  "
        f"{DIM}harvested{RESET} {s.ers_harvested/1e6:4.2f}  "
        f"{DIM}mode{RESET} {state.ers_mode_name}")
    out.append(
        f"  {s.speed*3.6:5.1f} km/h   gear {s.gear}   "
        f"MGU-K {s.p_mguk/1e3:5.1f} kW   "
        f"{DIM}lap{RESET} {s.lap_time:6.2f}s"
        + (f"   {CYAN}OVERRIDE{RESET}" if s.overtake_active else ""))

    if state.plan is not None and state.track is not None:
        out.append("")
        i = min(int(s.lap_dist / state.track.length * state.track.n),
                state.track.n - 1) if state.track.length else 0
        planned = float(state.plan.cum_energy[i])
        delta = s.ers_deployed - planned
        col = GREEN if abs(delta) < 3e5 else (YELLOW if delta > 0 else CYAN)
        out.append(f"  {DIM}vs plan{RESET} {col}{delta/1e6:+5.2f} MJ{RESET}"
                   f"   {DIM}energy price{RESET} {state.plan_lam*1e6:.3f} s/MJ")

    if getattr(state, "safety_car", 0) in (1, 2):
        out.append("")
        out.append(f"  {BOLD}{YELLOW}▲ SAFETY CAR — bank everything, "
                   f"the restart is where it pays{RESET}")

    if state.cue:
        out.append("")
        out.append(f"  {BOLD}{YELLOW}▲ {state.cue}{RESET}")

    for r in getattr(state, "race", [])[:2]:
        out.append("")
        out.append(f"  {BOLD}{YELLOW}◆ {r.where}:  {r.action}{RESET}")
        out.append(f"    {DIM}{r.why}{RESET}")

    nxt = getattr(state, "next_tip", None)
    if nxt is not None:
        out.append("")
        out.append(f"  {BOLD}{CYAN}▶ {nxt.where}:  {nxt.action}{RESET}"
                   f"  {DIM}+{nxt.gain:.2f}s{RESET}")
        out.append(f"    {DIM}{nxt.why}{RESET}")

    lap = state.last_lap
    if lap is not None and state.report is not None:
        out.append("")
        out.append(GREY + "─" * inner + RESET)
        r = state.report
        out.append(f"{BOLD}LAP {r.lap_num}{RESET}  {r.lap_time:.3f}s   "
                   f"{DIM}optimal ERS would give{RESET} "
                   f"{BOLD}{r.ers_loss + r.harvest_loss:.2f}s{RESET}"
                   f"   {DIM}model error {r.fidelity*100:.1f}%{RESET}")
        sw = max(inner - 14, 20)
        pk = max(float(lap.p_mguk.max()), float(r.plan.u.max()), 1.0)
        out.append(f"  {DIM}speed  {RESET}{sparkline(lap.v, sw)}")
        out.append(f"  {DIM}you    {RESET}{YELLOW}"
                   f"{sparkline(lap.p_mguk, sw, 0, pk)}{RESET}")
        out.append(f"  {DIM}optimal{RESET}{GREEN}"
                   f"{sparkline(r.plan.u, sw, 0, pk)}{RESET}")
        out.append(f"  {DIM}charge {RESET}{CYAN}"
                   f"{sparkline(lap.soc, sw, 0, cap)}{RESET}")

        if r.verdict:
            out.append("")
            v = r.verdict
            tag = "" if v.pool == "lap" else f" {DIM}(costs next lap){RESET}"
            out.append(f"  {BOLD}{RED}BIGGEST LOSS{RESET} "
                       f"{BOLD}{v.cost:.2f}s{RESET}{tag}  {v.name}")
            out.append(f"  {v.detail}")
            if v.where[1] > v.where[0]:
                out.append(f"  {DIM}worst stretch {v.where[0]:.0f}–"
                           f"{v.where[1]:.0f} m{RESET}")
            out.append(f"  {CYAN}→ {v.advice}{RESET}")
            for other in r.issues[1:3]:
                out.append(f"  {DIM}also {other.cost:.2f}s  {other.name}{RESET}")

        if getattr(state, "tips", None):
            out.append("")
            out.append(f"{BOLD}FIX THESE, IN ORDER{RESET}")
            for t in state.tips:
                out.append(f"  {CYAN}{t.where:<20}{RESET}{BOLD}{t.action}{RESET}"
                           f"  {DIM}+{t.gain:.2f}s{RESET}")
                out.append(f"  {DIM}{'':<20}{t.why}{RESET}")

    out.append("")
    out.append(f"{DIM}ctrl-c to stop and print the session report{RESET}")
    return CLEAR + HOME + "\n".join(out)


PACKET_NAMES = {1: "session", 2: "lap", 6: "telemetry", 7: "status",
                16: "telemetry2"}
NEEDED = (1, 2, 6, 7, 16)


def check_screen(st) -> str:
    """Diagnostic view: is the game talking to us, and does it decode?

    Deliberately shows the raw plumbing (source address, packet format, which
    packet types have arrived) before any of the derived numbers, because when
    this screen is up something is usually wrong and that is the order you need
    to debug it in.
    """
    L = [f"{BOLD}TELEMETRY CHECK{RESET}   listening on 0.0.0.0:{st['port']}", ""]

    if st["error"]:
        L += [f"  {RED}{st['error']}{RESET}", ""]
        return "\n".join(L)

    if not st["total"]:
        secs = int(st["waited"])
        L += [f"  {YELLOW}no packets yet{RESET}  ({secs}s)", "",
              f"  {DIM}On the PC, check:{RESET}",
              f"  {DIM}  · UDP Telemetry .... On{RESET}",
              f"  {DIM}  · UDP IP Address ... this Mac{RESET}",
              f"  {DIM}  · UDP Format ....... 2026{RESET}",
              f"  {DIM}  · and that you are on track, not in a menu{RESET}"]
        return "\n".join(L)

    L += [f"  {GREEN}receiving{RESET} from {st['src']}   "
          f"format {st['format']}   game year {st['year']}   "
          f"{st['rate']:.0f} packets/s", ""]

    seen = st["types"]
    marks = []
    for pid in NEEDED:
        name = PACKET_NAMES[pid]
        ok = seen.get(pid, 0) > 0
        marks.append(f"{GREEN}✓{RESET} {name}" if ok
                     else f"{RED}✗{RESET} {DIM}{name}{RESET}")
    L += ["  " + "   ".join(marks), ""]

    missing = [PACKET_NAMES[p] for p in NEEDED if not seen.get(p)]
    if missing:
        L += [f"  {YELLOW}missing: {', '.join(missing)}{RESET}",
              f"  {DIM}some of these only send once you are driving a session{RESET}",
              ""]

    s = st["sample"]
    if s is not None:
        L += [f"  lap {s.lap}   sector {s.sector + 1}   "
              f"{s.lap_dist:6.0f} m of {st['track_len']:.0f}",
              f"  {s.speed * 3.6:5.0f} km/h   gear {s.gear}   "
              f"throttle {s.throttle * 100:3.0f}%   brake {s.brake * 100:3.0f}%",
              f"  battery {s.ers_store / 1e6:4.2f} MJ   "
              f"deployed {s.ers_deployed / 1e6:4.2f} MJ   "
              f"harvested {s.ers_harvested / 1e6:4.2f} MJ",
              f"  MGU-K {s.p_mguk / 1000:4.0f} kW   ICE {s.p_ice / 1000:4.0f} kW"
              f"   mode {st['mode']}", ""]
        if st["all_zero"]:
            L += [f"  {YELLOW}values are all zero — car is stationary or in a "
                  f"menu; drive a lap{RESET}", ""]
        else:
            L += [f"  {GREEN}decoding correctly{RESET} — "
                  f"ready to run for real", ""]

    L.append(f"  {DIM}Ctrl-C to stop{RESET}")
    return "\n".join(L)


def session_report(metrics) -> str:
    m = metrics.summary()
    if not m:
        return "No complete laps recorded."
    lines = [
        "",
        f"{BOLD}SESSION REPORT{RESET}",
        GREY + "─" * 58 + RESET,
        f"  laps analysed            {m['laps']}",
        f"  best lap                 {m['best_lap']:.3f}s",
        f"  ERS loss, first laps     {m['ers_loss_first']:.3f}s per lap",
        f"  ERS loss, later laps     {m['ers_loss_last']:.3f}s per lap",
        f"  time available           {m['predicted_gain']:.3f}s per lap",
        f"  model error              {m['fidelity']*100:.2f}%",
    ]
    if m["controlled"]:
        lines += [
            f"  measured against         {m['baseline_laps']} uncoached laps",
            f"  improvement realised     {m['realised_gain']:.3f}s per lap",
            f"  of predicted captured    {m['capture']*100:.0f}%",
        ]
    else:
        lines += [
            GREY + "  no baseline: run --baseline N to measure a real "
            "before/after" + RESET,
        ]
    lines.append("")
    return "\n".join(lines)
