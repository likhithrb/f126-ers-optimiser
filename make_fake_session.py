"""Generates a synthetic F1 26 recording so the whole pipeline can be tested
without the game running. Writes real packet bytes in the 2026 layout.

    python3 make_fake_session.py session.f1
    python3 -m f126ers.app --replay session.f1

Laps 1-3 dump the battery early (the mistake). Laps 4-6 deploy on corner exits.
A correct pipeline should report a large ERS loss on the early laps, a small one
on the later laps, and a realised improvement close to the predicted one.
"""

from __future__ import annotations

import struct
import sys

import numpy as np

from f126ers import telemetry as T
from f126ers.optimiser import simulate, solve
from f126ers.track import DS, TrackModel

N = 200
V_STRAIGHT, V_APEX = 115.0, 30.0
APEXES = (50, 150)
CAPACITY = 4.0e6


def toy_track() -> tuple[TrackModel, np.ndarray]:
    tm = TrackModel(length=N * DS, n=N)
    d = np.arange(N)
    v_env = np.full(N, V_STRAIGHT)
    for apex in APEXES:
        dist = np.minimum(np.abs(d - apex), N - np.abs(d - apex))
        v_env = np.minimum(v_env, V_APEX + 2.2 * dist)
    tm.v_env = v_env
    tm.p_ice = np.full(N, 5.0e5)
    tm.a, tm.b, tm.c = 1.1e-3, 9.0e-4, 0.20
    tm.db = -2.0e-4  # straight-line mode sheds drag
    tm.aero = (v_env > 80.0).astype(float)
    tm.drag = tm.b + tm.db * tm.aero
    tm.capacity = CAPACITY
    harvest = np.zeros(N)
    for apex in APEXES:
        harvest[apex - 10:apex] = 1.0e5
    return tm, harvest


def _header(ptype: int, t: float, frame: int) -> bytes:
    return T._HEADER.pack(T.PACKET_FORMAT, 26, 1, 0, 1, ptype, 7, t, frame,
                          frame, 3, 255)


def _session_packet(t: float, frame: int) -> bytes:
    body = bytearray(T.SESSION_BODY if hasattr(T, "SESSION_BODY") else 897)
    struct.pack_into("<BHB", body, 3, 12, int(N * DS), 10)  # laps, length, race
    return _header(T.SESSION, t, frame) + bytes(body)


def _lap_packet(t, frame, lap, dist, lap_time, sector):
    body = bytearray(24 * T.LAP_ITEM + 2)
    T._LAP.pack_into(body, 3 * T.LAP_ITEM, 0, int(lap_time * 1000), 0, 0, 0, 0,
                     0, 0, 0, 0, float(dist), 0.0, 0.0, 1, lap, 0, 0, sector, 0)
    return _header(T.LAP_DATA, t, frame) + bytes(body)


def _status_packet(t, frame, soc, deployed, harvested, p_k, fuel=100.0, age=3):
    body = bytearray(24 * T.STATUS_ITEM)
    T._STATUS.pack_into(
        body, 3 * T.STATUS_ITEM, 0, 0, 0, 55, 0, float(fuel), 110.0, 20.0, 15000,
        4000, 8, 1, 0, 16, 16, int(age), 0, 5.0e5, float(p_k), float(soc), 2,
        float(harvested), 0.0, 4.0e6, float(deployed), False)
    return _header(T.CAR_STATUS, t, frame) + bytes(body)


def _tel_packet(t, frame, speed, throttle, brake, gear):
    body = bytearray(24 * T.TEL_ITEM + 3)
    T._TEL.pack_into(body, 3 * T.TEL_ITEM, int(speed * 3.6), float(throttle),
                     0.0, float(brake), 0, gear, 11000, 0)
    return _header(T.CAR_TELEMETRY, t, frame) + bytes(body)


def _tel2_packet(t, frame, overtake, aero=0):
    body = bytearray(24 * T.TEL2_ITEM)
    T._TEL2.pack_into(body, 3 * T.TEL2_ITEM, int(aero), True, 0, True,
                      bool(overtake), 0, True, False)
    return _header(T.CAR_TELEMETRY2, t, frame) + bytes(body)


def bad_profile(tm) -> np.ndarray:
    """Dump everything from the start line until the battery is flat."""
    u = np.zeros(N)
    u[:70] = tm.p_max(np.minimum(84.0, tm.v_env[:70]))
    return u


def charge_limited(tm, u, v0, soc, harvest, passes: int = 3) -> np.ndarray:
    """Trims a deployment profile to energy the car actually has.

    The real car cuts deployment when the battery empties; the simulator does
    not enforce that lower bound, so a naive dump profile would write telemetry
    showing joules that never existed -- and a fixture that is physically
    impossible cannot validate anything. Deployment changes speed, which changes
    the time spent in each bin, which changes the energy drawn, so this iterates
    a few times rather than solving in one shot.
    """
    u = u.copy()
    for _ in range(passes):
        sim = simulate(tm, u, v0, soc, harvest)
        charge = soc
        for i in range(N):
            avail = max(charge, 0.0)
            draw = u[i] * sim.dt[i]
            if draw > avail:
                u[i] = avail / sim.dt[i] if sim.dt[i] > 0 else 0.0
                draw = avail
            charge = charge - draw + harvest[i]
    return u


def good_profile(tm, harvest, soc, v0) -> np.ndarray:
    return solve(tm, soc, harvest, v0).u


def main(path: str = "session.f1", laps: int = 6) -> None:
    tm, harvest = toy_track()
    rec = T.Recorder(path)
    t, frame, soc = 0.0, 0, 3.0e6
    v_line = 84.0  # settles at the drag-limited speed across the line

    for lap in range(1, laps + 1):
        fuel = 100.0 - 8.0 * (lap - 1)  # burns off through the run
        u = bad_profile(tm) if lap <= 3 else good_profile(tm, harvest, soc, v_line)
        # Early laps also brake late, so they recover less energy.
        lap_harvest = harvest * (0.6 if lap <= 3 else 1.0)
        u = charge_limited(tm, u, v_line, soc, lap_harvest)
        sim = simulate(tm, u, v_line, soc, lap_harvest)
        # Pedals must agree with the speed trace: the car is on the brakes
        # wherever it is losing speed, not wherever we declared a braking zone.
        dv = np.diff(sim.v, append=sim.v[0])
        brake = (dv < -0.3).astype(float)

        deployed = harvested = 0.0
        lap_t = 0.0
        for i in range(N):
            deployed += u[i] * sim.dt[i]
            harvested += lap_harvest[i]
            for half in range(2):  # two frames per bin
                lap_t += sim.dt[i] / 2
                t += sim.dt[i] / 2
                frame += 1
                dist = (i + 0.5 * half) * DS
                sector = min(int(i * 3 / N), 2)
                if frame % 120 == 1:
                    rec.write(_session_packet(t, frame))
                rec.write(_lap_packet(t, frame, lap, dist, lap_t, sector))
                rec.write(_status_packet(t, frame, sim.soc[i], deployed,
                                         harvested, u[i],
                                         fuel=fuel, age=lap))
                rec.write(_tel2_packet(t, frame, False, tm.aero[i]))
                rec.write(_tel_packet(t, frame, sim.v[i],
                                      1.0 - brake[i], brake[i],
                                      int(np.clip(sim.v[i] / 12, 1, 8))))
        soc = float(sim.soc[-1] - u[-1] * sim.dt[-1] + lap_harvest[-1])
        soc = float(np.clip(soc, 0.0, CAPACITY))
        print(f"lap {lap}: {sim.lap_time:.3f}s  deployed {deployed/1e6:.2f} MJ  "
              f"end charge {soc/1e6:.2f} MJ")

    rec.close()
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "session.f1")
