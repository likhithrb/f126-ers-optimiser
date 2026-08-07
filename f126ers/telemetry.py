"""UDP telemetry reader for EA F1 25: 2026 Season Pack (packet format 2026).

Byte layouts ported from the official EA 2026 UDP spec as implemented by
github.com/volodymyr-fed/F1Game.UDP. All offsets verified against the stated
packet sizes (header 29 B, 24 cars per array).
"""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass, replace

PACKET_FORMAT = 2026
HEADER_SIZE = 29
MAX_CARS = 24

# Packet ids we consume.
SESSION, LAP_DATA, CAR_TELEMETRY, CAR_STATUS, CAR_TELEMETRY2 = 1, 2, 6, 7, 16

# Per-car struct sizes (checked: 29 + 24*size + trailing == documented packet size).
LAP_ITEM = 57
# Session packet offsets, counted from the start of the body (after the header):
# 21 marshal zones of 5 bytes each sit between numMarshalZones and safetyCarStatus.
GAME_PAUSED_OFF = 14
SAFETY_CAR_OFF = 19 + 21 * 5
# ... and the pit window sits past 64 weather-forecast samples of 8 bytes each,
# plus four one-byte fields and three uint32 link identifiers. Built up in
# pieces rather than written as one number, so it can be checked by eye.
_AFTER_SC = SAFETY_CAR_OFF + 1 + 1 + 1        # networkGame, numForecastSamples
_AFTER_FORECAST = _AFTER_SC + 64 * 8          # the samples themselves
PIT_IDEAL_OFF = _AFTER_FORECAST + 1 + 1 + 4 * 3   # accuracy, aiDifficulty, 3 ids
PIT_LATEST_OFF = PIT_IDEAL_OFF + 1
TEL_ITEM = 59
STATUS_ITEM = 59
TEL2_ITEM = 10

_HEADER = struct.Struct("<HBBBBBQfIIBB")
# LapData: we stop after `sector`, which is the last field we care about (offset 37).
_LAP = struct.Struct("<IIHBHBHBHBfffBBBBBB")
# CarTelemetry: stop after drs flag (offset 19).
_TEL = struct.Struct("<HfffBbHB")
# CarStatus: the whole 59 bytes; ERS block starts at offset 29.
_STATUS = struct.Struct("<BBBBBfffHHBBHBBBBfffBffff?")
_TEL2 = struct.Struct("<B?H??H??")
_MODE_OFF = struct.calcsize("<BBBBBfffHHBBHBBBBfff")  # ersDeployMode

ERS_MODES = ("None", "Medium", "Hotlap", "Overtake")
OVERTAKE_MODE = 3  # ERS_MODES index that means the manual override is engaged

# Measured from the game rather than taken from the regulations: the 2026 rules
# say 350 kW, the game caps the MGU-K at exactly this.
P_MGUK_OBSERVED = 3.15e5


@dataclass(frozen=True, slots=True)
class Sample:
    """One merged telemetry frame for the player's car."""

    t: float = 0.0  # session time, seconds
    lap: int = 0
    lap_dist: float = 0.0  # metres from start line (negative on the grid)
    lap_time: float = 0.0  # current lap time, seconds
    sector: int = 0
    lap_invalid: bool = False
    pit: int = 0
    delta_front: float = 0.0  # seconds to car ahead (0 == no car / leader)
    speed: float = 0.0  # m/s
    throttle: float = 0.0
    brake: float = 0.0
    gear: int = 0
    drs: bool = False
    ers_store: float = 0.0  # J
    ers_mode: int = 0
    ers_deployed: float = 0.0  # J this lap
    ers_harvested: float = 0.0  # J this lap, MGU-K
    ers_harvest_limit: float = 0.0  # J per lap
    p_ice: float = 0.0  # W
    p_mguk: float = 0.0  # W
    overtake_available: bool = False
    overtake_active: bool = False  # ERS mode is Overtake, i.e. the override
    aero_mode: int = 0  # 0 = corner (high downforce), 1 = straight
    track_length: float = 0.0
    session_type: int = 0
    total_laps: int = 0  # scheduled race distance; 0 outside a race
    safety_car: int = 0  # 0 none, 1 full SC, 2 VSC, 3 formation lap
    game_paused: bool = False
    fuel: float = 0.0  # kg in tank; the car loses ~100 kg over a race
    tyre_age: int = 0  # laps on this set
    position: int = 0
    pit_ideal_lap: int = 0  # lap the game reckons is the best to pit on
    pit_latest_lap: int = 0
    traction_control: int = 0  # 0 off, 1 medium, 2 full -- changes corner exits
    abs_on: bool = False
    gap_behind: float = 0.0  # s to the car behind (0 == nobody / last)
    lapped_behind: bool = False  # the car behind is a lap down: blue flags


def _car_slice(payload: bytes, idx: int, item: int) -> bytes:
    off = idx * item
    return payload[off:off + item]


# LapData field offsets within one car's slice, for the all-cars scan. Computed
# from the struct rather than written out, so they cannot drift away from _LAP.
_LAP_DIST_OFF = struct.calcsize("<IIHBHBHBHB")  # float lapDistance
_LAP_POS_OFF = struct.calcsize("<IIHBHBHBHBfff")  # byte carPosition
_LAP_NUM_OFF = _LAP_POS_OFF + 1  # byte currentLapNum


def _behind(body: bytes, player: int, my_dist: float, my_lap: int,
            track_len: float, my_speed: float) -> dict:
    """Gap to the car behind on the road, and whether they are a lap down.

    The game reports a delta to the car *ahead* but not the one behind, and
    defending is an ERS decision too. Every car's lap distance is already in
    this packet, so the gap is derivable.

    Measured around the track, not by race position: a car a lap down can be
    right on your gearbox, and that is exactly the case worth knowing about --
    they are about to be shown blue flags, so there is no point spending a
    megajoule defending from them.
    """
    if track_len <= 0.0:
        return {"gap_behind": 0.0, "lapped_behind": False}
    best_gap, lapped = 0.0, False
    for i in range(MAX_CARS):
        if i == player:
            continue
        sl = _car_slice(body, i, LAP_ITEM)
        if len(sl) < LAP_ITEM or sl[_LAP_POS_OFF] == 0:
            continue  # not present or retired
        dist = struct.unpack_from("<f", sl, _LAP_DIST_OFF)[0]
        gap_m = (my_dist - dist) % track_len
        if gap_m <= 0.0 or gap_m > track_len / 2:
            continue  # ahead of us, or so far back they are not a factor
        if best_gap == 0.0 or gap_m < best_gap:
            best_gap, lapped = gap_m, sl[_LAP_NUM_OFF] < my_lap
    if best_gap == 0.0:
        return {"gap_behind": 0.0, "lapped_behind": False}
    # Metres to seconds at the speed we are actually doing. A fixed divisor was
    # out by a factor of two through the slow corners, which matters because the
    # result is compared against a one-second threshold to decide whether the
    # car behind is a threat at all.
    return {"gap_behind": min(best_gap / max(my_speed, 15.0), 99.0),
            "lapped_behind": lapped}


class Merger:
    """Folds packets of different types into a single evolving Sample.

    The game sends packet types independently; we keep the latest of each and
    emit a frame whenever car telemetry (the fastest, most jitter-sensitive
    channel) arrives.
    """

    def __init__(self) -> None:
        self.state = Sample()

    def feed(self, data: bytes) -> Sample | None:
        """Returns a Sample when this packet completes a frame, else None."""
        if len(data) < HEADER_SIZE:
            return None
        fmt, _yr, _maj, _min, _pv, ptype, _uid, stime, _fid, _ofid, player, _p2 = \
            _HEADER.unpack_from(data)
        if fmt != PACKET_FORMAT:
            raise ValueError(
                f"packet format {fmt}, expected {PACKET_FORMAT}. "
                "Set the game's UDP Format option to 2026."
            )
        body = data[HEADER_SIZE:]
        s = self.state

        if ptype == CAR_TELEMETRY:
            speed, thr, _steer, brk, _clutch, gear, _rpm, drs = \
                _TEL.unpack_from(_car_slice(body, player, TEL_ITEM))
            self.state = replace(
                s, t=stime, speed=speed / 3.6, throttle=thr, brake=brk,
                gear=gear, drs=bool(drs),
            )
            return self.state

        if ptype == LAP_DATA:
            (_last, cur_ms, _s1ms, _s1m, _s2ms, _s2m, df_ms, df_m, _dl, _dlm,
             dist, _total, _sc, _pos, lap, pit, _stops, sector, invalid) = \
                _LAP.unpack_from(_car_slice(body, player, LAP_ITEM))
            # LapData advances the lap number; CarStatus resets the per-lap ERS
            # counters. They are separate packets, so for one frame at every
            # line the new lap still carries the old lap's totals -- which then
            # get attributed to the wrong lap. Zero them here, where the lap
            # number actually changes, and let the next CarStatus refill them.
            rolled = {} if lap == s.lap else {
                "ers_deployed": 0.0, "ers_harvested": 0.0}
            self.state = replace(
                s, lap=lap, lap_dist=dist, lap_time=cur_ms / 1000.0,
                sector=sector, lap_invalid=bool(invalid), pit=pit, **rolled,
                delta_front=df_m * 60.0 + df_ms / 1000.0,
                position=_pos,
                **_behind(body, player, dist, lap, s.track_length, s.speed),
            )
        elif ptype == CAR_STATUS:
            f = _STATUS.unpack_from(_car_slice(body, player, STATUS_ITEM))
            self.state = replace(
                s, p_ice=f[17], p_mguk=f[18], ers_store=f[19], ers_mode=f[20],
                # Overtake mode is the override. Taken from here rather than
                # from CarTelemetry2's overtake bytes: on a real race those read
                # as active for 90-100% of some laps and 0% of others, which no
                # driver does. The mode field was checked against a lap the
                # driver described from memory and matched it exactly.
                overtake_active=f[20] == OVERTAKE_MODE,
                ers_harvested=f[21], ers_harvest_limit=f[23], ers_deployed=f[24],
                fuel=f[5], tyre_age=f[15],
                # Assist settings change what "at the limit" means on a corner
                # exit, so they belong with the data rather than in someone's
                # memory of which race this was.
                traction_control=f[0], abs_on=bool(f[1]),
            )
        elif ptype == CAR_TELEMETRY2:
            # Only aero_mode is taken from this packet. It reads ~30% straight
            # line over a Bahrain race, which is right; the overtake bytes in
            # the same packet do not survive the same sanity check, so the
            # override is read from CarStatus instead.
            aero, _aero_avail, _aero_dist, ot_avail, _ot_act, _ot_dist, _regs, \
                _wrong_way = _TEL2.unpack_from(_car_slice(body, player, TEL2_ITEM))
            self.state = replace(
                s, aero_mode=aero, overtake_available=bool(ot_avail),
            )
        elif ptype == SESSION:
            # totalLaps sits at offset 3, immediately before trackLength.
            total_laps, track_len, sess_type = struct.unpack_from("<BHB", body, 3)
            paused = struct.unpack_from("<B", body, GAME_PAUSED_OFF)[0]
            sc = struct.unpack_from("<B", body, SAFETY_CAR_OFF)[0]
            ideal, latest = struct.unpack_from("<BB", body, PIT_IDEAL_OFF)
            self.state = replace(s, track_length=float(track_len),
                                 session_type=sess_type, total_laps=total_laps,
                                 safety_car=sc, game_paused=bool(paused),
                                 pit_ideal_lap=ideal, pit_latest_lap=latest)
        return None


class Recorder:
    """Length-prefixed raw packet log: replay is byte-identical by construction.

    Flushed on a timer rather than left to the operating system. A session you
    cannot get back is worth more than the syscalls: if the app crashes -- and
    on unfamiliar telemetry it will -- everything up to the last flush is still
    on disk, and the recording is the only way to reproduce whatever went wrong.
    """

    MAGIC = b"F126\x01"
    FLUSH_EVERY = 1.0  # seconds

    def __init__(self, path: str) -> None:
        self.path = path
        self.f = open(path, "wb")
        self.f.write(self.MAGIC)
        self.t0 = time.monotonic()
        self._flushed = self.t0
        self.packets = 0

    def write(self, data: bytes) -> None:
        now = time.monotonic()
        self.f.write(struct.pack("<fI", now - self.t0, len(data)))
        self.f.write(data)
        self.packets += 1
        if now - self._flushed >= self.FLUSH_EVERY:
            self.f.flush()
            self._flushed = now

    def close(self) -> None:
        if not self.f.closed:
            self.f.flush()
            self.f.close()


def replay(path: str, realtime: bool = False):
    """Yields (offset_seconds, packet_bytes) from a recording."""
    with open(path, "rb") as f:
        if f.read(len(Recorder.MAGIC)) != Recorder.MAGIC:
            raise ValueError(f"{path} is not an F126 recording")
        start = time.monotonic()
        while True:
            head = f.read(8)
            if len(head) < 8:
                return
            offset, n = struct.unpack("<fI", head)
            data = f.read(n)
            if len(data) < n:
                return
            if realtime:
                lag = offset - (time.monotonic() - start)
                if lag > 0:
                    time.sleep(lag)
            yield offset, data


def listen(port: int = 20777, recorder: "Recorder | None" = None,
           timeout: float = 1.0):
    """Yields Samples from the live game. Blocks on the socket.

    The caller owns the Recorder. It used to be created here, inside a
    generator, so its cleanup depended on the generator being finalised -- and
    an unhandled exception in the consumer meant that never reliably happened
    and the whole session was lost. The caller can close it in a finally that
    actually runs.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError as exc:
        # Every caller binds this one fixed port, so a clash has one likely
        # cause and one fix. A stack trace here buries both.
        sock.close()
        raise OSError(
            f"UDP port {port} is already in use — most likely another copy of "
            f"this tool (a --check you left running in another tab).\n"
            f"  find it:  lsof -nP -iUDP:{port}\n"
            f"  stop it:  kill <PID>"
        ) from exc
    sock.settimeout(timeout)
    merger = Merger()
    rec = recorder
    try:
        while True:
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                yield None  # let callers redraw / handle shutdown
                continue
            if rec:
                rec.write(data)
            sample = merger.feed(data)
            if sample is not None:
                yield sample
    finally:
        sock.close()


def samples_from_log(path: str, realtime: bool = False):
    merger = Merger()
    for _offset, data in replay(path, realtime):
        sample = merger.feed(data)
        if sample is not None:
            yield sample


def _self_check() -> None:
    """Round-trips a synthetic packet of each type through the Merger."""
    def header(ptype: int, t: float) -> bytes:
        return _HEADER.pack(PACKET_FORMAT, 26, 1, 0, 1, ptype, 1, t, 0, 0, 3, 255)

    m = Merger()

    body = bytearray(24 * STATUS_ITEM)
    _STATUS.pack_into(
        body, 3 * STATUS_ITEM, 0, 0, 0, 55, 0, 100.0, 110.0, 20.0, 15000, 4000,
        8, 1, 0, 16, 16, 3, 0, 400_000.0, 350_000.0, 3.5e6, 2, 1.1e6, 0.0,
        4.0e6, 2.2e6, False,
    )
    assert m.feed(header(CAR_STATUS, 1.0) + bytes(body)) is None
    assert m.state.ers_store == 3.5e6, m.state.ers_store
    assert m.state.p_mguk == 350_000.0
    assert m.state.ers_deployed == 2.2e6
    assert m.state.ers_harvest_limit == 4.0e6
    assert m.state.fuel == 100.0 and m.state.tyre_age == 3, (
        m.state.fuel, m.state.tyre_age)
    assert m.state.ers_mode == 2 and not m.state.overtake_active
    # Mode 3 is the override. Read from here, not from CarTelemetry2.
    body[3 * STATUS_ITEM + _MODE_OFF] = OVERTAKE_MODE
    assert m.feed(header(CAR_STATUS, 1.0) + bytes(body)) is None
    assert m.state.overtake_active, "ERS mode Overtake means the override is on"
    body[3 * STATUS_ITEM + _MODE_OFF] = 2

    # Session: safety car and pause sit past 21 marshal zones, so their offsets
    # are the easiest thing in this file to get quietly wrong.
    body = bytearray(897)
    # Session type 15 is Race in the 2026 numbering: the sprint-shootout
    # sessions occupy 10-14, which is where Race used to sit.
    struct.pack_into("<BHB", body, 3, 58, 5891, 15)  # totalLaps, length, race
    body[GAME_PAUSED_OFF] = 1
    body[SAFETY_CAR_OFF] = 2  # virtual safety car
    body[PIT_IDEAL_OFF] = 24
    body[PIT_LATEST_OFF] = 31
    assert m.feed(header(SESSION, 1.0) + bytes(body)) is None
    assert m.state.total_laps == 58 and m.state.track_length == 5891.0
    assert m.state.session_type == 15, m.state.session_type
    assert m.state.safety_car == 2, m.state.safety_car
    assert m.state.game_paused is True
    assert PIT_LATEST_OFF < 897, PIT_LATEST_OFF  # must land inside the packet
    assert m.state.pit_ideal_lap == 24 and m.state.pit_latest_lap == 31, (
        m.state.pit_ideal_lap, m.state.pit_latest_lap)

    body = bytearray(24 * LAP_ITEM)
    _LAP.pack_into(body, 3 * LAP_ITEM, 90_000, 42_500, 0, 0, 0, 0, 1200, 0, 0, 0,
                   1234.5, 0.0, 0.0, 4, 7, 0, 0, 2, 0)
    assert m.feed(header(LAP_DATA, 1.0) + bytes(body)) is None
    assert m.state.lap == 7 and m.state.sector == 2
    assert abs(m.state.lap_dist - 1234.5) < 1e-3
    assert abs(m.state.lap_time - 42.5) < 1e-6
    assert abs(m.state.delta_front - 1.2) < 1e-6
    assert m.state.position == 4, m.state.position
    assert m.state.gap_behind == 0.0  # no other car in the packet yet

    # Two cars behind: the nearer one wins, and a lap down means blue flags.
    _LAP.pack_into(body, 5 * LAP_ITEM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   1134.5, 0.0, 0.0, 5, 7, 0, 0, 2, 0)   # 100 m behind, same lap
    _LAP.pack_into(body, 6 * LAP_ITEM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   900.0, 0.0, 0.0, 6, 7, 0, 0, 2, 0)    # further back
    assert m.feed(header(LAP_DATA, 1.0) + bytes(body)) is None
    assert abs(m.state.gap_behind - 100.0 / max(m.state.speed, 15.0)) < 1e-6, \
        (m.state.gap_behind, m.state.speed)
    assert m.state.lapped_behind is False
    _LAP.pack_into(body, 5 * LAP_ITEM, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   1200.0, 0.0, 0.0, 5, 6, 0, 0, 2, 0)   # a lap down
    assert m.feed(header(LAP_DATA, 1.0) + bytes(body)) is None
    assert m.state.lapped_behind is True, "lapped car behind should raise blue flags"

    body = bytearray(24 * TEL_ITEM + 3)
    _TEL.pack_into(body, 3 * TEL_ITEM, 288, 1.0, 0.0, 0.0, 0, 7, 11000, 1)
    out = m.feed(header(CAR_TELEMETRY, 1.0) + bytes(body))
    assert out is not None and abs(out.speed - 80.0) < 1e-6, out
    assert out.gear == 7 and out.drs and out.ers_store == 3.5e6

    body = bytearray(24 * TEL2_ITEM)
    _TEL2.pack_into(body, 3 * TEL2_ITEM, 1, True, 300, True, True, 250, True,
                    False)
    assert m.feed(header(CAR_TELEMETRY2, 1.0) + bytes(body)) is None
    assert m.state.aero_mode == 1

    # Crossing the line must not carry the finished lap's ERS totals forward.
    m.state = replace(m.state, ers_deployed=9.1e6, ers_harvested=8.5e6, lap=7)
    body = bytearray(24 * LAP_ITEM)
    _LAP.pack_into(body, 3 * LAP_ITEM, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0,
                   1.4, 0.0, 0.0, 4, 8, 0, 0, 0, 0)  # lap 8 now
    assert m.feed(header(LAP_DATA, 1.0) + bytes(body)) is None
    assert m.state.ers_deployed == 0.0 and m.state.ers_harvested == 0.0, (
        "per-lap ERS counters must reset with the lap number, not one frame "
        "later when CarStatus catches up")

    try:
        m.feed(struct.pack("<H", 2025) + bytes(HEADER_SIZE))
    except ValueError as e:
        assert "2026" in str(e)
    else:
        raise AssertionError("stale packet format not rejected")

    print("telemetry self-check ok")


if __name__ == "__main__":
    _self_check()
