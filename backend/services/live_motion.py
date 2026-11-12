"""Live motion v2 for the SimTracker: play-flow motion + NHL rules staging (presentation only).

Two clocks
----------
* ``gc`` - game clock (seconds since the opening faceoff). It is the engine's clock: every real
  attempt / goal / penalty happens at its real ``t`` (= gc). The game clock FREEZES during stoppages.
* ``pt`` - presentation time. It never stops. Stoppages, goal celebrations, faceoff set-ups,
  intermissions and shootout staging advance ``pt`` while ``gc`` holds.

Frames are emitted every 0.5 s of ``pt`` (contiguous across /advance calls) and carry both clocks.
Events and commentary carry their engine ``t`` (gc) plus ``pt``. The client plays ``pt`` and displays
``gc``. Derived rule events (offside, icing, puck over the glass, faceoffs, line changes, delayed
penalty, hits) are inserted only in the gaps between real engine events and are ``authority: derived``.
Nothing here can change a result: the engine pass is final before any of this runs.
"""
from __future__ import annotations

import logging
import math
import random
import zlib
from bisect import bisect_right
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SIDES = ("home", "away")
MX, MY = 97.0, 40.0
NET_X = 89.0
BLUE = 25.0
ZONE_DOT_U, DOT_V, NZ_DOT_U = 69.0, 22.0, 20.0
# Broadcast-pace motion (round 6): cruise 12-16 ft/s, rush 18-22, sprint <= 26 only on breakaways / races.
V_CRUISE, V_RUSH, V_SPRINT, V_DRIFT = 14.0, 20.0, 25.0, 9.0
A_NORM, A_HARD = 10.0, 16.0
V_SKATE, A_SKATE, V_GOALIE = V_SPRINT, A_HARD, 16.0
CARRY, CARRY_RUSH, PASS, DUMP, SHOT = 13.0, 19.0, 58.0, 58.0, 100.0
MIN_CARRY_LEG, MIN_SHOT_FLIGHT, RETARGET_S = 1.8, 0.35, 1.2
CREASE_R, MIN_SEP = 7.0, 5.0  # 5 ft: markers and name tags stay readable, bodies never share a spot
BENCH_DOOR = {"home": (-12.0, 39.0), "away": (12.0, 39.0)}
BOX_SEAT = {"home": (-7.0, -42.0), "away": (7.0, -42.0)}
BOX_DOOR = {"home": (-7.0, -39.0), "away": (7.0, -39.0)}
ANCHOR_TYPES = frozenset({"attempt", "goal", "penalty", "period_start", "period_end", "shootout_attempt", "final"})
DEAD_KINDS = frozenset({"dead", "faceoff"})
TERMINAL_PT = 10.0


A_BRAKE = 22.0  # hard stops decelerate faster than skaters accelerate ("snow spray")
SHOT_V = {"slap": 110.0, "one_timer": 105.0, "snap": 95.0, "wrist": 85.0, "backhand": 70.0, "tip": 75.0, "wraparound": 38.0,
          "deke": 60.0}
SHOT_LIFT = {"slap": 1.2, "one_timer": 1.0, "snap": 1.4, "wrist": 1.8, "backhand": 2.0, "tip": 0.8, "wraparound": 0.3, "deke": 1.5}


def _kb(k: str) -> str:
    """Waypoint kind without its path flavour ('pass:bank' -> 'pass')."""
    return k.split(":", 1)[0]


def _kf(k: str) -> str:
    return k.split(":", 1)[1] if ":" in k else ""


def _end_contour(e: float) -> List[Tuple[float, float]]:
    """Board line around one end (rounded corners, 1 ft inside the boards), from the +y side wall to the -y side wall."""
    pts: List[Tuple[float, float]] = []
    for x in (55.0, 62.0, 68.0):
        pts.append((e * x, 41.0))
    for i in range(0, 10):
        th = math.pi / 2 - i * (math.pi / 2) / 9.0
        pts.append((e * (72.0 + 26.5 * math.cos(th)), 14.5 + 26.5 * math.sin(th)))
    for y in (7.0, 0.0, -7.0):
        pts.append((e * 98.5, y))
    for i in range(0, 10):
        th = -i * (math.pi / 2) / 9.0
        pts.append((e * (72.0 + 26.5 * math.cos(th)), -14.5 + 26.5 * math.sin(th)))
    for x in (68.0, 62.0, 55.0):
        pts.append((e * x, -41.0))
    return pts


_CONTOUR = {1.0: _end_contour(1.0), -1.0: _end_contour(-1.0)}


def rim_poly(p0: Tuple[float, float], p1: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
    """Puck rimmed around the boards from p0 to p1 in the same end (follows the board curve)."""
    if p0[0] * p1[0] <= 0 or abs(p1[0]) < 50.0:
        return None
    c = _CONTOUR[1.0 if p1[0] > 0 else -1.0]
    first = min(range(len(c)), key=lambda i: math.hypot(c[i][0] - p0[0], c[i][1] - p0[1]) + (0.0 if abs(p0[0]) > 50 else abs(c[i][0]) * 0.3))
    last = min(range(len(c)), key=lambda i: math.hypot(c[i][0] - p1[0], c[i][1] - p1[1]))
    step = 1 if last >= first else -1
    mid = [c[i] for i in range(first, last + step, step)]
    return [p0] + mid + [p1]


def bank_poly(p0: Tuple[float, float], p1: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Pass banked off the side boards (angle in = angle out)."""
    b = 41.5 if (p0[1] + p1[1]) >= 0 else -41.5
    y1r = 2 * b - p1[1]
    t = (b - p0[1]) / (y1r - p0[1]) if abs(y1r - p0[1]) > 1e-6 else 0.5
    t = max(0.05, min(0.95, t))
    return [p0, (p0[0] + t * (p1[0] - p0[0]), b), p1]


def poly_len(poly: List[Tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(poly, poly[1:]))


def poly_at(poly: List[Tuple[float, float]], a: float) -> Tuple[float, float]:
    total = poly_len(poly) or 1.0
    want = max(0.0, min(1.0, a)) * total
    for p, q in zip(poly, poly[1:]):
        d = math.hypot(q[0] - p[0], q[1] - p[1])
        if want <= d or q is poly[-1]:
            f = 0.0 if d <= 1e-9 else min(1.0, want / d)
            return (p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f)
        want -= d
    return poly[-1]


def _h01(*parts: Any) -> float:
    return (zlib.crc32("|".join(str(x) for x in parts).encode()) % 100000) / 100000.0


def bench_spot(side: str, pid: str) -> Tuple[float, float]:
    """Each player's own spot along his bench (benches span ~22 ft each side of centre ice)."""
    bx, by = BENCH_DOOR[side]
    off = ((zlib.crc32(pid.encode()) % 1000) / 1000.0 - 0.5) * 20.0
    return (bx + off, by + 0.5)


def opp(s: Optional[str]) -> str:
    return "away" if s == "home" else "home"


def rink_clamp(x: float, y: float, margin: float = 3.0) -> Tuple[float, float]:
    """Inside the boards, rounded corners included (rink 200 x 85 ft, corner radius 28 ft)."""
    X, Y, R = 100.0 - margin, 42.5 - margin, 28.0 - margin
    x, y = max(-X, min(X, x)), max(-Y, min(Y, y))
    cx, cy = 72.0, 14.5
    if abs(x) > cx and abs(y) > cy:
        dx, dy = abs(x) - cx, abs(y) - cy
        d = math.hypot(dx, dy)
        if d > R:
            x = math.copysign(cx + dx / d * R, x)
            y = math.copysign(cy + dy / d * R, y)
    return x, y


def clampxy(x: float, y: float) -> Tuple[float, float]:
    return rink_clamp(x, y, 2.5)


def assign_roles(ids: List[str], pos_of) -> List[Tuple[str, str]]:
    ds = [p for p in ids if pos_of(p) == "D"]
    fs = [p for p in ids if p not in ds]
    if not ds and len(fs) >= 4:
        ds, fs = fs[-2:], fs[:-2]
    out: List[Tuple[str, str]] = list(zip(ds[:2], ("LD", "RD")))
    out += [(p, "D") for p in ds[2:]]
    order = sorted(fs, key=lambda p: {"C": 0, "LW": 1, "RW": 2}.get(pos_of(p), 3))
    for j, p in enumerate(order):
        out.append((p, ("C", "LW", "RW")[j] if j < 3 else "F"))
    return out


def _seg(k: str, g0: float, g1: float, p0: Tuple[float, float], p1: Tuple[float, float], **kw: Any) -> Dict[str, Any]:
    s = {"k": k, "g0": float(g0), "g1": float(max(g1, g0)), "p0": p0, "p1": p1, "side": None, "car": None, "rcv": None,
         "ph": "play", "fo": None, "why": None, "ev": [], "d": 0.0}
    s.update(kw)
    return s


def _dead(g: float, d: float, p0: Tuple[float, float], p1: Tuple[float, float], **kw: Any) -> Dict[str, Any]:
    k = kw.pop("k", "dead")
    s = _seg(k, g, g, p0, p1, **kw)
    s["d"] = float(d)
    s.setdefault("ph", "stoppage")
    if s["ph"] == "play":
        s["ph"] = "stoppage"
    if s["ph"] == "stoppage" and s.get("reason") in ("offside", "icing", "penalty"):
        s["ph"] = s["reason"]
    return s


class MotionV2:
    """See module docstring. Public surface used by the live loop and snapshot code:
    generate_to(gc, skip) -> motion block, pt_at(gc, inclusive), pt_for_raw(raw), drain_events(),
    drain_hints(), buildup(raw_key), replay(raw_key), last_frame, roles, last_whistle, t_last,
    situation_key(), zone(), fo_tally, _cur_seg."""

    def __init__(self, loop: Any) -> None:
        self.loop = loop
        self.plans: List[Dict[str, Any]] = []
        self.flat_g: List[float] = []          # seg g0 (for bisect)
        self.flat: List[Dict[str, Any]] = []
        self.t_last: Optional[float] = None    # pt of last frame
        self.gc_last: float = 0.0
        self.pos: Dict[str, List[float]] = {}
        self.onice: Dict[str, List[str]] = {"home": [], "away": []}
        self.goalie_on: Dict[str, Optional[str]] = {"home": None, "away": None}
        self.leaving: Dict[str, Dict[str, Any]] = {}
        self.boxed: Dict[str, Dict[str, Any]] = {}
        self.last_frame: Optional[Dict[str, Any]] = None
        self.roles: Dict[str, str] = {}
        self.last_whistle: Optional[Tuple[str, Dict[str, Any]]] = None
        self.fo_tally = {"home": [0, 0], "away": [0, 0]}
        self._cur_seg: Optional[Dict[str, Any]] = None
        self._seg_idx = 0
        self._events: List[Dict[str, Any]] = []
        self._hints: List[Dict[str, Any]] = []
        self._emitted_ev: set = set()
        self._tl_key: Any = None
        self._anchors: List[Dict[str, Any]] = []
        self._atts: List[Dict[str, Any]] = []
        self._bu: Dict[str, Dict[str, Any]] = {}
        self._shot: Dict[str, Dict[str, Any]] = {}
        self._replay: Dict[str, Dict[str, Any]] = {}
        self._fo_rat: Optional[Dict[str, float]] = None
        self._freeze: Dict[str, List[Tuple[float, float]]] = {"home": [], "away": []}  # iced team: no change (gc intervals)
        self._lineup_cache: Dict[float, Dict[str, List[str]]] = {}
        self._extra: Dict[str, Optional[str]] = {"home": None, "away": None}
        self._last_change_ev: Dict[str, float] = {"home": -99.0, "away": -99.0}
        self.stats: Dict[str, float] = {"retargets": 0.0, "player_sec": 0.0}
        self.live_mode = True          # round 9: forward play simulation between anchors
        self.play: Any = None
        self._resets = 0
        self._play_stats = {"forced": 0, "out_region": 0, "resets": 0, "icings": 0, "live_segs": 0}
        self._hold: Dict[str, Tuple[Any, ...]] = {}
        self._marks: Dict[str, str] = {}

    # ───────────────────────────── basics
    def _book_pos(self, pid: str) -> str:
        return self.loop.book.pos(pid)

    def _period(self, gc: float) -> int:
        so = self._so_gc()
        if so is not None and gc >= so - 1e-9:
            return 5
        return self.loop.clock_at(max(0.0, gc - 1e-6))[0]

    def _so_gc(self) -> Optional[float]:
        for r in self._anchors:
            if r.get("type") == "period_start" and int(r.get("period") or 0) == 5:
                return float(r["t"])
        return None

    def dir(self, side: str, gc: float, period: Optional[int] = None) -> float:
        p = period if period is not None else self._period(gc)
        d = 1.0 if p in (1, 3, 5) else -1.0
        return d if side == "home" else -d

    def ab(self, side: str, gc: float, u: float, v: float, period: Optional[int] = None) -> Tuple[float, float]:
        d = self.dir(side, gc, period)
        return clampxy(u * d, v * d)

    def rel(self, side: str, gc: float, x: float, y: float, period: Optional[int] = None) -> Tuple[float, float]:
        d = self.dir(side, gc, period)
        return x * d, y * d

    def _rng(self, *parts: Any) -> random.Random:
        return random.Random(zlib.crc32(("|".join(str(p) for p in (self.loop.lg.seed, "mv2") + parts)).encode()))

    def _refresh_anchors(self) -> bool:
        tl = self.loop._timeline
        key = (id(tl), len(tl))
        if key == self._tl_key:
            return False
        changed = self._tl_key is not None
        self._tl_key = key
        self._anchors = [r for r in tl if r.get("type") in ANCHOR_TYPES]
        self._atts = [r for r in tl if r.get("type") in ("attempt", "goal") and (r.get("att") or {}).get("atk_ids")]
        self._lineup_cache = {}
        return changed

    # ───────────────────────────── lineups
    def lineup_at(self, gc: float) -> Dict[str, List[str]]:
        # exact key: a half-second bucket served the old unit through a whole stoppage that began inside it
        k = round(gc, 3)
        hit = self._lineup_cache.get(k)
        if hit is not None:
            return hit
        prev = nxt = None
        i_prev = -1
        for i_r, r in enumerate(self._atts):
            if float(r["t"]) <= gc + 1e-9:
                prev, i_prev = r, i_r
            else:
                nxt = r
                break
        use = prev or nxt
        out: Dict[str, List[str]] = {"home": [], "away": []}
        if use is not None:
            if prev is not None and nxt is not None:
                tp, tn = float(prev["t"]), float(nxt["t"])
                sw = max(tp + 0.35 * (tn - tp), tn - 16.0)  # the shooting unit is out early enough to set up at broadcast pace
                if self.live_mode:
                    sw = min(sw, max(tp + 0.5, tn - 7.0))   # and early enough to skate into the play from the bench
                if (prev.get("att") or {}).get("outcome") == "GOAL":
                    sw = min(sw, tp + 0.5)
                for r in self._anchors:
                    if tp < float(r["t"]) < tn and r["type"] in ("penalty", "period_start"):
                        # change during the whistle (the clock is stopped at the call), not on the ice after the draw
                        sw = min(sw, float(r["t"]) - 0.01)
                        break
                if gc >= sw:
                    use = nxt
            a = use.get("att") or {}
            s = a.get("side")
            if s in SIDES:
                out[s] = list(a.get("atk_ids") or [])
                out[opp(s)] = list(a.get("def_ids") or [])
        lines = self.loop.lines
        # Long stretch with no attempt: the engine only tells us who was out at each attempt, so without this one
        # unit would play the whole gap (5-6 minute shifts). Rotate the other lines through the middle of the gap;
        # the units at both attempts are still exactly the engine's.
        if prev is not None and nxt is not None and self.live_mode:
            tp, tn = float(prev["t"]), float(nxt["t"])
            # only while a penalty is actually being served (special-teams units come from the engine); the
            # even-strength stretch either side of it rotates like any other
            pen_gap = self.loop._cap is not None and any(pen["start"] - 2.0 <= gc < pen["start"] + pen["dur"] + 2.0 for pen in self.loop._cap.penalties)
            for side in SIDES:
                if pen_gap or gc >= 3600.0 or tn - tp < 45.0 or not (tp + 8.0 <= gc <= tn - 20.0):   # at least a 15 s shift for the fresh unit
                    break
                # how long has this unit been out? walk back through earlier attempts that list the same players
                unit = set(out[side])
                t0 = tp
                for r_ in reversed(self._atts[:i_prev]):
                    a2 = r_.get("att") or {}
                    ids2 = set((a2.get("atk_ids") if a2.get("side") == side else a2.get("def_ids")) or [])
                    if len(unit & ids2) >= 3 and tp - float(r_["t"]) < 200.0:
                        t0 = float(r_["t"])
                    else:
                        break
                long_gap = tn - tp > 75.0 and gc >= tp + 42.0
                tired = (gc - t0) > 55.0
                if not (long_gap or tired):
                    continue
                k_ = int((gc - (tp + 8.0)) // 42.0)          # a fresh unit every ~42 s off the change
                if True:
                    ln = lines[side]
                    Fl = [list(g) for g in (ln.get("F") or []) if g]
                    Dl = [list(g) for g in (ln.get("D") or []) if g]
                    if len(Fl) < 2 or len(Dl) < 2 or len(out[side]) != 5:
                        continue
                    cur_ = set(out[side])
                    fi = next((i for i, g in enumerate(Fl) if cur_ & set(g)), 0)
                    di = next((i for i, g in enumerate(Dl) if cur_ & set(g)), 0)
                    nF = Fl[(fi + 1 + k_) % min(len(Fl), 4)]
                    nD = Dl[(di + 1 + k_) % min(len(Dl), 3)]
                    out[side] = (nF + nD)[:5]
        for side in SIDES:
            if not out[side]:
                ln = lines[side]
                out[side] = (list((ln.get("F") or [[]])[0] or []) + list((ln.get("D") or [[]])[0] or []))[:5]
        need = {"home": 5, "away": 5}
        boxed = set()
        if gc >= 3600.0 and not self.loop.lg.is_playoff and (self.loop._result or {}).get("overtime"):
            need = {"home": 3, "away": 3}
        elif self.loop._cap is not None:
            for pen in self.loop._cap.penalties:
                if pen["start"] <= gc < pen["start"] + pen["dur"]:
                    need[pen["side"]] = max(3, need[pen["side"]] - 1)
                    if pen.get("player_id") and not self._pen_conflict(pen):
                        boxed.add(pen["player_id"])
        for side in SIDES:
            ids = [p for p in out[side] if p not in boxed]
            pool = [p for grp in (lines[side].get("F") or []) + (lines[side].get("D") or []) for p in grp]
            while len(ids) < need[side]:
                have_d = sum(1 for p in ids if self._book_pos(p) == "D")
                cand = [p for p in pool if p not in ids and p not in boxed]
                pref = [p for p in cand if (self._book_pos(p) == "D") == (have_d < 2)] or cand
                if not pref:
                    break
                ids.append(pref[0])
            keep = (use.get("att") or {}).get("shooter_id") if use is not None else None
            while len(ids) > need[side]:
                fw = [p for p in ids if self._book_pos(p) != "D" and p != keep]
                rest = [p for p in ids if p != keep]
                ids.remove(fw[-1] if len(fw) > 1 else rest[-1])
            out[side] = ids
        if len(self._lineup_cache) > 6000:
            self._lineup_cache.clear()
        self._lineup_cache[k] = out
        return out

    def _goalie_wanted(self, side: str, gc: float, seg: Optional[Dict[str, Any]]) -> Optional[str]:
        if seg is not None and seg.get("delayed") == side:
            return None  # extra attacker during a delayed penalty
        pulled = False
        for (tt, s, v) in self.loop._pull_hist:
            if s == side and tt <= gc + 1e-9:
                pulled = v
        if pulled:
            return None
        g = self.loop.lines[side].get("G") or []
        return g[0] if g else None

    # ───────────────────────────── faceoff ratings
    def _fo_rating(self, pid: Optional[str]) -> float:
        if not pid:
            return 50.0
        if self._fo_rat is None:
            self._fo_rat = {}
            cap = self.loop._cap
            for side in SIDES:
                for p in list((cap.ctx.get(f"{side}_dressed") if cap is not None else None) or []):
                    rt = getattr(p, "ratings", None)
                    val = None
                    if isinstance(rt, dict):
                        for k, v in rt.items():
                            if "face" in str(k).lower():
                                try:
                                    val = float(v)
                                    break
                                except (TypeError, ValueError):
                                    pass
                    pid0 = str(getattr(p, "id", "") or "")
                    if pid0 and val is not None:
                        self._fo_rat[pid0] = val * 99.0 if val <= 1.5 else val
        return float(self._fo_rat.get(pid, 50.0))

    # ───────────────────────────── movement personality (deterministic, from ratings + seeded jitter)
    _TRAIT_KEYS = {"speed": ("speed", "skat", "accel"), "agility": ("agil", "edge", "balance"), "hands": ("hand", "puck", "deke", "stick"),
                   "oiq": ("off", "sense", "iq", "vision", "creat"), "diq": ("def", "posit"), "phys": ("phys", "check", "strength", "aggress")}

    def _rating_rows(self) -> Dict[str, Dict[str, float]]:
        if getattr(self, "_rat_rows", None) is not None:
            return self._rat_rows
        rows: Dict[str, Dict[str, float]] = {}
        cap = self.loop._cap
        for side in SIDES:
            for pl in list((cap.ctx.get(f"{side}_dressed") if cap is not None else None) or []):
                rt = getattr(pl, "ratings", None)
                pid0 = str(getattr(pl, "id", "") or "")
                if not pid0 or not isinstance(rt, dict):
                    continue
                out: Dict[str, float] = {}
                for trait, keys in self._TRAIT_KEYS.items():
                    vals = []
                    for k, v in rt.items():
                        kl = str(k).lower()
                        if any(x in kl for x in keys):
                            try:
                                fv = float(v)
                            except (TypeError, ValueError):
                                continue
                            vals.append(fv / 99.0 if fv > 1.5 else fv)
                    if vals:
                        out[trait] = max(0.0, min(1.0, sum(vals) / len(vals)))
                rows[pid0] = out
        # Rank each trait among tonight's skaters and blend it with the raw rating, so the fastest and
        # smartest players on the ice visibly are, even when the two rosters are rated close together.
        for trait in self._TRAIT_KEYS:
            have = sorted((q for q in rows if trait in rows[q]), key=lambda q: rows[q][trait])
            n_ = max(1, len(have) - 1)
            for i_, q in enumerate(have):
                rows[q][trait] = max(0.0, min(1.0, 0.3 * rows[q][trait] + 0.7 * (i_ / n_)))
        self._rat_rows = rows
        return rows

    def _traits(self, pid: str) -> Dict[str, float]:
        tc = self.__dict__.setdefault("_trait_cache", {})
        t = tc.get(pid)
        if t is not None:
            return t
        r = self._rating_rows().get(pid) or self._rating_rows().get(pid.replace("PLAYER_", "")) or {}
        rng = self._rng("trait", pid)
        g = {k: (r[k] if k in r else max(0.0, min(1.0, 0.5 + rng.uniform(-0.2, 0.2)))) for k in self._TRAIT_KEYS}
        star = (g["speed"] + g["agility"] + g["hands"] + g["oiq"]) / 4.0 > 0.74
        t = {"v": 0.84 + 0.30 * g["speed"] + (0.02 if star else 0.0) + rng.uniform(-0.01, 0.01),     # top speed about +/-15 %
             "a": 0.78 + 0.44 * g["agility"], "turn": 0.6 + 0.8 * g["agility"],
             "stride": rng.uniform(0.6, 0.9), "sph": rng.uniform(0.0, 6.283), "samp": rng.uniform(0.04, 0.08),
             "lane": max(-1.0, min(1.0, rng.uniform(-0.7, 0.7) + (g["hands"] - 0.5) * 0.8 + (0.3 if star else 0.0))),  # + middle, - wall
             "deke": min(1.0, 0.2 + 0.8 * g["hands"] + (0.15 if star else 0.0)), "dper": rng.uniform(0.45, 0.75),
             "star": 1.0 if star else 0.0, "speed": g["speed"], "hands": g["hands"], "oiq": g["oiq"], "phys": g["phys"],
             "wob": rng.uniform(0.3, 0.5), "wf": rng.uniform(0.8, 1.25), "w1": rng.uniform(0, 6.283), "w2": rng.uniform(0, 6.283),
             "sc": rng.uniform(5.5, 9.5), "sco": rng.uniform(0.0, 9.0)}
        tc[pid] = t
        return t

    def _center(self, side: str, gc: float) -> Optional[str]:
        ids = self.lineup_at(gc)[side]
        roles = {r: p for p, r in assign_roles(ids, self._book_pos)}
        return roles.get("C") or (ids[0] if ids else None)

    def _role_pid(self, side: str, gc: float, role: str) -> Optional[str]:
        ids = self.lineup_at(gc)[side]
        if role in ("STAR", "STAR2"):   # the line's best puck handlers run the cycle
            fw = sorted((p for p in ids if self._book_pos(p) != "D"), key=lambda p: -(self._traits(p)["oiq"] + self._traits(p)["hands"]))
            pick = fw[0:1] if role == "STAR" else fw[1:2]
            if pick:
                return pick[0]
            role = "W" if role == "STAR" else "W2"
        roles = {r: p for p, r in assign_roles(ids, self._book_pos)}
        for r in {"D1": ("LD", "RD", "D"), "D2": ("RD", "LD", "D"), "W": ("LW", "RW", "C", "F"), "W2": ("RW", "LW", "F", "C"),
                  "C": ("C", "LW", "RW", "F")}.get(role, (role,)):
            if r in roles:
                return roles[r]
        return ids[0] if ids else None

    # ───────────────────────────── plan building
    def _faceoff(self, g: float, dot: Tuple[float, float], prefer: Optional[str], rng: random.Random, reason: str,
                 set_d: float = 1.6, offender: Optional[str] = None, force: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Faceoff staging (dead, players set) + drop; winner from faceoff ratings (derived)."""
        segs: List[Dict[str, Any]] = [_dead(g, set_d, dot, dot, k="faceoff", ph="faceoff", fo=dot)]
        ch, ca = self._center("home", g + 0.01), self._center("away", g + 0.01)
        rh, ra = self._fo_rating(ch), self._fo_rating(ca)
        p_home = 0.5 + 0.004 * (rh - ra)
        if prefer in SIDES:
            p_home += 0.08 if prefer == "home" else -0.08
        p_home = max(0.25, min(0.75, p_home))
        win = "home" if rng.random() < p_home else "away"
        if force in SIDES:
            win = force
        wpid, lpid = (ch, ca) if win == "home" else (ca, ch)
        du, dv = self.rel(win, g, *dot)
        back = self.ab(win, g, du - 18.0, dv * 0.6 + rng.uniform(-5, 5))
        d1 = self._role_pid(win, g + 0.01, "D1")
        dist = math.hypot(back[0] - dot[0], back[1] - dot[1])
        tp = max(0.45, dist / PASS)
        drop = _seg("pass", g, g + tp, dot, back, side=win, car=wpid, rcv=d1)
        drop["ev"].append({"type": "faceoff", "meta": {"reason": reason, "winner": win, "winner_side": win, "winner_id": wpid, "loser_id": lpid,
                                                       "location": self._dot_name(dot, g), "dot": [round(dot[0], 1), round(dot[1], 1)],
                                                       "offender": offender}})
        drop["fo_drop"] = {"dot": [round(dot[0], 1), round(dot[1], 1)], "win": win}
        segs.append(drop)
        return segs, {"g": g + tp, "puck": back, "side": win, "car": d1}

    def _dot_name(self, dot: Tuple[float, float], g: float) -> str:
        if abs(dot[0]) < 1 and abs(dot[1]) < 1:
            return "center"
        hx = dot[0] * self.dir("home", g)
        if hx < -40:
            return "home_end"
        if hx > 40:
            return "away_end"
        return "neutral"

    # ───────────────────────────── play-pattern banks (weighted by tactics/traits, no repeats within ~2 min per team)
    def _pick_pattern(self, X: str, g0: float, cat: str, weights: Dict[str, float], rng: random.Random) -> str:
        mem = self.__dict__.setdefault("_pat_mem", {}).setdefault((X, cat), [])
        w = {}
        for n, wt in weights.items():
            if wt <= 0:
                continue
            recent = any(nm == n and 0.0 <= g0 - t_ < 120.0 for t_, nm in mem)
            w[n] = wt * (0.0 if recent else 1.0)
        if not any(v > 0 for v in w.values()):  # everything used recently: least recently used wins
            last_use = {n: max([i_ for i_, (t_, nm) in enumerate(mem) if nm == n] or [-1]) for n in weights if weights[n] > 0}
            w = {min(last_use, key=last_use.get): 1.0}
        tot = sum(w.values()) or 1.0
        r, acc, pick = rng.random() * tot, 0.0, next(iter(w))
        for n, wt in w.items():
            acc += wt
            if r <= acc:
                pick = n
                break
        mem.append((g0, pick))
        del mem[:-12]
        self.__dict__.setdefault("pattern_log", []).append((round(g0, 1), X, cat, pick))
        return pick

    def _route(self, X: str, g0: float, p0: Tuple[float, float], car: Optional[str], gB: float,
               L: Optional[Tuple[float, float]], shooter: Optional[str], rng: random.Random, *, rush: bool = False,
               pp: bool = False, shot: Optional[str] = None, odd: bool = False) -> List[Dict[str, Any]]:
        u0, v0 = self.rel(X, g0, *p0)
        s = 1.0 if rng.random() < 0.5 else -1.0
        wps: List[Tuple[str, float, float, str]] = []
        T = max(0.5, gB - g0)
        tac = (getattr(self.loop, "tactics", None) or {}) if X == self.loop.lg.user_side else {}
        fc = tac.get("forecheck")
        tr = self._traits(car) if car else {"speed": 0.5, "hands": 0.5, "star": 0.0, "lane": 0.0}
        if rush and L is not None:
            lu, lv = self.rel(X, g0, *L)
            if u0 < 0:
                wps.append(("carry" if car == shooter else "pass:stretch", max(u0 + 10, -5.0), lv * 0.5, "SHOOTER"))
            wps.append(("carry", lu, lv, "SHOOTER"))
            return self._route_segs(X, g0, (u0, v0), car, wps, T, shooter, rng, rush=True, shot=shot)
        if u0 < -30.0:
            bo = self._pick_pattern(X, g0, "breakout", {"d2d": 1.0, "wheel": 0.6 + 0.4 * tr["speed"], "rim": 0.8, "stretch": 0.35 + (0.35 if fc == "aggressive" else 0.0),
                                                        "reverse": 0.5}, rng)
            wps += {"d2d": [("pass", -80.0, -s * 14.0, "D2"), ("pass", -32.0, s * 30.0, "W"), ("carry", 4.0, s * 24.0, "W")],
                    "wheel": [("carry", -93.0, s * 4.0, "CAR"), ("carry", -62.0, -s * 37.0, "CAR"), ("carry", 2.0, -s * 30.0, "CAR")],
                    "rim": [("pass:rim", -52.0, s * 39.0, "W"), ("carry", 4.0, s * 28.0, "W")],
                    "stretch": [("pass:stretch", 14.0, s * 20.0, "C"), ("carry", 22.0, s * 16.0, "C")],
                    "reverse": [("carry", -88.0, s * 30.0, "CAR"), ("pass:reverse", -88.0, -s * 28.0, "D2"), ("pass", -36.0, -s * 32.0, "W2"),
                                ("carry", 4.0, -s * 26.0, "W2")]}[bo]
        elif u0 < 22.0:
            wps.append(("carry", 8.0, max(-25.0, min(25.0, v0)), "CAR"))
        if u0 < 25.0:
            ew = {"carry_wide": 1.0 + 0.6 * tr["speed"] - 0.3 * tr["lane"], "carry_mid": 0.5 + 0.5 * tr["hands"] + 0.4 * tr["lane"] + 0.4 * tr["star"],
                  "dump_chase": 0.45 + (0.7 if fc == "aggressive" else (-0.3 if fc == "passive" else 0.0)), "chip_chase": 0.4 + (0.3 if fc == "aggressive" else 0.0),
                  "drop_pass": 0.35, "stretch": 0.25}
            if pp:
                ew.update(dump_chase=0.15, chip_chase=0.1, drop_pass=0.8)
            en = self._pick_pattern(X, g0, "entry", ew, rng)
            wps += {"carry_wide": [("carry", 30.0, s * 30.0, "CAR"), ("entry", 30.0, s * 30.0, "-")],
                    "carry_mid": [("carry", 30.0, s * 6.0, "CAR"), ("entry", 30.0, s * 6.0, "-")],
                    "dump_chase": [("dump:rim", 88.0, -s * 33.0, "-"), ("carry", 80.0, -s * 30.0, "W")],
                    "chip_chase": [("dump:chip", 52.0, s * 34.0, "-"), ("carry", 68.0, s * 32.0, "W")],
                    "drop_pass": [("carry", 20.0, s * 12.0, "CAR"), ("pass:drop", 13.0, s * 15.0, "W"), ("carry", 32.0, s * 20.0, "W"), ("entry", 32.0, s * 20.0, "-")],
                    "stretch": [("pass:stretch", 21.0, s * 27.0, "W"), ("carry", 33.0, s * 27.0, "W"), ("entry", 33.0, s * 27.0, "-")]}[en]
        lu = lv = 0.0
        if L is not None:
            lu, lv = self.rel(X, g0, *L)
        # Odd-man rush (the engine rolled a rush chance and the puck starts outside their zone): break out,
        # then a second forward carries up the far lane and feeds the shooter across. No zone set-up.
        odd = bool(odd and L is not None and u0 < 12.0 and T >= 4.5 and lu > 40.0 and not pp)
        if odd:
            far = -1.0 if lv >= 0 else 1.0
            wps = [w_ for w_ in wps if _kb(w_[0]) != "entry" and w_[1] < 12.0 and _kb(w_[0]) != "dump"]
            tail = [("carry", 6.0, far * 20.0, "MATE"), ("carry", max(30.0, lu - 20.0), far * 17.0, "MATE"), ("pass", lu, lv, "SHOOTER")]
            return self._route_segs(X, g0, (u0, v0), car, wps + tail, T, shooter, rng, shot=shot, odd_tail=3)

        def oz_seq(name: str, sg: float) -> List[Tuple[str, float, float, str]]:
            return {
                "low_cycle": [("carry", 80.0, sg * 34.0, "W"), ("pass", 30.0, sg * 25.0, "D1"), ("carry", 31.0, sg * 14.0, "D1"), ("pass", 64.0, sg * 31.0, "W"),
                              ("carry", 82.0, sg * 26.0, "W")],
                "overload": [("carry", 82.0, sg * 33.0, "W"), ("pass", 63.0, sg * 27.0, "C"), ("carry", 66.0, sg * 20.0, "C"), ("pass", 31.0, sg * 24.0, "D1"),
                             ("pass:bank", 70.0, sg * 33.0, "W")],
                "umbrella": [("pass", 30.0, sg * 22.0, "D1"), ("carry", 31.0, sg * 10.0, "D1"), ("pass", 31.0, -sg * 14.0, "D2"), ("carry", 32.0, -sg * 6.0, "D2")],
                "walkout": [("carry", 85.0, sg * 30.0, "W"), ("carry", 93.0, sg * 8.0, "W"), ("carry", 93.0, -sg * 8.0, "W"), ("carry", 80.0, -sg * 20.0, "W")],
                "net_front": [("carry", 84.0, sg * 32.0, "W"), ("pass", 30.0, sg * 20.0, "D1"), ("carry", 32.0, sg * 8.0, "D1"), ("pass:saucer", 62.0, sg * 30.0, "W")],
                "quick_strike": [],
                # east-west looks: the puck changes sides through the seam, across the top or behind the net
                "royal_road": [("carry", 68.0, sg * 31.0, "STAR"), ("pass:saucer", 66.0, -sg * 27.0, "STAR2"), ("carry", 79.0, -sg * 28.0, "STAR2"),
                               ("pass", 31.0, -sg * 16.0, "D2")],
                "high_switch": [("pass", 31.0, sg * 22.0, "D1"), ("carry", 33.0, -sg * 8.0, "D1"), ("pass", 58.0, -sg * 31.0, "W2"),
                                ("carry", 81.0, -sg * 30.0, "W2")],
                "behind_net": [("carry", 86.0, sg * 27.0, "STAR"), ("carry", 94.0, sg * 6.0, "STAR"), ("pass", 86.0, -sg * 24.0, "C"),
                               ("carry", 70.0, -sg * 31.0, "C"), ("pass", 31.0, -sg * 10.0, "D2")],
                "weak_side": [("pass", 30.0, sg * 20.0, "D1"), ("pass", 31.0, -sg * 16.0, "D2"), ("pass", 64.0, -sg * 31.0, "W2"),
                              ("carry", 82.0, -sg * 27.0, "W2")],
                "pp_umbrella": [("pass", 31.0, 0.0, "D1"), ("carry", 33.0, sg * 6.0, "D1"), ("pass", 60.0, sg * 27.0, "W"), ("carry", 63.0, sg * 25.0, "W")],
                "pp_overload": [("pass", 60.0, sg * 27.0, "W"), ("carry", 66.0, sg * 30.0, "W"), ("pass", 80.0, sg * 18.0, "C"), ("pass:drop", 62.0, sg * 26.0, "W")],
                "pp_low": [("carry", 84.0, sg * 20.0, "W"), ("pass:saucer", 84.0, -sg * 16.0, "W2"), ("carry", 80.0, -sg * 24.0, "W2"), ("pass", 32.0, -sg * 8.0, "D1")],
            }[name]

        def oz_pick() -> List[Tuple[str, float, float, str]]:
            if T < 8.0:
                return []  # no time to set anything up: straight to the shot (not logged as a pattern)
            if pp:
                w = {"pp_umbrella": 1.0, "pp_overload": 0.8, "pp_low": 0.7}
            else:
                w = {"low_cycle": 1.0, "overload": 0.7, "umbrella": 0.6 + (0.7 if 0 < lu < 42 else 0.0), "walkout": 0.4 + 0.6 * tr["hands"],
                     "net_front": 0.6 + (0.5 if shot == "tip" else 0.0), "quick_strike": 0.25,
                     "royal_road": 0.5 + 0.7 * tr["hands"], "high_switch": 0.8, "behind_net": 0.7, "weak_side": 0.8}
            return oz_seq(self._pick_pattern(X, g0, "ozone", w, rng), 1.0 if rng.random() < 0.5 else -1.0)

        final = None
        pre_final: List[Tuple[str, float, float, str]] = []
        if L is not None:
            fk = "pass" if shooter and shooter != car else "carry"
            if shot == "tip":
                pre_final = [("carry", 31.0, max(-20.0, min(20.0, lv * 0.4)), "D1")]
                fk = "pass:pointshot"
            elif shot == "wraparound":
                pre_final = [("carry", 94.0, (1.0 if lv >= 0 else -1.0) * -7.0, "SHOOTER")]
                fk = "carry"
            elif shot == "one_timer":
                fk = "pass:onetimer"
            final = (fk, lu, lv, "SHOOTER")

        def dur(k: str, a: Tuple[float, float], b: Tuple[float, float]) -> float:
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            kb = _kb(k)
            if kb == "entry":
                return 0.0
            return max(0.5, d / PASS) if kb in ("pass", "dump") else max(MIN_CARRY_LEG, d / CARRY)

        def total(lst: List[Tuple[str, float, float, str]]) -> float:
            tt, prv = 0.0, (u0, v0)
            for k, u, v, _r in lst:
                tt += dur(k, prv, (u, v))
                if _kb(k) != "entry":
                    prv = (u, v)
            return tt

        body = list(wps)
        cyc, ci, guard = oz_pick(), 0, 0
        tail = pre_final + ([final] if final is not None else [])
        while guard < 30 and (total(body + tail) < (0.85 if final is not None else 0.9) * T):
            if ci >= len(cyc):
                cyc, ci = oz_pick(), 0  # switch to another pattern instead of looping the same one
                if not cyc:
                    break
                continue
            body.append(cyc[ci])
            ci += 1
            guard += 1
        seq = body + tail
        while len(seq) > 1 and total(seq) > T:
            seq.pop(0)
        return self._route_segs(X, g0, (u0, v0), car, seq, T, shooter, rng, shot=shot)

    def _route_segs(self, X: str, g0: float, start_uv: Tuple[float, float], car: Optional[str], seq: List[Tuple[str, float, float, str]],
                    T: float, shooter: Optional[str], rng: random.Random, rush: bool = False, shot: Optional[str] = None,
                    odd_tail: int = 0) -> List[Dict[str, Any]]:
        # resolve who holds the puck at each step so kinds are right before timing (a "pass" to the
        # player who already has it is a carry), then drop early waypoints rather than speed things up
        def who_of(role: str, hold_: Optional[str]) -> Optional[str]:
            if role == "SHOOTER":
                return shooter or hold_
            if role == "MATE":   # the fastest forward who isn't the shooter leads the rush
                fw_ = [q for q in self.lineup_at(g0 + T)[X] if q != shooter and self._book_pos(q) != "D"]
                return max(fw_, key=lambda q: self._traits(q)["speed"]) if fw_ else hold_
            if role == "CAR":
                return hold_ or self._role_pid(X, g0 + T, "C")
            if role == "-":
                return None
            return self._role_pid(X, g0 + T, role)  # the unit on the ice at the shot

        def resolve(sq: List[Tuple[str, float, float, str]]) -> List[Tuple[str, float, float, str]]:
            out, hold_ = [], car
            for k, u, v, role in sq:
                who = who_of(role, hold_)
                if _kb(k) == "pass" and who is not None and who == hold_:
                    k = "carry"
                out.append((k, u, v, role))
                if _kb(k) in ("pass", "carry") and who is not None:
                    hold_ = who
                elif _kb(k) == "dump":
                    hold_ = None
            return out

        def timing(sq: List[Tuple[str, float, float, str]]) -> List[float]:
            ds, prv = [], start_uv
            for k, u, v, _r in sq:
                d = math.hypot(u - prv[0], v - prv[1])
                kb, kf = _kb(k), _kf(k)
                if kb == "entry":
                    ds.append(0.0)
                elif kb in ("pass", "dump"):
                    sp_ = DUMP if kb == "dump" else PASS
                    if kf in ("saucer", "chip"):
                        sp_ *= 0.8          # floated: slower
                    elif kf in ("rim", "bank"):
                        d *= 1.25           # longer path along / off the boards
                    elif kf == "pointshot":
                        sp_ = 85.0
                    ds.append(max(0.5, d / sp_))
                else:
                    ds.append(max(1.0 if rush else MIN_CARRY_LEG, d / (CARRY_RUSH if rush else CARRY)))
                if kb != "entry":
                    prv = (u, v)
            return ds

        if not seq:  # nothing planned (very short gap): the holder protects the puck where he is
            seq = [("carry", max(-85.0, min(85.0, start_uv[0] + 4.0)), start_uv[1] * 0.95, "CAR")]
        seq = resolve(seq)
        if shooter and len(seq) >= 2 and seq[-1][3] == "SHOOTER" and shot not in ("tip", "wraparound", "one_timer") and not odd_tail:
            # the shooter already touched it earlier: he keeps it and carries to his spot from there
            hold_, last_i = car, None
            for i, (k, u, v, role) in enumerate(seq[:-1]):
                who = who_of(role, hold_)
                if _kb(k) in ("pass", "carry") and who is not None:
                    hold_ = who
                elif _kb(k) == "dump":
                    hold_ = None
                if hold_ == shooter and _kb(k) != "dump":
                    last_i = i
            if car == shooter and last_i is None and not any(r_ == "-" for (_k, _u, _v, r_) in seq[:-1]):
                last_i = -1
            if last_i is not None and last_i < len(seq) - 2:
                fk, fu, fv, _fr = seq[-1]
                seq = seq[:last_i + 1] + [("carry", fu, fv, "SHOOTER")]

        def entry_fix(sq: List[Tuple[str, float, float, str]]) -> List[Tuple[str, float, float, str]]:
            # a pass into the zone to a shooter who had to stay onside: carry it over the line first
            if sq and _kb(sq[-1][0]) == "pass" and sq[-1][1] >= BLUE:
                prv = sq[-2] if len(sq) >= 2 else ("carry", start_uv[0], start_uv[1], "CAR")
                who_ = "CAR" if len(sq) < 2 else prv[3]
                if prv[1] < BLUE + 10.0 and _kb(prv[0]) != "dump":
                    sq = sq[:-1] + [("carry", BLUE + 14.0, max(-30.0, min(30.0, prv[2])), who_), sq[-1]]
                elif sq[-1][1] - prv[1] > 25.0 and _kb(prv[0]) != "dump" and _kf(sq[-1][0]) != "pointshot":
                    sq = sq[:-1] + [("carry", prv[1] + 8.0, prv[2], who_), sq[-1]]  # a beat in the zone while the shooter gets there
            return sq

        seq = resolve(seq if odd_tail else entry_fix(seq))
        durs = timing(seq)
        while len(seq) > max(2, odd_tail) and sum(durs) > T * 1.15:
            seq = resolve(seq[1:] if odd_tail else entry_fix(seq[1:]))
            durs = timing(seq)
        if odd_tail:   # the rush itself runs at rush speed
            for i in range(max(0, len(seq) - odd_tail), len(seq)):
                if _kb(seq[i][0]) == "carry":
                    durs[i] *= CARRY / CARRY_RUSH
        tot = sum(durs) or 1.0
        if tot < T:
            # Spare time goes to the earlier carries only. The last move to the shot (and an odd-man rush)
            # keeps its natural speed, so nobody walks into a shooting lane and waits there.
            keep = max(1, odd_tail) if shooter else 0
            idx = [i for i, (k, *_x) in enumerate(seq) if _kb(k) == "carry" and i < len(seq) - keep]
            if not idx and len(seq) > keep:
                idx = [0]
            extra = T - tot
            if idx:
                ct = sum(durs[i] for i in idx) or 1.0
                for i in idx:
                    durs[i] += extra * (durs[i] / ct)
            elif durs:
                durs[0] += extra
        else:
            sc = T / tot
            durs = [d * sc for d in durs]
        segs: List[Dict[str, Any]] = []
        g = g0
        cur = start_uv
        holder = car
        for (k, u, v, role), d in zip(seq, durs):
            kb, kf = _kb(k), _kf(k)
            if kb == "entry":
                if segs:
                    segs[-1]["ev_end_hint"] = "entry"
                continue
            g1 = g + d
            pa = self.ab(X, g, *cur)
            pb = self.ab(X, g, u, v)
            n_before = len(segs)
            if role == "SHOOTER":
                who = shooter or holder
                if kb == "pass" and who == holder:
                    kb = "carry"  # the shooter already has it: he carries to his spot
            elif role == "MATE":
                who = who_of("MATE", holder)
            elif role == "CAR":
                who = holder or self._role_pid(X, g, "C")
            elif role == "-":
                who = None
            else:
                who = self._role_pid(X, g1, role)
            if kb == "carry":
                if who and holder and who != holder:
                    mid = (pa[0] + 0.5 * (pb[0] - pa[0]), pa[1] + 0.5 * (pb[1] - pa[1]))
                    pt_ = min(g1 - 0.1, g + max(0.35, math.hypot(mid[0] - pa[0], mid[1] - pa[1]) / PASS))
                    sg = _seg("pass", g, pt_, pa, mid, side=X, car=holder, rcv=who, rush=rush)
                    self._pass_path(sg, "", rng)
                    segs.append(sg)
                    g, pa = pt_, mid
                cs = _seg("carry", g, g1, pa, pb, side=X, car=who or holder, rush=rush)
                tr = self._traits(who or holder) if (who or holder) else None
                if tr is not None:  # stickhandling while carrying (bigger dekes for skilled hands / breakaways)
                    cs["deke"] = min(1.5, 0.35 + 1.0 * tr["deke"] * (1.4 if rush else 1.0))
                    cs["dper"] = tr["dper"]
                segs.append(cs)
                holder = who or holder
            elif kb == "pass" and (who is None or who == holder):  # no self-passes: he just carries it there
                segs.append(_seg("carry", g, g1, pa, pb, side=X, car=holder, rush=rush))
            elif kb == "pass":
                sg = _seg("pass", g, g1, pa, pb, side=X, car=holder, rcv=who, rush=rush)
                self._pass_path(sg, kf, rng)
                segs.append(sg)
                holder = who
            else:  # dump-in / chip (legal: from beyond the red line)
                sg = _seg("loose", g, g1, pa, pb, side=X, dump=True)
                if kf == "rim":
                    poly = rim_poly(pa, pb)
                    if poly:
                        sg["poly"], sg["ptype"] = poly, "rim"
                elif kf == "chip":
                    sg["ptype"], sg["lift"] = "chip", 5.0
                if "ptype" not in sg:
                    sg["ptype"] = "dump"
                segs.append(sg)
                holder = None
            if odd_tail and role in ("MATE", "SHOOTER"):
                for sg_ in segs[n_before:]:
                    sg_["rush"] = True
                    sg_["odd"] = shooter
                    sg_["car_fast"] = True
            cur = (u, v)
            g = g1
        if shooter:
            for sg_ in segs[-2:]:
                sg_["pre_shot"] = True   # the last touches before a real shot: forwards go to the net
        return segs

    def _pass_path(self, sg: Dict[str, Any], kf: str, rng: random.Random) -> None:
        """Give a pass its path: bank / rim / saucer / stretch / drop / bounce / straight (presentation only)."""
        pa, pb = sg["p0"], sg["p1"]
        d = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        dur = max(0.05, sg["g1"] - sg["g0"])
        X = sg.get("side") or "home"
        dxu = (pb[0] - pa[0]) * self.dir(X, sg["g0"])
        typ = kf
        if not typ:
            if d > 75.0:
                typ = "stretch"
            elif d < 18.0 and dxu < -4.0:
                typ = "drop"
            elif abs(pa[1]) > 20.0 and abs(pb[1]) > 20.0 and pa[1] * pb[1] > 0 and d > 25.0 and rng.random() < 0.4:
                typ = "bank"
            elif d > 30.0 and rng.random() < 0.2:
                typ = "saucer"
            elif rng.random() < 0.15:
                typ = "bounce"
            else:
                typ = "line"
        if typ == "bank":
            poly = bank_poly(pa, pb)
            if poly_len(poly) / dur > 72.0:
                typ, poly = ("line", None)
            else:
                sg["poly"] = poly
        elif typ == "rim":
            poly = rim_poly(pa, pb)
            if poly and poly_len(poly) / dur <= 72.0:
                sg["poly"] = poly
            else:
                typ = "line" if not poly else "rim"
                if poly:
                    sg["poly"] = poly
        elif typ == "saucer":
            sg["arc"], sg["lift"] = (1.0 if rng.random() < 0.5 else -1.0) * min(2.0, 0.04 * d), 1.2
        elif typ == "bounce":
            sg["jit"] = rng.uniform(0.3, 0.6)
        elif typ in ("reverse",):
            typ = "reverse"
        sg["ptype"] = typ

    def _carry_to(self, X: str, g0: float, p0: Tuple[float, float], car: Optional[str], u: float, v: float) -> Tuple[List[Dict[str, Any]], float, Tuple[float, float], Optional[str]]:
        pa = p0
        pb = self.ab(X, g0, u, v)
        d = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        dur = max(MIN_CARRY_LEG, d / CARRY)
        car = car or self._role_pid(X, g0, "C")
        return [_seg("carry", g0, g0 + dur, pa, pb, side=X, car=car)], g0 + dur, pb, car

    def _derived_stoppage(self, g0: float, state: Dict[str, Any], gB: float, rng: random.Random,
                          next_side: Optional[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Maybe insert icing / offside / puck-over-glass / hit within a long gap (presentation only)."""
        gap = gB - g0  # gB here is the deadline: the build-up to the next real shot keeps >= 12 s
        if gap < 9.0:
            return [], state
        mp = self._manpower(g0)
        sh_side = mp.get("short")
        if any(self._goalie_wanted(s, g0, None) is None for s in SIDES):
            return [], state
        r = rng.random()
        side = state.get("side") if state.get("side") in SIDES else (next_side or "home")
        segs: List[Dict[str, Any]] = []
        p = state["puck"]
        car = state.get("car")
        g = g0
        if r < 0.22:  # icing / shorthanded clear
            Y = side
            pre, g, p, car = self._carry_to(Y, g, p, car, -42.0, rng.uniform(-18, 18))
            segs += pre
            uY, vY = self.rel(Y, g, *p)
            tgt = self.ab(Y, g, 97.0, (1 if vY >= 0 else -1) * rng.uniform(8, 26))
            tau = math.hypot(tgt[0] - p[0], tgt[1] - p[1]) / 62.0
            sg = _seg("loose", g, g + tau, p, tgt, side=Y, dump=True, ice=True)
            segs.append(sg)
            g += tau
            rim = self.ab(Y, g, 93.0, (1 if vY >= 0 else -1) * 36.0)
            segs.append(_seg("loose", g, g + 1.2, tgt, rim))
            g += 1.2
            if Y == sh_side:
                segs[-2]["ev"].append({"type": "rule", "meta": {"rule": "sh_clear", "side": Y}})
                return segs, {"g": g, "puck": rim, "side": None, "car": None}
            if rng.random() < 0.15:  # hybrid race won by the attacker: waved off
                segs[-1]["ev"].append({"type": "rule", "meta": {"rule": "icing_waved", "side": Y}})
                return segs, {"g": g, "puck": rim, "side": Y, "car": self._role_pid(Y, g, "W")}
            dot = self.ab(Y, g, -ZONE_DOT_U, DOT_V if vY >= 0 else -DOT_V)
            wh = _dead(g, rng.uniform(8.0, 11.0), rim, dot, why=("icing", {"side": Y}), fo=dot, reason="icing", freeze_side=Y)
            wh["ev"].append({"type": "stoppage", "meta": {"reason": "icing", "side": Y}})
            self._freeze[Y].append((g, g + 0.05))  # whistle to the faceoff drop (gc frozen in between)
            segs.append(wh)
            fos, st = self._faceoff(g, dot, next_side, rng, "icing", offender=Y)
            return segs + fos, st
        if r < 0.33:  # offside (staged: a winger crosses ahead of the puck)
            X = side
            pre, g, p, car = self._carry_to(X, g, p, car, -45.0, rng.uniform(-20, 20))  # regroup first
            segs += pre
            pre, g, p, car = self._carry_to(X, g, p, car, 15.0, rng.uniform(-15, 15))
            segs += pre
            uX, vX = self.rel(X, g, *p)
            over = self.ab(X, g, 26.5, vX)
            segs.append(_seg("carry", g, g + 0.6, p, over, side=X, car=car, offside_pid=self._role_pid(X, g, "W") if car != self._role_pid(X, g, "W") else self._role_pid(X, g, "W2")))
            g += 0.6
            dot = self.ab(X, g, NZ_DOT_U, DOT_V if vX >= 0 else -DOT_V)
            wh = _dead(g, rng.uniform(6.5, 9.0), over, dot, why=("offside", {"side": X}), fo=dot, reason="offside")
            wh["ev"].append({"type": "stoppage", "meta": {"reason": "offside", "side": X, "player_id": segs[-1].get("offside_pid")}})
            segs.append(wh)
            fos, st = self._faceoff(g, dot, next_side, rng, "offside", offender=X)
            return segs + fos, st
        if r < 0.355:  # puck over the glass from the defensive zone
            Y = side
            pre, g, p, car = self._carry_to(Y, g, p, car, -74.0, rng.uniform(-20, 20))
            segs += pre
            uY, vY = self.rel(Y, g, *p)
            out = self.ab(Y, g, -58.0, 42.0 if vY >= 0 else -42.0)
            segs.append(_seg("shot", g, g + 0.5, p, (out[0], 42.4 if out[1] > 0 else -42.4), side=Y))
            g += 0.5
            dot = self.ab(Y, g, -ZONE_DOT_U, DOT_V if vY >= 0 else -DOT_V)
            wh = _dead(g, rng.uniform(8.0, 10.0), (out[0], out[1]), dot, why=("glass", {"side": Y}), fo=dot, reason="glass")
            wh["ev"].append({"type": "stoppage", "meta": {"reason": "glass", "side": Y}})
            segs.append(wh)
            fos, st = self._faceoff(g, dot, next_side, rng, "glass", offender=Y)
            return segs + fos, st
        if r < 0.47 and side in SIDES:  # dump-in and a hit on the retrieval (no stat effect)
            X = side
            pre, g, p, car = self._carry_to(X, g, p, car, 8.0, rng.uniform(-15, 15))
            segs += pre
            tgt = self.ab(X, g, 90.0, (1 if rng.random() < 0.5 else -1) * 30.0)
            tau = math.hypot(tgt[0] - p[0], tgt[1] - p[1]) / DUMP
            segs.append(_seg("loose", g, g + tau, p, tgt, side=X, dump=True))
            g += tau
            hitter = self._role_pid(X, g, "W")
            victim = self._role_pid(opp(X), g, "D1")
            hs = _seg("loose", g, g + 1.0, tgt, tgt, hit=(hitter, victim))
            hs["ev"].append({"type": "hit", "meta": {"hitter_id": hitter, "victim_id": victim, "side": X}})
            segs.append(hs)
            g += 1.0
            return segs, {"g": g, "puck": tgt, "side": None, "car": None}
        return [], state

    def _manpower(self, gc: float) -> Dict[str, Any]:
        h = a = 5
        if self.loop._cap is not None:
            for pen in self.loop._cap.penalties:
                if pen["start"] <= gc < pen["start"] + pen["dur"]:
                    if pen["side"] == "home":
                        h -= 1
                    else:
                        a -= 1
        h, a = max(3, h), max(3, a)
        return {"home": h, "away": a, "short": "home" if h < a else ("away" if a < h else None)}

    def _buildup(self, g0: float, state: Dict[str, Any], B: Optional[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
        if B is None:
            return [_dead(g0, 99999.0, state["puck"], state["puck"], term=True)]
        gB = float(B["t"])
        segs: List[Dict[str, Any]] = []
        if B["type"] in ("attempt", "goal"):
            a = B.get("att") or {}
            X = a.get("side") if a.get("side") in SIDES else "home"
            L = (float((B.get("shot_loc") or {}).get("x", 0.0)), float((B.get("shot_loc") or {}).get("y", 0.0)))
            shooter = a.get("shooter_id")
            rush = a.get("chance") == "breakaway"
            odd = (not rush) and "RUSH" in str(a.get("chance_raw") or "").upper()
        elif B["type"] == "shootout_attempt" or (B["type"] == "period_start" and int(B.get("period") or 0) == 5):
            return [_seg("hold", g0, gB, state["puck"], state["puck"], ph="shootout")] if gB > g0 else []
        else:
            X = state.get("side") if state.get("side") in SIDES else ("home" if rng.random() < 0.5 else "away")
            L, shooter, rush = None, None, False
            odd = False
        st = dict(state)
        # derived stoppage in long gaps (before the build-up to B)
        if B["type"] in ("attempt", "goal") and gB - g0 >= 20.0:
            ds, st2 = self._derived_stoppage(g0, st, gB - 12.0, rng, X)
            if ds and float(st2.get("g", g0)) <= gB - 12.0:
                segs += ds
                st = st2
        g = float(st.get("g", g0)) if segs else g0
        p, side, car = st["puck"], st.get("side"), st.get("car")
        # delayed penalty: the non-offending team keeps the puck with an extra attacker
        if B["type"] == "penalty" and gB - g >= 12.0:
            P = (B.get("pen") or {}).get("side") or "home"
            N = opp(P)
            dly = rng.uniform(4.0, 7.0)
            pre_end = gB - dly
            if side != N:
                segs.append(_seg("loose", g, g + 0.5, p, p))
                g += 0.5
                side, car = N, self._role_pid(N, g, "C")
            segs += self._route(N, g, p, car, pre_end, None, None, rng)
            pz = self.ab(N, pre_end, 60.0, rng.uniform(-20, 20))
            pz_from = segs[-1]["p1"] if segs else p
            dseg = _seg("carry", pre_end, gB, pz_from, pz, side=N, car=self._role_pid(N, pre_end, "W"), delayed=N)
            dseg["ev"].append({"type": "delayed_penalty", "meta": {"side": P, "beneficiary": N, "delayed": True,
                                                                     "player_id": (B.get("pen") or {}).get("player_id")}})
            segs.append(dseg)
            return segs
        if side is None or car is None:
            rec = X if rng.random() < 0.6 or gB - g < 6.0 else opp(X)
            dt = min(1.2, max(0.4, (gB - g) * 0.2))
            segs.append(_seg("loose", g, g + dt, p, p))
            g, side, car = g + dt, rec, self._role_pid(rec, g + dt, "C")
        if side != X and gB - g > 8.0:
            dY = min(0.35 * (gB - g), 12.0)
            Y = side
            yu, yv = self.rel(Y, g, *p)
            q = self.ab(Y, g, min(yu + rng.uniform(25.0, 45.0), 20.0), max(-28.0, min(28.0, yv + rng.uniform(-10, 10))))
            segs.append(_seg("carry", g, g + dY, p, q, side=Y, car=car))
            segs.append(_seg("loose", g + dY, g + dY + 0.6, q, q, steal=X))
            g, p = g + dY + 0.6, q
            side, car = X, self._role_pid(X, g, "C")
        elif side != X:
            segs.append(_seg("loose", g, g + 0.4, p, p))
            g += 0.4
            side, car = X, self._role_pid(X, g, "C")
        pp = self._manpower(g).get("short") == opp(X)
        if L is not None:
            shot, defl = self._choose_shot(B, X, L, rng, gB - g)
            if shot in ("one_timer", "tip"):
                tq = gB                       # the puck arrives and is redirected / one-timed at once
            elif shot == "slap":
                tq = gB - 0.35                # wind-up
            else:
                tq = gB - rng.uniform(0.1, 0.3)
            if tq <= g + 0.2:
                tq = gB
            r_segs = self._route(X, g, p, car, tq, L, shooter, rng, rush=rush, pp=pp, shot=shot,
                                 odd=odd)
            segs += r_segs
            last = r_segs[-1] if r_segs else None
            if shot in ("one_timer", "tip") and not (last is not None and last["k"] == "pass" and last.get("rcv") == shooter):
                shot = "snap" if shot == "one_timer" else "wrist"
            if last is not None and last["k"] == "pass" and last.get("rcv") == shooter and shot == "one_timer":
                last["ptype"] = "onetimer" if last.get("ptype") in ("line", None, "") else last["ptype"]
            if tq < gB:
                segs.append(_seg("hold", tq, gB, L, L, side=X, car=shooter, windup=(shot == "slap")))
            self._shot[B["key"]] = {"shot_type": shot, "deflected": bool(defl)}
            for s_ in reversed(segs):  # the shooter gets to his spot before the puck arrives
                if s_["k"] in DEAD_KINDS or s_["g1"] <= gB - 6.5:
                    break
                involved = s_.get("car") == shooter or s_.get("rcv") == shooter
                final_leg = s_["k"] in ("hold",) or (s_["k"] in ("pass", "carry") and s_["p1"] == L)
                if involved and not final_leg:
                    break
                s_["pre"] = (shooter, L, X)
            self._note_buildup(B, segs, X, shooter)
        else:
            segs += self._route(X, g, p, car, gB, None, None, rng, pp=pp)
        return segs

    def _choose_shot(self, B: Dict[str, Any], X: str, L: Tuple[float, float], rng: random.Random, avail: float) -> Tuple[str, bool]:
        """Shot type from the engine's chance type, the spot and the shooter (presentation only)."""
        a = B.get("att") or {}
        ch = a.get("chance")
        lu, lv = self.rel(X, float(B["t"]), *L)
        tr = self._traits(a.get("shooter_id")) if a.get("shooter_id") else {"star": 0.0, "hands": 0.5}
        if ch == "breakaway":
            w = {"wrist": 3.0, "deke": 1.5 + 2.0 * tr["star"] + tr["hands"], "snap": 1.5, "backhand": 1.0}
        elif ch == "point" or lu < 42.0:
            w = {"slap": 3.0, "wrist": 2.0, "snap": 1.0}
        elif ch == "rebound":
            w = {"backhand": 1.5, "wrist": 2.0, "snap": 1.0, "tip": 0.6}
        elif lu >= 84.0 and abs(lv) >= 3.0:
            w = {"wraparound": 3.0, "backhand": 1.0}
        elif ch == "high_danger":
            w = {"wrist": 2.0, "tip": 1.5, "backhand": 1.2, "one_timer": 1.0}
        elif ch == "slot":
            w = {"one_timer": 2.0, "snap": 2.0, "wrist": 2.0}
        else:
            w = {"wrist": 2.0, "snap": 2.0, "slap": 0.6}
        if avail < 4.0:
            w.pop("tip", None)
            w.pop("one_timer", None)
        tot = sum(w.values())
        r, acc, pick = rng.random() * tot, 0.0, "wrist"
        for n, wt in w.items():
            acc += wt
            if r <= acc:
                pick = n
                break
        defl = (pick not in ("tip", "wraparound", "deke") and a.get("outcome") in ("SAVED", "MISSED") and 55.0 < lu < 82.0 and rng.random() < 0.12)
        return pick, defl

    def shot_info(self, key: Optional[str]) -> Dict[str, Any]:
        return dict(self._shot.get(key) or {}) if key else {}

    def _note_buildup(self, B: Dict[str, Any], segs: List[Dict[str, Any]], X: str, shooter: Optional[str]) -> None:
        """Remember who moved the puck before the shot (for build-up calls and replay breakdowns)."""
        moves = [s for s in segs if s["k"] in ("carry", "pass") and s.get("side") == X]
        if not moves:
            return
        last = moves[-1]
        prev = moves[-2] if len(moves) >= 2 else None
        info = {"shooter": shooter, "kind": "carry", "passer": None, "from_xy": None, "rush": bool(last.get("rush")), "seg": last}
        if last["k"] == "pass" and last.get("rcv") == shooter:
            info.update(kind="pass", passer=last.get("car"), from_xy=last["p0"])
        elif prev is not None and prev["k"] == "pass" and prev.get("rcv") == shooter:
            info.update(kind="pass", passer=prev.get("car"), from_xy=prev["p0"], seg=prev)
        info["start_seg"] = info["seg"]
        pseg = info["seg"] if info["kind"] == "pass" else None
        info["ptype"] = (pseg or {}).get("ptype") if pseg is not None else None
        info["shot_type"] = (self._shot.get(B["key"]) or {}).get("shot_type")
        self._bu[B["key"]] = info

    def _aftermath(self, A: Dict[str, Any], gB: float, nxt: Optional[Dict[str, Any]], rng: random.Random,
                   state: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        g = float(A["t"])
        typ = A["type"]
        segs: List[Dict[str, Any]] = []
        prefer = (nxt.get("att") or {}).get("side") if nxt is not None else None
        avail = gB - g
        if typ == "period_start":
            per = int(A.get("period") or 1)
            if per == 5:
                return [_dead(g, 3.0, (0.0, 0.0), (0.0, 0.0), ph="shootout", per=5)], {"g": g, "puck": (0.0, 0.0), "side": None, "car": None}
            segs.append(_dead(g, 6.0 if per > 1 else 4.0, (0.0, 0.0), (0.0, 0.0), fo=(0.0, 0.0), ph="stoppage", per=per, warmup=True))
            fos, st = self._faceoff(g, (0.0, 0.0), prefer, rng, "period_start")
            for s in fos:
                s["per"] = per
            return segs + fos, st
        if typ == "period_end":
            per = int(A.get("period") or 1)
            p0 = state["puck"]
            last = nxt is not None and nxt.get("type") == "final" and abs(float(nxt["t"]) - g) < 1e-6
            if last:  # game over: everyone to the benches, no intermission
                segs.append(_dead(g, 6.0, p0, p0, ph="stoppage", why=("final", {}), per=per, to_bench=True))
            else:
                segs.append(_dead(g, 60.0, p0, (0.0, 0.0), ph="intermission", why=("period_end", {}), per=per, to_bench=True))
            return segs, {"g": g, "puck": (0.0, 0.0), "side": None, "car": None}
        if typ == "final":
            return [_dead(g, 99999.0, state["puck"], state["puck"], term=True, why=("final", {}), to_bench=True)], state
        if typ == "shootout_attempt":
            r = A.get("so") or {}
            side = r.get("side") or "home"
            start = (0.0, 0.0)
            near = self.ab(side, g, 72.0, rng.uniform(-6, 6), period=5)
            net = self.ab(side, g, 89.5 if r.get("goal") else 86.0, rng.uniform(-2, 2), period=5)
            segs.append(_dead(g, 5.5, start, near, k="dead", ph="shootout", so=r, carry_by=r.get("shooter_id")))
            sh = _dead(g, 0.5, near, net, k="dead", ph="shootout", so=r)
            sh["shot_dead"] = True
            segs.append(sh)
            segs.append(_dead(g, 2.0, net, net, k="dead", ph="shootout", so=r))
            return segs, {"g": g, "puck": net, "side": None, "car": None}
        if typ == "penalty":
            pen = A.get("pen") or {}
            P = pen.get("side") if pen.get("side") in SIDES else "home"
            p0 = state["puck"]
            dot = self.ab(P, g, -ZONE_DOT_U, DOT_V if rng.random() < 0.5 else -DOT_V)
            wh = _dead(g, rng.uniform(12.0, 16.0), p0, dot, fo=dot, why=("penalty", {"player_id": pen.get("player_id")}), reason="penalty",
                       to_box=pen.get("player_id"), box_side=P)
            segs.append(wh)
            fos, st = self._faceoff(g, dot, opp(P), rng, "penalty", offender=P)
            return segs + fos, st
        # attempts / goals
        a = A.get("att") or {}
        X = a.get("side") if a.get("side") in SIDES else "home"
        L = (float((A.get("shot_loc") or {}).get("x", 0.0)), float((A.get("shot_loc") or {}).get("y", 0.0)))
        lu, lv = self.rel(X, g, *L)
        oc = a.get("outcome")
        shooter = a.get("shooter_id")
        si = self._shot.get(A.get("key")) or {}
        stype = si.get("shot_type") or "wrist"
        vshot = SHOT_V.get(stype, SHOT)

        def shot_seg(g_: float, p0_: Tuple[float, float], p1_: Tuple[float, float], **kw: Any) -> Dict[str, Any]:
            """Shot leg with its type: speed, lift, wraparound path round the post, deflection near the net."""
            poly = None
            if stype == "wraparound":
                e = 1.0 if p1_[0] > 0 else -1.0
                sy = 1.0 if p0_[1] >= 0 else -1.0
                poly = [p0_, (e * 91.5, -sy * 3.5), p1_]
            elif si.get("deflected") and not kw.get("blocked"):
                k = 0.82
                kink = (p0_[0] + (p1_[0] - p0_[0]) * k, p0_[1] + (p1_[1] - p0_[1]) * k)
                off = rng.uniform(2.5, 5.0) * (1.0 if rng.random() < 0.5 else -1.0)
                poly = [p0_, kink, (p1_[0], max(-40.0, min(40.0, p1_[1] + off)))]
            ln = poly_len(poly) if poly else math.hypot(p1_[0] - p0_[0], p1_[1] - p0_[1])
            tau_ = max(MIN_SHOT_FLIGHT if not kw.get("blocked") else 0.12, ln / vshot)
            sg = _seg("shot", g_, g_ + tau_, p0_, poly[-1] if poly else p1_, side=X, car=shooter, ptype=stype, lift=SHOT_LIFT.get(stype, 1.0), **kw)
            if poly:
                sg["poly"] = poly
            return sg

        if oc == "GOAL":
            net = self.ab(X, g, 90.5, max(-2.5, min(2.5, lv * 0.08)))
            sg = shot_seg(g, L, net)
            sg["p1"] = net if stype != "wraparound" else sg["p1"]
            if sg.get("poly"):
                sg["poly"][-1] = net
                sg["p1"] = net
            segs.append(sg)
            tau = sg["g1"] - sg["g0"]
            g += tau
            ot = a.get("seg") != "reg"
            cel = _dead(g, 12.0, net, net, ph="goal", why=("goal", {"player_id": shooter}), scorer=shooter, scorer_side=X, term=ot)
            segs.append(cel)
            if ot:
                return segs, {"g": g, "puck": net, "side": None, "car": None}
            segs.append(_dead(g, 8.0, net, (0.0, 0.0), fo=(0.0, 0.0), ph="stoppage", reason="goal"))
            fos, st = self._faceoff(g, (0.0, 0.0), prefer, rng, "after_goal")
            return segs + fos, st
        if oc == "SAVED" and not a.get("empty_net"):
            gp = self.ab(X, g, 86.0, max(-3.0, min(3.0, lv * 0.12)))
            sg = shot_seg(g, L, gp)
            if sg.get("poly"):
                sg["poly"][-1] = gp
                sg["p1"] = gp
            segs.append(sg)
            tau = sg["g1"] - sg["g0"]
            g += tau
            if avail >= 11.0 and rng.random() < 0.5:
                dot = self.ab(X, g, ZONE_DOT_U, DOT_V if lv >= 0 else -DOT_V)
                wh = _dead(g, rng.uniform(8.0, 10.0), gp, dot, fo=dot, why=("freeze", {"goalie_id": a.get("goalie_id")}), reason="freeze", scrum=X)
                wh["ev"].append({"type": "stoppage", "meta": {"reason": "freeze", "side": opp(X), "player_id": a.get("goalie_id")}})
                segs.append(wh)
                fos, st = self._faceoff(g, dot, prefer, rng, "freeze", offender=X)
                return segs + fos, st
            reb = self.ab(X, g, rng.uniform(70.0, 80.0), rng.uniform(-16.0, 16.0))
            segs.append(_seg("loose", g, g + 0.5, gp, reb, jit=rng.uniform(0.4, 0.9), ptype="rebound"))
            return segs, {"g": g + 0.5, "puck": reb, "side": None, "car": None}
        if oc == "BLOCKED":
            net = self.ab(X, g, 89.0, 0.0)
            bp = (L[0] + 0.12 * (net[0] - L[0]), L[1] + 0.12 * (net[1] - L[1]))
            sg = shot_seg(g, L, bp, blocked=True)
            segs.append(sg)
            tau = sg["g1"] - sg["g0"]
            dfl = clampxy(bp[0] + rng.uniform(-6, 6), bp[1] + rng.uniform(-6, 6))
            segs.append(_seg("loose", g + tau, g + tau + 0.6, bp, dfl))
            return segs, {"g": g + tau + 0.6, "puck": dfl, "side": None, "car": None}
        sgn = 1.0 if lv >= 0 else -1.0
        wide = self.ab(X, g, 97.0, sgn * rng.uniform(6.0, 22.0))
        sg = shot_seg(g, L, wide)
        segs.append(sg)
        wide = sg["p1"]
        g += sg["g1"] - sg["g0"]
        rim = self.ab(X, g, 90.0, sgn * 37.0)
        trim = max(0.5, math.hypot(rim[0] - wide[0], rim[1] - wide[1]) / 55.0)  # rims around the boards at 50-65 ft/s
        rs = _seg("loose", g, g + trim, wide, rim, ptype="rim")
        rpoly = rim_poly(wide, rim)
        if rpoly:
            rs["poly"] = rpoly
        segs.append(rs)
        g += trim
        if avail >= 14.0 and rng.random() < 0.18:
            dot = self.ab(X, g, ZONE_DOT_U, DOT_V * sgn)
            wh = _dead(g, rng.uniform(7.0, 9.0), rim, dot, fo=dot, why=("out", {}), reason="out")
            wh["ev"].append({"type": "stoppage", "meta": {"reason": "out", "side": X}})
            segs.append(wh)
            fos, st = self._faceoff(g, dot, prefer, rng, "out", offender=X)
            return segs + fos, st
        return segs, {"g": g, "puck": rim, "side": None, "car": None}

    def _next_anchor(self, after_idx: int) -> Optional[Dict[str, Any]]:
        return self._anchors[after_idx + 1] if after_idx + 1 < len(self._anchors) else None

    def _build_next_plan(self) -> bool:
        """Append the plan for the next anchor interval. Returns False when nothing is left."""
        if self.plans:
            last = self.plans[-1]
            ai = last["ai"] + 1
            state = last["end"]
        else:
            ai = 0
            state = {"g": 0.0, "puck": (0.0, 0.0), "side": None, "car": None}
        if ai >= len(self._anchors):
            return False
        if self.flat and self.flat[-1]["k"] == "live" and not self.flat[-1].get("done"):
            return False
        A = self._anchors[ai]
        B = self._next_anchor(ai)
        gA = float(A["t"])
        gB = float(B["t"]) if B is not None else gA + 99999.0
        rng = self._rng(A.get("key"), B.get("key") if B else None)
        if ai == 0 or float(state.get("g", gA)) < gA - 1e-6:
            pass
        segs, st = self._aftermath(A, gB, B, rng, state)
        gend = float(st.get("g", gA))
        live = None
        if B is not None and gB > gend + 0.05 and not any(s.get("term") for s in segs):
            if self.live_mode and B["type"] in ("attempt", "goal", "penalty", "period_end", "final"):
                if segs and segs[-1]["k"] == "loose" and segs[-1].get("ptype") == "rim":
                    px_ = float(segs[-1]["p1"][0])   # the rim carries on up the wall toward the point
                    st = dict(st)
                    st["pv"] = (-math.copysign(rng.uniform(26.0, 34.0), px_), 0.0)
                live = {"k": "live", "g0": gend, "g1": gB, "B": B, "ph": "play", "state": dict(st), "side": st.get("side"),
                        "car": None, "rcv": None, "fo": None, "why": None, "ev": [], "p0": st.get("puck") or (0.0, 0.0), "p1": st.get("puck") or (0.0, 0.0)}
            else:
                segs += self._buildup(gend, st, B, rng)
        if self.live_mode and live is None and B is not None and B["type"] in ("attempt", "goal") and gB - gend < 1.0:
            # a shot straight off the draw: it is taken from where the shooter lines up for the faceoff
            a_ = B.get("att") or {}
            S_, X_ = a_.get("shooter_id"), a_.get("side")
            dot_ = st.get("puck")
            if S_ and X_ in SIDES and dot_ is not None:
                try:
                    ids_ = list(self.onice.get(X_) or []) if S_ in (self.onice.get(X_) or []) else list(self.lineup_at(gB).get(X_) or [])
                    fs_ = self._faceoff_spots(X_, gB, tuple(dot_), ids_)
                    du_, dv_ = self.rel(X_, gB, float(dot_[0]), float(dot_[1]))
                    loc_ = fs_.get(S_)
                    if loc_ is None or math.hypot(loc_[0] - dot_[0], loc_[1] - dot_[1]) > 17.0:
                        loc_ = self.ab(X_, gB, du_ - 2.4, dv_)      # off the draw, right at the dot
                    B["shot_loc"] = {"x": round(loc_[0], 1), "y": round(loc_[1], 1), "est": True, "from_motion": True}
                except Exception:
                    pass
        self._commit_plan(ai, A, B, segs, live=live)
        return True

    def _commit_plan(self, ai: int, A: Optional[Dict[str, Any]], B: Optional[Dict[str, Any]], segs: List[Dict[str, Any]],
                     live: Optional[Dict[str, Any]] = None) -> None:
        pt = self.flat[-1]["pt1"] if self.flat else 0.0
        g = self.flat[-1]["g1"] if self.flat else 0.0
        clean: List[Dict[str, Any]] = []
        for s in segs:
            if s["k"] in DEAD_KINDS:
                s["g0"] = s["g1"] = max(g, s["g0"]) if not clean else clean[-1]["g1"]
                s["pt0"], s["pt1"] = pt, pt + s["d"]
            else:
                g0 = clean[-1]["g1"] if clean else max(g, s["g0"])
                g1 = max(g0, s["g1"])
                s["g0"], s["g1"] = g0, g1
                s["pt0"], s["pt1"] = pt, pt + (g1 - g0)
                if g1 - g0 <= 1e-9:
                    continue
            pt = s["pt1"]
            clean.append(s)
        if live is not None:
            live["g0"] = clean[-1]["g1"] if clean else g
            live["pt0"], live["pt1"] = pt, 1e18
            if clean:
                live["state"]["g"] = live["g0"]
            clean.append(live)
            self._play_stats["live_segs"] += 1
        elif B is not None and clean and B["type"] not in ("shootout_attempt",) and not clean[-1].get("term"):
            gB = float(B["t"])
            if clean[-1]["g1"] < gB - 1e-6:  # pad to the anchor so every real event keeps its exact t
                s = _seg("hold", clean[-1]["g1"], gB, clean[-1]["p1"], clean[-1]["p1"], side=clean[-1].get("side"), car=clean[-1].get("car") if clean[-1]["k"] in ("carry", "hold") else None)
                s["pt0"], s["pt1"] = pt, pt + (gB - clean[-1]["g1"])
                clean.append(s)
        end = {"g": clean[-1]["g1"] if clean else g, "puck": clean[-1]["p1"] if clean else (0.0, 0.0),
               "_live": live is not None,
               "side": (clean[-1].get("side") if clean and clean[-1]["k"] in ("carry", "pass", "hold") else None),
               "car": ((clean[-1].get("rcv") or clean[-1].get("car")) if clean and clean[-1]["k"] in ("carry", "pass", "hold") else None)}
        self.plans.append({"ai": ai, "A": A.get("key") if A else None, "B": B.get("key") if B else None, "segs": clean, "end": end})
        for s in clean:
            self.flat.append(s)
            self.flat_g.append(s["g0"])
        bu = self._bu.get(B.get("key")) if B else None
        if bu is not None:
            bu["pt"] = bu["start_seg"].get("pt0")

    def _ensure(self, gc: float) -> None:
        if self._refresh_anchors() and self.plans:
            self._rebuild_after_change()
        guard = 0
        while (not self.flat or self.flat[-1]["g1"] < gc + 1e-6 or (self.flat[-1]["k"] in DEAD_KINDS and self.flat[-1]["g0"] <= gc and not self.flat[-1].get("term"))) and guard < 400:
            if self.flat and self.flat[-1]["k"] == "live" and not self.flat[-1].get("done"):
                break
            if not self._build_next_plan():
                break
            guard += 1

    def _rebuild_after_change(self) -> None:
        """Coach re-sim spliced a new future: keep everything already shown, rebuild from the head."""
        if self.flat and self.flat[-1]["k"] == "live" and not self.flat[-1].get("done"):
            sg = self.flat[-1]
            gnow = self.play.gc if (self.play is not None and self.play.seg is sg) else sg["g0"]
            nxt = next((r for r in self._anchors if float(r["t"]) > gnow + 1e-6), None)
            sg["B"] = nxt
            sg["g1"] = float(nxt["t"]) if nxt is not None else gnow + 99999.0
            ai = -1
            for i, r in enumerate(self._anchors):
                if float(r["t"]) <= gnow + 1e-6:
                    ai = i
            if self.plans:
                self.plans[-1]["ai"] = ai
                self.plans[-1]["B"] = nxt.get("key") if nxt is not None else None
            return
        head_pt = self.t_last if self.t_last is not None else 0.0
        keep_plans: List[Dict[str, Any]] = []
        for pl in self.plans:
            if pl["segs"] and pl["segs"][0]["pt0"] <= head_pt + 1e-9:
                keep_plans.append(pl)
        if not keep_plans:
            self.plans, self.flat, self.flat_g = [], [], []
            return
        cur = keep_plans[-1]
        segs = [s for s in cur["segs"] if s["pt0"] <= head_pt + 1e-9]
        if segs:
            s = segs[-1]
            if s["pt1"] > head_pt and s["k"] not in DEAD_KINDS:
                frac = (head_pt - s["pt0"]) / max(1e-9, s["pt1"] - s["pt0"])
                s["p1"] = (s["p0"][0] + (s["p1"][0] - s["p0"][0]) * frac, s["p0"][1] + (s["p1"][1] - s["p0"][1]) * frac)
                s["g1"] = s["g0"] + (head_pt - s["pt0"])
                s["pt1"] = head_pt
        cur["segs"] = segs
        self.plans = keep_plans
        self.flat = [s for pl in self.plans for s in pl["segs"]]
        self.flat_g = [s["g0"] for s in self.flat]
        g_head = self.flat[-1]["g1"] if self.flat else 0.0
        ai = -1
        for i, r in enumerate(self._anchors):
            if float(r["t"]) <= g_head + 1e-6:
                ai = i
        B = self._next_anchor(ai)
        last = self.flat[-1] if self.flat else None
        state = {"g": g_head, "puck": last["p1"] if last else (0.0, 0.0),
                 "side": last.get("side") if last is not None and last["k"] in ("carry", "pass", "hold") else None,
                 "car": (last.get("rcv") or last.get("car")) if last is not None and last["k"] in ("carry", "pass", "hold") else None}
        rng = self._rng("cont", round(g_head, 2), B.get("key") if B else None)
        if B is not None and state["side"] is None:
            state["side"] = (B.get("att") or {}).get("side")
            state["car"] = self._role_pid(state["side"], g_head, "C") if state["side"] in SIDES else None
        self.plans[-1]["end"] = state
        self.plans[-1]["ai"] = ai
        if B is not None and self.live_mode and B["type"] in ("attempt", "goal", "penalty", "period_end", "final"):
            live = {"k": "live", "g0": g_head, "g1": float(B["t"]), "B": B, "ph": "play", "state": dict(state), "side": state.get("side"),
                    "car": None, "rcv": None, "fo": None, "why": None, "ev": [], "p0": state["puck"], "p1": state["puck"]}
            self._commit_plan(ai, None, B, [], live=live)
        else:
            segs2 = self._buildup(g_head, state, B, rng) if B is not None else [_dead(g_head, 99999.0, state["puck"], state["puck"], term=True)]
            self._commit_plan(ai, None, B, segs2)

    # ───────────────────────────── clocks
    @staticmethod
    def _live_pt(s: Dict[str, Any], gc: float) -> float:
        hg, hp = s.get("hist_g") or [s["g0"]], s.get("hist_pt") or [s["pt0"]]
        if gc >= hg[-1]:
            return hp[-1] + (gc - hg[-1])        # beyond what has been played: extrapolate at 1x
        j = bisect_right(hg, gc) - 1
        j = max(0, min(j, len(hg) - 2))
        g0_, g1_ = hg[j], hg[j + 1]
        f = 0.0 if g1_ - g0_ <= 1e-9 else (gc - g0_) / (g1_ - g0_)
        return hp[j] + (hp[j + 1] - hp[j]) * max(0.0, min(1.0, f))

    def pt_at(self, gc: float, inclusive: bool = False) -> float:
        self._ensure(gc)
        if not self.flat:
            return float(gc)
        i = bisect_right(self.flat_g, gc + 1e-9) - 1
        if i < 0:
            return float(gc)
        if self.flat[i]["k"] == "live":
            return self._live_pt(self.flat[i], gc)
        # walk back to the first segment starting at this gc (strict) or forward to the last dead one (inclusive)
        s = self.flat[i]
        if s["k"] not in DEAD_KINDS and s["g0"] <= gc < s["g1"]:
            return s["pt0"] + (gc - s["g0"])
        j = i
        while j > 0 and abs(self.flat[j - 1]["g0"] - gc) < 1e-9 and self.flat[j - 1]["k"] in DEAD_KINDS:
            j -= 1
        if not inclusive:
            if self.flat[j]["k"] in DEAD_KINDS and abs(self.flat[j]["g0"] - gc) < 1e-9:
                return self.flat[j]["pt0"]
            return s["pt0"] + max(0.0, gc - s["g0"]) if s["k"] not in DEAD_KINDS else s["pt1"]
        k = i
        while k + 1 < len(self.flat) and abs(self.flat[k + 1]["g0"] - gc) < 1e-9 and self.flat[k + 1]["k"] in DEAD_KINDS and not self.flat[k + 1].get("term"):
            k += 1
        s = self.flat[k]
        if s.get("term"):
            return s["pt0"]
        return s["pt1"] if s["k"] in DEAD_KINDS else s["pt0"] + max(0.0, gc - s["g0"])

    def pt_for_raw(self, raw: Dict[str, Any]) -> float:
        typ = raw.get("type")
        gc = float(raw.get("t", 0.0))
        try:
            if typ == "period_start":
                return round(self.pt_at(gc, inclusive=True), 2)
            if typ == "shootout_attempt":
                self._ensure(gc)
                for s in self.flat:
                    if s.get("shot_dead") and s.get("so") is raw.get("so"):
                        return round(s["pt0"], 2)
                return round(self.pt_at(gc), 2)
            if typ in ("final", "period_end"):
                p = self.pt_at(gc)
                return round(p + (12.5 if self._goal_at(gc) else 0.0) + (0.5 if typ == "final" else 0.0), 2)
            return round(self.pt_at(gc), 2)
        except Exception:
            logger.warning("pt_for_raw failed", exc_info=True)
            return round(gc, 2)

    def _goal_at(self, gc: float) -> bool:
        return any(r.get("type") == "goal" and abs(float(r["t"]) - gc) < 0.6 for r in self._atts)

    def gc_for_pt(self, pt: float) -> float:
        """Game clock at presentation time pt (builds plans as needed)."""
        if self.live_mode and self.t_last is not None and pt > self.t_last:
            return self.gc_last + (pt - self.t_last)   # not played yet: 1x estimate
        guard = 0
        while (not self.flat or (self.flat[-1]["pt1"] < pt and not self.flat[-1].get("term"))) and guard < 400:
            if self._refresh_anchors() and self.plans:
                self._rebuild_after_change()
            if not self._build_next_plan():
                break
            guard += 1
        save = self._seg_idx
        try:
            return self.gc_at(pt)
        finally:
            self._seg_idx = save

    def stoppage_after(self, gc: float) -> Optional[str]:
        """Whistle reason of a stoppage that starts at gc (e.g. a goalie freeze right after a save)."""
        self._ensure(gc + 0.6)
        i = bisect_right(self.flat_g, gc - 0.05)
        for s in self.flat[i:i + 6]:
            if s["g0"] > gc + 1.5:
                break
            if s["k"] in DEAD_KINDS and s.get("why"):
                return s["why"][0]
        return None

    def gc_at(self, pt: float) -> float:
        s = self._seg_for_pt(pt)
        if s is None:
            return self.gc_last
        if s["k"] == "live":
            hp, hg = s.get("hist_pt") or [s["pt0"]], s.get("hist_g") or [s["g0"]]
            if pt >= hp[-1]:
                return hg[-1] + (pt - hp[-1])
            j = max(0, min(bisect_right(hp, pt) - 1, len(hp) - 2))
            f = 0.0 if hp[j + 1] - hp[j] <= 1e-9 else (pt - hp[j]) / (hp[j + 1] - hp[j])
            return hg[j] + (hg[j + 1] - hg[j]) * max(0.0, min(1.0, f))
        if s["k"] in DEAD_KINDS:
            return s["g0"]
        return s["g0"] + max(0.0, min(s["g1"] - s["g0"], pt - s["pt0"]))

    def _seg_for_pt(self, pt: float) -> Optional[Dict[str, Any]]:
        if not self.flat:
            return None
        i = min(self._seg_idx, len(self.flat) - 1)
        while i > 0 and self.flat[i]["pt0"] > pt + 1e-9:
            i -= 1
        while i + 1 < len(self.flat) and self.flat[i + 1]["pt0"] <= pt + 1e-9:
            i += 1
        self._seg_idx = i
        return self.flat[i]

    # ───────────────────────────── per-frame puck / targets
    @staticmethod
    def _seg_a(s: Dict[str, Any], pt: float) -> float:
        span = s["pt1"] - s["pt0"]
        a = 0.0 if span <= 1e-9 else max(0.0, min(1.0, (pt - s["pt0"]) / span))
        if s.get("term"):
            a = 0.0
        if s["k"] == "carry":
            a = a * a * (3.0 - 2.0 * a)
        if s["k"] == "dead" and s.get("d", 0) > 3.0 and not s.get("carry_by") and not s.get("shot_dead"):
            a = max(0.0, min(1.0, (pt - s["pt0"] - 1.5) / max(0.5, s["d"] - 2.5)))  # linesman brings it to the dot
        return a

    @staticmethod
    def _puck(s: Dict[str, Any], pt: float, deke: bool = True) -> Tuple[float, float]:
        a = MotionV2._seg_a(s, pt)
        p0, p1 = s["p0"], s["p1"]
        if s.get("poly"):
            x, y = poly_at(s["poly"], a)
        else:
            x, y = p0[0] + (p1[0] - p0[0]) * a, p0[1] + (p1[1] - p0[1]) * a
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        dl = math.hypot(dx, dy)
        if dl > 1e-6:
            nx, ny = -dy / dl, dx / dl
            off = 0.0
            if s.get("arc"):                       # saucer pass: floats on a slight arc
                off += s["arc"] * math.sin(math.pi * a)
            if s.get("jit"):                       # bouncing puck
                off += s["jit"] * math.sin(a * 6.0 * math.pi) * math.sin(math.pi * a)
            if deke and s["k"] == "carry" and s.get("deke"):  # stickhandling: puck moves side to side in front of the body
                ph = (pt - s["pt0"]) / max(0.3, s.get("dper", 0.6))
                off += s["deke"] * math.sin(2.0 * math.pi * ph) * math.sin(math.pi * a)
            if off:
                x, y = clampxy(x + nx * off, y + ny * off)
        return (x, y)

    @staticmethod
    def _puck_z(s: Dict[str, Any], pt: float) -> float:
        """Puck height (ft) for floated passes, chips and shots (presentation hint for the client)."""
        lift = s.get("lift")
        if not lift:
            return 0.0
        a = MotionV2._seg_a(s, pt)
        if s["k"] == "shot":
            return lift * a
        return lift * math.sin(math.pi * a)

    def _faceoff_spots(self, side: str, gc: float, dot: Tuple[float, float], ids: List[str], period: Optional[int] = None) -> Dict[str, Tuple[float, float]]:
        """Real alignments: centres at the dot, wingers on the hash marks (outside the circle), defence at the
        top of the circle in their own end, on the blue line when attacking, back at centre ice; others outside."""
        du, dv = self.rel(side, gc, dot[0], dot[1], period)
        sg = 1.0 if dv >= 0 else -1.0
        own_end, att_end = du < -40.0, du > 40.0
        roles = assign_roles(ids, self._book_pos)
        out_uv: Dict[str, Tuple[float, float]] = {}
        wings = [p for p, r in roles if r in ("LW", "RW", "F")]
        ds = [p for p, r in roles if r in ("LD", "RD", "D")]
        cs = [p for p, r in roles if r == "C"]
        if cs:
            out_uv[cs[0]] = (du - 2.4, dv)
        wing_pts = [(du - 3.2, dv + 15.8), (du - 3.2, dv - 15.8)]
        for p, pt_ in zip(sorted(wings, key=lambda q: 0 if self.roles.get(q) == "LW" else 1), wing_pts):
            out_uv[p] = pt_
        if own_end:
            d_pts = [(du - 13.0, dv - sg * 9.0), (du - 11.0, dv + sg * 11.5)]   # low, between the dot and the net
        elif att_end:
            d_pts = [(BLUE + 4.0, dv * 0.5 + 12.0 * sg), (BLUE + 4.0, dv * 0.5 - 12.0 * sg)]  # on the points
        else:
            d_pts = [(du - 26.0, dv + 12.0), (du - 26.0, dv - 12.0)]
        for p, pt_ in zip(ds, d_pts):
            out_uv[p] = pt_
        extra = [p for p in ids if p not in out_uv]
        for i, p in enumerate(extra):
            out_uv[p] = (du - 18.0, dv + (8.0 if i % 2 == 0 else -8.0) * (1 + i // 2))
        return {p: clampxy(*self.ab(side, gc, u, v, period)) for p, (u, v) in out_uv.items()}

    # ───────────────────────────── formation systems (puck-anchored, role-based, real structures)
    def _assign(self, side: str, gc: float, pids: List[str], spots: List[Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        """Greedy nearest assignment of players to formation spots (keeps continuity, no crossing swaps)."""
        out: Dict[str, Tuple[float, float]] = {}
        free = list(pids)
        for (u, v) in spots:
            if not free:
                break
            ax, ay = self.ab(side, gc, u, v)
            best = min(free, key=lambda q: math.hypot((self.pos.get(q) or [ax, ay])[0] - ax, (self.pos.get(q) or [ax, ay])[1] - ay))
            out[best] = (u, v)
            free.remove(best)
        return out

    def _spots(self, side: str, gc: float, seg: Dict[str, Any], puck: Tuple[float, float], ids: List[str], n_opp: int, opp_ids: List[str]) -> Dict[str, Tuple[float, float]]:
        """Every skater's target from the puck, possession, zone, manpower and role:
        O-zone 2-1-2 cycle, D-zone (D on the carrier / net-front, C low, wingers on the points, collapse to the slot),
        neutral-zone lanes with a trailer vs 1-2-2 forecheck or trap, breakout (wingers on the wall at the hash marks,
        C swinging low, D to D), PP 1-3-1 / umbrella, PK box / triangle, 4v4, 3v3, 6v5."""
        roles = assign_roles(ids, self._book_pos)
        self.roles.update({p: r for p, r in roles})
        D_ids = [p for p, r in roles if r in ("LD", "RD", "D")]
        F_ids = [p for p, r in roles if r not in ("LD", "RD", "D")]
        pu, pv = self.rel(side, gc, *puck)
        s = 1.0 if pv >= 0 else -1.0
        n = len(ids)
        k = seg["k"]
        poss = seg.get("side") if k in ("carry", "pass", "shot", "hold") else None
        if poss is None and k == "loose" and seg.get("side") in SIDES:
            poss = seg["side"]                     # dump / chip: the dumping team chases in
        if poss is None:                           # loose puck: whoever's attacking zone it is in plays it as attack
            poss = side if pu > BLUE else (opp(side) if pu < -BLUE else None)
        car = seg.get("car") if k in ("carry", "hold") else (seg.get("rcv") if k == "pass" else None)
        if side == self.loop.lg.user_side:
            tac = getattr(self.loop, "tactics", None) or {}
        else:
            # the other bench has a system too, fixed for the night, so the two teams don't play the same shape
            h_t = zlib.crc32(f"{getattr(self.loop.lg, 'live_id', '')}:{side}".encode())
            opts = ("passive", "balanced", "aggressive")
            tac = {"forecheck": opts[h_t % 3], "defense": opts[(h_t // 3) % 3]}
        t_def, t_fc = tac.get("defense"), tac.get("forecheck")
        spot: Dict[str, Tuple[float, float]] = {}

        def lim_onside(u: float) -> float:
            return min(u, BLUE - 2.0) if pu < BLUE else u

        def toward(u: float, v: float, tu: float, tv: float, d: float) -> Tuple[float, float]:
            du, dv = tu - u, tv - v
            m = math.hypot(du, dv) or 1.0
            return (u + du / m * d, v + dv / m * d)

        if poss == side:
            if pu >= BLUE or seg.get("delayed") == side:
                if n > n_opp and n >= 4:
                    # power play (or delayed penalty / extra attacker): 1-3-1 or umbrella
                    if tac.get("pp_style") == "passive" or (tac.get("pp_style") is None and (zlib.crc32(side.encode()) % 2 == 0)):
                        pts = [(33.0, 0.2 * pv), (60.0, 28.0 * s), (60.0, -28.0 * s), (67.0, 0.15 * pv), (84.0, -3.0 * s), (74.0, 14.0 * s)]
                    else:
                        pts = [(32.0, 16.0 * s), (32.0, -14.0 * s), (60.0, 28.0 * s), (83.0, -3.0 * s), (66.0, -24.0 * s), (72.0, 6.0 * s)]
                    rest = [p for p in ids if p != car]
                    d_pts, f_pts = pts[:2] if len(D_ids) >= 2 else pts[:1], pts[2:] if len(D_ids) >= 2 else pts[1:]
                    spot.update(self._assign(side, gc, [p for p in D_ids if p != car], d_pts))
                    spot.update(self._assign(side, gc, [p for p in rest if p not in spot], f_pts + d_pts))
                else:
                    # 2-1-2 cycle: F1 on the puck, F2 low support, F3 high slot, D on the points shading to the puck
                    # strong-side D walks the wall with the puck, weak-side D holds the far point: the whole blue line is covered
                    d_pts = [(30.0, max(-30.0, min(30.0, 0.5 * pv + 14.0 * s))), (31.0, max(-28.0, min(28.0, 0.12 * pv - 19.0 * s)))]
                    if n <= 3:   # 3v3: one D high, two forwards
                        d_pts = d_pts[:1]
                    spot.update(self._assign(side, gc, [p for p in D_ids if p != car], d_pts))
                    fw = [p for p in F_ids if p != car and p not in spot]
                    if car is not None and car in D_ids:   # D has it at the point: screen, half wall, weak side
                        f_pts = [(83.0, -4.0 * s), (62.0, 30.0 * s), (70.0, -24.0 * s)]
                    else:
                        low = toward(pu, pv, 85.0, -6.0 * s, 12.0) if pu < 80.0 else (pu - 13.0, pv * 0.45)
                        f_pts = [low, (64.0, 0.1 * pv), (76.0, -25.0 * s)]   # F2 low support, F3 high slot, weak-side winger stays wide
                    spot.update(self._assign(side, gc, fw, f_pts))
                    for p in [q for q in ids if q not in spot and q != car]:
                        spot[p] = (70.0, -20.0 * s)
                    # Off the puck, forwards go to the net instead of holding a spot. On the last touches before
                    # a real shot one screens and one takes the rebound lane; otherwise each forward dives
                    # back-door on his own rhythm, and the higher his offensive IQ the more often he goes.
                    busy = {car, seg.get("rcv"), seg.get("car")}
                    free = [q for q in fw if q not in busy and q in spot]
                    if seg.get("pre_shot") and free:
                        free.sort(key=lambda q: math.hypot(spot[q][0] - 83.0, spot[q][1]))
                        spot[free[0]] = (83.0, -5.0 * s)
                        if len(free) > 1:
                            spot[free[1]] = (72.0, 5.0 * s)
                    else:
                        for q in free:
                            iq = self._traits(q)["oiq"]
                            h_ = zlib.crc32(q.encode())
                            per = 6.0 + (h_ % 5) + 6.0 * (1.0 - iq)
                            if (gc + (h_ % 13)) % per < 2.4:
                                v_ = spot[q][1]
                                spot[q] = (79.0 + 4.0 * iq, (-6.5 if v_ >= 0 else 6.5) if abs(v_) < 14.0 else v_ * 0.42)
            elif pu > -BLUE:
                # neutral zone with the puck: width lanes, a trailer, D trailing the play (nobody offside)
                lanes = [(lim_onside(pu + 10.0), 26.0), (lim_onside(pu + 10.0), -26.0), (pu - 12.0, 0.3 * pv)]
                spot.update(self._assign(side, gc, [p for p in F_ids if p != car], lanes))
                spot.update(self._assign(side, gc, [p for p in D_ids if p != car], [(max(-86.0, pu - 28.0), 16.0), (max(-86.0, pu - 30.0), -16.0)]))
            else:
                # breakout: wingers on the walls at the hash marks, C swinging low, D to D
                f_pts = [(-66.0, 37.0 * s), (-60.0, -36.0 * s), (-78.0, -12.0 * s)]
                spot.update(self._assign(side, gc, [p for p in F_ids if p != car], f_pts))
                spot.update(self._assign(side, gc, [p for p in D_ids if p != car], [(-84.0, -16.0 * s), (-70.0, 18.0 * s)]))
            if seg.get("rush"):
                for pid in list(spot):
                    u, v = spot[pid]
                    spot[pid] = (min(u, pu - 6.0), v)  # support trails the breakaway
            if seg.get("odd"):
                # odd-man rush: the shooter drives the far lane level with the puck, the third forward trails, D join late
                tgt = seg.get("odd")
                k_ = 0
                for pid in list(spot):
                    if pid == tgt:
                        spot[pid] = (lim_onside(min(pu + 4.0, 80.0)), -17.0 * s)
                    elif pid in D_ids:
                        spot[pid] = (pu - 32.0, 14.0 if k_ % 2 == 0 else -14.0)
                        k_ += 1
                    else:
                        spot[pid] = (pu - 15.0, 3.0 * s)
        else:
            # defending
            if pu <= -BLUE:
                if n < n_opp:
                    sh = 0.25 * pv
                    if n >= 4:
                        pts_f, pts_d = [(-58.0, 11.0 + sh), (-58.0, -11.0 + sh)], [(-79.0, 9.5 + 0.2 * pv), (-79.0, -9.5 + 0.2 * pv)]
                    else:
                        pts_f, pts_d = [(-60.0, sh)], [(-79.0, 9.0), (-79.0, -9.0)]
                    spot.update(self._assign(side, gc, D_ids, pts_d))
                    spot.update(self._assign(side, gc, [p for p in ids if p not in spot], pts_f + pts_d))
                else:
                    # strong-side D on the carrier, weak-side D net-front, C low support, wingers on the points
                    # system: aggressive pressures the puck and the points, passive packs the slot and gives up the outside
                    gap_c, gap_w, sag = (3.0, 4.0, 0.08) if t_def == "aggressive" else ((5.5, 13.0, 0.42) if t_def == "passive" else (3.8, 7.0, 0.2))
                    car_ = seg.get("car")
                    # the checker plays the body he expects, not the one he saw: lead the carrier by his own speed
                    cst = self.pos.get(car_) if car_ else None
                    lu, lv = pu, pv
                    if cst is not None and len(cst) >= 4:
                        ax_, ay_ = cst[0] + cst[2] * 0.35, cst[1] + cst[3] * 0.35
                        lu, lv = self.rel(side, gc, ax_, ay_)
                    on_puck = toward(lu, lv, -89.0, 0.0, gap_c)
                    low = pu < -50.0            # puck below the tops of the circles: a defenceman takes it
                    att = [q for q in opp_ids if q in self.pos and self.loop.book.pos(q) != "G" and q != car_]
                    rel_att = {q: self.rel(side, gc, *self.pos[q][:2]) for q in att}
                    # attackers standing in the slot get a body on them, goal side and tight
                    slot_att = sorted([q for q in att if rel_att[q][0] < -60.0 and abs(rel_att[q][1]) < 20.0],
                                      key=lambda q: math.hypot(rel_att[q][0] + 89.0, rel_att[q][1]))
                    def _front(q: str) -> Tuple[float, float]:
                        return toward(rel_att[q][0], rel_att[q][1], -89.0, 0.0, 2.8)
                    fixed: set = set()
                    d_net = _front(slot_att[0]) if slot_att else (-82.0, -5.0 * s)
                    d_pts = [on_puck if low else (-79.0, 7.0 * s), d_net]
                    got = self._assign(side, gc, D_ids, d_pts)
                    spot.update(got)
                    for p_, uv_ in got.items():
                        if slot_att and uv_ == d_net:
                            self._marks[p_] = slot_att[0]
                            fixed.add(p_)
                    pointmen = sorted([q for q in att if q not in slot_att[:2]], key=lambda q: -rel_att[q][0])[:2]  # highest attackers
                    w_pts = []
                    for q in pointmen:
                        qu, qv = rel_att[q]
                        weak = (qv * s) < 0.0
                        # puck low: the weak-side winger sags to the slot instead of standing beside his point man
                        w_pts.append(toward(qu, qv, -89.0, 0.0, max(gap_w, 15.0) if (low and weak) else gap_w))
                    while len(w_pts) < 2:
                        w_pts.append((-42.0, (20.0 if len(w_pts) == 0 else -20.0)))
                    if not low:
                        # puck up high: the nearest winger goes out to the carrier, the defencemen hold the front of the net
                        w_pts[0 if (not pointmen or (rel_att[pointmen[0]][1] * s) >= 0.0) else min(1, len(w_pts) - 1)] = on_puck
                    c_pt = _front(slot_att[1]) if len(slot_att) > 1 else ((-76.0, 0.45 * pv) if low else (-72.0, 4.0 * s))
                    fw = [p for p in F_ids if p not in spot]
                    cpid = next((p for p, r in roles if r == "C" and p in fw), fw[0] if fw else None)
                    if cpid:
                        spot[cpid] = c_pt
                        if len(slot_att) > 1:
                            self._marks[cpid] = slot_att[1]
                            fixed.add(cpid)
                    spot.update(self._assign(side, gc, [p for p in fw if p != cpid], w_pts))
                    for i_, q in enumerate(pointmen):
                        for p_, (u_, v_) in spot.items():
                            if abs(u_ - w_pts[i_][0]) < 0.01 and abs(v_ - w_pts[i_][1]) < 0.01 and (u_, v_) != on_puck:
                                self._marks[p_] = q
                    for pid, (u, v) in list(spot.items()):   # the unit collapses toward the slot
                        if pid not in self._marks and pid not in fixed and (u, v) != on_puck:
                            spot[pid] = (u + (-72.0 - u) * sag, v * (1.0 - sag))
            elif pu < BLUE:
                aggressive = t_fc == "aggressive"
                if aggressive:   # 1-2-2 forecheck high
                    f_pts = [(pu - 5.0, pv), (pu - 16.0, 20.0 * s), (pu - 30.0, -12.0 * s)]
                    d_pts = [(max(-80.0, pu - 45.0), 15.0), (max(-80.0, pu - 45.0), -15.0)]
                else:            # neutral-zone trap
                    f_pts = [(pu - 8.0, pv * 0.8), (min(pu - 22.0, 5.0), 20.0), (min(pu - 22.0, 5.0), -20.0)]
                    d_pts = [(min(pu - 42.0, -28.0), 14.0), (min(pu - 42.0, -28.0), -14.0)]
                spot.update(self._assign(side, gc, F_ids, f_pts))
                spot.update(self._assign(side, gc, D_ids, d_pts))
            else:
                # forecheck in their end: F1 on the puck, F2 cuts off the strong wall, F3 high, D hold the line
                if t_fc == "aggressive":      # 2-1-2: two forwards hunt below the dots, D pinch down the wall
                    f_pts = [(pu - 4.0, pv), (max(pu - 9.0, 52.0), 24.0 * s if abs(pv) < 20.0 else 0.5 * pv), (58.0, 0.15 * pv)]
                    d_line = [(40.0 if abs(pv) > 26.0 else 33.0, 27.0 * s), (31.0, -12.0 * s)]
                elif t_fc == "passive":       # 1-4: one forward shadows, everyone else waits in the neutral zone
                    f_pts = [(pu - 12.0, 0.7 * pv), (22.0, 22.0 * s), (20.0, -20.0 * s)]
                    d_line = [(6.0, 13.0 * s), (4.0, -13.0 * s)]
                else:                         # 1-2-2: F1 on the puck, F2 cuts off the strong wall, F3 high, D hold the line
                    pinch = 36.0 if abs(pv) > 30.0 and t_def != "passive" else 30.0
                    f_pts = [(pu - 5.0, pv), (max(pu - 15.0, 42.0), 30.0 * s), (55.0, 0.2 * pv)]
                    d_line = [(pinch, 26.0 * s), (30.0, -12.0 * s)]
                spot.update(self._assign(side, gc, F_ids, f_pts))
                spot.update(self._assign(side, gc, D_ids, d_line))
            if seg.get("rush") and poss == opp(side):
                for pid in list(spot):
                    spot[pid] = (min(spot[pid][0], pu - 9.0), spot[pid][1] * 0.6)  # defenders trail the breakaway
            if seg.get("odd") and poss == opp(side) and ids:
                # one defender is back, between the puck and the net, taking the pass away; the rest are chasing
                pool = D_ids or list(ids)
                back = min(pool, key=lambda q: self.rel(side, gc, *self.pos[q][:2])[0] if q in self.pos else 0.0)
                k_ = 0
                for pid in ids:
                    if pid == back:
                        spot[pid] = (max(-82.0, pu - 15.0), 0.45 * pv)
                    else:
                        lag = 10.0 + 7.0 * k_ - 7.0 * self._traits(pid)["speed"]   # quick backcheckers close the gap
                        spot[pid] = (pu + lag, 0.6 * pv + (9.0 if k_ % 2 == 0 else -9.0))
                        k_ += 1
        for pid in ids:
            if pid not in spot:
                spot[pid] = (lim_onside(pu - 10.0) if poss == side else pu - 10.0, -10.0 * s)
        out: Dict[str, Tuple[float, float]] = {}
        for pid, (u, v) in spot.items():
            if abs(u - pu) > 35.0:   # the unit flows with the puck: far spots are drawn toward it
                u = pu + math.copysign(35.0 + (abs(u - pu) - 35.0) * 0.6, u - pu)
            du, dv = u - pu, v - pv
            dd = math.hypot(du, dv)
            if dd > 85.0:   # the unit moves as one: nobody's spot is farther than ~85 ft from the puck
                u, v = pu + du / dd * 85.0, pv + dv / dd * 85.0
            x, y = self.ab(side, gc, u, v)
            out[pid] = rink_clamp(x, y, 3.0)
        return out

    # ───────────────────────────── stepping
    def _desired(self, gc: float, seg: Dict[str, Any]) -> Dict[str, List[str]]:
        want = {s: list(v) for s, v in self.lineup_at(gc).items()}
        pl = self.play if (self.live_mode and seg.get("live")) else None
        if pl is not None:
            # nobody changes with the puck on his stick (or a pass on its way to him): the change waits a moment
            # while the play gets rid of it (the play sim moves the puck at once)
            busy = pl.owner if pl.mode == "carry" else ((pl.pass_ or {}).get("rcv") if pl.mode == "pass" else None)
            for side in SIDES:
                if busy and busy in self.onice[side] and busy not in want[side]:
                    cw = self.__dict__.setdefault("_chg_wait", {})
                    t0 = cw.get((side, busy))
                    if t0 is None or gc - t0 > 8.0:
                        t0 = cw[(side, busy)] = gc
                    if gc - t0 < 3.0:
                        want[side] = list(self.onice[side])
                        pl.changing = busy
        for side in SIDES:
            if self.onice[side] and any(a_ <= gc <= b_ for a_, b_ in self._freeze.get(side, [])):
                want[side] = [p for p in self.onice[side] if p in want[side] or True][:len(want[side]) or 5]
            g = self._goalie_wanted(side, gc, seg)
            if g is None and want[side]:
                pool = [p for grp in (self.loop.lines[side].get("F") or []) for p in grp if p not in want[side] and p not in self.boxed]
                if pool:
                    want[side].append(pool[0])
        return want

    def _sync(self, gc: float, seg: Dict[str, Any], dead: bool) -> None:
        if seg.get("to_bench"):
            for side in SIDES:
                for pid in list(self.onice[side]):
                    self._send_off(pid, side)
                self.onice[side] = []
                g = self.goalie_on.get(side)
                if g:
                    self._send_off(g, side)
                    self.goalie_on[side] = None
            return
        if seg.get("ph") == "shootout":
            so = seg.get("so") or {}
            sh = so.get("shooter_id") or seg.get("carry_by")
            for side in SIDES:
                for pid in list(self.onice[side]):
                    if pid != sh:
                        self._send_off(pid, side)
                self.onice[side] = [sh] if sh and self.loop.book.side.get(sh) == side else []
                if sh and self.loop.book.side.get(sh) == side and sh not in self.pos:
                    self.pos[sh] = [0.0, 0.0, 0.0, 0.0]
                gw = self._goalie_wanted(side, gc, None)
                self._sync_goalie(side, gw)
            return
        want = self._desired(gc, seg)
        for side in SIDES:
            cur = self.onice[side]
            leave = [p for p in cur if p not in want[side]]
            enter = [p for p in want[side] if p not in cur]
            if leave or enter:
                if cur and len(enter) >= 3 and gc - self._last_change_ev[side] > 20.0:
                    self._last_change_ev[side] = gc
                    roles = {r: p for p, r in assign_roles(want[side], self._book_pos)}
                    self._events.append({"type": "line_change", "t": gc, "pt": max(0.0, self._pt_now), "meta": {"side": side, "center_id": roles.get("C"), "on_the_fly": not dead,
                                                                                                    "n": len(enter)}})
                for p in leave:
                    self._send_off(p, side)
                for p in enter:
                    self.leaving.pop(p, None)
                    if p not in self.pos:
                        bx, by = BOX_DOOR[side] if p in self.boxed_recent else bench_spot(side, p)
                        self.pos[p] = [bx, by, 0.0, 0.0]
                        if dead:  # full change at a stoppage trickles in over a few seconds
                            self.__dict__.setdefault("_enter_hold", {})[p] = self._pt_now + 0.6 * enter.index(p)
                        elif self.live_mode:
                            # on the fly: he steps on only when the man he replaces is at the bench door (NHL: within 5 ft)
                            self.__dict__.setdefault("_enter_hold", {})[p] = self._pt_now + 0.4 * enter.index(p)
                            i_ = enter.index(p)
                            if i_ < len(leave):
                                self.__dict__.setdefault("_enter_pair", {})[p] = leave[i_]
                self.onice[side] = list(want[side])
            self._sync_goalie(side, self._goalie_wanted(side, gc, seg))

    @property
    def boxed_recent(self) -> set:
        return set(self.boxed)

    def _sync_goalie(self, side: str, want: Optional[str]) -> None:
        cur = self.goalie_on.get(side)
        if cur == want:
            return
        if cur:
            self._send_off(cur, side)
        if want:
            self.leaving.pop(want, None)
            if want not in self.pos:
                bx, by = BENCH_DOOR[side]
                self.pos[want] = [bx, by, 0.0, 0.0]
            self._goalie_returning = getattr(self, "_goalie_returning", set()) | {want}
        self.goalie_on[side] = want

    def _send_off(self, pid: str, side: str) -> None:
        if pid in self.pos and pid not in self.boxed:
            self.leaving[pid] = {"side": side, "since": self._pt_now}

    def _pen_conflict(self, pen: Dict[str, Any]) -> bool:
        """The engine's penalty record names a player who still takes part in attempts during that penalty:
        the engine's on-ice lists win for the picture (nobody is shown in the box for it)."""
        pid = pen.get("player_id")
        a0, a1 = float(pen["start"]), float(pen["start"]) + float(pen["dur"])
        for r in self._atts:
            t = float(r["t"])
            if a0 < t < a1:
                a = r.get("att") or {}
                # only a player who actually acts in the play (shoots, assists, blocks) keeps him out of the box;
                # a bare on-ice listing from the engine is not enough to show a penalized man playing
                if pid in (a.get("shooter_id"), a.get("blocker_id")) or pid in (a.get("assist_ids") or []):
                    return True
        return False

    def _box(self, pid: Optional[str], side: str, gc: float) -> None:
        if not pid or pid in self.boxed:
            return
        if self.loop._cap is not None and any(pen.get("player_id") == pid and pen["start"] - 1.0 <= gc <= pen["start"] + pen["dur"] and self._pen_conflict(pen)
                                              for pen in self.loop._cap.penalties):
            return
        until = gc
        if self.loop._cap is not None:
            for pen in self.loop._cap.penalties:
                if pen.get("player_id") == pid and pen["start"] <= gc + 0.5 < pen["start"] + pen["dur"] + 0.5:
                    until = pen["start"] + pen["dur"]
        self.boxed[pid] = {"side": side, "until": until, "walk": True}
        for s in SIDES:
            if pid in self.onice[s]:
                self.onice[s].remove(pid)
        self.leaving.pop(pid, None)

    def _clamp(self, x: float, y: float) -> Tuple[float, float]:
        return clampxy(x, y)

    def _hint(self, h: Dict[str, Any]) -> None:
        self._hints.append(h)

    def _step(self, pt: float, gc: float, dt: float, seg: Dict[str, Any], puck_override: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        puck = puck_override if puck_override is not None else self._puck(seg, pt)
        dead = seg["k"] in DEAD_KINDS
        self._pt_now = pt
        self._sync(gc, seg, dead)
        if seg.get("to_box") and seg.get("pt0", 0) <= pt:
            self._box(seg["to_box"], seg.get("box_side") or "home", gc)
        for pid in [p for p, b in self.boxed.items() if gc >= b["until"] - 1e-6 and not dead]:
            b = self.boxed.pop(pid)
            self.pos[pid] = [BOX_DOOR[b["side"]][0], BOX_DOOR[b["side"]][1], 0.0, 0.0]
            if pid not in self.onice[b["side"]]:
                self._send_off(pid, b["side"])  # back to the bench unless his unit is on (then he joins the play)
        targets: Dict[str, Tuple[float, float]] = {}
        spec: Dict[str, Tuple[str, Any, float, float]] = {}  # pid -> (kind, key, vmax, amax)
        ph = seg.get("ph")
        fo = seg.get("fo")
        self._marks = {}
        for side in SIDES:
            ids = self.onice[side]
            if not ids:
                continue
            if dead and fo is not None and (seg["k"] == "faceoff" or pt - seg["pt0"] >= (1.6 if seg.get("scrum") else 0.0)):
                per = seg.get("per")
                fs = self._faceoff_spots(side, gc + (0.05 if seg.get("warmup") or seg.get("per") else 0.0), fo, ids, period=per if seg.get("warmup") or seg["k"] == "faceoff" and per else None)
                targets.update(fs)
                spec.update({p_: ("fo", tuple(fo), V_CRUISE, A_NORM) for p_ in fs})
            elif ph == "goal":
                scorer = seg.get("scorer")
                sx, sy = (self.pos.get(scorer) or [puck[0], puck[1]])[:2]
                for i, pid in enumerate(ids):
                    if side == seg.get("scorer_side"):
                        ang = i * 1.3
                        near_ = sorted(ids, key=lambda q: math.hypot((self.pos.get(q) or [sx, sy])[0] - sx, (self.pos.get(q) or [sx, sy])[1] - sy))
                        if pid == scorer or near_.index(pid) < 3 or not self.live_mode:
                            targets[pid] = clampxy(sx - 3.0 * math.cos(ang) * self.dir(side, gc), sy + 3.0 * math.sin(ang))
                        else:   # the last man arrives later and taps gloves from the edge of the group
                            targets[pid] = clampxy(sx - 19.0 * self.dir(side, gc), sy * 0.6 + (5.0 if i % 2 else -5.0))
                    else:  # the other team drifts back toward its own blue line, each on his own path
                        st0 = self.pos.get(pid) or [0.0, 0.0]
                        u0, v0 = self.rel(side, gc, st0[0], st0[1])
                        targets[pid] = clampxy(*self.ab(side, gc, max(-40.0, min(u0, -10.0) - 8.0), v0 * 0.7 + (i - 2) * 3.0))
                    spec[pid] = ("goal", side, 10.0, A_NORM)
            elif dead and seg.get("scrum"):
                X = seg["scrum"]
                gx = NET_X * self.dir(X, gc)
                for i, pid in enumerate(ids[:2]):
                    targets[pid] = clampxy(gx - 9.0 * self.dir(X, gc), (i * 2 - 1) * 4.0)
                    spec[pid] = ("scrum", X, 8.0, A_NORM)
            elif dead:
                for pid in ids:
                    st = self.pos.get(pid)
                    if st:
                        targets[pid] = (st[0], st[1])
                        spec[pid] = ("stop", None, 6.0, A_NORM)
            else:
                sp_ = self._spots(side, gc, seg, seg.get("fpuck") or puck, ids, len(self.onice[opp(side)]), self.onice[opp(side)])
                targets.update(sp_)
                rush = bool(seg.get("rush"))
                # the checker whose job is the puck closes at speed; cruising at him never got there
                presser = None
                if seg.get("side") in SIDES and side != seg.get("side") and sp_:
                    pp_, dd_ = min(((q_, math.hypot(t_[0] - puck[0], t_[1] - puck[1])) for q_, t_ in sp_.items()), key=lambda z: z[1])
                    if dd_ < 9.0:
                        presser = pp_
                for p_, (tx_, ty_) in sp_.items():
                    if p_ == presser:
                        spec[p_] = ("press", seg.get("car"), V_RUSH, A_HARD)
                    elif p_ in self._marks:
                        spec[p_] = ("mark", self._marks[p_], 16.0, A_NORM)
                    else:
                        far = math.hypot(tx_ - puck[0], ty_ - puck[1]) > 45.0
                        if seg.get("live"):   # live play: a skater well off his spot hurries to it (the unit moves as one)
                            st_ = self.pos.get(p_)
                            off_ = math.hypot(tx_ - st_[0], ty_ - st_[1]) if st_ is not None else 0.0
                            spec[p_] = ("spot", None, V_RUSH if (rush or off_ > 22.0) else V_CRUISE, A_HARD if off_ > 22.0 else A_NORM)
                        else:
                            spec[p_] = ("spot", None, V_RUSH if rush else (V_DRIFT if far else V_CRUISE), A_NORM)
        if not dead and seg.get("side") in SIDES:  # the next receiver / carrier gets open early (passes go to a player, not to ice)
            X = seg["side"]
            pu_, _pv = self.rel(X, gc, *puck)
            i0 = self._seg_idx
            if 0 <= i0 < len(self.flat) and self.flat[i0] is seg:
                busy = {seg.get("car"), seg.get("rcv")}
                for s2 in self.flat[i0 + 1:i0 + 3]:
                    if s2["k"] in DEAD_KINDS or s2.get("side") != X:
                        break
                    who, spot_ = (s2.get("rcv"), s2["p1"]) if s2["k"] == "pass" else (s2.get("car"), s2["p0"])
                    if who and who in targets and who not in busy:
                        lu, lv = self.rel(X, gc, *spot_)
                        if pu_ < BLUE:
                            lu = min(lu, BLUE - 2.0)
                        targets[who] = self.ab(X, gc, lu, lv)
                        spec[who] = ("next", id(s2), V_RUSH, A_HARD)
                        busy.add(who)
        pre = seg.get("pre")
        if pre and not dead and pre[0] in targets and seg.get("car") != pre[0]:
            X = pre[2]
            lu, lv = self.rel(X, gc, *pre[1])
            pu_, _pv = self.rel(X, gc, *puck)
            if pu_ < BLUE:
                lu = min(lu, BLUE - 2.0)  # stay onside until the puck is in
            targets[pre[0]] = self.ab(X, gc, lu, lv)
            dpre = math.hypot(targets[pre[0]][0] - self.pos[pre[0]][0], targets[pre[0]][1] - self.pos[pre[0]][1]) if pre[0] in self.pos else 0.0
            spec[pre[0]] = ("pre", tuple(pre[1]), V_SPRINT if dpre > 30.0 else V_RUSH, A_HARD)
        if not dead and seg["k"] in ("carry", "hold") and seg.get("car") in targets:
            lead = 0.2 if seg["k"] == "carry" else 0.0
            targets[seg["car"]] = seg["ctarget"] if seg.get("live") and seg.get("ctarget") else self._puck(seg, pt + lead, deke=False)
            dcar = math.hypot(targets[seg["car"]][0] - self.pos[seg["car"]][0], targets[seg["car"]][1] - self.pos[seg["car"]][1]) if seg["car"] in self.pos else 0.0
            if seg.get("live"):
                spec[seg["car"]] = ("car", seg.get("car"), 20.0 if (seg.get("rush") or seg.get("car_fast")) else 15.0, A_HARD)
            else:
                spec[seg["car"]] = ("car", id(seg), V_SPRINT if (dcar > 12.0 or seg.get("rush")) else 17.0, A_HARD)
        if seg["k"] == "pass" and seg.get("rcv") in targets:
            targets[seg["rcv"]] = seg["p1"]
            spec[seg["rcv"]] = ("rcv", id(seg), V_RUSH, A_HARD)
        if seg.get("offside_pid") in targets:
            X = seg.get("side") or "home"
            u, v = self.rel(X, gc, *puck)
            targets[seg["offside_pid"]] = self.ab(X, gc, 30.0, v * 0.5)
            spec[seg["offside_pid"]] = ("offside", id(seg), V_RUSH, A_HARD)
        if seg["k"] == "loose" and not dead:
            # nobody lets the puck travel alone: the nearest skater of each team goes for it at once (to where it
            # is going on dumps / rims / clears), the dumping team sends a second man
            longball = bool(seg.get("ice") or seg.get("dump") or seg.get("ptype") in ("rim", "chip"))
            aim = seg["p1"] if longball else puck
            for side in SIDES:
                ranked = sorted((math.hypot(q[0] - aim[0], q[1] - aim[1]), pid) for pid in self.onice[side]
                                for q in [self.pos.get(pid)] if q is not None)
                n_ch = 2 if (longball and seg.get("side") == side) else 1
                for _d, pid in ranked[:n_ch]:
                    targets[pid] = aim
                    spec[pid] = ("chase", id(seg), V_SPRINT if (longball or seg.get("live")) else V_RUSH, A_HARD)
        if seg.get("carry_by") and seg["carry_by"] in self.pos:
            targets[seg["carry_by"]] = puck
            spec[seg["carry_by"]] = ("so", id(seg), V_RUSH, A_HARD)
        # leaving players head for the door (bench or box)
        for pid, info in list(self.leaving.items()):
            targets[pid] = bench_spot(info["side"], pid)
            st_l = self.pos.get(pid)
            far_l = st_l is not None and math.hypot(st_l[0] - targets[pid][0], st_l[1] - targets[pid][1]) > 25.0
            spec[pid] = ("door", info["side"], V_RUSH if (far_l and self.live_mode) else V_CRUISE, A_HARD if far_l else A_NORM)   # hurry off: his team plays short until he's there
        pairs_ = self.__dict__.setdefault("_enter_pair", {})
        for pid, until in list(self.__dict__.get("_enter_hold", {}).items()):
            mate = pairs_.get(pid)
            if mate is not None:
                mst = self.pos.get(mate)
                lv_ = self.leaving.get(mate)
                bd_ = BENCH_DOOR[lv_["side"]] if lv_ is not None else None
                at_boards = bd_ is not None and mst is not None and abs(mst[1]) > 31.0 and mst[1] * bd_[1] > 0 and abs(mst[0] - bd_[0]) < 16.0
                if mate not in self.leaving or mst is None or at_boards or (lv_ is not None and
                        math.hypot(mst[0] - bench_spot(lv_["side"], mate)[0], mst[1] - bench_spot(lv_["side"], mate)[1]) < 9.0):
                    pairs_.pop(pid, None)
                    mate = None
            pl_ = self.play
            busy_ = {seg.get("car"), seg.get("rcv")}
            if pl_ is not None:
                busy_ |= {pl_.owner, (pl_.pass_ or {}).get("rcv")}
            if pid in busy_:
                mate = None
                until = pt   # the play needs him now: no one with the puck waits on the bench
            if (pt >= until and mate is None) or pid not in self.pos:
                self._enter_hold.pop(pid, None)
                pairs_.pop(pid, None)
            elif pid in targets:
                targets[pid] = (self.pos[pid][0], self.pos[pid][1])   # waits at the boards for his turn
                spec[pid] = ("wait", None, 2.0, A_NORM)
        for pid, b in self.boxed.items():
            if b.get("walk"):
                targets[pid] = BOX_DOOR[b["side"]]
                spec[pid] = ("box", b["side"], V_CRUISE, A_NORM)
        for pid, xy in (seg.get("force") or {}).items():
            if pid in targets:
                targets[pid] = xy
                spec[pid] = ("force", None, V_RUSH, A_HARD)
        targets = self._hold_targets(targets, spec, seg, pt, dt, dead)
        before = {pid: (st[0], st[1]) for pid, st in self.pos.items()}
        onside: Dict[str, bool] = {}
        X_on = seg.get("side") if (not dead and seg["k"] in ("carry", "pass", "hold") and seg.get("side") in SIDES) else None
        if X_on is not None:
            dX = self.dir(X_on, gc)
            if puck[0] * dX < BLUE:
                for pid in self.onice[X_on]:
                    st = self.pos.get(pid)
                    if st is not None:
                        onside[pid] = st[0] * dX <= BLUE - 0.2
        for pid, (tx, ty) in targets.items():
            st = self.pos.get(pid)
            if st is None:
                continue
            kind, _key, vmax, amax = spec.get(pid, ("spot", None, V_CRUISE, A_NORM))
            tr = self._traits(pid)
            vmax, amax = vmax * tr["v"], amax * tr["a"]
            dist = math.hypot(tx - st[0], ty - st[1])
            spd = math.hypot(st[2], st[3])
            if kind == "spot" and dist < 6.0 and seg.get("live") and not dead:
                # nobody parks on his spot: he keeps a slow loop around it (open ice, a stick on the ice, a look over
                # the shoulder), on his own rhythm, so off-puck play is always curving and never frozen
                rad = 3.0 + 2.5 * tr["samp"] / 0.08
                w_ = 0.55 + 0.35 * (tr["sph"] % 1.0)
                tx, ty = clampxy(tx + rad * math.cos(w_ * pt + tr["sph"]), ty + rad * 0.8 * math.sin(w_ * pt * 1.3 + tr["sph"]))
                dist = math.hypot(tx - st[0], ty - st[1])
            if (kind == "spot" and dist < 12.0) or (kind == "mark" and dist < 4.0):   # glide into a support spot; a marker stays tight to his man
                vmax, amax = min(vmax, 10.0), amax * 0.55
            if spd > 8.0:                                  # stride rhythm: speed swells and eases with each push
                vmax *= 1.0 + tr["samp"] * math.sin(6.2832 * pt / tr["stride"] + tr["sph"])
            if kind in ("spot", "mark") and (pt + tr["sco"]) % tr["sc"] < 0.35:
                vmax *= 0.7                                # shoulder check
            if self._book_pos(pid) == "D" and spd > 4.0 and dist > 3.0:
                if (st[2] * (tx - st[0]) + st[3] * (ty - st[1])) < -0.2 * spd * dist:
                    vmax = min(vmax, 5.0)                  # pivot: slow through the turn, then re-accelerate
            self._steer(st, tx, ty, dt, min(vmax, V_SPRINT), amax, tr["turn"])
        self._goalies(gc, puck, dt, seg)
        if not dead and not seg.get("live") and seg["k"] in ("carry", "hold") and seg.get("car") in self.pos:
            st = self.pos[seg["car"]]
            X = seg.get("side") or "home"
            pb_ = self._puck(seg, pt, deke=False)
            want = clampxy(pb_[0] - 1.2 * self.dir(X, gc), pb_[1])
            dd = math.hypot(st[0] - want[0], st[1] - want[1])
            if 1e-6 < dd <= 8.0:  # the carrier keeps the puck on his stick
                vx, vy = (want[0] - st[0]) / dt, (want[1] - st[1]) / dt
                vm = math.hypot(vx, vy)
                cap = V_SPRINT
                if vm > cap:
                    vx, vy = vx / vm * cap, vy / vm * cap
                st[0], st[1] = st[0] + vx * dt, st[1] + vy * dt
                st[2], st[3] = 0.6 * st[2] + 0.4 * vx, 0.6 * st[3] + 0.4 * vy
        if seg["k"] == "shot" and seg.get("car") in self.pos and pt - seg["pt0"] < 0.3:
            st = self.pos[seg["car"]]
            if math.hypot(st[0] - seg["p0"][0], st[1] - seg["p0"][1]) <= 6.0:  # the shooter releases from shot_loc
                st[0], st[1] = seg["p0"]
        fixed = [self.pos[g] for g in self.goalie_on.values() if g and g in self.pos]
        allp = [p for side in SIDES for p in self.onice[side]] + list(self.leaving)
        self._separate(gc, [p for p in dict.fromkeys(allp) if p in self.pos and p not in self.boxed], fixed,
                       team_gap=(0.0 if dead and fo is not None else 6.0), puck=puck, min_sep=3.4)   # opponents contest body to body; teammates keep their spacing
        lim = (V_SPRINT + 0.5) * dt  # hard speed cap whatever moved them this step
        gset = {g for g in self.goalie_on.values() if g}
        for pid, st in self.pos.items():
            b0 = before.get(pid)
            if b0 is None or pid in gset:
                continue
            dx, dy = st[0] - b0[0], st[1] - b0[1]
            dd = math.hypot(dx, dy)
            if dd > lim:
                st[0], st[1] = b0[0] + dx / dd * lim, b0[1] + dy / dd * lim
        self._separate(gc, [p for p in dict.fromkeys(allp) if p in self.pos and p not in self.boxed], fixed, min_sep=3.15)  # hard floor
        if onside:  # hold the line: nobody who is onside crosses the offensive blue line ahead of the puck
            dX = self.dir(X_on, gc)
            for pid, was_on in onside.items():
                st = self.pos.get(pid)
                if was_on and st is not None and st[0] * dX > BLUE - 0.2 and pid not in (seg.get("car"), seg.get("offside_pid")):
                    st[0] = (BLUE - 0.2) * dX
                    if st[2] * dX > 0:
                        st[2] = 0.0
                elif was_on and st is not None and pid == seg.get("car") and pid != seg.get("offside_pid"):
                    lim_u = max(BLUE - 0.2, puck[0] * dX + 0.8)   # the carrier crosses with the puck, not ahead of it
                    if seg.get("hold_line"):
                        lim_u = BLUE - 2.5                         # a teammate is still in the zone: he waits for the tag-up
                    if st[0] * dX > lim_u:
                        st[0] = lim_u * dX
                        if st[2] * dX > 0:
                            st[2] = 0.0
        on_now = set(self.onice["home"]) | set(self.onice["away"]) | {g for g in self.goalie_on.values() if g}
        for pid in list(self.leaving):
            if pid in on_now:  # back on the ice before reaching the bench
                self.leaving.pop(pid, None)
                continue
            st = self.pos.get(pid)
            door = bench_spot(self.leaving[pid]["side"], pid)
            bd = BENCH_DOOR[self.leaving[pid]["side"]]
            over = self.live_mode and abs(st[1]) > 35.0 and st[1] * bd[1] > 0 and abs(st[0] - bd[0]) < 13.0 if st is not None else False  # hops over the boards
            if st is None or over or math.hypot(st[0] - door[0], st[1] - door[1]) < 3.0 or pt - self.leaving[pid]["since"] > 20.0:
                self.leaving.pop(pid, None)
                self.pos.pop(pid, None)
        for side in SIDES:  # safety: anyone on the ice has a body
            for pid in self.onice[side]:
                if pid not in self.pos and pid not in self.boxed:
                    bx, by = BENCH_DOOR[side]
                    self.pos[pid] = [bx, by, 0.0, 0.0]
        for pid, b in self.boxed.items():
            st = self.pos.get(pid)
            if b.get("walk") and st is not None and math.hypot(st[0] - BOX_DOOR[b["side"]][0], st[1] - BOX_DOOR[b["side"]][1]) < 2.5:
                b["walk"] = False
                self.pos.pop(pid, None)
            elif b.get("walk") and st is None:
                b["walk"] = False
        return puck

    def _hold_targets(self, targets: Dict[str, Tuple[float, float]], spec: Dict[str, Tuple[str, Any, float, float]],
                      seg: Dict[str, Any], pt: float, dt: float, dead: bool) -> Dict[str, Tuple[float, float]]:
        """Fewer target changes: a formation spot is kept ~1.2 s unless the puck changes hands. Moving references
        (carrier, receiver, the man being checked, the puck on a loose chase) are followed continuously but count as
        one decision. Also counts retargets (decision changes) for the diagnostics."""
        holder = (seg.get("side"), seg.get("car") if seg["k"] in ("carry", "hold") else seg.get("rcv")) if not dead else ("dead", seg.get("why"))
        out: Dict[str, Tuple[float, float]] = {}
        st_ = self.stats
        for pid, tgt in targets.items():
            kind, key, _v, _a = spec.get(pid, ("spot", None, V_CRUISE, A_NORM))
            h = self._hold.get(pid)
            if kind == "spot":
                same = h is not None and h[0] == "spot"
                if same and ((h[4] == holder and pt - h[3] < RETARGET_S) or math.hypot(tgt[0] - h[2][0], tgt[1] - h[2][1]) <= 6.0):
                    out[pid] = h[2]  # keep the current destination (small drifts are ignored)
                    if pid not in self.leaving:
                        st_["player_sec"] += dt
                    continue
                self._hold[pid] = ("spot", None, tgt, pt, holder)
                if pid not in self.leaving:
                    st_["retargets"] += 1
            else:
                if h is None or h[0] != kind or h[1] != key:
                    if pid not in self.leaving:
                        st_["retargets"] += 1
                self._hold[pid] = (kind, key, tgt, pt, holder)
            out[pid] = tgt
            if pid not in self.leaving:
                st_["player_sec"] += dt
        return out

    def _steer(self, st: List[float], tx: float, ty: float, dt: float, vmax: float, amax: float = A_NORM, turn: float = 1.0) -> None:
        """Arrive-steering: forward acceleration limit, a separate (personal) turning limit so players carve
        arcs with crossovers instead of zig-zagging, and harder braking for stops."""
        dx, dy = tx - st[0], ty - st[1]
        dist = math.hypot(dx, dy)
        want = min(vmax, math.sqrt(2.0 * A_BRAKE * 0.6 * max(0.0, dist - 0.5)) * 0.9)
        vx, vy = ((dx / dist * want, dy / dist * want) if dist > 1e-6 else (0.0, 0.0))
        ax, ay = vx - st[2], vy - st[3]
        spd = math.hypot(st[2], st[3])
        if spd > 1.0:
            tx_, ty_ = st[2] / spd, st[3] / spd
            at = ax * tx_ + ay * ty_
            nx_, ny_ = ax - at * tx_, ay - at * ty_
            ahead = (vx * tx_ + vy * ty_) > 0.0       # hard stop only when braking toward a spot ahead, not to reverse
            at = max(-(A_BRAKE if ahead else amax) * dt, min(amax * dt, at))
            nm = math.hypot(nx_, ny_)
            nlim = amax * turn * dt
            if nm > nlim:
                nx_, ny_ = nx_ / nm * nlim, ny_ / nm * nlim
            ax, ay = tx_ * at + nx_, ty_ * at + ny_
        else:
            am = math.hypot(ax, ay)
            lim = amax * dt
            if am > lim:
                ax, ay = ax / am * lim, ay / am * lim
        st[2] += ax
        st[3] += ay
        sp = math.hypot(st[2], st[3])
        if sp > V_SPRINT:
            st[2], st[3] = st[2] / sp * V_SPRINT, st[3] / sp * V_SPRINT
        st[0], st[1] = clampxy(st[0] + st[2] * dt, st[1] + st[3] * dt)

    def _separate(self, gc: float, ids: List[str], fixed: Optional[List[List[float]]] = None, min_sep: float = MIN_SEP,
                  team_gap: float = 0.0, puck: Optional[Tuple[float, float]] = None) -> None:
        """Min 3 ft between bodies; skaters stay out of both creases (goalies hold the crease)."""
        pts = [(pid, self.pos[pid]) for pid in ids if pid in self.pos]
        for _ in range(6):
            moved = False
            for i in range(len(pts)):
                a = pts[i][1]
                for j in range(i + 1, len(pts)):
                    b = pts[j][1]
                    dx, dy = b[0] - a[0], b[1] - a[1]
                    d = math.hypot(dx, dy)
                    need = min_sep
                    if team_gap > min_sep and self.loop.book.side.get(pts[i][0]) == self.loop.book.side.get(pts[j][0]):
                        battling = puck is not None and math.hypot(a[0] - puck[0], a[1] - puck[1]) < 10.0 and math.hypot(b[0] - puck[0], b[1] - puck[1]) < 10.0
                        if not battling:
                            need = team_gap  # teammates keep their spacing (no clumps, no single file)
                    if d < need:
                        if d < 1e-6:
                            dx, dy, d = 1.0, 0.0, 1.0
                        push = (need - d) / 2.0 + 0.01
                        if need > min_sep:
                            push = min(push, 0.6)  # spacing is a soft push, not a shove
                        nx, ny = dx / d, dy / d
                        ax0, ay0 = a[0], a[1]
                        a[0], a[1] = clampxy(a[0] - nx * push, a[1] - ny * push)
                        short = push - math.hypot(a[0] - ax0, a[1] - ay0)  # a is against the boards: b moves more
                        b[0], b[1] = clampxy(b[0] + nx * (push + max(0.0, short)), b[1] + ny * (push + max(0.0, short)))
                        moved = True
            for _pid, st in pts:
                for fx in fixed or []:
                    dx, dy = st[0] - fx[0], st[1] - fx[1]
                    d = math.hypot(dx, dy)
                    if d < min_sep:
                        if d < 1e-6:
                            dx, dy, d = (1.0 if fx[0] < 0 else -1.0), 0.0, 1.0
                        st[0], st[1] = clampxy(fx[0] + dx / d * (min_sep + 0.02), fx[1] + dy / d * (min_sep + 0.02))
                        moved = True
                for gx in (-NET_X, NET_X):
                    dx, dy = st[0] - gx, st[1]
                    d = math.hypot(dx, dy)
                    if d < CREASE_R:
                        if d < 1e-6:
                            dx, dy, d = (-1.0 if gx > 0 else 1.0), 0.0, 1.0
                        st[0], st[1] = clampxy(gx + dx / d * CREASE_R, dy / d * CREASE_R)
                        moved = True
            if not moved:
                break

    def _goalies(self, gc: float, puck: Tuple[float, float], dt: float, seg: Dict[str, Any]) -> None:
        ret = getattr(self, "_goalie_returning", set())
        for side in SIDES:
            g = self.goalie_on.get(side)
            if not g:
                continue
            st = self.pos.get(g)
            # the opening draw of a period sits at exactly the old period's end time: use the segment's own period
            od = self.dir(side, gc, seg.get("per") if (seg.get("warmup") or seg.get("k") == "faceoff") and seg.get("per") else None)
            nx = -NET_X * od
            if st is None:
                st = [nx, 0.0, 0.0, 0.0]
                self.pos[g] = st
            if g not in ret and math.hypot(st[0] - nx, st[1]) > 12.0 and not seg.get("to_bench") and seg.get("ph") != "intermission":
                ret.add(g)   # out of position (period start, after a change): straight back to the crease first
                self._goalie_returning = ret
            if g in ret:
                if math.hypot(st[0] - nx, st[1]) < 5.0:
                    ret.discard(g)
                else:
                    self._steer(st, nx + od * 1.5, 0.0, dt, 18.0, A_HARD)
                    continue
            vx, vy = puck[0] - nx, puck[1]
            if vx * od <= 0:
                tx, ty = nx + od * 1.0, max(-3.0, min(3.0, puck[1]))
            else:
                m = math.hypot(vx, vy) or 1.0
                tx, ty = nx + vx / m * 4.0, vy / m * 4.0
            dx, dy = tx - st[0], ty - st[1]
            d = math.hypot(dx, dy)
            step = min(d, V_GOALIE * dt)
            if d > 1e-6:
                st[0] += dx / d * step
                st[1] += dy / d * step
            st[2] = st[3] = 0.0

    # ───────────────────────────── public
    # ───────────────────────────── live play (forward simulation + director + time warp)
    def _live_substep(self, sg: Dict[str, Any], pp: float) -> None:
        try:
            from services.live_play import PlaySim  # type: ignore
        except ImportError:  # local tests
            from live_play import PlaySim  # type: ignore
        if self.play is None or self.play.seg is not sg:
            self.play = PlaySim(self, sg, sg.get("state") or {}, sg["pt0"])
            sg["hist_pt"], sg["hist_g"] = [sg["pt0"]], [sg["g0"]]
            sg["_last_pp"] = sg["pt0"]
        dt = pp - sg.get("_last_pp", sg["pt0"])
        if dt <= 1e-6:
            return
        sg["_last_pp"] = pp
        self._cur_seg = sg
        if sg.get("B") is None:   # nothing left (end of the timeline): idle play
            self.play.gc += dt
        else:
            self.play.substep(pp, dt)
        sg["hist_pt"].append(pp)
        sg["hist_g"].append(self.play.gc)
        if self.play.reset_req:
            self._live_reset(sg, pp, self.play.reset_req)
        elif self.play.done:
            self._live_finish(sg, pp)

    def _close_live(self, sg: Dict[str, Any], pp: float) -> None:
        sg["done"] = True
        sg["g1"] = self.play.gc
        sg["pt1"] = pp

    def _live_finish(self, sg: Dict[str, Any], pp: float) -> None:
        pl = self.play
        self._close_live(sg, pp)
        B = sg["B"]
        end = {"g": pl.gc, "puck": (pl.puck[0], pl.puck[1]), "side": pl.side if pl.mode == "carry" else None,
               "car": pl.owner if pl.mode == "carry" else None}
        if pl.release is not None and B is not None:
            X = pl._X() or "home"
            S = pl._shooter()
            loc = pl.release["loc"]
            B["shot_loc"] = {"x": round(loc[0], 1), "y": round(loc[1], 1), "est": True, "from_motion": True}
            rng = self._rng("shot", B.get("key"))
            shot, defl = self._choose_shot(B, X, loc, rng, 99.0)
            last_pass = next((m for m in reversed(pl.moves) if m["kind"] == "pass" and m.get("rcv") == S), None)
            recv = next((m for m in reversed(pl.moves) if m["kind"] == "recv" and m.get("pid") == S), None)
            fresh = recv is not None and pp - recv["pt"] < 0.8
            lu = self.rel(X, pl.gc, loc[0], loc[1])[0]
            if fresh and shot in ("snap", "wrist") and lu > 55.0 and rng.random() < 0.6:
                shot = "one_timer"
            if shot in ("one_timer", "tip") and not fresh:
                shot = "snap" if shot == "one_timer" else "wrist"
            if shot == "tip" and last_pass is not None and self.rel(X, pl.gc, *last_pass["from"])[0] > 45.0:
                shot = "wrist"
            self._shot[B["key"]] = {"shot_type": shot, "deflected": bool(defl)}
            hist_ = [h for h in pl.s_hist if pp - h[0] >= 2.5] or pl.s_hist[:1]
            bu_p0 = (hist_[-1][1], hist_[-1][2]) if hist_ else loc
            mv = last_pass if (last_pass is not None and pp - last_pass["pt"] < 4.0) else None
            start = next((m for m in reversed(pl.moves) if pp - m["pt"] <= 4.0), None)
            self._bu[B["key"]] = {"shooter": S, "kind": "pass" if mv else "carry", "passer": mv["pid"] if mv else None,
                                  "from_xy": mv["from"] if mv else None, "rush": bool(pl.rush), "ptype": mv.get("ptype") if mv else None,
                                  "shot_type": shot, "seg": {"p1": loc, "p0": bu_p0}, "start_seg": {"pt0": (start or {"pt": pp})["pt"]},
                                  "pt": (start or {"pt": pp})["pt"], "forced": bool(pl.release.get("forced"))}
            end = {"g": pl.gc, "puck": loc, "side": X, "car": S}
        if self.plans:
            self.plans[-1]["end"] = end
        self.play = None
        self._build_next_plan()

    def _append_segs(self, segs: List[Dict[str, Any]]) -> None:
        pt = self.flat[-1]["pt1"] if self.flat else 0.0
        g = self.flat[-1]["g1"] if self.flat else 0.0
        for s_ in segs:
            if s_["k"] in DEAD_KINDS:
                s_["g0"] = s_["g1"] = g
                s_["pt0"], s_["pt1"] = pt, pt + s_["d"]
            else:
                s_["g0"], s_["g1"] = g, g + max(0.0, s_["g1"] - s_["g0"])
                s_["pt0"], s_["pt1"] = pt, pt + (s_["g1"] - s_["g0"])
            pt, g = s_["pt1"], s_["g1"]
            if self.plans:
                self.plans[-1]["segs"].append(s_)
            self.flat.append(s_)
            self.flat_g.append(s_["g0"])

    def _live_reset(self, sg: Dict[str, Any], pp: float, reason: str) -> None:
        """A natural stoppage inside a live interval (icing, or the defending team puts the puck out of play);
        the game clock stops, there is a faceoff, and the same anchor is played toward again."""
        pl = self.play
        self._close_live(sg, pp)
        B = sg["B"]
        X = pl._X()
        g = pl.gc
        rng = self._rng("reset", B.get("key") if B else None, round(g, 1))
        puck = (pl.puck[0], pl.puck[1])
        if reason == "icing":
            Y = (pl.loose_info or {}).get("side") or pl.side or "home"
            self._play_stats["icings"] += 1
            sgn = 1.0 if self.rel(Y, g, *puck)[1] >= 0 else -1.0
            dot = self.ab(Y, g, -ZONE_DOT_U, DOT_V * sgn)
            wh = _dead(g, rng.uniform(8.0, 10.5), puck, dot, why=("icing", {"side": Y}), fo=dot, reason="icing")
            wh["ev"].append({"type": "stoppage", "meta": {"reason": "icing", "side": Y}})
            fos, st = self._faceoff(g, dot, X, rng, "icing", offender=Y, force=X if (X and pl.urgency() > 0.3) else None)
        else:
            Xs = X or "home"
            D = opp(Xs)
            self._resets += 1
            self._play_stats["resets"] += 1
            sgn = 1.0 if self.rel(Xs, g, *puck)[1] >= 0 else -1.0
            dot = self.ab(Xs, g, ZONE_DOT_U, DOT_V * sgn)
            wh = _dead(g, rng.uniform(7.0, 9.0), puck, dot, why=("out", {}), fo=dot, reason="out")
            wh["ev"].append({"type": "stoppage", "meta": {"reason": "out", "side": D}})
            fos, st = self._faceoff(g, dot, Xs, rng, "out", offender=Xs, force=Xs)
        self._append_segs([wh] + fos)
        self.play = None
        st = dict(st)
        live = {"k": "live", "g0": self.flat[-1]["g1"], "g1": float(B["t"]) if B else g + 99999.0, "B": B, "ph": "play", "state": st,
                "side": st.get("side"), "car": None, "rcv": None, "fo": None, "why": None, "ev": [], "p0": st.get("puck"), "p1": st.get("puck")}
        live["pt0"], live["pt1"] = self.flat[-1]["pt1"], 1e18
        if self.plans:
            self.plans[-1]["segs"].append(live)
        self.flat.append(live)
        self.flat_g.append(live["g0"])

    def realized_gc(self) -> float:
        return float(self.gc_last)

    def frame_at(self, pt: float) -> Optional[Dict[str, Any]]:
        seg = self._seg_for_pt(pt)
        if seg is None:
            return None
        for sub in (0.25, 0.5):
            pp = pt - 0.5 + sub
            sg = self._seg_for_pt(pp) or seg
            if sg["k"] == "live" and not sg.get("done"):
                self._live_substep(sg, pp)
                continue
            gg = sg["g0"] if sg["k"] in DEAD_KINDS else sg["g0"] + max(0.0, min(sg["g1"] - sg["g0"], pp - sg["pt0"]))
            if sg["k"] == "live":
                gg = self.gc_at(pp)
            self._cur_seg = sg
            self._emit_seg_events(sg)
            self._step(pp, gg, 0.25, sg)
        seg = self._seg_for_pt(pt) or seg
        self._cur_seg = seg
        live_now = seg["k"] == "live"
        if live_now:
            gc = self.gc_at(pt)
            pl = self.play if (self.play is not None and self.play.seg is seg) else None
            puck = tuple(pl.puck) if pl is not None else tuple(seg.get("p1") or (0.0, 0.0))
        else:
            gc = seg["g0"] if seg["k"] in DEAD_KINDS else seg["g0"] + max(0.0, min(seg["g1"] - seg["g0"], pt - seg["pt0"]))
            puck = self._puck(seg, pt)
        if seg.get("why") is not None and seg["k"] in DEAD_KINDS:
            self.last_whistle = seg["why"]
        c = seg.get("car") if seg["k"] in ("carry", "hold") or seg.get("carry_by") else None
        if seg.get("carry_by"):
            c = seg["carry_by"]
        side = seg.get("side")
        if live_now:
            pl = self.play if (self.play is not None and self.play.seg is seg) else None
            c = pl.owner if (pl is not None and pl.mode == "carry") else None
            side = pl.side if pl is not None else None
            seg_k = "carry" if c else ("pass" if pl is not None and pl.mode == "pass" else "loose")
        else:
            seg_k = seg["k"]
        if c is not None and c not in self.pos:
            c = None
        pos_side = side if seg_k in ("carry", "pass", "shot", "hold") and side in SIDES else (self.loop.book.side.get(c) if c else None)
        players, lv, sp = [], [], []
        gset = {g for g in self.goalie_on.values() if g}
        spd_prev = self.__dict__.setdefault("_spd_prev", {})

        def drawn(pid: str, st: List[float]) -> List[Any]:
            """Body position as drawn: physics position plus a small personal wobble (stick checks, weight shifts),
            bigger when standing still so nobody is ever frozen. Hard decelerations are flagged for a snow puff."""
            tr = self._traits(pid)
            spd = math.hypot(st[2], st[3])
            prev = spd_prev.get(pid)
            last_sp = self.__dict__.setdefault("_sp_last", {})
            if prev is not None and prev > 14.0 and spd < prev - 8.0 and pid not in gset and pt - last_sp.get(pid, -99.0) > 6.0:
                sp.append(pid)
                last_sp[pid] = pt
            spd_prev[pid] = spd
            amp = (0.22 if pid in gset else tr["wob"]) * (1.0 if spd < 2.5 else 0.35)
            wx = amp * math.sin(6.2832 * 0.37 * tr["wf"] * pt + tr["w1"])
            wy = amp * math.sin(6.2832 * 0.29 * tr["wf"] * pt + tr["w2"])
            return [pid, round(st[0] + wx, 1), round(st[1] + wy, 1)]

        held = self.__dict__.get("_enter_hold", {})
        for s in SIDES:
            for pid in list(self.onice[s]) + ([self.goalie_on[s]] if self.goalie_on.get(s) else []):
                st = self.pos.get(pid)
                waiting = self.live_mode and (held.get(pid, -1.0) > pt or (pid in held and pid in self.__dict__.get("_enter_pair", {})))
                if st is not None and not waiting:   # still on the bench, waiting his turn (or for the man he replaces)
                    players.append(drawn(pid, st))
        for pid in self.leaving:
            st = self.pos.get(pid)
            if st is not None:
                players.append(drawn(pid, st))
                lv.append(pid)
        bx = []
        for pid, b in self.boxed.items():
            st = self.pos.get(pid)
            if b.get("walk") and st is not None:
                players.append([pid, round(st[0], 1), round(st[1], 1)])
                lv.append(pid)
            else:
                bx.append([pid, BOX_SEAT[b["side"]][0] + (len(bx) * 2.5), BOX_SEAT[b["side"]][1]])
        hx = puck[0] * self.dir("home", gc)
        f: Dict[str, Any] = {"t": round(pt, 2), "pt": round(pt, 2), "gc": round(gc, 2), "ph": seg.get("ph") or "play",
                             "pk": [round(puck[0], 1), round(puck[1], 1)], "c": c, "pos": pos_side, "p": players,
                             "z": "h" if hx < -BLUE else ("a" if hx > BLUE else "n"), "mp": self._mp_code(gc)}
        if lv:
            f["lv"] = lv
        if sp:
            f["sp"] = sp
        pzv = self._puck_z(seg, pt)
        if pzv > 0.3:
            f["pz"] = round(pzv, 1)
        if seg.get("ptype") and seg["k"] in ("pass", "shot", "loose"):
            f["pp"] = seg["ptype"]
        if bx:
            f["bx"] = bx
        if seg.get("fo_drop") and abs(pt - seg["pt0"]) < 0.26:
            f["fo"] = seg["fo_drop"]
        if seg["k"] in DEAD_KINDS and seg.get("why") and abs(pt - seg["pt0"]) < 0.26:
            f["wh"] = seg["why"][0]
        if seg.get("delayed"):
            f["dp"] = opp(seg["delayed"])
        self.last_frame = f
        try:
            self._presence(gc)
        except Exception:
            logger.debug("presence accounting failed", exc_info=True)
        self.gc_last = gc
        return f

    def _mp_code(self, gc: float) -> str:
        m = self._manpower(gc)
        sh = m.get("short")
        if gc >= 3600.0 and (self.loop._result or {}).get("overtime") and not self.loop.lg.is_playoff:
            return "3v3"
        if sh:
            return f"PP:{opp(sh)}"
        return "EV" if m["home"] == 5 else f"{m['home']}v{m['away']}"

    def _emit_seg_events(self, seg: Dict[str, Any]) -> None:
        sid = id(seg)
        if sid in self._emitted_ev:
            return
        self._emitted_ev.add(sid)
        for e in seg.get("ev") or []:
            ev = dict(e)
            ev.setdefault("t", seg["g0"])
            ev["pt"] = round(seg["pt0"], 2)
            if ev["type"] == "faceoff":
                w = ev["meta"].get("winner")
                if w in SIDES:
                    self.fo_tally[w][0] += 1
                    self.fo_tally["home"][1] += 1
                    self.fo_tally["away"][1] += 1
                try:   # zone starts per line (the unit on the ice for the draw)
                    P = self._pres()
                    dot = ev["meta"].get("dot") or [0.0, 0.0]
                    for sd in SIDES:
                        du = self.rel(sd, seg["g0"] + 0.01, dot[0], dot[1])[0]
                        zone = "oz" if du > 40.0 else ("dz" if du < -40.0 else None)
                        if zone:
                            ev["meta"].setdefault("zone_for", {})[sd] = zone
                            for key in [k_ for k_ in (P["cur"][sd].get("F"), P["cur"][sd].get("D")) if k_]:
                                P["starts"].setdefault(f"{sd}:{key}", {"oz": 0, "dz": 0})[zone] += 1
                except Exception:
                    logger.debug("zone start accounting failed", exc_info=True)
            self._events.append(ev)
        if seg["k"] in ("carry", "pass") and seg.get("side") in SIDES:
            self._hints.append({"pt": round(seg["pt0"], 2), "t": seg["g0"], "kind": seg["k"], "side": seg["side"], "pid": seg.get("car"),
                                "rcv": seg.get("rcv"), "from": seg["p0"], "to": seg["p1"], "rush": bool(seg.get("rush")),
                                "entry": seg.get("ev_end_hint") == "entry", "dump": False, "ptype": seg.get("ptype")})
        elif seg.get("dump") and seg.get("side") in SIDES:
            self._hints.append({"pt": round(seg["pt0"], 2), "t": seg["g0"], "kind": "dump", "side": seg["side"], "pid": None, "from": seg["p0"], "to": seg["p1"],
                                "rush": False, "entry": False, "dump": True, "ptype": seg.get("ptype")})

    # ───────────────────────────── on-ice presence (TOI / shifts / lines / matchups from the motion track)
    def _pres(self) -> Dict[str, Any]:
        P = self.__dict__.get("pres")
        if P is None:
            P = {"gc": None, "p": {}, "line": {}, "match": {}, "cur": {"home": {}, "away": {}}, "starts": {},
                 "units": {"home": {"PP": [], "PK": []}, "away": {"PP": [], "PK": []}}}
            self.pres = P
        return P

    def _presence(self, gc: float) -> None:
        P = self._pres()
        prev = P["gc"]
        dg = gc - prev if prev is not None and 0.0 < gc - prev <= 2.0 else 0.0
        P["gc"] = gc
        mp = self._manpower(gc)
        for side in SIDES:
            on = list(self.onice[side])
            g = self.goalie_on.get(side)
            allon = set(on) | ({g} if g else set())
            for pid in allon:
                r = P["p"].setdefault(pid, {"side": side, "toi": 0.0, "shifts": 0, "on": False, "since": gc, "last_off": None})
                if not r["on"]:
                    r["on"], r["since"] = True, gc
                    r["shifts"] += 1
                r["toi"] += dg
            for pid, r in P["p"].items():
                if r["side"] == side and r["on"] and pid not in allon:
                    r["on"], r["last_off"] = False, gc
            lines = self.loop.lines.get(side) or {}
            onset = set(on)
            fk = next((f"F{j + 1}" for j, ln in enumerate(lines.get("F") or []) if ln and sum(1 for q in ln if q in onset) >= 2), None)
            dk = next((f"D{j + 1}" for j, pr in enumerate(lines.get("D") or []) if pr and sum(1 for q in pr if q in onset) >= 2), None)
            uk = None
            kind = "PK" if mp.get("short") == side else ("PP" if mp.get("short") == opp(side) else None)
            if kind and onset:
                reg = P["units"][side][kind]
                idx = next((i for i, u in enumerate(reg) if len(u & onset) >= 3), None)
                if idx is None:
                    if len(reg) < 2:
                        reg.append(frozenset(onset))
                        idx = len(reg) - 1
                    else:
                        idx = max(range(len(reg)), key=lambda i: len(reg[i] & onset))
                uk = f"{kind}{idx + 1}"
            cur = {"F": fk, "D": dk, "U": uk}
            old = P["cur"][side]
            for slot, key in cur.items():
                if not key:
                    continue
                L = P["line"].setdefault(f"{side}:{key}", {"toi": 0.0, "shifts": 0, "since": gc, "last_shift": 0.0, "ids": []})
                if old.get(slot) != key:
                    L["shifts"] += 1
                    L["since"] = gc
                L["toi"] += dg
                L["last_shift"] = gc - L["since"]
                ids_ = [q for q in onset if (slot != "F" or self._book_pos(q) != "D") and (slot != "D" or self._book_pos(q) == "D")] if slot != "U" else sorted(onset)
                if ids_:
                    L["ids"] = ids_
            P["cur"][side] = cur
        hk, ak = P["cur"]["home"].get("F") or P["cur"]["home"].get("U"), P["cur"]["away"].get("F") or P["cur"]["away"].get("U")
        if hk and ak and dg:
            P["match"][(hk, ak)] = P["match"].get((hk, ak), 0.0) + dg

    def presence(self) -> Dict[str, Any]:
        return self._pres()

    def drain_events(self) -> List[Dict[str, Any]]:
        out, self._events = self._events, []
        return out

    def drain_hints(self) -> List[Dict[str, Any]]:
        out, self._hints = self._hints, []
        return out

    def buildup(self, key: str) -> Optional[Dict[str, Any]]:
        return self._bu.get(key)

    def generate_to(self, gc_end: float, *, skip: bool = False) -> Dict[str, Any]:
        self._ensure(gc_end)
        pt = 0.0 if self.t_last is None else self.t_last + 0.5
        frames: List[Dict[str, Any]] = []
        if skip and self.live_mode:
            # sim-to-end: the rest of the game is not animated; hand back the last picture at the final clock
            f = dict(self.last_frame) if self.last_frame else None
            frames = []
            if f is not None:
                pt = (self.t_last or 0.0) + 0.5
                f.update(t=round(pt, 2), pt=round(pt, 2), gc=round(float(gc_end), 2), ph="stoppage")
                frames.append(f)
                self.t_last = pt
            return {"from_t": frames[0]["pt"] if frames else 0.0, "to_t": frames[-1]["pt"] if frames else 0.0, "from_pt": frames[0]["pt"] if frames else 0.0,
                    "to_pt": frames[-1]["pt"] if frames else 0.0, "step": 0.5, "frames": frames, "authority": "model_estimated", "skipped": True,
                    "clock": "frames carry pt (presentation, contiguous) and gc (game clock, frozen in stoppages)"}
        if skip:
            end_pt = self.pt_at(gc_end, inclusive=True)
            if end_pt - pt > 30.0:
                pt = math.floor((end_pt - 10.0) * 2.0) / 2.0
                self.pos, self.leaving, self.boxed = {}, {}, {}
                self.onice = {"home": [], "away": []}
                self.goalie_on = {"home": None, "away": None}
                self._emitted_ev |= {id(s) for s in self.flat if s["pt0"] < pt}
        guard = 0
        int_per = self.loop._int_period if self.loop.intermission else None
        if int_per is None and self.live_mode and hasattr(self.loop, "_pending_intermission_t"):
            pend = self.loop._pending_intermission_t()
            if pend is not None and pend <= gc_end + 1e-9:
                int_per = self.loop.clock_at(max(0.0, pend - 1e-6))[0]
        while guard < 4000:
            seg = self._seg_for_pt(pt)
            if seg is None:
                break
            if seg["k"] == "live" and not seg.get("done"):
                gnow = self.play.gc if (self.play is not None and self.play.seg is seg) else seg["g0"]
                if gnow + 0.375 > gc_end + 1e-9 and float(seg["g1"]) > gc_end + 1e-9:
                    break     # stepping could run the clock past what may be shown (unless the anchor itself is reachable)
                gc = gnow
            else:
                gc = seg["g0"] if seg["k"] in DEAD_KINDS else seg["g0"] + max(0.0, min(seg["g1"] - seg["g0"], pt - seg["pt0"]))
            if seg["k"] not in DEAD_KINDS and seg["k"] != "live" and gc > gc_end + 1e-9:
                # A play that straddles the horn (a shot in the last half-second, its loose puck running past 20:00)
                # must not hold the clock short of it forever: the horn is reached and the period ends.
                if seg["g0"] <= gc_end + 1e-9 and self.gc_last < gc_end - 1e-9 and any(
                        r.get("type") in ("period_end", "final") and abs(float(r["t"]) - gc_end) < 1e-6 for r in self._anchors):
                    self.gc_last = float(gc_end)
                break
            if seg["k"] in DEAD_KINDS and seg["g0"] > gc_end + 1e-9:
                break
            if seg.get("term") and pt - seg["pt0"] > TERMINAL_PT:
                break
            if int_per is not None and seg.get("per") is not None and int(seg["per"]) > int(int_per) and seg.get("ph") != "intermission":
                break
            if seg["pt1"] <= pt and seg is self.flat[-1] and not seg.get("term"):
                self._ensure(seg["g1"] + 0.5)
                if self.flat[-1] is seg:
                    break
                continue
            try:
                f = self.frame_at(pt)
            except Exception:
                logger.warning("motion v2 frame failed at pt=%.1f", pt, exc_info=True)
                f = dict(self.last_frame, t=round(pt, 2), pt=round(pt, 2)) if self.last_frame else None
            if f is not None:
                frames.append(f)
            self.t_last = pt
            pt += 0.5
            guard += 1
        start = frames[0]["pt"] if frames else (self.t_last if self.t_last is not None else 0.0)
        end = frames[-1]["pt"] if frames else start
        return {"from_t": start, "to_t": end, "from_pt": start, "to_pt": end, "step": 0.5, "frames": frames,
                "authority": "model_estimated", "skipped": bool(skip), "clock": "frames carry pt (presentation, contiguous) and gc (game clock, frozen in stoppages)"}

    def situation_key(self) -> Optional[Tuple[str, Optional[str]]]:
        f = self.last_frame
        if not f:
            return None
        if f["ph"] in ("stoppage", "faceoff", "offside", "icing", "penalty", "goal", "intermission"):
            return ("stop", None)
        if f["ph"] != "play" or f.get("pos") not in SIDES or not f.get("pk"):
            return None
        side = f["pos"]
        u, _v = self.rel(side, f["gc"], f["pk"][0], f["pk"][1])
        return ("oz" if u > BLUE else ("dz" if u < -BLUE else "nz"), side)

    def zone(self) -> Optional[str]:
        f = self.last_frame
        if not f or not f.get("pk"):
            return None
        return {"h": "home_d", "a": "away_d", "n": "neutral"}.get(f.get("z"), None)
