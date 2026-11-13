"""Forward play simulation for the SimTracker (round 9): the puck is carried and passed by players who decide,
and a director bends the play (and a variable-rate game clock) toward the next real engine event.

* The carrier skates lanes and reads pressure from the formation/steering system; he carries, protects, passes to an
  open teammate (lane, distance, pressure), dumps, or (shooting team only, director-gated) shoots.
* Passes and loose pucks are physical: receivers meet passes, defenders intercept or poke the puck loose (derived
  possession changes). Everybody reacts to the puck every substep.
* Director: knows the next anchor (team, shooter, chance type, outcome, or a whistle at t). It tilts possession toward
  the shooting team, steers its attack into the chance region and releases the shot when the play gets there.
* Time warp: presentation time pt drives the visuals; the game clock gc runs at 0.75-1.35x (briefly up to 1.6x)
  so that it reads exactly the engine t at the release. Too early: the shooter protects. Too late: the director pushes
  (direct lanes, quick passes) and, if needed, a natural reset stoppage (puck out of play by the defending team).
Presentation only: nothing here feeds the engine.
"""
from __future__ import annotations

import math
import random
import zlib
from typing import Any, Dict, List, Optional, Tuple

try:
    from services.live_motion import bank_poly, poly_at, poly_len  # type: ignore
except Exception:  # pragma: no cover - local tests import the module directly
    from live_motion import bank_poly, poly_at, poly_len  # type: ignore

SIDES = ("home", "away")
BLUE = 25.0
NET_X = 89.0
R_MIN, R_MAX, R_BURST = 0.75, 1.35, 1.6
R_READY = 2.4   # clock rate while the engine's shooter is set in his spot waiting for the scheduled shot
ANTIC = 1.0   # seconds of look-ahead for the units' positioning
PASS_V, DUMP_V, CARRY_V, RUSH_V = 66.0, 58.0, 14.0, 19.0

# chance-type regions in the shooting team's attack coordinates: (u_min, u_max, |v| min, |v| max)
REGION: Dict[str, Tuple[float, float, float, float]] = {
    "slot": (64.0, 80.0, 0.0, 14.0), "high_danger": (76.0, 84.0, 0.0, 9.0), "rebound": (78.0, 85.0, 0.0, 8.0),
    "breakaway": (68.0, 82.0, 0.0, 10.0), "point": (27.0, 40.0, 0.0, 34.0), "perimeter": (44.0, 78.0, 18.0, 37.0),
}
REGION_DEFAULT = (60.0, 80.0, 0.0, 20.0)


def _opp(s: Optional[str]) -> str:
    return "away" if s == "home" else "home"


def region_of(chance: Optional[str]) -> Tuple[float, float, float, float]:
    return REGION.get(str(chance or ""), REGION_DEFAULT)


def in_region(u: float, v: float, reg: Tuple[float, float, float, float], slack: float = 0.0) -> bool:
    if not (reg[0] - slack <= u <= reg[1] + slack and reg[2] - slack <= abs(v) <= reg[3] + slack):
        return False
    return math.hypot(NET_X - u, v) >= 7.5 or u < NET_X - 7.5   # never inside the crease


def project(u: float, v: float, reg: Tuple[float, float, float, float]) -> Tuple[float, float]:
    pu = max(reg[0] + 1.0, min(reg[1] - 1.0, u))
    av = max(reg[2] + 1.0, min(reg[3] - 1.0, abs(v)))
    pv = math.copysign(av, v if v != 0 else 1.0)
    d = math.hypot(NET_X - pu, pv)
    if d < 7.6:
        pu = NET_X - 7.6 * (NET_X - pu) / max(d, 1e-6) if d > 1e-6 else NET_X - 7.6
    return pu, pv


class PlaySim:
    """One live interval (from the end of an aftermath to the next anchor). Owned by MotionV2."""

    def __init__(self, mt: Any, seg: Dict[str, Any], state: Dict[str, Any], pt0: float) -> None:
        self.mt = mt
        self.seg = seg
        self.gc = float(seg["g0"])
        self.rate = 1.0
        p = state.get("puck") or (0.0, 0.0)
        self.puck = [float(p[0]), float(p[1])]
        self.pv = [float(v) for v in (state.get("pv") or (0.0, 0.0))]
        self.owner: Optional[str] = state.get("car") if state.get("car") in mt.pos else None
        self.mode = "carry" if self.owner else "loose"
        self.side = mt.loop.book.side.get(self.owner) if self.owner else state.get("side")
        self.pass_: Optional[Dict[str, Any]] = None
        self.loose_info: Dict[str, Any] = {"src": "start"}
        self.last_touch: Optional[str] = self.owner
        self.next_decide = pt0 + 0.5
        self.held_since = pt0
        self.ctarget: Optional[Tuple[float, float]] = None
        self.moves: List[Dict[str, Any]] = []        # carry / pass history (commentary build-ups, hints)
        self.rng = random.Random(zlib.crc32(f"{mt.loop.lg.seed}|play|{seg.get('B', {}).get('key')}|{round(self.gc, 2)}".encode()))
        self.done = False
        self.release: Optional[Dict[str, Any]] = None
        self.reset_req: Optional[str] = None
        self.rush = False
        # The engine rolled a rush chance: when the shooting team wins the puck outside the zone the play
        # goes straight up ice as an odd-man rush (set in virtual_seg once the final approach starts).
        self.odd = "RUSH" in str(((seg.get("B") or {}).get("att") or {}).get("chance_raw") or "").upper()
        self._odd_on = False
        self.ozpat: Dict[str, Tuple[int, float, float, float]] = {}
        self.lane = {}
        self.stats = {"forced": 0, "out_region": 0}
        self.changing: Optional[str] = None
        self.s_hist: List[Tuple[float, float, float]] = []

    # ── helpers
    @property
    def B(self) -> Optional[Dict[str, Any]]:
        return self.seg.get("B")

    def _X(self) -> Optional[str]:
        B = self.B
        if B is None or B.get("type") not in ("attempt", "goal"):
            return None
        a = B.get("att") or {}
        return a.get("side") if a.get("side") in SIDES else None

    def _shooter(self) -> Optional[str]:
        B = self.B
        return (B.get("att") or {}).get("shooter_id") if B is not None and B.get("type") in ("attempt", "goal") else None

    def rel(self, side: str, x: float, y: float) -> Tuple[float, float]:
        return self.mt.rel(side, self.gc, x, y)

    def ab(self, side: str, u: float, v: float) -> Tuple[float, float]:
        return self.mt.ab(side, self.gc, u, v)

    def _team(self, pid: Optional[str]) -> Optional[str]:
        return self.mt.loop.book.side.get(pid) if pid else None

    def _skaters(self, side: str) -> List[str]:
        # a man still waiting at the bench for his turn is not on the ice yet: he can't touch the puck or check
        hold = self.mt.__dict__.get("_enter_hold") or {}
        pair = self.mt.__dict__.get("_enter_pair") or {}
        now = getattr(self.mt, "_pt_now", 0.0)
        return [p for p in self.mt.onice.get(side) or [] if p in self.mt.pos
                and not (p in hold and (hold[p] > now or p in pair) and p != self.owner)]

    def _nearest_opp_dist(self, side: str, x: float, y: float) -> float:
        best = 99.0
        for q in self._skaters(_opp(side)):
            st = self.mt.pos[q]
            best = min(best, math.hypot(st[0] - x, st[1] - y))
        return best

    def _seg_dist(self, ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        return math.hypot(ax + t * dx - px, ay + t * dy - py)

    # ── director
    def _tr(self, pid: Optional[str]) -> Dict[str, float]:
        try:
            return self.mt._traits(pid) if pid else {}
        except Exception:
            return {}

    @staticmethod
    def _danger(reg: Tuple[float, float, float, float]) -> bool:
        """Slot / net-front regions: the places a player shouldn't hold the puck without shooting."""
        return reg[0] >= 58.0 and reg[3] <= 22.0

    _OZ_PATTERNS = (
        ((82.0, 30.0), (62.0, 33.0), (74.0, 18.0)),                       # work the wall low to high
        ((84.0, 27.0), (94.0, 7.0), (94.0, -7.0), (82.0, -27.0)),         # behind the net, out the far side
        ((62.0, 32.0), (46.0, 12.0), (47.0, -14.0), (66.0, -31.0)),       # up the wall and across the top
        ((80.0, 29.0), (66.0, 20.0), (58.0, 31.0)),                       # curl off the half wall and back
    )

    def _oz_point(self, own: str, lane: float) -> Tuple[float, float]:
        """Where the carrier takes the puck in the zone: a pattern picked per carrier, on his own tempo."""
        cur = self.ozpat.get(own)
        tr = self._tr(own)
        if cur is None or self.gc - cur[1] >= cur[2] * len(self._OZ_PATTERNS[cur[0]]):
            hands, oiq = float(tr.get("hands", 0.5)), float(tr.get("oiq", 0.5))
            w = [1.0, 0.5 + 0.8 * hands, 0.5 + 0.8 * oiq, 0.7]
            if cur is not None:
                w[cur[0]] *= 0.25                                         # rarely the same look twice in a row
            r, acc, pick = self.rng.random() * sum(w), 0.0, 0
            for i, wt in enumerate(w):
                acc += wt
                if r <= acc:
                    pick = i
                    break
            sgn = math.copysign(1.0, lane)
            if cur is not None and self._OZ_PATTERNS[cur[0]][-1][1] < 0:
                sgn = -sgn                                                # he finished on the far side: start from there
                self.lane[own] = -lane
            cur = (pick, self.gc, self.rng.uniform(2.6, 4.6) * (1.15 - 0.3 * float(tr.get("speed", 0.5))), sgn)
            self.ozpat[own] = cur
        pts = self._OZ_PATTERNS[cur[0]]
        u, v = pts[min(len(pts) - 1, int((self.gc - cur[1]) / cur[2]))]
        return u, v * cur[3]

    def _eta(self) -> float:
        """Estimated presentation seconds until the shooter can release from the chance region."""
        X, S = self._X(), self._shooter()
        B = self.B
        if X is None:
            return 0.0
        reg = region_of((B.get("att") or {}).get("chance"))
        pu, pv = self.rel(X, *self.puck)
        tu, tv = project(pu, pv, reg)
        d_puck = math.hypot(tu - pu, tv - pv)
        st = self.mt.pos.get(S) if S else None
        d_s = 0.0
        if st is not None:
            su, sv = self.rel(X, st[0], st[1])
            ru, rv = project(su, sv, reg)
            d_s = math.hypot(ru - su, rv - sv)
        v_att = 20.0   # carries and passes mixed when the attack is on
        if self.mode == "carry" and self.side == X:
            if self.owner == S:
                return d_puck / 16.0
            return max(d_puck / v_att, d_s / 20.0) + 0.7
        if self.mode == "pass" and self.side == X:
            return max(d_s / 20.0, 0.5) + 0.5
        return 1.6 + d_puck / v_att + 0.7

    def _final_approach(self) -> bool:
        """The director starts the last move toward the chance when the remaining clock matches the distance."""
        B = self.B
        if B is None:
            return False
        R = float(B["t"]) - self.gc
        return R <= self._eta() * 1.8 + 6.0   # start a little earlier: fewer last-instant snaps of the puck to the shooter

    def ready(self) -> bool:
        X, S = self._X(), self._shooter()
        if X is None or self.mode != "carry" or self.owner != S:
            return False
        reg = region_of((self.B.get("att") or {}).get("chance"))
        st = self.mt.pos.get(S)
        if st is None:
            return False
        u, v = self.rel(X, st[0], st[1])       # where the shooter stands (the puck is on his stick)
        return in_region(u, v, reg, slack=2.5)

    def urgency(self) -> float:
        """0 (lots of time) .. 1 (must happen now) for the shooting team's attack."""
        B = self.B
        if B is None:
            return 0.0
        R = float(B["t"]) - self.gc
        need = self._eta() * R_MIN
        return max(0.0, min(1.0, (need - R + 12.0) / 10.0))

    def _rate_step(self, dt: float, R: float) -> float:
        eta = self._eta()
        rdy = False
        if self.B is not None and self.B.get("type") in ("attempt", "goal"):
            if self.ready():
                rdy = True
                # shooter set in his spot: the clock hurries to his shot instead of him standing there with it
                want = min(R_READY, max(1.0, R / 0.55))
            elif eta + 0.5 > R:
                want = R / max(0.4, eta + 0.5)            # late: slow the clock, the director pushes
            else:
                want = 1.0
        else:
            want = 1.0
        top = R_READY if rdy else R_MAX
        want = max(R_MIN, min(top, want))
        lim = (2.4 if rdy or self.rate > R_MAX else 0.5) * dt   # smooth: at most 0.5 per second change (faster for a set shooter)
        self.rate += max(-lim, min(lim, want - self.rate))
        self.rate = max(R_MIN, min(top if rdy else max(R_MAX, self.rate), self.rate))
        return self.rate

    # ── decisions
    def _pass_candidates(self) -> List[Tuple[float, str]]:
        T = self.side
        if T not in SIDES or not self.owner:
            return []
        X, S = self._X(), self._shooter()
        pu, pv = self.rel(T, *self.puck)
        out = []
        for q in self._skaters(T):
            if q == self.owner:
                continue
            st = self.mt.pos[q]
            qu, qv = self.rel(T, st[0], st[1])
            if pu < BLUE and qu > BLUE - 1.0:
                continue   # no pass to a player who would be offside
            if pu >= BLUE and qu < BLUE + 1.0:
                continue   # the puck stays in the zone (a pass out would make everyone tag up)
            d = math.hypot(st[0] - self.puck[0], st[1] - self.puck[1])
            if d < 5.0 or d > 95.0:
                continue
            openness = min(20.0, self._nearest_opp_dist(T, st[0], st[1]))
            lane_block = min((self._seg_dist(self.puck[0], self.puck[1], st[0], st[1], self.mt.pos[o][0], self.mt.pos[o][1])
                              for o in self._skaters(_opp(T))), default=30.0)
            fa = T == X and self._final_approach()
            back = qu - pu
            score = openness + 0.25 * back - 0.08 * max(0.0, d - 45.0) - (14.0 if lane_block < 4.0 else 0.0)
            if back < -5.0:
                score -= 15.0 if fa else 5.0      # the attack goes forward when the director wants the chance
            if pu >= BLUE and lane_block >= 6.0:
                score += min(6.0, 0.14 * abs(qv - pv))          # an open cross-ice look is worth taking
            score += 6.0 * (float(self._tr(q).get("oiq", 0.5)) - 0.5)   # the puck finds the better players
            score += 4.0 * (float(self._tr(self.owner).get("hands", 0.5)) - 0.5) if lane_block < 8.0 else 0.0  # skilled passers thread it
            if T == X and q == S:
                B = self.B
                reg = region_of((B.get("att") or {}).get("chance"))
                if in_region(qu, qv, reg, slack=6.0):
                    score += 10.0 + 25.0 * self.urgency()
            out.append((score, q))
        out.sort(reverse=True)
        return out

    def _carrier_target(self) -> Tuple[float, float]:
        tx, ty = self._carrier_target_raw()
        T, own = self.side, self.owner
        st = self.mt.pos.get(own) if own else None
        if T in SIDES and st is not None and self.rel(T, st[0], st[1])[0] >= BLUE:
            tu, tv = self.rel(T, tx, ty)
            if tu < BLUE + 3.0:
                tx, ty = self.ab(T, BLUE + 3.0, tv)   # in the zone he keeps it in
        return tx, ty

    def _carrier_target_raw(self) -> Tuple[float, float]:
        T, own = self.side, self.owner
        X, S = self._X(), self._shooter()
        st = self.mt.pos.get(own)
        if T not in SIDES or st is None:
            return (self.puck[0], self.puck[1])
        cu, cv = self.rel(T, st[0], st[1])
        lane = self.lane.setdefault(own, (1.0 if cv >= 0 else -1.0) * self.rng.uniform(10.0, 28.0))
        if T == X and own == S and self._final_approach():
            reg = region_of((self.B.get("att") or {}).get("chance"))
            R = float(self.B["t"]) - self.gc
            tu0, tv0 = project(cu, cv, reg)
            d_reg = math.hypot(tu0 - cu, tv0 - cv)
            if self._danger(reg) and not self._odd_on and cu > BLUE + 4.0 and R > d_reg / 15.0 + (1.3 if in_region(cu, cv, reg) else 0.9):
                # too early to shoot: he works the half wall just outside the scoring area and attacks it on time,
                # instead of standing in the slot with the puck
                side_ = math.copysign(1.0, cv if abs(cv) > 3.0 else lane)
                ang = (self.gc * 0.7) % 6.283
                return self.ab(T, max(56.0, min(76.0, tu0 - 3.0)) + 3.0 * math.cos(ang), side_ * (27.0 + 3.0 * math.sin(ang)))
            if in_region(cu, cv, reg) and R > 0.4:
                # early: protect inside the region (small arcs), keep the puck away from the checker
                ang = (self.gc * 0.9) % 6.283
                tu, tv = project(cu + 2.5 * math.cos(ang), cv + 2.5 * math.sin(ang), reg)
                return self.ab(T, tu, tv)
            if cu > BLUE - 4.0 or self.urgency() > 0.3:
                tu, tv = project(cu, cv, reg)
                if cu < BLUE:
                    tu, tv = (BLUE - 3.0 if self._teammate_offside(T) else BLUE + 6.0), max(-30.0, min(30.0, tv))
                return self.ab(T, tu, tv)
        if T == X and own != S and S and cu >= BLUE + 2.0 and self._final_approach():
            # the feeder for the engine's chance works the wall or the goal line, not the slot: an open man
            # in the slot passing it away is exactly what doesn't happen in a real game
            side_ = math.copysign(1.0, cv if abs(cv) > 3.0 else lane)
            if cu > 70.0:
                return self.ab(T, 82.0, side_ * 27.0)
            return self.ab(T, max(48.0, min(64.0, cu)), side_ * 31.0)
        if T == X and self._final_approach() and cu < BLUE + 2.0:   # direct line to the zone
            if self._teammate_offside(T):
                return self.ab(T, min(cu + 12.0, BLUE - 3.0), lane * 0.6)
            return self.ab(T, BLUE + 10.0, max(-24.0, min(24.0, lane * 0.6)))
        if cu < -BLUE:            # breakout: up the lane
            return self.ab(T, cu + 30.0, lane)
        if cu < BLUE + 2.0:       # through the neutral zone toward the entry
            if self._teammate_offside(T):
                return self.ab(T, min(cu + 10.0, BLUE - 3.0), lane)   # wait for the tag-up
            return self.ab(T, min(cu + 26.0, BLUE + 12.0), lane)
        # in the zone: wall cycle, behind the net, across the top or a curl, on the carrier's own tempo
        return self.ab(T, *self._oz_point(own, lane))

    def _do_pass(self, rcv: str, pt: float, ptype: Optional[str] = None) -> None:
        st = self.mt.pos[rcv]
        flight0 = math.hypot(st[0] - self.puck[0], st[1] - self.puck[1]) / PASS_V
        tx, ty = st[0] + st[2] * flight0, st[1] + st[3] * flight0
        tx, ty = self.mt._clamp(tx, ty)
        T = self.side
        if self.rel(T, *self.puck)[0] < BLUE and self._teammate_offside(T):
            tu_, tv_ = self.rel(T, tx, ty)
            if tu_ > BLUE - 2.0:
                tx, ty = self.ab(T, BLUE - 2.0, tv_)     # nothing goes in while a teammate is still offside
        d = math.hypot(tx - self.puck[0], ty - self.puck[1])
        du = (self.rel(T, tx, ty)[0] - self.rel(T, *self.puck)[0])
        if ptype is None:
            if d > 75.0:
                ptype = "stretch"
            elif d < 18.0 and du < -4.0:
                ptype = "drop"
            elif abs(self.puck[1]) > 20.0 and abs(ty) > 20.0 and self.puck[1] * ty > 0 and d > 25.0 and self.rng.random() < 0.35:
                ptype = "bank"
            elif d > 30.0 and self.rng.random() < 0.2:
                ptype = "saucer"
            else:
                ptype = "line"
        poly = None
        if ptype == "bank":
            poly = bank_poly((self.puck[0], self.puck[1]), (tx, ty))
            d = poly_len(poly)
        speed = PASS_V * (0.8 if ptype == "saucer" else 1.0)
        dur = max(0.3, d / speed)
        self.pass_ = {"from": (self.puck[0], self.puck[1]), "to": (tx, ty), "rcv": rcv, "passer": self.owner, "pt0": pt, "dur": dur,
                      "poly": poly, "ptype": ptype, "arc": (1.0 if self.rng.random() < 0.5 else -1.0) * min(2.0, 0.04 * d) if ptype == "saucer" else 0.0}
        self.moves.append({"kind": "pass", "pid": self.owner, "rcv": rcv, "pt": pt, "gc": self.gc, "ptype": ptype, "from": self.pass_["from"], "to": (tx, ty)})
        self.mt._hint({"pt": round(pt, 2), "t": self.gc, "kind": "pass", "side": T, "pid": self.owner, "rcv": rcv, "from": self.pass_["from"],
                       "to": (tx, ty), "rush": self.rush, "entry": False, "dump": False, "ptype": ptype})
        self.mode, self.owner = "pass", None

    def _do_dump(self, pt: float, clear: bool = False, chip: bool = False) -> None:
        T = self.side
        sgn = 1.0 if self.rng.random() < 0.5 else -1.0
        if chip:   # a zone clear up the near wall that dies in the neutral zone (never an icing)
            pv0 = self.rel(T, *self.puck)[1]
            sgn = 1.0 if pv0 >= 0.0 else -1.0
        tu, tv = (8.0, sgn * 39.0) if chip else ((96.0, sgn * 20.0) if clear else (90.0, sgn * 30.0))
        tx, ty = self.mt._clamp(*self.ab(T, tu, tv))
        d = math.hypot(tx - self.puck[0], ty - self.puck[1])
        v = 44.0 if chip else (62.0 if clear else DUMP_V)
        self.pv = [(tx - self.puck[0]) / d * v, (ty - self.puck[1]) / d * v] if d > 1e-6 else [0.0, 0.0]
        pu = self.rel(T, *self.puck)[0]
        self.loose_info = {"side": T, "dump": True, "from_u": pu, "icing_ok": pu < 0.0 and not self._short(T) and not chip, "dest": (tx, ty), "pt0": pt}
        self.moves.append({"kind": "dump", "pid": self.owner, "pt": pt, "gc": self.gc})
        self.mt._hint({"pt": round(pt, 2), "t": self.gc, "kind": "dump", "side": T, "pid": None, "from": (self.puck[0], self.puck[1]), "to": (tx, ty),
                       "rush": False, "entry": False, "dump": True, "ptype": "rim" if not clear else "dump"})
        self.last_touch = self.owner
        self.mode, self.owner = "loose", None

    def _teammate_offside(self, T: str) -> bool:
        pu = self.rel(T, *self.puck)[0]
        if pu >= BLUE:
            return False
        for q in self._skaters(T):
            if q != self.owner and self.rel(T, self.mt.pos[q][0], self.mt.pos[q][1])[0] > BLUE - 0.5:
                return True
        return False

    def _delayed_offside(self) -> set:
        """Teams under a delayed offside (NHL rule 83.3): the puck went fully out of their attacking zone with a
        teammate still inside and has come back in. It lifts once every one of them is back at the blue line, or
        the other team takes the puck."""
        dl = self.__dict__.setdefault("_dl_off", {})
        out = set()
        for T in SIDES:
            pu = self.rel(T, *self.puck)[0]
            inside = [q for q in self._skaters(T) if self.rel(T, self.mt.pos[q][0], self.mt.pos[q][1])[0] > BLUE + 0.5]
            if pu < BLUE - 1.0:                    # puck completely out
                dl[T] = bool(inside)
            elif not inside:                       # everyone tagged up
                dl[T] = False
            if dl.get(T) and pu >= BLUE:
                out.add(T)
        return out

    def _short(self, side: str) -> bool:
        return self.mt._manpower(self.gc).get("short") == side

    def _decide(self, pt: float) -> None:
        if self.mode != "carry" or not self.owner or self.owner not in self.mt.pos:
            return
        T = self.side
        X, S = self._X(), self._shooter()
        st = self.mt.pos[self.owner]
        pressure = self._nearest_opp_dist(T, st[0], st[1])
        pu, _pv = self.rel(T, *self.puck)
        held = pt - self.held_since
        urg = self.urgency()
        cands = self._pass_candidates()
        if self.changing != self.owner and self._open_look(pt, T, X, S, pressure, held):
            return
        if self.changing == self.owner:
            self.changing = None
            if self._open_look(pt, T, X, S, pressure, held, force=True):
                return                              # open in the slot at the end of his shift: he shoots, then changes
            if cands:
                self._do_pass(cands[0][1], pt)      # gives it up before heading to the bench
            elif -BLUE < pu and not self._teammate_offside(T) and pu < BLUE:
                self._do_dump(pt)
            return
        if T == X:
            R0 = float(self.B["t"]) - self.gc if self.B is not None else 0.0
            if self.owner == S and self._final_approach() and (R0 <= 4.0 or self._odd_on):
                return  # the shooter keeps it: he walks into the region / protects until the director releases
            if cands and cands[0][1] == S and self._final_approach() and (cands[0][0] > 6.0 or urg > 0.4):
                self._do_pass(S, pt)
                return
            if self._final_approach() and S in self.mt.pos and any(q == S for _sc, q in cands):
                reg = region_of((self.B.get("att") or {}).get("chance"))
                su, sv = self.rel(X, self.mt.pos[S][0], self.mt.pos[S][1])
                if in_region(su, sv, reg, slack=6.0):
                    self._do_pass(S, pt)    # the shooter is in his spot: find him
                    return
            R = float(self.B["t"]) - self.gc if self.B is not None else 99.0
            if cands and self._wide_open(cands[0], held) and (R > 1.6 or cands[0][1] == S):
                self._do_pass(cands[0][1], pt)      # a teammate all alone with a clear lane: he gets it now
                return
            if cands and (pressure < 7.0 or held > self.rng.uniform(2.0, 3.5)) and cands[0][0] > 6.0 and (R > 1.6 or cands[0][1] == S):
                self._do_pass(cands[0][1], pt)
                return
            if -BLUE < pu < BLUE and pressure < 6.0 and urg < 0.5 and self.rng.random() < 0.25 and not self._teammate_offside(T):
                self._do_dump(pt)
            return
        # the other team has it: it can move the puck, but under urgency it gets rid of it
        if X is not None and urg > 0.45 and self.rng.random() < 0.5:
            # under the attack's pressure the puck gets away from him (bobbled, a soft pass picked off)
            near = min(self._skaters(X), key=lambda q: math.hypot(self.mt.pos[q][0] - st[0], self.mt.pos[q][1] - st[1]), default=None)
            if near is not None:
                nx, ny = self.mt.pos[near][0], self.mt.pos[near][1]
                d = math.hypot(nx - self.puck[0], ny - self.puck[1]) or 1.0
                v = min(16.0, 4.0 + d * 0.6)
                self.pv = [(nx - self.puck[0]) / d * v, (ny - self.puck[1]) / d * v]
                self.loose_info = {"side": None, "src": "giveaway"}
                self.moves.append({"kind": "giveaway", "pid": self.owner, "pt": pt, "gc": self.gc})
                self.last_touch = self.owner
                self.mode, self.owner = "loose", None
                return
        if pu < -BLUE and pressure < 8.0 and held > 0.5 and self.rng.random() < 0.45:
            if pressure < 5.0 and self.rng.random() < 0.07 and not self._short(T):
                self._do_dump(pt, clear=True)   # hemmed in: fires it the length of the ice (icing if nobody touches it)
            else:
                self._do_dump(pt, chip=True)    # off the glass and out
            return
        if cands and (self._wide_open(cands[0], held) or ((pressure < 6.0 or held > self.rng.uniform(1.8, 3.2)) and cands[0][0] > 5.0)):
            self._do_pass(cands[0][1], pt)
            return
        if -BLUE < pu < BLUE and pressure < 7.0 and self.rng.random() < 0.3 and not self._teammate_offside(T):
            self._do_dump(pt)

    def _wide_open(self, cand: Tuple[float, str], held: float) -> bool:
        """The best option is a teammate with nobody near him, a clean lane and a better spot than the carrier's."""
        if held < 0.45:
            return False
        q = cand[1]
        st = self.mt.pos.get(q)
        if st is None:
            return False
        T = self.side
        if self._nearest_opp_dist(T, st[0], st[1]) < 13.0:
            return False
        own = self.mt.pos.get(self.owner)
        if own is not None:
            ou, ov = self.rel(T, own[0], own[1])
            if 54.0 <= ou < 86.0 and abs(ov) < 24.0 and self._nearest_opp_dist(T, own[0], own[1]) > 7.5:
                # he is open in the slot himself: only a tap-in at the far post beats his own shot
                qu_, qv_ = self.rel(T, st[0], st[1])
                if math.hypot(89.0 - qu_, qv_) > 14.0:
                    return False
        lane = min((self._seg_dist(self.puck[0], self.puck[1], st[0], st[1], self.mt.pos[o][0], self.mt.pos[o][1])
                    for o in self._skaters(_opp(T))), default=30.0)
        if lane < 6.0:
            return False
        qu, qv = self.rel(T, st[0], st[1])
        pu, pv = self.rel(T, *self.puck)
        better = qu > pu + 6.0 or (qu >= BLUE and math.hypot(89.0 - qu, qv) + 8.0 < math.hypot(89.0 - pu, pv))
        return better or cand[0] > 22.0

    def _open_look(self, pt: float, T: str, X: Optional[str], S: Optional[str], pressure: float, held: float, force: bool = False) -> bool:
        """A carrier with time and a lane shoots instead of passing it away.

        The engine decides which shots count, so a look the engine did not schedule is shown as an attempt that
        misses the net or is blocked: nothing reaches the goalie and the box score is unchanged. The engine's own
        shooter, close to his scheduled shot, is left to the director."""
        st = self.mt.pos.get(self.owner)
        if st is None:
            return False
        u, v = self.rel(T, st[0], st[1])
        dist_net = math.hypot(89.0 - u, v)
        slot = 54.0 <= u < 86.0 and abs(v) < 24.0 and dist_net < 34.0
        point = BLUE + 2.0 <= u < 54.0 and abs(v) < 30.0
        if not (slot or point):
            return False
        R = float(self.B["t"]) - self.gc if self.B is not None else 99.0
        if T == X:
            if self.owner == S:
                if R < 8.0 or self._final_approach():
                    return False          # his real shot is coming: the director has him
            elif R < 4.0 or (self._final_approach() and R < 6.0):
                return False              # the puck has to reach the engine's shooter in time
        last = self.__dict__.setdefault("_look_gc", {}).get(T, -99.0)
        if force:
            if not slot or pressure <= 6.0:
                return False
        elif self.gc - last < (2.5 if slot else 12.0):
            return False
        tr = self._tr(self.owner)
        oiq, hands = float(tr.get("oiq", 0.5)), float(tr.get("hands", 0.5))
        if force:
            p = 1.0
        elif slot:
            if pressure <= 7.5 or held < 0.3:
                return False
            p = (0.72 if dist_net < 25.0 else 0.5) + 0.2 * oiq + (0.08 if pressure > 14.0 else 0.0)
        else:
            if pressure <= 12.0 or held < 1.0:
                return False
            p = 0.18 + 0.15 * hands
        if self.rng.random() > p:
            if slot:
                # he holds it and walks in rather than giving it up while he's open
                self.held_since = min(self.held_since, pt - 0.2)
                self.next_decide = pt + self.rng.uniform(0.3, 0.5)
                return True
            return False
        self._look_gc[T] = self.gc
        # aim just outside a post; a defender in the lane gets a piece of it
        side_ = 1.0 if self.rng.random() < 0.5 else -1.0
        miss = self.rng.uniform(3.6, 7.0) - 2.0 * hands
        aim_u, aim_v = 89.0 + self.rng.uniform(0.5, 3.0), side_ * (3.0 + max(0.8, miss))
        ax, ay = self.ab(T, aim_u, aim_v)
        blocker = None
        for q in self._skaters(_opp(T)):
            qx, qy = self.mt.pos[q][0], self.mt.pos[q][1]
            if self._seg_dist(st[0], st[1], ax, ay, qx, qy) < 2.4 and math.hypot(qx - st[0], qy - st[1]) > 2.0:
                blocker = q
                break
        if blocker is not None:
            bx, by = self.mt.pos[blocker][0], self.mt.pos[blocker][1]
            d = math.hypot(bx - self.puck[0], by - self.puck[1]) or 1.0
            self.pv = [(bx - self.puck[0]) / d * 45.0, (by - self.puck[1]) / d * 45.0]
            self.loose_info = {"side": T, "src": "blocked", "look": True}
        else:
            d = math.hypot(ax - self.puck[0], ay - self.puck[1]) or 1.0
            sp = 78.0 + 20.0 * hands
            self.pv = [(ax - self.puck[0]) / d * sp, (ay - self.puck[1]) / d * sp]
            self.loose_info = {"side": T, "src": "wide", "look": True}
        self.moves.append({"kind": "look", "pid": self.owner, "pt": pt, "gc": self.gc, "blocked": blocker})
        self.mt._hint({"pt": round(pt, 2), "t": self.gc, "kind": "look", "side": T, "pid": self.owner, "from": (self.puck[0], self.puck[1]),
                       "to": (ax, ay) if blocker is None else (self.mt.pos[blocker][0], self.mt.pos[blocker][1]),
                       "rush": False, "entry": False, "dump": False, "ptype": "shot", "blocked": blocker})
        self.last_touch = self.owner
        self.mode, self.owner = "loose", None
        return True

    # ── one substep
    def virtual_seg(self) -> Dict[str, Any]:
        X, S = self._X(), self._shooter()
        vs: Dict[str, Any] = {"k": "loose", "live": True, "side": None, "car": None, "rcv": None, "p0": tuple(self.puck), "p1": tuple(self.puck),
                              "ph": "play", "fo": None, "why": None, "ev": [], "force": {}}
        if self.mode == "carry" and self.owner:
            vs.update(k="carry", side=self.side, car=self.owner, rush=self.rush, car_fast=(self.side == X and (self.urgency() > 0.4 or self._final_approach())))
            self.ctarget = self._carrier_target()
            vs["ctarget"] = self.ctarget
        elif self.mode == "pass" and self.pass_:
            vs.update(k="pass", side=self._team(self.pass_["passer"]), car=self.pass_["passer"], rcv=self.pass_["rcv"],
                      p0=self.pass_["from"], p1=self.pass_["to"], rush=self.rush)
        else:
            li = self.loose_info
            if li.get("dump") and math.hypot(*self.pv) > 10.0:
                vs.update(side=li.get("side"), dump=True, p1=li.get("dest") or tuple(self.puck))
            elif li.get("dump"):
                vs.update(side=li.get("side"))      # it died short of the corner: everyone plays the puck itself
        vs["fpuck"] = self._anticipate()
        if X is not None and S and S in self.mt.pos and self.owner != S:
            reg = region_of((self.B.get("att") or {}).get("chance"))
            st = self.mt.pos[S]
            su, sv = self.rel(X, st[0], st[1])
            tu, tv = project(su, sv, reg)
            R_ = float(self.B["t"]) - self.gc
            # He times his cut: into a scoring area only when the shot is close, not parked there waiting.
            if not self._danger(reg) or R_ <= math.hypot(tu - su, tv - sv) / 17.0 + 2.2:
                vs["pre"] = (S, self.ab(X, tu, tv), X)
        if X is not None and self.side == X and self.B is not None and self.B.get("type") in ("attempt", "goal") \
                and float(self.B["t"]) - self.gc < 2.6 and self.rel(X, *self.puck)[0] >= BLUE:
            vs["pre_shot"] = True     # last moments before the shot: a forward screens, another takes the rebound lane
        if self.odd and X is not None and self.side == X and self.mode in ("carry", "pass") and self._final_approach():
            pu_, pv_ = self.rel(X, *self.puck)
            if not self._odd_on and pu_ < BLUE + 4.0:
                self._odd_on = True
            if self._odd_on and pu_ < 72.0:
                vs["rush"], vs["odd"], vs["car_fast"] = True, S, True
                if S and S in self.mt.pos and self.owner != S and pu_ < 60.0:
                    vs.pop("pre", None)   # he drives the far lane beside the carrier until the pass
        elif self.side != X:
            self._odd_on = False
        B = self.B
        if B is not None and (B.get("att") or {}).get("outcome") == "BLOCKED" and self.owner == S and S:
            blk = (B.get("att") or {}).get("blocker_id")
            if blk and blk in self.mt.pos and self._team(blk) == _opp(X):
                net = self.ab(X, NET_X, 0.0)
                bx, by = self.puck[0] + (net[0] - self.puck[0]) * 0.3, self.puck[1] + (net[1] - self.puck[1]) * 0.3
                vs["force"][blk] = (bx, by)   # the engine's blocker gets into the lane
        if self.rush and self.side != X:
            vs["rush"] = False
        if self.mode == "carry" and self.side in SIDES:
            if self._teammate_offside(self.side):
                self._tag_off_gc = self.gc
            if self.gc - getattr(self, "_tag_off_gc", -99.0) < 0.6:
                vs["hold_line"] = True      # waits until everyone is clearly back onside
        # tag-up: teammates caught inside the zone hurry back onside while the puck is outside
        if self.mode == "carry" and self.side in SIDES and self.rel(self.side, *self.puck)[0] < BLUE:
            for q in self._skaters(self.side):
                if q != self.owner:
                    qu, qv = self.rel(self.side, self.mt.pos[q][0], self.mt.pos[q][1])
                    if qu > BLUE - 0.5:
                        vs["force"][q] = self.ab(self.side, BLUE - 4.0, qv)
        # possession tilt: when the shooting team needs the puck its two nearest skaters go after the carrier
        if X is not None and self.mode == "carry" and self.side != X and self.owner in self.mt.pos and self.urgency() > 0.1:
            st = self.mt.pos[self.owner]
            hunters = sorted(self._skaters(X), key=lambda q: math.hypot(self.mt.pos[q][0] - st[0], self.mt.pos[q][1] - st[1]))[:2]
            for q in hunters:
                vs["force"][q] = (st[0], st[1])
        # a loose puck the shooting team needs: its two nearest skaters swarm it
        if X is not None and self.mode == "loose" and self.urgency() > 0.3 and math.hypot(*self.pv) < 30.0:
            near = sorted(self._skaters(X), key=lambda q: math.hypot(self.mt.pos[q][0] - self.puck[0], self.mt.pos[q][1] - self.puck[1]))[:2]
            for q in near:
                vs["force"][q] = (self.puck[0], self.puck[1])
        return vs

    def _anticipate(self) -> Tuple[float, float]:
        """Where the play is going (the units read it and move with it, not behind it)."""
        mt = self.mt
        if self.mode == "carry" and self.owner in mt.pos:
            st = mt.pos[self.owner]
            x, y = self.puck[0] + st[2] * ANTIC, self.puck[1] + st[3] * ANTIC
        elif self.mode == "pass" and self.pass_:
            x, y = self.pass_["to"]
        elif self.loose_info.get("dest") and math.hypot(*self.pv) > 10.0:
            x, y = self.loose_info["dest"]
        else:
            x, y = self.puck[0] + self.pv[0] * 1.2, self.puck[1] + self.pv[1] * 1.2
        return mt._clamp(x, y)

    def _validate_owner(self) -> None:
        on = set(self.mt.onice.get("home") or []) | set(self.mt.onice.get("away") or [])
        if self.mode == "carry" and (self.owner not in self.mt.pos or self.owner not in on):
            self.loose_info = {"side": None, "src": "change"}
            self.mode, self.owner = "loose", None          # he went off on a change: the puck is loose where he left it
        if self.mode == "pass" and self.pass_ and (self.pass_["rcv"] not in self.mt.pos or self.pass_["rcv"] not in on):
            self.pass_["rcv_gone"] = True

    def substep(self, pt: float, dt: float) -> None:
        mt = self.mt
        self._validate_owner()
        B = self.B
        tB = float(B["t"]) if B is not None else self.gc + 999.0
        R = tB - self.gc
        X, S = self._X(), self._shooter()
        vs = self.virtual_seg()
        mt._step(pt, self.gc, dt, vs, puck_override=(self.puck[0], self.puck[1]))
        if S and S in mt.pos:   # the shooter's path (build-up calls describe where he actually carried it)
            self.s_hist.append((pt, mt.pos[S][0], mt.pos[S][1]))
            if len(self.s_hist) > 30:
                del self.s_hist[0]
        self._puck_update(pt, dt)
        if pt >= self.next_decide or (self.changing is not None and self.changing == self.owner):
            self._decide(pt)
            self.next_decide = pt + 0.5
        # clock
        rate = self._rate_step(dt, R)
        is_shot = B is not None and B.get("type") in ("attempt", "goal")
        if is_shot and self.ready() and R <= max(R_BURST, rate) * dt + 1e-9:
            self.gc = tB
            self._release(pt, forced=False)
            return
        step = rate * dt
        if not is_shot and self.gc + step >= tB - 1e-9:
            self.gc = tB
            self.done = True
            return
        if is_shot and self.gc + step >= tB - 1e-9:
            # out of time without a clean chance: release from where the play is (shooter steered to the puck)
            self.gc = tB
            self._release(pt, forced=True)
            return
        # late beyond repair: a natural reset (the defending team puts the puck out of play in its own end)
        if is_shot and R > 2.5 and self._eta() > R / R_MIN + 1.0 and not (self.side == X and self.mode == "carry") and mt._resets < 6:
            self.reset_req = "out"
        self.gc += step

    def _puck_update(self, pt: float, dt: float) -> None:
        mt = self.mt
        X = self._X()
        if self.mode == "carry" and self.owner in mt.pos:
            st = mt.pos[self.owner]
            sp = math.hypot(st[2], st[3])
            hx, hy = ((st[2] / sp, st[3] / sp) if sp > 1.0 else (mt.dir(self.side, self.gc), 0.0))
            tr = mt._traits(self.owner)
            amp = min(1.5, 0.3 + 1.0 * tr["deke"]) if sp > 3.0 else 0.4
            off = amp * math.sin(2.0 * math.pi * pt / max(0.3, tr["dper"]))
            px, py = st[0] + hx * 1.6 - hy * off, st[1] + hy * 1.6 + hx * off
            dX = mt.dir(self.side, self.gc)
            cu = st[0] * dX
            if BLUE - 4.0 < cu < BLUE + 4.0 and px * dX < cu + 1.0:
                px = (cu + 1.0) * dX         # at the line the puck goes in on the forehand, ahead of his skates
            self.puck = list(mt._clamp(px, py))
            self._poke_checks(pt, dt)
            return
        if self.mode == "pass" and self.pass_:
            P = self.pass_
            a = min(1.0, (pt - P["pt0"]) / P["dur"])
            if P.get("poly"):
                x, y = poly_at(P["poly"], a)
            else:
                x = P["from"][0] + (P["to"][0] - P["from"][0]) * a
                y = P["from"][1] + (P["to"][1] - P["from"][1]) * a
                if P.get("arc"):
                    dx, dy = P["to"][0] - P["from"][0], P["to"][1] - P["from"][1]
                    L = math.hypot(dx, dy) or 1.0
                    x, y = x - dy / L * P["arc"] * math.sin(math.pi * a), y + dx / L * P["arc"] * math.sin(math.pi * a)
            self.puck = [x, y]
            T = self._team(P["passer"])
            # interceptions
            for q in self._skaters(_opp(T)):
                stq = mt.pos[q]
                if math.hypot(stq[0] - x, stq[1] - y) < 2.6 and 0.15 < a < 0.95:
                    if self.rng.random() < self._steal_p(_opp(T), 0.35):
                        self._gain(q, pt, how="intercept")
                        return
            if a >= 1.0:
                rcv = P["rcv"]
                st = mt.pos.get(rcv)
                reach = 6.5 if (T == X and self.urgency() > 0.3) else 4.5   # a stretched stick when the chance must come
                if st is not None and math.hypot(st[0] - x, st[1] - y) < reach:
                    self.moves.append({"kind": "recv", "pid": rcv, "pt": pt, "gc": self.gc})
                    self._gain(rcv, pt, how="pass")
                else:   # missed connection: the puck slides on
                    d = math.hypot(P["to"][0] - P["from"][0], P["to"][1] - P["from"][1]) or 1.0
                    self.pv = [(P["to"][0] - P["from"][0]) / d * 25.0, (P["to"][1] - P["from"][1]) / d * 25.0]
                    self.loose_info = {"side": T, "src": "missed"}
                    self.mode, self.pass_ = "loose", None
            return
        # loose: slides with friction, bounces off the boards
        x, y = self.puck[0] + self.pv[0] * dt, self.puck[1] + self.pv[1] * dt
        cx, cy = mt._clamp(x, y)
        if abs(cx - x) > 0.01:
            self.pv[0] = -0.6 * self.pv[0]
        if abs(cy - y) > 0.01:
            self.pv[1] = -0.6 * self.pv[1]
        for gx in (-NET_X, NET_X):   # a loose puck doesn't sit in the crease: it squirts out
            dxn, dyn = cx - gx, cy
            dn = math.hypot(dxn, dyn)
            if dn < 8.0:
                ux, uy = ((dxn / dn, dyn / dn) if dn > 1e-6 else ((-1.0 if gx > 0 else 1.0), 0.0))
                cx, cy = self.mt._clamp(gx + ux * 8.2, uy * 8.2)
                self.pv = [ux * 8.0, uy * 8.0]
        self.puck = [cx, cy]
        # a puck moving fast glides (ice barely slows it: clears and rims reach the far end); slow pucks in a
        # scramble among skates and sticks die quickly
        sp_now = math.hypot(*self.pv)
        f = (0.965 if sp_now > 30.0 else (0.9 if sp_now > 14.0 else 0.82)) ** (dt / 0.25)
        self.pv = [self.pv[0] * f, self.pv[1] * f]
        li = self.loose_info
        if li.get("icing_ok") and li.get("side"):
            u = self.rel(li["side"], *self.puck)[0]
            if u > 88.5:
                self.reset_req = "icing"
                li["icing_ok"] = False
                return
        speed = math.hypot(*self.pv)
        best, bd = None, 2.6
        frozen = self._delayed_offside()
        for side in SIDES:
            if side in frozen:
                continue   # delayed offside: the attacking team can't touch it inside the zone until everyone tags up
            for q in self._skaters(side):
                st = mt.pos[q]
                d = math.hypot(st[0] - self.puck[0], st[1] - self.puck[1])
                if X is not None and side != X and self.urgency() > 0.5 and d > 1.2:
                    continue   # the defending side only bats at it while the attack swarms the loose puck
                if d < bd and speed < 45.0:
                    # contested: the team the director needs wins ties
                    bias = -0.8 if (X is not None and side == X and self.urgency() > 0.3) else 0.0
                    if d + bias < bd:
                        best, bd = q, d + bias
        if best is not None:
            if li.get("dump") and li.get("side") and self._team(best) != li.get("side"):
                pass
            self._gain(best, pt, how="loose")

    def _steal_p(self, thief_side: str, base: float) -> float:
        X = self._X()
        B = self.B
        R = (float(B["t"]) - self.gc) if B is not None else 99.0
        urg = self.urgency()
        if X is None:
            return base * 0.6
        if thief_side == X:
            return min(0.95, base * (0.6 + 5.0 * urg))
        # The defending side can win the puck whenever the next chance is far enough off for the attack to earn it
        # back. It used to be shut out for the 22 s before every chance, which is most of the game: checks never landed.
        if R < 9.0 or self._final_approach():
            return 0.0
        return base * (0.7 + min(1.1, (R - 9.0) / 16.0))

    def _poke_checks(self, pt: float, dt: float) -> None:
        mt = self.mt
        T = self.side
        st = mt.pos.get(self.owner)
        if st is None or T not in SIDES:
            return
        X = self._X()
        reach = 8.0 if (X is not None and _opp(T) == X and self.urgency() > 0.6) else 5.2   # stick lift when it must happen
        for q in self._skaters(_opp(T)):
            sq = mt.pos[q]
            if math.hypot(sq[0] - st[0], sq[1] - st[1]) < reach:   # bodies keep >= 4 ft, sticks reach ~5 ft
                p = self._steal_p(_opp(T), 0.06) * (dt / 0.25)
                # a smart checker against poor hands wins the puck far more often than the reverse
                p *= max(0.4, 1.0 + 0.9 * (float(self._tr(q).get("diq", 0.5)) - float(self._tr(self.owner).get("hands", 0.5))))
                if self.rng.random() < p:
                    ang = self.rng.uniform(0, 6.283)
                    self.pv = [math.cos(ang) * 10.0, math.sin(ang) * 10.0]
                    self.loose_info = {"side": None, "poke": q}
                    self.moves.append({"kind": "poke", "pid": q, "pt": pt, "gc": self.gc})
                    self.last_touch = q
                    self.mode, self.owner = "loose", None
                    return

    def _gain(self, pid: str, pt: float, how: str) -> None:
        prev_side = self.side
        self.owner, self.mode, self.pass_ = pid, "carry", None
        self.side = self._team(pid)
        dl = self.__dict__.get("_dl_off")
        if dl:
            dl[_opp(self.side)] = False   # the defending team has the puck: the delayed offside is over
        self.pv = [0.0, 0.0]
        self.held_since = pt
        self.next_decide = pt + self.rng.uniform(0.6, 1.2)
        self.last_touch = pid
        li = self.loose_info
        if li.get("dump") and li.get("side") and self.side != li.get("side") and self.rng.random() < 0.3:
            hitter = min(self._skaters(li["side"]), key=lambda q: math.hypot(self.mt.pos[q][0] - self.puck[0], self.mt.pos[q][1] - self.puck[1]), default=None)
            if hitter and math.hypot(self.mt.pos[hitter][0] - self.puck[0], self.mt.pos[hitter][1] - self.puck[1]) < 6.0:
                self.mt._events.append({"type": "hit", "t": self.gc, "pt": round(pt, 2), "meta": {"hitter_id": hitter, "victim_id": pid, "side": li["side"]}})
        self.loose_info = {}
        X = self._X()
        # a breakaway chance turns into a rush when the shooting team wins it in the neutral zone
        B = self.B
        self.rush = bool(X is not None and self.side == X and (B.get("att") or {}).get("chance") == "breakaway" and self.rel(X, *self.puck)[0] < BLUE)
        self.moves.append({"kind": "gain", "pid": pid, "pt": pt, "gc": self.gc, "how": how, "turnover": prev_side not in (None, self.side)})

    def _release(self, pt: float, forced: bool) -> None:
        X, S = self._X(), self._shooter()
        mt = self.mt
        if forced:
            self.stats["forced"] += 1
            mt._play_stats["forced"] += 1
            why = ("x_has" if self.side == X and self.mode == "carry" else ("pass" if self.mode == "pass" else ("loose" if self.mode == "loose" else "opp_has")))
            if self.side == X and self.owner == S:
                why = "s_has_outside"
            reg_ = region_of((self.B.get("att") or {}).get("chance"))
            pu_, pv_ = self.rel(X, *self.puck)
            tu_, tv_ = project(pu_, pv_, reg_)
            mt._play_stats.setdefault("trace", []).append((round(float(self.B["t"]) - self.seg["g0"], 1), (self.B.get("att") or {}).get("chance"),
                                                          self.mode, self.side == X, self.owner == S, round(math.hypot(tu_ - pu_, tv_ - pv_)), round(pu_)))
            mt._play_stats.setdefault("why", {})
            mt._play_stats["why"][why] = mt._play_stats["why"].get(why, 0) + 1
            st = mt.pos.get(S) if S else None
            if self.owner != S and st is not None:
                # the puck finds the shooter (quick feed) so the engine's shooter takes it
                self.moves.append({"kind": "pass", "pid": self.owner, "rcv": S, "pt": pt, "gc": self.gc, "ptype": "line",
                                   "from": tuple(self.puck), "to": (st[0], st[1])})
                self.puck = [st[0], st[1]]
                self.owner, self.side, self.mode = S, X, "carry"
        shooter_pos = mt.pos.get(S) if S else None
        loc = (self.puck[0], self.puck[1]) if self.owner == S or shooter_pos is None else (shooter_pos[0], shooter_pos[1])
        for gx in (-NET_X, NET_X):   # shots are released outside the crease
            dn = math.hypot(loc[0] - gx, loc[1])
            if dn < 7.6:
                ux, uy = (((loc[0] - gx) / dn, loc[1] / dn) if dn > 1e-6 else ((-1.0 if gx > 0 else 1.0), 0.0))
                loc = (gx + ux * 7.6, uy * 7.6)
        reg = region_of((self.B.get("att") or {}).get("chance"))
        u, v = self.rel(X, loc[0], loc[1]) if X else (0.0, 0.0)
        if X and u < BLUE + 2.0 and not (self.B.get("att") or {}).get("empty_net"):
            # a shot on goal from outside the zone only happens at an empty net: the shooter has stepped inside the line
            u = BLUE + 2.0 + self.rng.uniform(0.0, 3.0)
            loc = self.ab(X, u, v)
        if X and not in_region(u, v, reg, slack=3.0):
            self.stats["out_region"] += 1
            mt._play_stats["out_region"] += 1
        self.release = {"loc": loc, "pt": pt, "forced": forced}
        self.done = True
