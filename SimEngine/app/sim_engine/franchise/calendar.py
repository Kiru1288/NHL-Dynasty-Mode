"""
NHL-style franchise calendar: real dates from mid-September through late June.

This module:
- Builds a hockey-year calendar.
- Marks preseason, regular season, playoffs, and offseason.
- Marks league events/breaks.
- Maps abstract schedule slate IDs onto real dates using NHL-like month balance.

Major scheduling goal:
Do NOT allow the season to be dead for the first three months and then overloaded
later. Regular-season abstract slates are distributed month-by-month in a stable,
NHL-style pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# =============================================================================
# Calendar model
# =============================================================================


@dataclass(frozen=True)
class CalendarDay:
    index: int
    iso: str
    weekday: str
    segment: str  # preseason | regular | playoffs | offseason
    allows_games: bool
    tags: Tuple[str, ...]
    ui_phase: str
    ui_note: str


# =============================================================================
# Date helpers
# =============================================================================


def _daterange(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _season_end_year(season_start_year: int) -> int:
    return int(season_start_year) + 1


def _empty_date_window(year: int) -> Tuple[date, date]:
    start = date(int(year), 1, 1)
    return start, start - timedelta(days=1)


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _dedupe_keep_order(items: Sequence[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for item in items:
        item = int(item)
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _in_window(d: date, start: date, end: date) -> bool:
    return start <= d <= end


def _month_key_from_iso(iso: str) -> Tuple[int, int]:
    y, m, _ = iso.split("-")
    return int(y), int(m)


# =============================================================================
# Season event rules
# =============================================================================


def _olympic_winter_season_by_end_year(end_year: int) -> bool:
    return int(end_year) in (2022, 2026, 2030, 2034)


def _four_nations_season(season_start_year: int) -> bool:
    end_year = _season_end_year(season_start_year)
    if _olympic_winter_season_by_end_year(end_year):
        return False
    return int(season_start_year) % 2 == 1


def _all_star_window(season_start_year: int) -> Tuple[date, date]:
    end_year = _season_end_year(season_start_year)

    if _olympic_winter_season_by_end_year(end_year):
        return _empty_date_window(end_year)

    return date(end_year, 1, 30), date(end_year, 2, 2)


def _olympic_window(season_start_year: int) -> Tuple[date, date]:
    end_year = _season_end_year(season_start_year)

    if not _olympic_winter_season_by_end_year(end_year):
        return _empty_date_window(end_year)

    return date(end_year, 2, 7), date(end_year, 2, 22)


def _four_nations_window(season_start_year: int) -> Tuple[date, date]:
    end_year = _season_end_year(season_start_year)

    if not _four_nations_season(season_start_year):
        return _empty_date_window(end_year)

    return date(end_year, 2, 10), date(end_year, 2, 16)


def _regular_season_bounds(season_start_year: int) -> Tuple[date, date]:
    return date(season_start_year, 10, 8), date(season_start_year + 1, 4, 18)


def _preseason_bounds(season_start_year: int) -> Tuple[date, date]:
    return date(season_start_year, 9, 15), date(season_start_year, 10, 7)


def _playoff_bounds(season_start_year: int) -> Tuple[date, date]:
    return date(season_start_year + 1, 4, 19), date(season_start_year + 1, 6, 20)


# =============================================================================
# Calendar classification
# =============================================================================


def classify_calendar_day(
    d: date,
    season_start_year: int,
) -> Tuple[str, bool, Tuple[str, ...], str, str]:
    season_start_year = int(season_start_year)
    tags: List[str] = []

    pre_start, pre_end = _preseason_bounds(season_start_year)
    reg_start, reg_end = _regular_season_bounds(season_start_year)
    po_start, po_end = _playoff_bounds(season_start_year)

    allstar_start, allstar_end = _all_star_window(season_start_year)
    olympic_start, olympic_end = _olympic_window(season_start_year)
    four_start, four_end = _four_nations_window(season_start_year)

    # -------------------------------------------------------------------------
    # Preseason
    # -------------------------------------------------------------------------

    if pre_start <= d <= pre_end:
        allows = True
        ui_phase = "Preseason"
        ui_note = "Evaluation camps · roster battles · exhibition slate"

        if d.weekday() >= 5:
            tags.append("preseason_weekend")

        if d >= date(season_start_year, 9, 28):
            tags.append("roster_battle")
            ui_note = "Final roster battles · waiver decisions · exhibition slate"

        return "preseason", allows, tuple(tags), ui_phase, ui_note

    # -------------------------------------------------------------------------
    # Regular season
    # -------------------------------------------------------------------------

    if reg_start <= d <= reg_end:
        allows = True
        ui_phase = "Regular Season"
        ui_note = "Standings grind · travel · divisional races"

        # Christmas pause.
        if d.month == 12 and d.day in (24, 25, 26):
            allows = False
            tags.append("christmas_break")
            ui_phase = "Christmas Break"
            ui_note = "League pause — no NHL games"

        # World Juniors.
        if (d.month == 12 and d.day >= 26) or (d.month == 1 and d.day <= 5):
            tags.append("world_juniors")
            if allows:
                ui_phase = "World Juniors Window"
                ui_note = "Prospect spotlight · international tournament buzz"

        # Outdoor showcases (franchise narrative dates; also mirrored in season_anchor_event_markers for UI).
        if d.month == 1 and d.day == 1:
            tags.append("winter_classic")
            ui_phase = "Winter Classic"
            ui_note = "Outdoor showcase · league spotlight game"

        if d.month == 2 and d.day == 18:
            tags.append("heritage_classic")
            ui_phase = "Heritage Classic"
            ui_note = "Canadian legacy game · outdoor showcase"

        # Breaks.
        if _in_window(d, allstar_start, allstar_end):
            allows = False
            tags.append("allstar_break")
            ui_phase = "All-Star Break"
            ui_note = "Mid-season pause · skills showcase · league reset"

        if _in_window(d, olympic_start, olympic_end):
            allows = False
            tags.append("olympics")
            ui_phase = "Olympic Break"
            ui_note = "International tournament · NHL schedule pause"

        if _in_window(d, four_start, four_end):
            allows = False
            tags.append("four_nations")
            ui_phase = "4 Nations Face-Off"
            ui_note = "International best-on-best · NHL schedule pause"

        # Trade deadline.
        if (d.month == 2 and d.day >= 25) or (d.month == 3 and d.day <= 10):
            tags.append("trade_deadline")
            if allows:
                ui_phase = "Trade Deadline Window"
                ui_note = "Trade deadline pressure · contender arms race"

        # Final race.
        if d >= date(season_start_year + 1, 4, 1):
            tags.append("playoff_race")
            if allows:
                ui_note = "Playoff race · clinching scenarios · desperation hockey"

        return "regular", allows, tuple(tags), ui_phase, ui_note

    # -------------------------------------------------------------------------
    # Playoffs
    # -------------------------------------------------------------------------

    if po_start <= d <= po_end:
        tags = []

        if d.month == 4:
            tags.append("round_one_window")
        elif d.month == 5 and d.day <= 15:
            tags.append("round_two_window")
        elif d.month == 5 and d.day > 15:
            tags.append("conference_final_window")
        elif d.month == 6:
            tags.append("cup_final_window")

        return (
            "playoffs",
            False,
            tuple(tags),
            "Stanley Cup Playoffs",
            "Series hockey · maximum stakes",
        )

    # -------------------------------------------------------------------------
    # Offseason
    # -------------------------------------------------------------------------

    tags = []
    ui_phase = "Offseason"
    ui_note = "Draft · free agency · development"

    if d.month == 6 and d.day >= 21:
        tags.append("draft_window")
        ui_phase = "Draft Window"
        ui_note = "Draft floor · prospect decisions · franchise direction"

    if d.month == 7 and d.day <= 10:
        tags.append("free_agency_opening")
        ui_phase = "Free Agency"
        ui_note = "Market opens · roster construction · cap pressure"

    return "offseason", False, tuple(tags), ui_phase, ui_note


# =============================================================================
# Calendar construction / serialization
# =============================================================================


def build_season_calendar(season_start_year: int) -> List[CalendarDay]:
    season_start_year = int(season_start_year)
    start = date(season_start_year, 9, 15)
    end = date(season_start_year + 1, 6, 30)

    out: List[CalendarDay] = []

    for i, cur in enumerate(_daterange(start, end)):
        segment, allows, tags, ui_phase, ui_note = classify_calendar_day(
            cur,
            season_start_year,
        )

        out.append(
            CalendarDay(
                index=i,
                iso=cur.isoformat(),
                weekday=cur.strftime("%a"),
                segment=segment,
                allows_games=bool(allows),
                tags=tuple(tags),
                ui_phase=ui_phase,
                ui_note=ui_note,
            )
        )

    return out


def calendar_day_to_dict(d: CalendarDay) -> Dict[str, Any]:
    return {
        "index": d.index,
        "iso": d.iso,
        "weekday": d.weekday,
        "segment": d.segment,
        "allows_games": d.allows_games,
        "tags": list(d.tags),
        "ui_phase": d.ui_phase,
        "ui_note": d.ui_note,
    }


# =============================================================================
# Calendar index helpers
# =============================================================================


def eligible_game_indices(
    calendar: List[CalendarDay],
    *,
    segments: Optional[Tuple[str, ...]] = None,
) -> List[int]:
    segs = segments or ("preseason", "regular")
    return [
        d.index
        for d in calendar
        if d.segment in segs and d.allows_games
    ]


def last_regular_season_index(calendar: List[CalendarDay]) -> int:
    last = 0
    for d in calendar:
        if d.segment in ("preseason", "regular"):
            last = d.index
    return last


def _calendar_by_index(calendar: List[CalendarDay]) -> Dict[int, CalendarDay]:
    return {int(d.index): d for d in calendar}


def _indices_for_segment(calendar: List[CalendarDay], segment: str) -> List[int]:
    return [
        d.index
        for d in calendar
        if d.segment == segment and d.allows_games
    ]


def _indices_with_tag(calendar: List[CalendarDay], tag: str) -> List[int]:
    return [
        d.index
        for d in calendar
        if tag in d.tags
    ]


# =============================================================================
# NHL-style slate distribution
# =============================================================================


def _regular_day_weight(day: CalendarDay) -> int:
    """
    Daily NHL density preference.

    This only controls which calendar dates receive abstract slates. It does not
    know which teams are playing.
    """
    if not day.allows_games or day.segment != "regular":
        return 0

    if day.weekday == "Sat":
        weight = 6
    elif day.weekday in ("Tue", "Thu"):
        weight = 5
    elif day.weekday in ("Fri", "Sun"):
        weight = 3
    else:
        weight = 2

    tags = set(day.tags)

    if "winter_classic" in tags or "heritage_classic" in tags:
        weight = min(weight, 3)

    if "trade_deadline" in tags:
        weight += 1

    if "playoff_race" in tags:
        weight += 1

    return max(1, weight)


def _preseason_day_weight(day: CalendarDay) -> int:
    if not day.allows_games or day.segment != "preseason":
        return 0

    if day.weekday == "Sat":
        return 4
    if day.weekday in ("Fri", "Sun"):
        return 3
    if day.weekday in ("Tue", "Thu"):
        return 2
    return 1


def _month_target_weight(month: int) -> float:
    """
    NHL-style month balance.

    This is the biggest fix.

    Old behavior:
        The mapper sampled across a weighted pool and could leave the first
        few months too empty, then overload later.

    New behavior:
        Each month gets a reasonable chunk of the season before we pick dates.
    """
    if month == 10:
        return 0.86  # Oct starts after opening week, but should still be busy.
    if month == 11:
        return 1.05
    if month == 12:
        return 0.98  # Christmas pause trims it slightly.
    if month == 1:
        return 1.02
    if month == 2:
        return 0.78  # All-Star/Olympics/Four Nations can reduce volume.
    if month == 3:
        return 1.16  # Heavy stretch run.
    if month == 4:
        return 0.62  # Regular season ends mid-April.
    return 1.0


def _group_indices_by_month(
    calendar: List[CalendarDay],
    indices: Sequence[int],
) -> Dict[Tuple[int, int], List[int]]:
    by_index = _calendar_by_index(calendar)
    grouped: Dict[Tuple[int, int], List[int]] = {}

    for idx in indices:
        day = by_index.get(int(idx))
        if not day:
            continue

        key = _month_key_from_iso(day.iso)
        grouped.setdefault(key, []).append(int(idx))

    return grouped


def _allocate_counts_by_month(
    calendar: List[CalendarDay],
    eligible: Sequence[int],
    total_count: int,
) -> Dict[Tuple[int, int], int]:
    """
    Allocate regular-season abstract slates by calendar month.

    The allocation considers:
    - how many playable dates the month has
    - NHL-style monthly intensity
    - minimum early-season presence
    - no insane late-season dump
    """
    total_count = max(0, int(total_count))
    if total_count <= 0:
        return {}

    grouped = _group_indices_by_month(calendar, eligible)
    if not grouped:
        return {}

    weighted: Dict[Tuple[int, int], float] = {}
    total_weight = 0.0

    for key, days in grouped.items():
        _, month = key
        playable_days = len(days)
        month_weight = playable_days * _month_target_weight(month)
        weighted[key] = month_weight
        total_weight += month_weight

    if total_weight <= 0:
        keys = sorted(grouped.keys())
        base = total_count // max(1, len(keys))
        rem = total_count % max(1, len(keys))
        return {key: base + (1 if i < rem else 0) for i, key in enumerate(keys)}

    raw: Dict[Tuple[int, int], float] = {
        key: (weight / total_weight) * total_count
        for key, weight in weighted.items()
    }

    counts: Dict[Tuple[int, int], int] = {
        key: int(raw[key])
        for key in raw
    }

    used = sum(counts.values())
    remaining = total_count - used

    remainders = sorted(
        raw.keys(),
        key=lambda key: raw[key] - int(raw[key]),
        reverse=True,
    )

    for key in remainders:
        if remaining <= 0:
            break
        counts[key] += 1
        remaining -= 1

    # Early-season floor: October/November/December cannot be starved.
    # For a normal slate count, these months need meaningful volume.
    month_keys = {month: key for key in counts for _, month in [key]}

    if total_count >= 40:
        minimum_share = {
            10: 0.105,
            11: 0.135,
            12: 0.125,
            1: 0.125,
        }

        for month, share in minimum_share.items():
            key = month_keys.get(month)
            if key is None:
                continue

            min_count = max(1, int(round(total_count * share)))
            if counts.get(key, 0) < min_count:
                needed = min_count - counts.get(key, 0)
                counts[key] = min_count

                # Pull from the most overloaded later months first.
                for donor_month in (3, 2, 4, 1, 12, 11, 10):
                    if needed <= 0:
                        break

                    donor_key = month_keys.get(donor_month)
                    if donor_key is None or donor_key == key:
                        continue

                    while needed > 0 and counts.get(donor_key, 0) > 1:
                        counts[donor_key] -= 1
                        needed -= 1

    # Final correction to exact total.
    diff = total_count - sum(counts.values())

    while diff > 0:
        best_key = max(
            counts.keys(),
            key=lambda key: weighted.get(key, 0.0) / max(1, counts.get(key, 0)),
        )
        counts[best_key] += 1
        diff -= 1

    while diff < 0:
        best_key = max(
            counts.keys(),
            key=lambda key: counts.get(key, 0),
        )
        if counts[best_key] > 0:
            counts[best_key] -= 1
            diff += 1
        else:
            break

    return counts


def _weighted_pool_for_indices(
    calendar: List[CalendarDay],
    indices: Sequence[int],
    *,
    segment: str,
) -> List[int]:
    by_index = _calendar_by_index(calendar)
    pool: List[int] = []

    for idx in indices:
        day = by_index.get(int(idx))
        if not day:
            continue

        if segment == "preseason":
            weight = _preseason_day_weight(day)
        elif segment == "regular":
            weight = _regular_day_weight(day)
        else:
            weight = 1 if day.allows_games else 0

        for _ in range(max(0, int(weight))):
            pool.append(int(idx))

    return pool or [int(x) for x in indices]


def _sample_evenly_from_pool(pool: Sequence[int], count: int) -> List[int]:
    count = max(0, int(count))
    if count <= 0 or not pool:
        return []

    if count == 1:
        return [int(pool[len(pool) // 2])]

    out: List[int] = []
    last = len(pool) - 1

    for i in range(count):
        pos = int(round((i * last) / max(1, count - 1)))
        pos = max(0, min(last, pos))
        out.append(int(pool[pos]))

    return out


def _limit_identical_runs(values: List[int], max_run: int = 2) -> List[int]:
    if not values:
        return values

    max_run = max(1, int(max_run))
    out = list(values)

    run_start = 0
    while run_start < len(out):
        run_end = run_start + 1

        while run_end < len(out) and out[run_end] == out[run_start]:
            run_end += 1

        run_len = run_end - run_start

        if run_len > max_run:
            for j in range(run_start + max_run, run_end):
                out[j] = max(out[j], out[j - 1] + 1)

        run_start = run_end

    return out


def _ensure_unique_mapping_targets(
    targets: List[int],
    eligible: Sequence[int],
) -> List[int]:
    """
    Spread duplicate calendar indices across nearby eligible dates.

    Multiple abstract slate days must not map to the same concrete calendar day;
    merging slates onto one date double-books teams that were safe on separate
    abstract days.
    """
    if not targets:
        return []

    eligible_sorted = sorted(_dedupe_keep_order([int(x) for x in eligible]))
    if not eligible_sorted:
        return list(targets)

    eligible_set = set(eligible_sorted)
    used: set = set()
    out: List[int] = []

    def _nearest_unused(anchor: int) -> int:
        anchor = int(anchor)
        for radius in range(0, len(eligible_sorted) + 1):
            for cand in (anchor + radius, anchor - radius):
                if cand in eligible_set and cand not in used:
                    return int(cand)
        for cand in eligible_sorted:
            if cand not in used:
                return int(cand)
        return int(eligible_sorted[-1])

    for raw in targets:
        raw = int(raw)
        if raw not in eligible_set:
            cursor = 0
            while cursor < len(eligible_sorted) and eligible_sorted[cursor] < raw:
                cursor += 1
            if cursor < len(eligible_sorted):
                raw = int(eligible_sorted[cursor])
            else:
                raw = int(eligible_sorted[-1])

        if raw not in used:
            out.append(raw)
            used.add(raw)
            continue

        replacement = _nearest_unused(raw)
        out.append(replacement)
        used.add(replacement)

    return out


def _clamp_to_eligible(mapped: List[int], eligible: Sequence[int]) -> List[int]:
    if not mapped:
        return []

    eligible_sorted = sorted(_dedupe_keep_order([int(x) for x in eligible]))

    if not eligible_sorted:
        return mapped

    eligible_set = set(eligible_sorted)
    out: List[int] = []
    cursor = 0

    for raw in mapped:
        raw = int(raw)

        if raw in eligible_set:
            out.append(raw)
            while cursor < len(eligible_sorted) and eligible_sorted[cursor] < raw:
                cursor += 1
            continue

        while cursor < len(eligible_sorted) and eligible_sorted[cursor] < raw:
            cursor += 1

        if cursor < len(eligible_sorted):
            out.append(eligible_sorted[cursor])
        else:
            out.append(eligible_sorted[-1])

    return out


def _smooth_month_internal_gaps(
    calendar: List[CalendarDay],
    month_indices: Sequence[int],
    count: int,
) -> List[int]:
    """
    Pick dates inside a month.

    The goal is not one-game-per-day. The goal is:
    - frequent NHL rhythm
    - weekday weighting
    - no dead month
    - no same-date spam
    """
    if count <= 0:
        return []

    eligible = sorted(_dedupe_keep_order([int(x) for x in month_indices]))
    if not eligible:
        return []

    pool = _weighted_pool_for_indices(calendar, eligible, segment="regular")
    mapped = _sample_evenly_from_pool(pool, count)
    mapped = _limit_identical_runs(mapped, max_run=2)
    mapped = _clamp_to_eligible(mapped, eligible)
    mapped.sort()
    return mapped


def _build_regular_mapping_targets(
    calendar: List[CalendarDay],
    count: int,
) -> List[int]:
    eligible = _indices_for_segment(calendar, "regular")

    if count <= 0 or not eligible:
        return []

    grouped = _group_indices_by_month(calendar, eligible)
    monthly_counts = _allocate_counts_by_month(calendar, eligible, count)

    targets: List[int] = []

    for key in sorted(monthly_counts.keys()):
        month_days = grouped.get(key, [])
        month_count = monthly_counts.get(key, 0)
        targets.extend(_smooth_month_internal_gaps(calendar, month_days, month_count))

    targets.sort()
    targets = _protect_showcase_dates(calendar, targets)
    targets = _limit_total_month_overload(calendar, targets)
    targets.sort()

    return targets[:count]


def _build_preseason_mapping_targets(
    calendar: List[CalendarDay],
    count: int,
) -> List[int]:
    eligible = _indices_for_segment(calendar, "preseason")

    if count <= 0 or not eligible:
        return []

    pool = _weighted_pool_for_indices(calendar, eligible, segment="preseason")
    mapped = _sample_evenly_from_pool(pool, count)
    mapped = _limit_identical_runs(mapped, max_run=2)
    mapped = _clamp_to_eligible(mapped, eligible)
    mapped.sort()
    return mapped


def _protect_showcase_dates(
    calendar: List[CalendarDay],
    mapped_regular: List[int],
) -> List[int]:
    if not mapped_regular:
        return mapped_regular

    by_index = _calendar_by_index(calendar)

    showcase_indices: List[int] = []
    showcase_indices.extend(_indices_with_tag(calendar, "winter_classic"))
    showcase_indices.extend(_indices_with_tag(calendar, "heritage_classic"))

    showcase_indices = [
        idx
        for idx in _dedupe_keep_order(showcase_indices)
        if by_index.get(idx) is not None
        and by_index[idx].segment == "regular"
        and by_index[idx].allows_games
    ]

    if not showcase_indices:
        return mapped_regular

    out = list(mapped_regular)

    for showcase_idx in showcase_indices:
        if showcase_idx in out:
            continue

        closest_pos = min(
            range(len(out)),
            key=lambda i: abs(int(out[i]) - int(showcase_idx)),
        )
        out[closest_pos] = int(showcase_idx)

    out.sort()
    return out


def _limit_total_month_overload(
    calendar: List[CalendarDay],
    targets: List[int],
) -> List[int]:
    """
    Prevent the abstract slate mapper from creating an absurd month dump.

    This is not a perfect team schedule fixer, but it stops the league calendar
    from putting way too many abstract slates into one month.
    """
    if not targets:
        return []

    by_index = _calendar_by_index(calendar)
    grouped: Dict[Tuple[int, int], List[int]] = {}

    for idx in targets:
        day = by_index.get(int(idx))
        if not day:
            continue
        key = _month_key_from_iso(day.iso)
        grouped.setdefault(key, []).append(int(idx))

    total = len(targets)

    # Monthly cap for abstract slates. This is intentionally generous because
    # these are league-wide slate IDs, not user-team games.
    cap = max(4, int(round(total * 0.19)))

    overflow: List[int] = []

    for key in sorted(grouped.keys()):
        rows = sorted(grouped[key])
        if len(rows) > cap:
            grouped[key] = rows[:cap]
            overflow.extend(rows[cap:])

    if not overflow:
        return sorted(targets)

    # Move overflow into underfilled months with available eligible dates.
    regular_eligible = _indices_for_segment(calendar, "regular")
    eligible_by_month = _group_indices_by_month(calendar, regular_eligible)

    for extra in overflow:
        candidate_keys = sorted(
            eligible_by_month.keys(),
            key=lambda key: len(grouped.get(key, [])),
        )

        placed = False

        for key in candidate_keys:
            if len(grouped.get(key, [])) >= cap:
                continue

            month_days = eligible_by_month.get(key, [])
            if not month_days:
                continue

            closest = min(month_days, key=lambda idx: abs(int(idx) - int(extra)))
            grouped.setdefault(key, []).append(int(closest))
            placed = True
            break

        if not placed:
            original_day = by_index.get(int(extra))
            if original_day:
                original_key = _month_key_from_iso(original_day.iso)
                grouped.setdefault(original_key, []).append(int(extra))

    out: List[int] = []

    for key in sorted(grouped.keys()):
        out.extend(sorted(grouped[key]))

    return sorted(out)


def _estimate_preseason_abstract_count(
    total_abstract_count: int,
    preseason_slate_cap: int,
    preseason_eligible_count: int,
) -> int:
    """
    Conservative preseason estimate.

    Important:
    Many engines pass only regular-season abstract slate IDs into this mapper.
    If we steal too many of those and call them preseason, the actual regular
    calendar starts too slowly. So this uses a small preseason front block only.
    """
    total_abstract_count = max(0, int(total_abstract_count))
    preseason_slate_cap = max(0, int(preseason_slate_cap))
    preseason_eligible_count = max(0, int(preseason_eligible_count))

    if total_abstract_count <= 0:
        return 0

    if preseason_slate_cap <= 0 or preseason_eligible_count <= 0:
        return 0

    # Do not steal 8+ regular slates by default. Keep preseason short.
    estimated = min(
        preseason_slate_cap,
        preseason_eligible_count,
        max(2, round(total_abstract_count * 0.025)),
    )

    return max(0, int(estimated))


# =============================================================================
# Main mapping function
# =============================================================================


def map_abstract_schedule_to_calendar(
    calendar: List[CalendarDay],
    abstract_day_numbers: List[int],
    *,
    preseason_slate_cap: int = 0,
) -> Dict[int, int]:
    """
    Map abstract schedule day IDs to concrete calendar indices.

    Regular-season abstract slates must land on regular-segment calendar days.
    Mapping them onto preseason days caused ~50 games to be simmed then wiped
    at the preseason→regular stats split (playoff_ready with a short season).

    Preseason calendar days remain for exhibition UI; they do not steal
    regular-season matchups unless an explicit positive preseason_slate_cap
    is passed by a dedicated exhibition path.
    """
    abstract_sorted = sorted(_dedupe_keep_order([_safe_int(x) for x in abstract_day_numbers]))

    if not abstract_sorted:
        return {}

    preseason_eligible = _indices_for_segment(calendar, "preseason")
    regular_eligible = _indices_for_segment(calendar, "regular")

    if not regular_eligible and preseason_eligible:
        regular_eligible = list(preseason_eligible)

    if not preseason_eligible and not regular_eligible:
        return {old: 0 for old in abstract_sorted}

    total = len(abstract_sorted)

    # Default: map ALL regular-season abstract slates onto the regular segment.
    preseason_count = 0
    if int(preseason_slate_cap) > 0:
        preseason_count = _estimate_preseason_abstract_count(
            total_abstract_count=total,
            preseason_slate_cap=preseason_slate_cap,
            preseason_eligible_count=len(preseason_eligible),
        )
        preseason_count = min(preseason_count, total)

    regular_count = max(0, total - preseason_count)

    preseason_targets = _build_preseason_mapping_targets(calendar, preseason_count)
    regular_targets = _build_regular_mapping_targets(calendar, regular_count)

    if preseason_count > 0 and not preseason_targets:
        regular_count = total
        preseason_count = 0
        regular_targets = _build_regular_mapping_targets(calendar, regular_count)

    if regular_count > 0 and not regular_targets:
        # Prefer regular days; only fall back to preseason if no regular days exist.
        if regular_eligible:
            regular_count = total
            preseason_count = 0
            regular_targets = _build_regular_mapping_targets(calendar, regular_count)
        else:
            preseason_count = total
            regular_count = 0
            preseason_targets = _build_preseason_mapping_targets(calendar, preseason_count)

    preseason_targets = _ensure_unique_mapping_targets(
        preseason_targets,
        preseason_eligible or regular_eligible,
    )
    regular_targets = _ensure_unique_mapping_targets(
        regular_targets,
        regular_eligible or preseason_eligible,
    )

    targets = preseason_targets + regular_targets
    targets.sort()

    if len(targets) < total:
        fallback_pool = regular_targets or regular_eligible or preseason_targets or preseason_eligible or [0]
        while len(targets) < total:
            targets.append(int(fallback_pool[-1]))

    if len(targets) > total:
        targets = targets[:total]

    # Final guard: one abstract slate day -> one unique calendar index.
    # Prefer regular-eligible days so regular matchups are not remapped onto preseason.
    combined_eligible = _dedupe_keep_order(
        list(regular_eligible) + list(preseason_eligible)
    )
    targets = _ensure_unique_mapping_targets(targets, combined_eligible)

    return {
        int(old): int(new_idx)
        for old, new_idx in zip(abstract_sorted, targets)
    }


# =============================================================================
# Optional diagnostics
# =============================================================================


def summarize_calendar(calendar: List[CalendarDay]) -> Dict[str, Any]:
    segment_counts: Dict[str, int] = {}
    playable_counts: Dict[str, int] = {}
    tag_counts: Dict[str, int] = {}

    for d in calendar:
        segment_counts[d.segment] = segment_counts.get(d.segment, 0) + 1

        if d.allows_games:
            playable_counts[d.segment] = playable_counts.get(d.segment, 0) + 1

        for tag in d.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    return {
        "days": len(calendar),
        "segment_counts": segment_counts,
        "playable_counts": playable_counts,
        "tag_counts": tag_counts,
        "first_day": calendar[0].iso if calendar else None,
        "last_day": calendar[-1].iso if calendar else None,
    }


def validate_calendar_mapping(
    calendar: List[CalendarDay],
    mapping: Dict[int, int],
) -> Dict[str, Any]:
    by_index = _calendar_by_index(calendar)

    invalid: List[Dict[str, Any]] = []
    segment_counts: Dict[str, int] = {}
    tag_counts: Dict[str, int] = {}
    duplicate_date_counts: Dict[int, int] = {}
    month_counts: Dict[str, int] = {}

    for abstract_day, calendar_idx in sorted(mapping.items()):
        calendar_idx = int(calendar_idx)
        day = by_index.get(calendar_idx)

        duplicate_date_counts[calendar_idx] = duplicate_date_counts.get(calendar_idx, 0) + 1

        if day is None:
            invalid.append(
                {
                    "abstract_day": abstract_day,
                    "calendar_idx": calendar_idx,
                    "reason": "calendar index does not exist",
                }
            )
            continue

        month_label = day.iso[:7]
        month_counts[month_label] = month_counts.get(month_label, 0) + 1

        if not day.allows_games:
            invalid.append(
                {
                    "abstract_day": abstract_day,
                    "calendar_idx": calendar_idx,
                    "iso": day.iso,
                    "segment": day.segment,
                    "tags": list(day.tags),
                    "reason": "mapped to non-game day",
                }
            )

        segment_counts[day.segment] = segment_counts.get(day.segment, 0) + 1

        for tag in day.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    busiest = sorted(
        [
            {
                "calendar_idx": idx,
                "iso": by_index[idx].iso if idx in by_index else None,
                "slate_count": count,
            }
            for idx, count in duplicate_date_counts.items()
        ],
        key=lambda row: row["slate_count"],
        reverse=True,
    )[:10]

    return {
        "mapped_count": len(mapping),
        "invalid_count": len(invalid),
        "invalid": invalid[:25],
        "segment_counts": segment_counts,
        "tag_counts": tag_counts,
        "month_counts": dict(sorted(month_counts.items())),
        "busiest_dates": busiest,
    }


def season_anchor_event_markers(season_start_year: int) -> List[Dict[str, Any]]:
    """
    League storytelling dates for the franchise calendar UI.

    These are display-only markers (injuries/trades still come from calendar_events).
    Dates follow the user's season anchor list for a hockey year starting in ``season_start_year``.
    """
    y = int(season_start_year)
    y1 = y + 1

    def mk(
        eid: str,
        iso: str,
        etype: str,
        title: str,
        priority: str = "MEDIUM",
        description: str = "",
    ) -> Dict[str, Any]:
        desc = str(description or title or "")
        return {
            "id": f"season_anchor_{y}_{eid}",
            "kind": "season_anchor",
            "type": etype,
            "event_type": etype,
            "date": iso,
            "calendar_iso": iso,
            "title": title,
            "headline": title,
            "summary": desc,
            "description": desc,
            "priority": str(priority or "MEDIUM").upper(),
        }

    return [
        mk("preseason_start", f"{y}-09-10", "preseason_start", "Preseason Begins", "MEDIUM"),
        mk("preseason_finale", f"{y}-09-30", "preseason_finale", "Preseason Finale", "MEDIUM"),
        mk("opening_night", f"{y}-10-08", "opening_night", "Opening Night", "HIGH"),
        mk(
            "home_opener_window",
            f"{y}-10-14",
            "home_opener",
            "Home Opener Window",
            "MEDIUM",
            "Typical NHL home opener cluster: Oct 8–20.",
        ),
        mk("thanksgiving", f"{y}-11-28", "thanksgiving_checkpoint", "Thanksgiving Checkpoint", "MEDIUM"),
        mk(
            "roster_freeze",
            f"{y}-12-19",
            "roster_freeze",
            "Holiday Roster Freeze",
            "MEDIUM",
            "Holiday roster freeze window: Dec 19–27.",
        ),
        mk("wjc_start", f"{y}-12-26", "wjc_start", "World Juniors Begin", "HIGH"),
        mk("wjc_semis", f"{y1}-01-04", "wjc_semifinals", "World Juniors Semifinals", "HIGH"),
        mk("wjc_final", f"{y1}-01-05", "wjc_final", "World Juniors Final", "HIGH"),
        mk("winter_classic", f"{y1}-01-01", "winter_classic", "Winter Classic", "HIGH"),
        mk("allstar_weekend", f"{y1}-02-01", "all_star_weekend", "All-Star Weekend", "HIGH", "All-Star Weekend: Feb 1–3."),
        mk("allstar_game", f"{y1}-02-03", "allstar_game", "All-Star Game", "HIGH"),
        mk(
            "four_nations_start",
            f"{y1}-02-10",
            "four_nations_tournament",
            "4 Nations / International Tournament Start",
            "CRITICAL",
        ),
        mk("heritage", f"{y1}-02-18", "heritage_classic", "Heritage Classic", "HIGH"),
        mk(
            "four_nations_final",
            f"{y1}-02-20",
            "four_nations_faceoff",
            "4 Nations / International Final",
            "CRITICAL",
        ),
        mk("stadium", f"{y1}-02-24", "stadium_series", "Stadium Series", "HIGH"),
        mk("trade_deadline", f"{y1}-03-07", "trade_deadline", "Trade Deadline", "CRITICAL"),
        mk("playoff_push", f"{y1}-03-20", "playoff_push", "Playoff Push Begins", "MEDIUM"),
        mk("reg_finale", f"{y1}-04-17", "regular_season_finale", "Regular Season Finale", "HIGH"),
        mk("playoffs_start", f"{y1}-04-20", "playoffs_start", "Stanley Cup Playoffs Begin", "CRITICAL"),
        mk("draft_lottery", f"{y1}-05-07", "draft_lottery", "Draft Lottery", "CRITICAL"),
        mk("ecf", f"{y1}-05-20", "conference_finals", "Conference Finals Begin", "HIGH"),
        mk("scf", f"{y1}-06-04", "stanley_cup_final", "Stanley Cup Final Begins", "CRITICAL"),
        mk("awards", f"{y1}-06-24", "nhl_awards", "NHL Awards", "MEDIUM"),
        mk("draft", f"{y1}-06-27", "nhl_draft", "NHL Draft", "CRITICAL", "NHL Draft: June 27–28."),
        mk("fa", f"{y1}-07-01", "free_agency", "Free Agency Opens", "CRITICAL"),
        mk(
            "dev_camp",
            f"{y1}-07-05",
            "development_camp",
            "Development Camp",
            "MEDIUM",
            "Development camp window: July 5–10.",
        ),
    ]
