/** WJC payload normalisation — no UI, so the app shell can use it without loading the menu. */

export function asArray(value) {
  return Array.isArray(value) ? value : [];
}

export function isWjcPayload(value) {
  if (!value || typeof value !== "object") return false;

  return (
    value.kind === "wjc_tournament" ||
    value.wjc_live === true ||
    Boolean(value.wjc_phase)
  );
}

export function findActiveWjcPopup(franchiseState) {
  const popups = [
    ...asArray(franchiseState?.pending_ui_popups),
    ...asArray(franchiseState?.pendingUiPopups),
  ];

  return popups.find((popup) => popup && isWjcPayload(popup)) || null;
}

export function findArchivedWjc(franchiseState) {
  const archive = asArray(franchiseState?.showcase_archive);
  const seasonYear = Number(
    franchiseState?.season_year || franchiseState?.seasonYear || 0
  );

  for (let index = archive.length - 1; index >= 0; index -= 1) {
    const entry = archive[index];
    if (!isWjcPayload(entry)) continue;
    if (!seasonYear) return entry;
    const label = String(entry.season_label || "");
    // Prefer this season's archive; skip prior-year WJC desks.
    if (!label || label.startsWith(String(seasonYear))) return entry;
  }

  return null;
}

export function normalizePlayoffs(rawPlayoffs) {
  const playoffs =
    rawPlayoffs && typeof rawPlayoffs === "object" ? rawPlayoffs : {};

  return {
    quarterfinals: asArray(playoffs.quarterfinals),
    semifinals: asArray(playoffs.semifinals),
    bronze:
      playoffs.bronze && typeof playoffs.bronze === "object"
        ? playoffs.bronze
        : null,
    gold:
      playoffs.gold && typeof playoffs.gold === "object"
        ? playoffs.gold
        : null,
  };
}

export function normalizeWjcFields(raw) {
  const source = raw && typeof raw === "object" ? raw : {};

  const wjcDay = source.wjc_day ?? source.day ?? null;
  const wjcDaysTotal = source.wjc_days_total ?? source.days_total ?? 11;
  const rawPhase = String(source.wjc_phase || source.phase || "").toLowerCase();

  const medalsFinal = Boolean(
    source.medals_final ||
      rawPhase === "complete" ||
      (wjcDay != null && Number(wjcDay) >= Number(wjcDaysTotal))
  );

  return {
    wjc_phase: medalsFinal
      ? "complete"
      : rawPhase || (wjcDay != null ? "live" : ""),

    calendar_iso: String(source.calendar_iso || source.iso || ""),
    wjc_day: wjcDay != null ? Number(wjcDay) : null,
    wjc_days_total: Number(wjcDaysTotal) || 11,

    title: String(source.title || ""),
    season_label: String(source.season_label || ""),

    countries: asArray(source.countries),

    round_robin_games: asArray(source.round_robin_games),
    round_robin_total:
      Number(source.round_robin_total) ||
      asArray(source.round_robin_games).length ||
      0,

    standings: asArray(source.standings),
    playoffs: normalizePlayoffs(source.playoffs),

    medal_labels:
      source.medal_labels && typeof source.medal_labels === "object"
        ? { ...source.medal_labels }
        : {},

    medals_final: medalsFinal,

    user_prospects: asArray(source.user_prospects),
    tournament_prospects: asArray(source.tournament_prospects),
    player_stats: asArray(source.player_stats),

    all_games: asArray(source.all_games),
    games_today: asArray(source.games_today),

    all_games_total:
      Number(source.all_games_total) ||
      asArray(source.all_games).length ||
      0,

    rr_days_total: Number(source.rr_days_total) || 9,
  };
}

export function buildCalendarFallback(franchiseState) {
  const hud = franchiseState?.draft_class_hud?.events?.wjc || {};
  const anchors = asArray(franchiseState?.season_anchor_events);

  const wjcAnchor =
    anchors.find((anchor) =>
      String(anchor?.key || "").toLowerCase().includes("wjc_start")
    ) ||
    anchors.find((anchor) =>
      String(anchor?.type || anchor?.id || "")
        .toLowerCase()
        .includes("wjc")
    ) ||
    null;

  const countdown = hud.display || hud.date || "";
  const daysUntil = hud.days_until ?? hud.daysUntil ?? null;
  const startDate = hud.date || wjcAnchor?.date || "";
  const nations =
    asArray(franchiseState?.wjc_nations).length > 0
      ? asArray(franchiseState.wjc_nations)
      : asArray(franchiseState?.wjc_tournament?.countries);

  return {
    ...normalizeWjcFields({ countries: nations }),
    countdown_display: String(countdown || ""),
    countdown_days: daysUntil,
    start_date: String(startDate || ""),
    anchor_title: String(
      wjcAnchor?.title || wjcAnchor?.label || "World Juniors"
    ),
  };
}

export function wjcPayloadHasTournamentData(raw) {
  if (!raw || typeof raw !== "object") return false;
  const phase = String(raw.wjc_phase || raw.phase || "").toLowerCase();
  if (phase === "live" || phase === "complete") return true;
  if (raw.wjc_day != null || raw.medals_final) return true;
  if (asArray(raw.all_games).length > 0) return true;
  if (asArray(raw.player_stats).length > 0) return true;
  if (asArray(raw.round_robin_games).length > 0) return true;
  if (asArray(raw.standings).length > 0) return true;
  if (asArray(raw.tournament_prospects).length > 0) return true;
  return false;
}

export function isPreTournamentPayload(raw) {
  if (!raw || typeof raw !== "object") return true;
  const phase = String(raw.wjc_phase || raw.phase || "").toLowerCase();
  if (phase === "upcoming") return true;
  if (phase === "live" || phase === "complete") return false;
  if (raw.wjc_day != null || raw.medals_final) return false;
  if (asArray(raw.all_games).length > 0 || asArray(raw.player_stats).length > 0) {
    return false;
  }
  return true;
}

export function resolveWorldJuniorsPayload(franchiseState, eventData) {
  const emptyPayload = {
    source: "none",
    hasData: false,
    isPreTournament: true,
    raw: normalizeWjcFields({}),
    ...normalizeWjcFields({}),
    countdown_display: "",
    countdown_days: null,
    start_date: "",
    anchor_title: "World Juniors",
  };

  if (!franchiseState && !eventData) {
    return emptyPayload;
  }

  let source = "calendar";
  let rawPayload = null;

  const activePopup = findActiveWjcPopup(franchiseState);
  const stateTournament =
    franchiseState?.wjc_tournament && isWjcPayload(franchiseState.wjc_tournament)
      ? franchiseState.wjc_tournament
      : null;

  if (activePopup) {
    source = "live";
    rawPayload = activePopup;
  } else if (eventData && isWjcPayload(eventData) && wjcPayloadHasTournamentData(eventData)) {
    source = "eventData";
    rawPayload = eventData;
  } else if (stateTournament && wjcPayloadHasTournamentData(stateTournament)) {
    source = "state";
    rawPayload = stateTournament;
  } else {
    const archivedPayload = findArchivedWjc(franchiseState);

    if (archivedPayload) {
      source = "archive";
      rawPayload = archivedPayload;
    } else if (stateTournament) {
      source = "state";
      rawPayload = stateTournament;
    } else if (eventData && isWjcPayload(eventData)) {
      source = "eventData";
      rawPayload = eventData;
    }
  }

  const calendarMeta = buildCalendarFallback(franchiseState);

  if (rawPayload) {
    const normalized = normalizeWjcFields(rawPayload);
    const countries =
      asArray(normalized.countries).length > 0
        ? normalized.countries
        : calendarMeta.countries;
    const hasData = wjcPayloadHasTournamentData({ ...normalized, countries });
    const isPreTournament = isPreTournamentPayload(normalized);

    return {
      source,
      hasData,
      isPreTournament,
      raw: rawPayload,
      ...normalized,
      countries,
      countdown_display: calendarMeta.countdown_display,
      countdown_days: calendarMeta.countdown_days,
      start_date: calendarMeta.start_date,
      anchor_title: calendarMeta.anchor_title,
    };
  }

  const hasCountdown = Boolean(
    calendarMeta.countdown_display ||
      calendarMeta.countdown_days != null ||
      calendarMeta.start_date
  );

  return {
    source: hasCountdown ? "calendar" : "none",
    hasData: false,
    isPreTournament: true,
    raw: null,
    ...calendarMeta,
  };
}
