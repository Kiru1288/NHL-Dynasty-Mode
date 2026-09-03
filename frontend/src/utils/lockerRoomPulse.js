/** Surface character / life / off-ice beats without opening a player dossier. */

function asObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function str(value) {
  return value == null ? "" : String(value).trim();
}

export function lookupHumanDossier(franchiseState, playerId) {
  const id = str(playerId);
  if (!id) return null;
  const universe = asObject(franchiseState?.narrative_universe);
  const direct = asObject(universe.human_dossiers);
  if (direct[id]) return direct[id];
  const row = asArray(universe.players).find((p) => str(p?.player_id) === id);
  return row?.human_dossier || null;
}

export function isPersonalLifeStory(raw) {
  const cat = str(raw?.category || raw?.type || "").toLowerCase();
  const cause = str(raw?.cause_type || "").toUpperCase();
  return (
    cat === "personal_life" ||
    cat.includes("life") ||
    cause.includes("LIFE_EVENT") ||
    cause === "POSITIVE_LIFE_EVENT" ||
    cause === "MINOR_LIFE_EVENT"
  );
}

export function playerCharacterChips(dossier) {
  const charBlock = asObject(dossier?.character);
  const tags = asArray(charBlock.descriptors)
    .map((t) => str(t))
    .filter(Boolean);
  const headline = str(charBlock.headline);
  if (headline && headline !== "Average") tags.unshift(headline);
  return tags.slice(0, 2);
}

export function playerRoomLine(dossier) {
  if (!dossier) return "";
  const life = str(asObject(dossier.life).summary);
  const morale = str(asObject(dossier.current_state).morale_tier);
  const summary = str(asObject(dossier.character).summary_line);
  const chips = playerCharacterChips(dossier);
  const bits = [];
  if (summary) bits.push(summary);
  else if (chips[0]) bits.push(chips[0]);
  if (life && life !== "Limited information") bits.push(life);
  if (morale && morale !== "Average") bits.push(`Morale: ${morale}`);
  return bits.slice(0, 3).join(" · ");
}

function rosterNameById(franchiseState) {
  const map = new Map();
  const orgs = asArray(franchiseState?.roster_browser?.organizations);
  orgs.forEach((org) => {
    asArray(org?.players).forEach((p) => {
      const id = str(p?.id || p?.player_id);
      const name = str(p?.name || p?.player_name);
      if (id && name) map.set(id, name);
    });
  });
  asArray(franchiseState?.roster).forEach((p) => {
    const id = str(p?.id || p?.player_id);
    const name = str(p?.name || p?.player_name);
    if (id && name) map.set(id, name);
  });
  return map;
}

export function collectLockerPulse(franchiseState, { limit = 8 } = {}) {
  const universe = asObject(franchiseState?.narrative_universe);
  const dossiers = asObject(universe.human_dossiers);
  const names = rosterNameById(franchiseState);
  const people = Object.entries(dossiers)
    .map(([key, dossier]) => {
      const ident = asObject(dossier?.identity);
      const playerId = str(dossier?.player_id || ident.player_id || key);
      const name = str(dossier?.player_name || ident.name || names.get(playerId) || "");
      if (!name) return null;
      return {
        playerId,
        name,
        position: str(ident.position || dossier?.position),
        line: playerRoomLine(dossier),
        chips: playerCharacterChips(dossier),
        life: str(asObject(dossier?.life).summary),
        morale: str(asObject(dossier?.current_state).morale_tier),
        character: str(asObject(dossier?.character).headline),
        pressure: str(asObject(dossier?.current_state).pressure_label),
      };
    })
    .filter(Boolean)
    .filter((row) => row.line || row.chips.length)
    .slice(0, 24);

  const lifeStories = asArray(franchiseState?.storyline_events)
    .filter(isPersonalLifeStory)
    .map((raw) => ({
      id: str(raw?.id || raw?.storyline_id),
      headline: str(raw?.headline || raw?.title),
      summary: str(raw?.summary || raw?.short_summary || raw?.description),
      playerName: str(raw?.player_name),
      teamId: str(raw?.team_id || raw?.team),
    }))
    .filter((row) => row.headline)
    .slice(-limit)
    .reverse();

  return { people, lifeStories };
}

export function isRoutineLeagueTrade(raw, userTeamId) {
  const headline = str(raw?.headline || raw?.title);
  if (!/\bacquires\b|\btraded to\b/i.test(headline)) return false;
  const uid = str(userTeamId);
  const tid = str(raw?.team_id || raw?.team);
  const related = asArray(raw?.related_teams || raw?.related_team_ids).map(str);
  if (uid && (tid === uid || related.includes(uid))) return false;
  return true;
}

export function storylineTickerLabel(raw) {
  const cat = str(raw?.category || raw?.type).toLowerCase();
  const cause = str(raw?.cause_type).toUpperCase();
  if (isPersonalLifeStory(raw) || cat.includes("life")) return "Off ice";
  if (cat.includes("locker") || cause.includes("LOCKER") || cause.includes("ROLE_FRUSTRATION")) return "Room";
  if (cat.includes("injur")) return "Injury";
  if (cat.includes("trade") || cat.includes("rumor")) return "Market";
  if (cat.includes("perform") || /HAT_TRICK|SHUTOUT|FIGHT/.test(cause)) return "Ice";
  return "League";
}

function tickerBucket(raw, userTeamId) {
  const cat = str(raw?.category || raw?.type).toLowerCase();
  const cause = str(raw?.cause_type).toUpperCase();
  if (isPersonalLifeStory(raw)) return "life";
  if (cat.includes("locker") || /LOCKER|ROLE_FRUSTRATION|WINNING_CONCERN|REPORTER/.test(cause)) return "room";
  if (["HAT_TRICK", "SHUTOUT", "TEAMMATE_FIGHT"].includes(cause) || cat === "performance" || cat.includes("injur")) {
    return "ice";
  }
  if (str(raw?.team_id || raw?.team) === str(userTeamId)) return "user";
  return "league";
}

/** Mixed league-wire items for the hub / newsroom tickers — not a trade dump. */
export function buildHubStoryTicker(franchiseState, { limit = 14 } = {}) {
  const uid = str(
    franchiseState?.user_team_id || franchiseState?.team?.id || franchiseState?.user_team?.id
  );
  const buckets = { life: [], room: [], ice: [], user: [], league: [] };
  const seen = new Set();

  const push = (raw, bucket) => {
    const headline = str(raw?.headline || raw?.title);
    if (!headline) return;
    if (isRoutineLeagueTrade(raw, uid)) return;
    const id = str(raw?.id || raw?.storyline_id || headline);
    if (!id || seen.has(id)) return;
    seen.add(id);
    buckets[bucket].push({
      id,
      headline,
      label: storylineTickerLabel(raw),
      teamId: str(raw?.team_id || raw?.team),
    });
  };

  asArray(franchiseState?.narrative_universe?.breaking_alerts).forEach((row) => {
    push(row, tickerBucket(row, uid));
  });

  asArray(franchiseState?.storyline_events)
    .slice(-180)
    .reverse()
    .forEach((raw) => {
      const heat = Number(raw?.heat) || 0;
      const level = str(raw?.breaking_level);
      const interesting =
        isPersonalLifeStory(raw) ||
        heat >= 28 ||
        Boolean(level) ||
        str(raw?.team_id) === uid;
      if (!interesting) return;
      push(raw, tickerBucket(raw, uid));
    });

  const out = [];
  const order = ["life", "ice", "room", "user", "league"];
  let round = 0;
  while (out.length < limit) {
    let added = false;
    for (const key of order) {
      const row = buckets[key][round];
      if (row) {
        out.push(row);
        added = true;
      }
      if (out.length >= limit) break;
    }
    if (!added) break;
    round += 1;
  }
  return out;
}
