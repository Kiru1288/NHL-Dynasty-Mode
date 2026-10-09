/** Shared cap / offer math for Contract Office + offseason desks. */

export function computeOfferCapHitM(aav, years, signingBonus = 0) {
  const y = Math.max(1, Number(years) || 1);
  return Math.round(((Number(aav) || 0) * y + (Number(signingBonus) || 0)) / y * 1000) / 1000;
}

function countedCurrentHit(capSnapshot, playerRow, replaceCurrentHit) {
  if (!replaceCurrentHit) return 0;
  const currentHit = Number(playerRow.aav_m ?? playerRow.cap_hit_m ?? playerRow.aav ?? 0) || 0;
  const inSeason = Boolean(capSnapshot.in_season_cap);
  const pending = Boolean(playerRow.pending_july1_expiry || playerRow.pendingJuly1Expiry);
  if (pending && !inSeason) return 0;
  return Math.max(0, currentHit);
}

export function projectNegotiationCap({
  capSnapshot = {},
  nextYearProjection = {},
  playerRow = {},
  offerCapHitM,
  replaceCurrentHit = true,
}) {
  const usable = Number(capSnapshot.usable_cap_space_m);
  const totalHit = Number(capSnapshot.total_cap_hit_m);
  const upper = Number(capSnapshot.upper_limit_m);
  const upperNext =
    Number(capSnapshot.projected_next_year_upper_limit_m) ||
    Number(nextYearProjection.upperLimit) ||
    Number(nextYearProjection.upper_limit_m) ||
    (Number.isFinite(upper) ? upper * 1.03 : NaN);

  const yrsLeft = Math.max(0, Number(playerRow.years_remaining ?? playerRow.yearsRemaining ?? 0));
  const offerHit = Number(offerCapHitM) || 0;
  const inSeason = Boolean(capSnapshot.in_season_cap);
  const pendingJuly = Boolean(playerRow.pending_july1_expiry || playerRow.pendingJuly1Expiry);
  const countedHit = countedCurrentHit(capSnapshot, playerRow, replaceCurrentHit);
  // Final-year extensions keep this year's AAV. The new cap hit starts next season.
  const dealStartsNextYear = replaceCurrentHit && yrsLeft >= 1 && !(pendingJuly && !inSeason);

  const empty = {
    capDeltaNowM: null,
    projectedAfterM: null,
    projectedNextSeasonM: null,
    nextYearRoomBeforeM: null,
    capUsedAfterM: null,
    capBarPct: null,
    inSeasonCap: inSeason,
    dealStartsNextYear,
  };

  if (!Number.isFinite(usable)) return empty;

  const capDeltaNow = dealStartsNextYear ? 0 : (inSeason || yrsLeft <= 1 || !replaceCurrentHit)
    ? Math.max(0, offerHit - countedHit)
    : 0;
  const projectedAfter = usable - capDeltaNow;
  const capUsedAfterM = Number.isFinite(totalHit)
    ? (dealStartsNextYear ? totalHit : totalHit - countedHit + offerHit)
    : null;

  let projectedNextSeason = null;
  const following = Number(capSnapshot.following_season_active_cap_hit_m);
  if (Number.isFinite(upperNext) && Number.isFinite(following)) {
    const deadNext =
      (Number(capSnapshot.buried_cap_hit_m) || 0) +
      (Number(capSnapshot.retained_salary_m) || 0) +
      (Number(capSnapshot.buyout_cap_hit_m) || 0) +
      (Number(capSnapshot.other_dead_cap_m) || 0);
    const booked = dealStartsNextYear ? Number(playerRow.extension_aav_m || 0) || 0 : 0;
    const nextOffer = dealStartsNextYear || !replaceCurrentHit ? offerHit : 0;
    projectedNextSeason = upperNext - (Math.max(0, following - booked) + nextOffer + deadNext);
  } else if (Number.isFinite(upperNext) && Number.isFinite(totalHit)) {
    const expiringRaw = Number(
      capSnapshot.expiring_roster_cap_hit_m ?? capSnapshot.expiringRosterCapHit,
    );
    const expiringHit = Number.isFinite(expiringRaw)
      ? Math.max(0, expiringRaw)
      : (yrsLeft <= 1 ? countedHit : 0);
    const nextOffer = yrsLeft <= 1 || !replaceCurrentHit ? offerHit : 0;
    projectedNextSeason = upperNext - (totalHit - expiringHit + nextOffer);
  }

  const nextOfferForRoom = dealStartsNextYear || !replaceCurrentHit ? offerHit : 0;
  const nextYearRoomBefore =
    projectedNextSeason != null && Number.isFinite(nextOfferForRoom)
      ? projectedNextSeason + nextOfferForRoom
      : null;

  const barLimit = dealStartsNextYear && Number.isFinite(upperNext) ? upperNext : upper;
  const barUsed = dealStartsNextYear && projectedNextSeason != null && Number.isFinite(barLimit)
    ? barLimit - projectedNextSeason
    : capUsedAfterM;
  const capBarPct =
    barUsed != null && Number.isFinite(barLimit) && barLimit > 0
      ? Math.max(0, Math.min(100, (barUsed / barLimit) * 100))
      : null;

  return {
    capDeltaNowM: capDeltaNow,
    projectedAfterM: projectedAfter,
    projectedNextSeasonM: projectedNextSeason,
    nextYearRoomBeforeM: nextYearRoomBefore,
    capUsedAfterM,
    capBarPct,
    inSeasonCap: inSeason,
    dealStartsNextYear,
  };
}

/** Upper bound for the AAV range slider (CBA max salary vs cap room for when the deal counts). */
export function maxNegotiationOfferAavM({
  capSnapshot = {},
  nextYearProjection = {},
  playerRow = {},
  cbaMaxSalaryM = 99,
  replaceCurrentHit = true,
}) {
  const cbaMax =
    Number.isFinite(Number(cbaMaxSalaryM)) && Number(cbaMaxSalaryM) > 0 ? Number(cbaMaxSalaryM) : 99;
  const capAtZero = projectNegotiationCap({
    capSnapshot,
    nextYearProjection,
    playerRow,
    offerCapHitM: 0,
    replaceCurrentHit,
  });
  const inSeason = Boolean(capSnapshot.in_season_cap);
  const yrsLeft = Math.max(0, Number(playerRow.years_remaining ?? playerRow.yearsRemaining ?? 0));
  const currentHit = Math.max(0, Number(playerRow.aav_m ?? playerRow.cap_hit_m ?? playerRow.aav ?? 0) || 0);
  const usable = Number(capSnapshot.usable_cap_space_m);

  let capRoomM = 99;
  if (capAtZero.dealStartsNextYear) {
    const before = capAtZero.nextYearRoomBeforeM;
    if (before != null && Number.isFinite(before)) {
      capRoomM = before;
    } else if (capAtZero.projectedNextSeasonM != null && Number.isFinite(capAtZero.projectedNextSeasonM)) {
      capRoomM = capAtZero.projectedNextSeasonM;
    } else {
      const snapNext = Number(capSnapshot.projected_cap_space_next_season_m);
      if (Number.isFinite(snapNext) && snapNext > 0) capRoomM = snapNext;
    }
  } else if (inSeason && Number.isFinite(usable)) {
    capRoomM = usable + (yrsLeft >= 1 ? currentHit : 0);
  } else if (Number.isFinite(usable)) {
    capRoomM = usable + currentHit;
  }

  return Math.min(cbaMax, Math.max(0.775, capRoomM));
}

export function interestMeterTone(interest) {
  const n = Number(interest) || 0;
  if (n >= 88) return "instant";
  if (n >= 62) return "good";
  if (n >= 40) return "mid";
  return "bad";
}

/** Client-side interest estimate while the agent preview is in flight (matches backend tanh curve). */
function protectionTier(ntcMode, nmc) {
  if (nmc) return 3;
  const mode = String(ntcMode || "NONE").toUpperCase();
  if (mode === "FULL") return 2;
  if (mode === "MODIFIED") return 1;
  return 0;
}

function preferredProtectionTier(preferredClause) {
  const pref = String(preferredClause || "None").toUpperCase().replace("_", "-");
  if (pref === "NMC") return 3;
  if (pref === "NTC") return 2;
  if (pref === "M-NTC" || pref === "MNTC") return 1;
  return 0;
}

function clauseInterestEstimate(preferredClause, ntcMode, nmc, security = 0.55) {
  const offer = protectionTier(ntcMode, nmc);
  const want = preferredProtectionTier(preferredClause);
  const sec = Math.max(0, Math.min(1, Number(security) || 0.55));
  if (want <= 0) {
    if (offer <= 0) return 0;
    const sweet = [0, 1.6, 2.8, 4.2][offer];
    return sweet + sec * (0.4 * offer);
  }
  const delta = offer - want;
  if (delta > 0) return 9 + delta * (3.5 + 2.5 * sec);
  if (delta === 0) return 10 + 4 * sec;
  return -Math.abs(delta) * (5.5 + 4 * sec);
}

export function estimateOfferInterestM({
  offerAavM,
  offerCapHitM,
  wantAavM,
  offerYears = 3,
  wantYears = 3,
  stayInterest,
  signingBonusM = 0,
  totalValueM,
  preferredClause = "None",
  ntcMode = "NONE",
  nmc = false,
  securityPref = 0.55,
}) {
  const want = Math.max(0.5, Number(wantAavM) || 3);
  const aav = Number(offerAavM) || 0;
  const gapM = aav - want;
  const scale = Math.max(2.5, want * 0.35);
  const salary = 46 * Math.tanh(gapM / scale);
  const termGap = Math.max(1, Number(offerYears) || 1) - Math.max(1, Number(wantYears) || 3);
  const term = 7 * Math.tanh(termGap * 0.45) * 0.65;
  const stayN = Number.isFinite(Number(stayInterest)) ? Number(stayInterest) : 50;
  let askAnchor = 56;
  if (stayN >= 70) askAnchor = 74;
  else if (stayN < 45) askAnchor = 50;
  const total = Number(totalValueM) || aav * Math.max(1, Number(offerYears) || 1);
  const bonusShare = total > 0 ? Math.max(0, Number(signingBonusM) || 0) / total : 0;
  const bonus = Math.min(4, 8 * bonusShare);
  const clause = clauseInterestEstimate(preferredClause, ntcMode, nmc, securityPref) * 0.3;
  const other = Math.max(-6, Math.min(6, term * 0.35 + bonus + clause));
  return Math.max(0, Math.min(100, askAnchor + salary + other));
}

export function resignInterestLabel(row) {
  if (!row) return "Medium";
  if (row.interest_label || row.interest_level) {
    return String(row.interest_label || row.interest_level);
  }
  if (row.stay_interest != null) {
    const n = Number(row.stay_interest);
    if (n >= 70) return "High";
    if (n < 45) return "Low";
    return "Medium";
  }
  const ovr = Number(row.overall ?? row.ovr ?? 0);
  if (ovr >= 88) return "High";
  if (ovr >= 78) return "Medium";
  return "Low";
}
