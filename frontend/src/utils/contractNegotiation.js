/** Shared cap / offer math for Contract Office + offseason desks. */

export function computeOfferCapHitM(aav, years, signingBonus = 0) {
  const y = Math.max(1, Number(years) || 1);
  return Math.round(((Number(aav) || 0) * y + (Number(signingBonus) || 0)) / y * 1000) / 1000;
}

export function projectNegotiationCap({
  capSnapshot = {},
  nextYearProjection = {},
  playerRow = {},
  offerCapHitM,
}) {
  const usable = Number(capSnapshot.usable_cap_space_m);
  const totalHit = Number(capSnapshot.total_cap_hit_m);
  const upper = Number(capSnapshot.upper_limit_m);
  const upperNext =
    Number(capSnapshot.projected_next_year_upper_limit_m) ||
    Number(nextYearProjection.upperLimit) ||
    Number(nextYearProjection.upper_limit_m) ||
    (Number.isFinite(upper) ? upper * 1.03 : NaN);

  const currentHit = Number(playerRow.aav_m ?? playerRow.cap_hit_m ?? 0);
  const yrsLeft = Math.max(0, Number(playerRow.years_remaining ?? playerRow.yearsRemaining ?? 0));
  const offerHit = Number(offerCapHitM) || 0;

  if (!Number.isFinite(usable)) {
    return {
      capDeltaNowM: null,
      projectedAfterM: null,
      projectedNextSeasonM: null,
    };
  }

  const capDeltaNow = yrsLeft > 1 ? 0 : Math.max(0, offerHit - currentHit);
  const projectedAfter = usable - capDeltaNow;

  let projectedNextSeason = null;
  if (Number.isFinite(upperNext) && Number.isFinite(totalHit)) {
    if (yrsLeft <= 1) {
      projectedNextSeason = upperNext - (totalHit - currentHit + offerHit);
    } else {
      projectedNextSeason = upperNext - totalHit;
    }
  }

  return {
    capDeltaNowM: capDeltaNow,
    projectedAfterM: projectedAfter,
    projectedNextSeasonM: projectedNextSeason,
  };
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
  const pct = (aav - want) / want;
  const salary = 36 * Math.tanh(pct / 0.07);
  const termGap = Math.max(1, Number(offerYears) || 1) - Math.max(1, Number(wantYears) || 3);
  const term = 10 * Math.tanh(termGap * 0.55);
  const stay = Number.isFinite(Number(stayInterest)) ? (Number(stayInterest) - 50) * 0.22 : 0;
  const total = Number(totalValueM) || aav * Math.max(1, Number(offerYears) || 1);
  const bonusShare = total > 0 ? Math.max(0, Number(signingBonusM) || 0) / total : 0;
  const bonus = bonusShare > 0 ? Math.min(18, 52 * bonusShare) : 0;
  const clause = clauseInterestEstimate(preferredClause, ntcMode, nmc, securityPref);
  return Math.max(0, Math.min(100, 48 + salary + term + stay + bonus + clause));
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
