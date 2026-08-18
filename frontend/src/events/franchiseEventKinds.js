/** Popup kinds handled by FranchiseEventOverlay — not ShowcasePopupLayer. */
export const FRANCHISE_CINEMATIC_POPUP_KINDS = new Set([
  "playoff_start",
  "playoffs_start",
  "awards",
  "awards_night",
  "retirements",
  "salary_cap",
  "development_report",
  "draft_lottery",
  "draft_combine",
  "draft",
  "draft_review",
  "prospect_rights",
  "entry_draft",
  "re_sign",
  "free_agency",
  "roster_cleanup",
  "next_season_reveal",
  "opening_night",
  "trade_deadline",
]);

export function isFranchiseCinematicPopup(popup) {
  if (!popup) return false;
  const kind = String(popup.kind || popup.type || "").toLowerCase();
  return FRANCHISE_CINEMATIC_POPUP_KINDS.has(kind);
}
