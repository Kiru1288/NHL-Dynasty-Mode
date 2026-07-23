/**
 * Franchise lifecycle timeline — maps backend stages to permanent event folders.
 */

export const FRANCHISE_LIFECYCLE = [
  { id: "awards", label: "Awards Night", folder: "awardsNight", backendStage: "awards", status: "active" },
  { id: "retirements", label: "Retirements", folder: "retirements", backendStage: "retirements", status: "active" },
  { id: "hall_of_fame", label: "Hall of Fame", folder: "hallOfFame", backendStage: null, status: "planned" },
  { id: "salary_cap", label: "Cap Report", folder: "salaryCap", backendStage: "salary_cap", status: "active" },
  { id: "development_report", label: "Development Review", folder: "developmentReport", backendStage: "development_report", status: "active" },
  { id: "draft_lottery", label: "Draft Lottery", folder: "draftLottery", backendStage: "draft_lottery", status: "active" },
  { id: "draft_combine", label: "Draft Combine", folder: "draftCombine", backendStage: "draft_combine", status: "active" },
  { id: "draft", label: "NHL Draft", folder: "nhlDraft", backendStage: "draft", status: "active" },
  { id: "draft_review", label: "Draft Review", folder: "nhlDraft", backendStage: "draft_review", status: "active" },
  { id: "prospect_rights", label: "Prospect Rights", folder: "prospectDevelopment", backendStage: "prospect_rights", status: "active" },
  { id: "re_sign", label: "Re-Sign", folder: "reSign", backendStage: "re_sign", status: "active" },
  { id: "free_agency", label: "Free Agency", folder: "freeAgency", backendStage: "free_agency", status: "active" },
  { id: "roster_cleanup", label: "Roster Check", folder: "rosterCleanup", backendStage: "roster_cleanup", status: "active" },
  { id: "next_season", label: "New Season", folder: "nextSeasonReveal", backendStage: "next_season_reveal", status: "active" },
  { id: "training_camp", label: "Training Camp", folder: "trainingCamp", backendStage: null, status: "planned" },
  { id: "preseason", label: "Preseason", folder: "preseason", backendStage: "preseason", status: "active" },
  { id: "regular", label: "Regular Season", folder: null, backendStage: "regular", status: "active" },
];

export function resolveTimelineIndex(franchiseState) {
  const stage = String(franchiseState?.offseason_stage || "").toLowerCase();
  const phase = String(franchiseState?.phase || franchiseState?.season_phase || "").toLowerCase();

  if (phase === "post_cup" || stage === "awards") {
    return FRANCHISE_LIFECYCLE.findIndex((s) => s.id === "awards");
  }

  const byStage = FRANCHISE_LIFECYCLE.findIndex((s) => s.backendStage === stage);
  if (byStage >= 0) return byStage;

  if (phase === "preseason") {
    return FRANCHISE_LIFECYCLE.findIndex((s) => s.id === "preseason");
  }

  if (phase === "regular") {
    return FRANCHISE_LIFECYCLE.findIndex((s) => s.id === "regular");
  }

  return 0;
}

export function getTimelineSteps(franchiseState) {
  const currentIdx = resolveTimelineIndex(franchiseState);
  return FRANCHISE_LIFECYCLE.map((step, index) => ({
    ...step,
    state: index < currentIdx ? "done" : index === currentIdx ? "current" : "upcoming",
  }));
}
