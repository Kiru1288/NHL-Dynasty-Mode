/**
 * Central resolver for franchise phase → cinematic event UI.
 */
import React from "react";
import { userMadePlayoffs } from "./playoffs/playoffUtils";

// Menus load on demand: statically importing them put every offseason/draft/playoff
// screen (tens of thousands of lines) into the boot bundle the hub waits on.
const lazyNamed = (loader, name) => React.lazy(() => loader().then((m) => ({ default: m[name] })));
const loadOffseason = () => import("./offseasonEventMenus");

const PlayoffStartEvent = React.lazy(() => import("./playoffs/PlayoffStartEvent"));
const AwardsNightEvent = React.lazy(() => import("./awardsNight/AwardsNightEvent"));
const EntryDraftMenu = React.lazy(() => import("./entryDraft/EntryDraftMenu"));
const BoardOfGovernorsMenu = React.lazy(() => import("./boardOfGovernors/BoardOfGovernorsMenu"));
const DraftCombineEvent = React.lazy(() => import("./draftCombine/DraftCombine"));
const OpeningNightMenu = React.lazy(() => import("./openingNight/OpeningNightMenu"));
const TradeDeadlineMenu = React.lazy(() => import("./tradeDeadline/TradeDeadlineMenu"));
const RetirementsEventMenu = lazyNamed(loadOffseason, "RetirementsEventMenu");
const DraftLotteryEventMenu = lazyNamed(loadOffseason, "DraftLotteryEventMenu");
const DraftReviewEventMenu = lazyNamed(loadOffseason, "DraftReviewEventMenu");
const ProspectRightsEventMenu = lazyNamed(loadOffseason, "ProspectRightsEventMenu");
const ReSignEventMenu = lazyNamed(loadOffseason, "ReSignEventMenu");
const FreeAgencyEventMenu = lazyNamed(loadOffseason, "FreeAgencyEventMenu");
const SalaryCapEventMenu = lazyNamed(loadOffseason, "SalaryCapEventMenu");
const DevelopmentReportEventMenu = lazyNamed(loadOffseason, "DevelopmentReportEventMenu");
const RosterCleanupEventMenu = lazyNamed(loadOffseason, "RosterCleanupEventMenu");
const NextSeasonRevealEventMenu = lazyNamed(loadOffseason, "NextSeasonRevealEventMenu");

const EVENT_MAP = {
  playoffs_start: {
    key: "playoffs_start",
    title: "Stanley Cup Playoffs",
    component: PlayoffStartEvent,
    ctaLabel: "Enter Playoffs",
    getEventData: (fs) => fs?.playoff_payload || fs?.playoffs || fs?.playoff_data || {},
  },
  awards: {
    key: "awards",
    title: "Awards Night",
    component: wrapMenu(AwardsNightEvent),
    ctaLabel: "Continue to Retirements",
    getEventData: (fs) => fs?.awards || {},
  },
  retirements: {
    key: "retirements",
    title: "Final Skate",
    component: wrapMenu(RetirementsEventMenu),
    ctaLabel: "Continue to Board of Governors",
    getEventData: (fs) => ({ retirements: fs?.retirements }),
  },
  board_of_governors: {
    key: "board_of_governors",
    title: "Board of Governors",
    component: wrapMenu(BoardOfGovernorsMenu),
    ctaLabel: "Continue to Salary Cap",
    getEventData: (fs) => ({ board_of_governors: fs?.board_of_governors }),
  },
  salary_cap: {
    key: "salary_cap",
    title: "Cap Report",
    component: wrapMenu(SalaryCapEventMenu),
    ctaLabel: "View Development",
    getEventData: (fs) => ({ salary_cap: fs?.salary_cap }),
  },
  development_report: {
    key: "development_report",
    title: "Prospect Development Review",
    component: wrapMenu(DevelopmentReportEventMenu),
    ctaLabel: "Continue to Lottery",
    getEventData: (fs) => ({ development_report: fs?.development_report }),
  },
  draft_lottery: {
    key: "draft_lottery",
    title: "Draft Lottery",
    component: wrapMenu(DraftLotteryEventMenu),
    ctaLabel: "Enter Combine",
    getEventData: (fs) => ({ draft_lottery: fs?.draft_lottery }),
  },
  draft_combine: {
    key: "draft_combine",
    title: "Draft Combine",
    component: wrapMenu(DraftCombineEvent),
    ctaLabel: "Enter Entry Draft",
    getEventData: (fs) => ({ draft_combine: fs?.draft_combine }),
  },
  draft: {
    key: "draft",
    title: "NHL Entry Draft",
    component: wrapMenu(EntryDraftMenu),
    ctaLabel: "Continue to Draft Review",
    getEventData: (fs) => ({ draft: fs?.draft, draft_board: fs?.draft_board }),
  },
  draft_review: {
    key: "draft_review",
    title: "Draft Review",
    component: wrapMenu(DraftReviewEventMenu),
    ctaLabel: "Open Prospect Rights",
    getEventData: (fs) => ({ draft_review: fs?.draft_review }),
  },
  prospect_rights: {
    key: "prospect_rights",
    title: "Prospect Rights",
    component: wrapMenu(ProspectRightsEventMenu),
    ctaLabel: "Continue to Re-Sign",
    getEventData: (fs) => ({ prospect_rights: fs?.prospect_rights }),
  },
  re_sign: {
    key: "re_sign",
    title: "Contract Table",
    component: wrapMenu(ReSignEventMenu),
    ctaLabel: "Open Free Agency",
    getEventData: (fs) => ({ contracts: fs?.contracts }),
  },
  free_agency: {
    key: "free_agency",
    title: "Market Opens",
    component: wrapMenu(FreeAgencyEventMenu),
    ctaLabel: "Generate Next Season",
    getEventData: (fs) => ({
      free_agents: fs?.free_agents,
      free_agency_market: fs?.free_agency_market,
    }),
  },
  roster_cleanup: {
    key: "roster_cleanup",
    title: "Roster Check",
    component: wrapMenu(RosterCleanupEventMenu),
    ctaLabel: "Generate Next Season",
    getEventData: (fs) => ({ roster_cleanup: fs?.roster_cleanup }),
  },
  next_season_reveal: {
    key: "next_season_reveal",
    title: "New Season",
    component: wrapMenu(NextSeasonRevealEventMenu),
    ctaLabel: "Enter Preseason",
    getEventData: (fs) => ({ next_season: fs?.next_season }),
  },
  opening_night: {
    key: "opening_night",
    title: "Opening Night",
    component: wrapMenu(OpeningNightMenu),
    ctaLabel: "Return to Hub",
    getEventData: (fs) => ({ opening_night: fs?.opening_night }),
  },
  trade_deadline: {
    key: "trade_deadline",
    title: "Trade Deadline",
    component: wrapMenu(TradeDeadlineMenu),
    ctaLabel: "Return to Hub",
    getEventData: (fs) => ({ trade_deadline: fs?.trade_deadline }),
  },
};

function wrapMenu(Menu) {
  return function WrappedEvent({ franchiseState, eventData, onContinue, onBack, onClose }) {
    return (
      <Menu
        franchiseState={franchiseState}
        eventData={eventData}
        onContinue={onContinue}
        onBack={onBack || onClose}
      />
    );
  };
}

function playoffsAreComplete(franchiseState) {
  return Boolean(
    franchiseState?.playoffs_done ||
      franchiseState?.flags?.playoffs_done ||
      franchiseState?.flags?.playoffs_simulated
  );
}

function resolveEventKey(franchiseState) {
  if (!franchiseState) return null;
  const phase = String(
    franchiseState.season_phase || franchiseState.phase || ""
  ).toLowerCase();
  const stage = String(franchiseState.offseason_stage || "").toLowerCase();
  const next = String(franchiseState.next_important_event || "").toLowerCase();

  if (phase === "post_cup") return "playoffs_start";
  if (phase === "offseason" && stage === "awards") return "awards";
  if (phase === "complete" && playoffsAreComplete(franchiseState)) return "playoffs_start";
  if (next === "awards" && phase === "offseason") return "awards";
  if (phase === "offseason" && stage && EVENT_MAP[stage]) return stage;
  if (phase === "offseason" && next && EVENT_MAP[next]) return next;
  if (phase === "playoffs" || phase === "playoff_ready" || next === "enter_playoffs" || next === "playoffs") {
    return "playoffs_start";
  }
  if (next && EVENT_MAP[next]) return next;

  return null;
}

export { userMadePlayoffs } from "./playoffs/playoffUtils";

export function getFranchisePhaseCta(franchiseState) {
  if (!franchiseState) return null;
  const phase = String(franchiseState.season_phase || franchiseState.phase || "").toLowerCase();
  const stage = String(franchiseState.offseason_stage || "").toLowerCase();
  const next = String(franchiseState.next_important_event || "").toLowerCase();

  if (phase === "post_cup") return "Continue to Awards";
  if (stage === "awards" || (phase === "offseason" && next === "awards")) {
    return "Resume Awards Night";
  }
  if (phase === "playoffs") return "Resume Playoff Bracket";
  if (phase === "playoff_ready" || next === "enter_playoffs") {
    return userMadePlayoffs(franchiseState) ? "Enter Playoffs" : "View Playoff Bracket";
  }
  if (phase === "offseason" && stage) {
    if (stage === "retirements") return "Resume Offseason Timeline";
    if (stage === "board_of_governors") return "Resume Board of Governors";
    if (stage === "salary_cap") return "Resume Offseason Timeline";
    if (stage === "development_report") return "Resume Offseason Timeline";
    if (stage === "draft_lottery") return "Resume Offseason Timeline";
    if (stage === "draft_combine") return "Resume Offseason Timeline";
    if (stage === "draft") {
      return franchiseState?.draft?.draft_completed
        ? "Resume Offseason Timeline"
        : "Resume Entry Draft";
    }
    if (stage === "draft_review") return "Resume Offseason Timeline";
    if (stage === "prospect_rights") return "Resume Offseason Timeline";
    if (stage === "re_sign") return "Resume Offseason Timeline";
    if (stage === "free_agency") return "Resume Offseason Timeline";
    if (stage === "free_agency" || next === "generate_next_season") return "Generate Next Season";
    if (stage === "next_season_reveal" || next === "preseason_start") return "Enter Preseason";
    return "Resume Offseason Timeline";
  }
  if (phase === "preseason" || phase === "regular") return "Advance Day";

  const ev = getCurrentFranchiseEvent(franchiseState);
  return ev?.ctaLabel || null;
}

export function getCurrentFranchiseEvent(franchiseState) {
  const key = resolveEventKey(franchiseState);
  if (!key || !EVENT_MAP[key]) return null;
  const spec = EVENT_MAP[key];
  return {
    key: spec.key,
    title: spec.title,
    component: spec.component,
    eventData: spec.getEventData ? spec.getEventData(franchiseState) : {},
    ctaLabel: spec.ctaLabel,
  };
}

export function registerFranchiseEventTypes(registerEventType) {
  if (typeof registerEventType !== "function") return;
  for (const [key, spec] of Object.entries(EVENT_MAP)) {
    registerEventType(key, spec);
  }
}

export { EVENT_MAP };
