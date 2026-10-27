import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import EntryDraftMenu from "./EntryDraftMenu";

jest.mock("../../game/GameUIContext", () => ({
  useGameUI: () => ({ mergeFranchiseState: jest.fn() }),
}));

jest.mock("../../services/franchiseService", () => ({
  acceptEntryDraftTrade: jest.fn(),
  completeEntryDraft: jest.fn(),
  getDraftProspectProfile: jest.fn(() => Promise.resolve(null)),
  getEntryDraftState: jest.fn(),
  simEntryDraftToUserPick: jest.fn(),
  startEntryDraft: jest.fn(),
  submitCpuDraftPick: jest.fn(),
  submitDraftPick: jest.fn(),
}));

function userPick(overrides) {
  return {
    team_id: "OTT",
    team_name: "Ottawa Senators",
    position: "C",
    round: 1,
    development_path: "NCAA",
    scouting_confidence: 72,
    drafted_overall: 80,
    drafted_potential: 88,
    current_ovr_range: [74, 81],
    ...overrides,
  };
}

function franchise(draft) {
  return {
    user_team_id: "OTT",
    season_year: 2026,
    draft,
  };
}

const completedDraft = {
  draft_completed: true,
  draft_started: true,
  draft_year: 2027,
  draft_order: [
    { team_id: "OTT", team_name: "Ottawa Senators" },
    { team_id: "TOR", team_name: "Toronto Maple Leafs" },
  ],
  completed_picks: [
    userPick({
      prospect_id: "reach",
      prospect_name: "Ada Reach",
      overall_pick: 5,
      final_rank: 19,
      public_rank_at_pick: 19,
      public_rank_delta: -14,
      scouting_confidence: 30,
    }),
    userPick({
      prospect_id: "value",
      prospect_name: "Bea Value",
      position: "D",
      overall_pick: 40,
      round: 2,
      final_rank: 24,
      public_rank_at_pick: 24,
      public_rank_delta: 16,
      development_path: "Junior",
    }),
    userPick({
      prospect_id: "expected",
      prospect_name: "Cal Expected",
      overall_pick: 70,
      round: 3,
      final_rank: 62,
      public_rank_at_pick: 62,
      public_rank_delta: 8,
      development_path: "Europe",
    }),
    userPick({
      prospect_id: "edge-value",
      prospect_name: "Dee Edge",
      overall_pick: 12,
      final_rank: 2,
      public_rank_delta: 10,
    }),
    userPick({
      prospect_id: "edge-expected",
      prospect_name: "Eli Near",
      overall_pick: 13,
      final_rank: 4,
      public_rank_delta: 9,
    }),
  ],
  recap: {
    user_grade: "D",
    user_grade_score: 59,
    user_class_rank: "24th of 32",
    user_pick_count: 5,
    user_steal_count: 0,
    user_reach_count: 3,
    user_value_count: 9,
    user_need_count: 1,
    user_avg_value_delta: -10.7,
    user_position_breakdown: { C: 4, D: 1 },
    needs_report: [{ category: "Young NHL Depth", filled: false, detail: "Still thin up front" }],
    best_steals: [
      {
        team_id: "TOR",
        team_name: "Toronto Maple Leafs",
        prospect_id: "steal-1",
        prospect_name: "League Steal",
        overall_pick: 20,
        final_rank: 4,
        public_rank_delta: 16,
      },
    ],
    biggest_reaches: [
      {
        team_id: "TOR",
        team_name: "Toronto Maple Leafs",
        prospect_id: "gus",
        prospect_name: "Gus Reach",
        overall_pick: 8,
        final_rank: 30,
        public_rank_delta: -22,
      },
    ],
    headlines: [
      "Biggest steal: Toronto landed League Steal at pick 20.",
      "Biggest reach: Ottawa took Ada Reach early.",
      "Goalie run: 2 goalies came off the board in Round 2.",
    ],
    still_on_board: [
      { name: "Left Behind", projected_rank: 40, projected: "#40", status: "Undrafted" },
      { name: "Also Left", projected_rank: 55, status: "Undrafted" },
      { name: "Third Left", projected_rank: 60, status: "Undrafted" },
      { name: "Fourth Left", projected_rank: 70, status: "Undrafted" },
    ],
    best_user_pick: { prospect_id: "value" },
  },
};

describe("draft complete recap", () => {
  test("shows each fact once and keeps verdicts consistent", async () => {
    render(<EntryDraftMenu franchiseState={franchise(completedDraft)} onContinue={jest.fn()} onBack={jest.fn()} />);

    expect(screen.getByRole("heading", { name: "Entry Draft" })).toBeInTheDocument();
    expect(screen.getByText("Draft complete")).toBeInTheDocument();
    expect(document.querySelector(".cap-report-cap-display strong")).toHaveTextContent("D");
    expect(screen.getByText(/59\/100/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Draft class rank 24 of 32/)).toBeInTheDocument();
    expect(screen.getAllByText("Avg value")).toHaveLength(1);
    expect(screen.getAllByText("-10.7")).toHaveLength(1);
    expect(screen.getAllByText("Steals")).toHaveLength(1);
    expect(screen.getAllByText("Reaches")).toHaveLength(1);
    expect(screen.getAllByText("Ada Reach")).toHaveLength(1);
    expect(screen.getAllByText("Bea Value")).toHaveLength(1);

    expect(screen.getByText("REACH −14")).toBeInTheDocument();
    expect(screen.getByText("VALUE +16")).toBeInTheDocument();
    expect(screen.getByText("EXPECTED +8")).toBeInTheDocument();
    expect(screen.getByText("VALUE +10")).toBeInTheDocument();
    expect(screen.getByText("EXPECTED +9")).toBeInTheDocument();

    expect(screen.getAllByText(/NCAA|Junior|Europe/).length).toBeGreaterThan(0);
    expect(screen.getByText(/30% confidence/)).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/~/);
    expect(screen.getByTitle("Still thin up front")).toBeInTheDocument();
    expect(screen.getByText("Goalie run: 2 goalies came off the board in Round 2.")).toBeInTheDocument();

    expect(screen.queryByText(/Info allocation/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Class storylines/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Value adds/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Needs report/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Scout hits/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Biggest steal:/i)).not.toBeInTheDocument();
    expect(screen.queryByText("Yours")).not.toBeInTheDocument();

    expect(screen.getAllByTitle(/Public rank is the consensus board/).length).toBeGreaterThan(0);
    expect(screen.getByTitle("Public rank #19")).toBeInTheDocument();
    expect(screen.queryByRole("list", { name: "Drafted players by team" })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /League draft board/ }));
    expect(screen.getByRole("list", { name: "Drafted players by team" })).toBeInTheDocument();

    expect(screen.queryByText("Fourth Left")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "View all" }));
    expect(screen.getByText("Fourth Left")).toBeInTheDocument();

    expect(screen.getByRole("button", { name: /Continue offseason/i })).toBeInTheDocument();
    expect(document.querySelector(".edraft-haul-pick")).toBeInTheDocument();
  });

  test("shows an empty class when the user made no picks", () => {
    render(
      <EntryDraftMenu
        franchiseState={franchise({
          draft_completed: true,
          draft_started: true,
          draft_year: 2027,
          completed_picks: [],
          draft_order: [],
          recap: {
            user_grade: "—",
            user_pick_count: 0,
            user_steal_count: 0,
            user_reach_count: 0,
            user_avg_value_delta: 0,
            needs_report: [],
            best_steals: [],
            biggest_reaches: [],
            headlines: [],
            still_on_board: [],
          },
        })}
        onContinue={jest.fn()}
        onBack={jest.fn()}
      />
    );

    expect(screen.getByText(/did not make a selection/i)).toBeInTheDocument();
    expect(screen.queryByText(/Needs:/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /League draft board/ })).not.toBeInTheDocument();
  });
});
