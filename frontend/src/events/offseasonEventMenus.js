import React from "react";
import CinematicEventShell from "./shared/CinematicEventShell";
import AwardsNight from "./awardsNight/AwardsNight";
import DraftLotteryNight from "./draftLottery/DraftLotteryNight";
import ProspectDevelopmentMenu from "./prospectDevelopment/ProspectDevelopmentMenu";
import RetirementsBoard from "./retirements/RetirementsBoard";
import CapReportBoard from "./salaryCap/CapReportBoard";
import {
  firstDefined,
  formatMoney,
  formatPick,
  getPlayerName,
  getPlayerOverall,
  getPlayerPosition,
  getTeamName,
  pickFranchiseData,
  safeArray,
} from "./shared/eventHelpers";

function seasonLabel(franchiseState) {
  const y = franchiseState?.season_year || franchiseState?.seasonYear;
  return y ? `${y}–${Number(y) + 1}` : "";
}

function Card({ prefix, rank, title, details, active }) {
  const p = prefix;
  return (
    <article className={`${p}-card ${active ? "is-active" : ""}`}>
      {rank != null ? <div className={`${p}-card-rank`}>{rank}</div> : null}
      <div className={`${p}-card-body`}>
        <strong>{title}</strong>
        <div className={`${p}-card-details`}>
          {safeArray(details).map((d, i) => (
            <span key={i}>{d}</span>
          ))}
        </div>
      </div>
    </article>
  );
}

export function AwardsEventMenu(props) {
  return <AwardsNight {...props} />;
}


export function RetirementsEventMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const raw = pickFranchiseData(franchiseState, eventData, ["retirements", "offseason.retirements"]);

  return (
    <RetirementsBoard
      franchiseState={franchiseState}
      retirees={raw}
      onContinue={onContinue}
      onBack={onBack}
    />
  );
}

export function DraftLotteryEventMenu(props) {
  return <DraftLotteryNight {...props} />;
}

export function DraftEventMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const prefix = "draft";
  const raw = pickFranchiseData(franchiseState, eventData, [
    "draft",
    "draft_board",
    "prospects",
    "offseason.draft",
  ]);
  const prospects = safeArray(raw?.prospects || raw?.board || raw);
  const userPick = raw?.current_pick || raw?.user_picks?.[0];
  const uid = String(franchiseState?.user_team_id || "");

  return (
    <CinematicEventShell
      prefix={prefix}
      phaseLabel="DRAFT FLOOR"
      seasonLabel={seasonLabel(franchiseState)}
      title="DRAFT FLOOR"
      eyebrow="Prospects on the board"
      ctaLabel="Continue to Re-Sign Phase"
      tickerItems={["DRAFT FLOOR", "ON THE CLOCK", "NEXT GENERATION", "FRANCHISE PICKS"]}
      onContinue={onContinue}
      onBack={onBack}
      heroContent={
        userPick ? (
          <>
            <p className={`${prefix}-hero-name`}>Your pick #{userPick.pick ?? "—"}</p>
            <div className={`${prefix}-meta`}>
              <span className={`${prefix}-chip`}>{userPick.team_name || "On the clock"}</span>
            </div>
          </>
        ) : (
          <p className={`${prefix}-empty`}>Draft board loading</p>
        )
      }
      railTitle="Prospect Board"
      railContent={
        prospects.length ? (
          prospects.slice(0, 24).map((p, i) => (
            <Card
              key={i}
              prefix={prefix}
              rank={formatPick(p.rank ?? p.overall ?? i + 1)}
              title={getPlayerName(p)}
              details={[
                getPlayerPosition(p),
                p.age != null ? `Age ${p.age}` : null,
                p.league || p.league_name,
                p.potential != null ? `Pot ${p.potential}` : null,
                p.scout_grade || p.grade,
                p.risk,
              ].filter(Boolean)}
            />
          ))
        ) : (
          <p className={`${prefix}-empty`}>No prospects on board</p>
        )
      }
    />
  );
}

export function ReSignEventMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const prefix = "resign";
  const raw = pickFranchiseData(franchiseState, eventData, [
    "contracts",
    "re_sign",
    "offseason.expiring_contracts",
    "pending_free_agents",
  ]) || {};
  const expiring = safeArray(raw?.expiring_contracts || raw?.expiring || raw);
  const summary = raw?.summary || {};
  const grouped = raw?.grouped || {};
  const ufa = safeArray(grouped.pending_ufa);
  const rfa = safeArray(grouped.pending_rfa);
  const cap = raw?.cap_snapshot || franchiseState?.team?.cap || franchiseState?.salary_cap;
  const warnings = safeArray(raw?.warning_reasons);
  const [filter, setFilter] = React.useState("all");
  const rows = filter === "ufa" ? ufa : filter === "rfa" ? rfa : expiring;

  return (
    <CinematicEventShell
      prefix={prefix}
      phaseLabel="CONTRACT TABLE"
      seasonLabel={seasonLabel(franchiseState)}
      title="CONTRACT TABLE"
      eyebrow="Organization-wide contract overview"
      ctaLabel="Open Free Agency"
      tickerItems={["CONTRACT TABLE", "CAP MATH", "CORE DECISIONS", "KEEP OR CUT"]}
      onContinue={onContinue}
      onBack={onBack}
      heroContent={
        <>
          <div className={`${prefix}-meta`}>
            <span className={`${prefix}-chip`}>Expiring {summary.expiringDeals ?? expiring.length}</span>
            <span className={`${prefix}-chip`}>UFA {summary.ufaCount ?? ufa.length}</span>
            <span className={`${prefix}-chip`}>RFA {summary.rfaCount ?? rfa.length}</span>
            <span className={`${prefix}-chip`}>Space {formatMoney(cap?.usable_cap_space_m ?? cap?.cap_space ?? franchiseState?.team?.cap_space)}</span>
            {raw?.contract_slots?.open != null ? (
              <span className={`${prefix}-chip`}>Slots {raw.contract_slots.open} open</span>
            ) : null}
          </div>
          {warnings.length ? <p className={`${prefix}-empty`}>{warnings[0]}</p> : null}
          <div className={`${prefix}-meta`} style={{ marginTop: 8 }}>
            {["all", "ufa", "rfa"].map((f) => (
              <button key={f} type="button" className={`${prefix}-chip`} onClick={() => setFilter(f)}>
                {f.toUpperCase()}
              </button>
            ))}
          </div>
        </>
      }
      railTitle="Pending Decisions"
      railContent={
        rows.length ? (
          rows.slice(0, 18).map((p, i) => (
            <Card
              key={p.player_id || i}
              prefix={prefix}
              title={getPlayerName(p)}
              details={[
                getPlayerPosition(p),
                getPlayerOverall(p) != null ? `OVR ${getPlayerOverall(p)}` : null,
                formatMoney(p.player_ask_aav_m ?? p.cap_hit ?? p.aav_m ?? p.aav),
                p.interest_label ? `Interest ${p.interest_label}` : null,
                p.clause_ask && p.clause_ask !== "None" ? p.clause_ask : null,
                p.rights || p.expiry_status || p.status,
              ].filter(Boolean)}
            />
          ))
        ) : (
          <p className={`${prefix}-empty`}>No pending contract decisions</p>
        )
      }
    />
  );
}

export function FreeAgencyEventMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const prefix = "fa";
  const market = pickFranchiseData(franchiseState, eventData, [
    "free_agency_market",
    "offseason.free_agency_market",
  ]) || franchiseState?.free_agency_market || {};
  const rows = safeArray(
    market.major_available?.length
      ? market.major_available
      : pickFranchiseData(franchiseState, eventData, ["free_agents", "offseason.free_agents"])
  );
  const bonus = market.signing_bonus || {};
  const recent = safeArray(market.recent_league_signings);

  return (
    <CinematicEventShell
      prefix={prefix}
      phaseLabel="MARKET OPENS"
      seasonLabel={seasonLabel(franchiseState)}
      title="MARKET OPENS"
      eyebrow="UFA / RFA market live"
      ctaLabel="Continue to Roster Cleanup"
      tickerItems={["MARKET OPENS", "BIDDING WARS", "NEW FACES", "BUILD YOUR ROSTER"]}
      onContinue={onContinue}
      onBack={onBack}
      heroContent={
        <>
          <div className={`${prefix}-meta`}>
            <span className={`${prefix}-chip`}>{market.market_status || "open"}</span>
            <span className={`${prefix}-chip`}>Available {market.available_count ?? rows.length}</span>
            <span className={`${prefix}-chip`}>Space {formatMoney(market.cap_space_m)}</span>
            <span className={`${prefix}-chip`}>
              Bonus {bonus.eligible ? `OK · ${Math.round((bonus.max_bonus_pct || 0) * 100)}%` : "Blocked <$110M"}
            </span>
            {market.pending_rfa_count ? (
              <span className={`${prefix}-chip`}>RFA pending {market.pending_rfa_count}</span>
            ) : null}
            {market.cpu_signings_count != null ? (
              <span className={`${prefix}-chip`}>CPU signs {market.cpu_signings_count}</span>
            ) : null}
          </div>
          <p className={`${prefix}-empty`}>Full board lives in Cap Ledger · Free Agency</p>
        </>
      }
      railTitle="Market Snapshot"
      railContent={
        <>
          {recent.slice(0, 4).map((s, i) => (
            <Card
              key={`r-${i}`}
              prefix={prefix}
              title={`${s.team_id || "CPU"} signed`}
              details={[s.player_id, formatMoney(s.aav_m), s.years != null ? `${s.years}y` : null].filter(Boolean)}
              active
            />
          ))}
          {rows.length ? (
            rows.slice(0, 10).map((p, i) => (
              <Card
                key={p.player_id || p.id || i}
                prefix={prefix}
                title={getPlayerName(p)}
                details={[
                  getPlayerPosition(p),
                  getPlayerOverall(p) != null ? `OVR ${getPlayerOverall(p)}` : null,
                  p.age != null ? `Age ${p.age}` : null,
                  formatMoney(p.askingAav ?? p.asking_price ?? p.ask_aav_m),
                ].filter(Boolean)}
              />
            ))
          ) : (
            <p className={`${prefix}-empty`}>Free agency not opened</p>
          )}
        </>
      }
    />
  );
}

export function SalaryCapEventMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  return (
    <CapReportBoard
      franchiseState={franchiseState}
      eventData={eventData}
      onContinue={onContinue}
      onBack={onBack}
    />
  );
}

export function DevelopmentReportEventMenu(props) {
  return <ProspectDevelopmentMenu {...props} />;
}

export function RosterCleanupEventMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const prefix = "cleanup";
  const raw = pickFranchiseData(franchiseState, eventData, [
    "roster_cleanup",
    "roster_issues",
    "offseason.roster_cleanup",
  ]) || {};
  const blocking = safeArray(raw.blocking?.length ? raw.blocking : (raw.issues || []).map((m) => ({ message: m })));
  const warnings = safeArray(raw.warnings?.length ? raw.warnings : (raw.warning_messages || []).map((m) => ({ message: m })));
  const canGenerate = Boolean(raw.valid ?? franchiseState?.flags?.can_generate_next_season);

  return (
    <CinematicEventShell
      prefix={prefix}
      phaseLabel="ROSTER CHECK"
      seasonLabel={seasonLabel(franchiseState)}
      title="ROSTER CHECK"
      eyebrow="Pre-season validation"
      ctaLabel={canGenerate ? "Generate Next Season" : "Resolve Blocking Issues"}
      tickerItems={["ROSTER CHECK", "COMPLIANCE", "FINAL REVIEW", "READY TO GO"]}
      onContinue={canGenerate ? onContinue : undefined}
      onBack={onBack}
      heroContent={
        <div className={`${prefix}-meta`}>
          <span className={`${prefix}-chip`}>NHL {raw.nhl_roster_count ?? "—"}</span>
          <span className={`${prefix}-chip`}>F {raw.forward_count ?? "—"}</span>
          <span className={`${prefix}-chip`}>D {raw.defense_count ?? "—"}</span>
          <span className={`${prefix}-chip`}>G {raw.goalie_count ?? "—"}</span>
          <span className={`${prefix}-chip`}>Space {formatMoney(raw.cap_space_m)}</span>
          <span className={`${prefix}-chip`}>
            Slots {raw.contract_slots_used ?? "—"}/{raw.contract_slots_limit ?? "—"}
          </span>
          <span className={`${prefix}-chip`}>{canGenerate ? "Ready" : "Blocked"}</span>
        </div>
      }
      railTitle="Compliance"
      railContent={
        <>
          {blocking.length ? (
            blocking.map((item, i) => (
              <Card
                key={`b-${i}`}
                prefix={prefix}
                title={String(item.message || item)}
                details={[item.route ? `Resolve in ${item.route}` : "Blocking"].filter(Boolean)}
                active
              />
            ))
          ) : null}
          {warnings.length ? (
            warnings.map((item, i) => (
              <Card
                key={`w-${i}`}
                prefix={prefix}
                title={String(item.message || item)}
                details={["Warning"]}
              />
            ))
          ) : null}
          {!blocking.length && !warnings.length ? (
            <p className={`${prefix}-empty`}>No roster issues flagged</p>
          ) : null}
          {!canGenerate ? (
            <p className={`${prefix}-empty`}>Generate stays locked until blocking issues clear.</p>
          ) : null}
        </>
      }
    />
  );
}

export function NextSeasonRevealEventMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const prefix = "nextseason";
  const raw = pickFranchiseData(franchiseState, eventData, [
    "next_season",
    "calendar_summary",
  ]) || {};
  const year = raw.season_year || franchiseState?.season_year;
  const markers = safeArray(raw.calendar_markers).slice(0, 6);

  return (
    <CinematicEventShell
      prefix={prefix}
      phaseLabel="NEW SEASON"
      seasonLabel={raw.season_label || (year ? `${year}–${Number(year) + 1}` : "")}
      title="NEW SEASON LOADING"
      eyebrow="Franchise continues"
      ctaLabel="Enter Preseason"
      tickerItems={["NEW YEAR", "CAMP OPENS", "THE CHASE RETURNS", "FRANCHISE CONTINUES"]}
      onContinue={onContinue}
      onBack={onBack}
      heroContent={
        <>
          <p className={`${prefix}-hero-name`}>{raw.season_label || year || "Next Season"}</p>
          <div className={`${prefix}-meta`}>
            {raw.opening_night ? <span className={`${prefix}-chip`}>Opening {raw.opening_night}</span> : null}
            {raw.preseason_start ? <span className={`${prefix}-chip`}>Camp {raw.preseason_start}</span> : null}
            {raw.first_opponent ? <span className={`${prefix}-chip`}>vs {raw.first_opponent}</span> : null}
            {raw.salary_cap_m != null ? <span className={`${prefix}-chip`}>Cap ${raw.salary_cap_m}M</span> : null}
            <span className={`${prefix}-chip`}>{raw.generation_status || "ready"}</span>
          </div>
        </>
      }
      railTitle="Season Preview"
      railContent={
        <>
          {raw.schedule_games != null ? (
            <Card prefix={prefix} title="Schedule Ready" details={[`${raw.schedule_games} games mapped`]} />
          ) : null}
          {markers.map((m, i) => (
            <Card
              key={m.key || i}
              prefix={prefix}
              title={m.label || m.key || "Marker"}
              details={[m.iso].filter(Boolean)}
            />
          ))}
          {!raw.schedule_games ? <p className={`${prefix}-empty`}>Generate next season to load calendar</p> : null}
        </>
      }
    />
  );
}

function draftReviewFirstId(picks) {
  const row = picks.find((p) => p?.prospect_id != null && String(p.prospect_id));
  return row ? String(row.prospect_id) : null;
}

function productionStatEntries(prod, isGoalie) {
  if (!prod || typeof prod !== "object" || prod.mode === "scouting") return [];
  if (isGoalie) {
    return [
      ["GP", prod.games],
      ["Starts", prod.starts],
      ["SV%", prod.save_percentage],
      ["GAA", prod.goals_against_average],
      ["SO", prod.shutouts],
    ].filter(([, v]) => v != null && v !== "");
  }
  return [
    ["GP", prod.games],
    ["G", prod.goals],
    ["A", prod.assists],
    ["PTS", prod.points],
    ["PPG", prod.points_per_game],
  ].filter(([, v]) => v != null && v !== "").slice(0, 5);
}

function findPickName(picks, id) {
  if (!id) return null;
  return picks.find((p) => String(p.prospect_id) === String(id))?.prospect_name || null;
}

export function DraftReviewEventMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const prefix = "draftreview";
  const raw = pickFranchiseData(franchiseState, eventData, ["draft_review", "offseason.draft_review"]) || {};
  const picks = safeArray(raw.user_picks);
  const haul = raw.haul_summary || {};
  const analysisChips = safeArray(haul.analysis_chips).filter(Boolean);
  const [selectedId, setSelectedId] = React.useState(() => draftReviewFirstId(picks));

  React.useEffect(() => {
    if (!picks.length) {
      setSelectedId(null);
      return;
    }
    const stillThere = picks.some((p) => String(p?.prospect_id) === String(selectedId));
    if (!stillThere) setSelectedId(draftReviewFirstId(picks));
  }, [picks, selectedId]);

  const pick =
    picks.find((p) => String(p?.prospect_id) === String(selectedId)) || picks[0] || null;
  const plan = pick?.development_plan || {};
  const pathSteps = safeArray(plan.path_steps);
  const production = pick?.production || {};
  const fit = pick?.organizational_fit || {};
  const rights = pick?.rights_card || {};
  const isGoalie = String(pick?.position || "").toUpperCase() === "G";
  const stats = productionStatEntries(production, isGoalie);
  const isScouting = production.mode === "scouting" || (!stats.length && (production.headline || production.notes));
  const scoutNotes = safeArray(production.notes);

  const closestName = findPickName(picks, haul.closest_to_nhl_pick_id);
  const bestValueName = findPickName(picks, haul.best_value_pick_id);

  const onRailKeyDown = (e, id) => {
    if (!picks.length) return;
    const ids = picks.map((p) => String(p.prospect_id));
    const idx = Math.max(0, ids.indexOf(String(selectedId)));
    if (e.key === "ArrowDown" || e.key === "ArrowRight") {
      e.preventDefault();
      setSelectedId(ids[Math.min(ids.length - 1, idx + 1)]);
    } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
      e.preventDefault();
      setSelectedId(ids[Math.max(0, idx - 1)]);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      setSelectedId(String(id));
    }
  };

  return (
    <CinematicEventShell
      prefix={prefix}
      phaseLabel="DRAFT REVIEW"
      seasonLabel={seasonLabel(franchiseState)}
      title="DRAFT REVIEW"
      eyebrow="Post-draft evaluation"
      ctaLabel="Open Prospect Rights"
      tickerItems={["DRAFT REVIEW", "VALUE & FIT", "DEVELOPMENT PLAN", "NEXT: RIGHTS"]}
      onContinue={onContinue}
      onBack={onBack}
      heroContent={
        pick ? (
          <div className={`${prefix}-workspace`}>
            <div className={`${prefix}-haul-block`}>
              <div className={`${prefix}-haul-strip`} aria-label="Haul summary">
                <span className={`${prefix}-haul-item`}>
                  <strong>{haul.total_picks ?? picks.length}</strong> picks
                </span>
                <span className={`${prefix}-haul-item is-gold`}>
                  <strong>{haul.haul_grade || raw.user_grade || "—"}</strong>
                  {haul.haul_grade_label || "haul"}
                </span>
                {haul.position_balance_label ? (
                  <span className={`${prefix}-haul-item`}>
                    <strong>Mix</strong>
                    {haul.position_balance_label}
                  </span>
                ) : null}
                {(safeArray(haul.needs_addressed)[0] || null) && (
                  <span className={`${prefix}-haul-item`}>
                    <strong>Need</strong>
                    {haul.needs_addressed[0]}
                  </span>
                )}
                {closestName ? (
                  <span className={`${prefix}-haul-item`}>
                    <strong>Closest</strong>
                    {closestName}
                  </span>
                ) : null}
                {bestValueName && bestValueName !== closestName ? (
                  <span className={`${prefix}-haul-item`}>
                    <strong>Best value</strong>
                    {bestValueName}
                  </span>
                ) : null}
                {haul.long_term_label ? (
                  <span className={`${prefix}-haul-item`}>
                    <strong>Timeline</strong>
                    {haul.long_term_label}
                  </span>
                ) : null}
              </div>
              {haul.haul_grade_reason ? (
                <p className={`${prefix}-haul-reason`}>{haul.haul_grade_reason}</p>
              ) : haul.summary_line ? (
                <p className={`${prefix}-haul-reason`}>{haul.summary_line}</p>
              ) : null}
              {analysisChips.length ? (
                <div className={`${prefix}-analysis-row`}>
                  {analysisChips.slice(0, 6).map((chip) => (
                    <span key={chip} className={`${prefix}-analysis-chip`}>
                      {chip}
                    </span>
                  ))}
                </div>
              ) : null}
            </div>

            <div className={`${prefix}-grid`}>
              <section className={`${prefix}-pane`}>
                <p className={`${prefix}-pane-label`}>Selected prospect</p>
                <p className={`${prefix}-pick-kicker`}>
                  #{pick.overall_pick ?? "—"} · Round {pick.round ?? "—"}
                  {pick.round_pick != null ? ` · Pick ${pick.round_pick}` : ""}
                </p>
                <h2 className={`${prefix}-pick-name`}>{pick.prospect_name || getPlayerName(pick)}</h2>
                <p className={`${prefix}-pick-sub`}>
                  {[
                    pick.position || getPlayerPosition(pick),
                    pick.archetype || pick.player_type,
                    pick.potential_label ? `Pot ${pick.potential_label}` : null,
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                </p>
                <p className={`${prefix}-pick-meta`}>
                  {[
                    pick.age != null ? `Age ${pick.age}` : null,
                    pick.height,
                    pick.weight != null ? `${pick.weight} lb` : null,
                    pick.shoots ? `Shoots ${pick.shoots}` : null,
                    pick.league,
                    pick.club,
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                </p>
                <div className={`${prefix}-grade-row`}>
                  {pick.selection_grade ? (
                    <span className={`${prefix}-grade-pill`}>
                      {pick.selection_grade}
                      {pick.selection_grade_label ? ` · ${pick.selection_grade_label}` : ""}
                    </span>
                  ) : null}
                  {pick.selection_verdict ? (
                    <span className={`${prefix}-verdict`}>{pick.selection_verdict}</span>
                  ) : null}
                  {pick.risk_level ? (
                    <span className={`${prefix}-risk`}>{pick.risk_level} risk</span>
                  ) : null}
                </div>
                {pick.selection_reason ? (
                  <p className={`${prefix}-context`}>{pick.selection_reason}</p>
                ) : null}
                {(pick.floor_label || pick.ceiling_label) && (
                  <p className={`${prefix}-context`}>
                    {[
                      pick.floor_label ? `Floor: ${pick.floor_label}` : null,
                      pick.ceiling_label ? `Ceiling: ${pick.ceiling_label}` : null,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </p>
                )}
                {pick.review_line ? <p className={`${prefix}-review-line`}>{pick.review_line}</p> : null}
              </section>

              <section className={`${prefix}-pane`}>
                <p className={`${prefix}-pane-label`}>Development blueprint</p>
                <p className={`${prefix}-dest-label`}>Next season</p>
                <p className={`${prefix}-dest-hero`}>
                  {plan.next_club || plan.next_destination || pick.league || "Development path"}
                </p>
                <p className={`${prefix}-pick-sub`}>
                  {[
                    plan.next_destination_label,
                    plan.next_destination && plan.next_destination !== plan.next_club
                      ? plan.next_destination
                      : null,
                  ]
                    .filter(Boolean)
                    .filter((v, i, arr) => arr.indexOf(v) === i)
                    .join(" · ")}
                </p>
                <div className={`${prefix}-plan-row`}>
                  <p className={`${prefix}-plan-kv`}>
                    <span>Role</span>
                    <strong>{plan.recommended_role || "Org prospect"}</strong>
                  </p>
                  <p className={`${prefix}-plan-kv`}>
                    <span>Deployment</span>
                    <strong>
                      {[plan.minutes_target, plan.special_teams_role].filter(Boolean).join(" · ") ||
                        "Standard minutes"}
                    </strong>
                  </p>
                  <p className={`${prefix}-plan-kv`}>
                    <span>ETA</span>
                    <strong>
                      {plan.eta_range || "3–5 years"}
                      {plan.eta_confidence ? ` · ${plan.eta_confidence}` : ""}
                    </strong>
                  </p>
                  <p className={`${prefix}-plan-kv`}>
                    <span>NHL outlook</span>
                    <strong>
                      {plan.nhl_projection || "Organizational depth"}
                      {plan.nhl_projection_confidence
                        ? ` · ${plan.nhl_projection_confidence}`
                        : ""}
                    </strong>
                  </p>
                </div>
                {plan.nhl_projection_reason ? (
                  <p className={`${prefix}-context`}>{plan.nhl_projection_reason}</p>
                ) : null}
                <p className={`${prefix}-plan-kv ${prefix}-obj`}>
                  <span>Season objective</span>
                  <strong>{plan.season_objective || "Develop tools"}</strong>
                </p>
                <div className={`${prefix}-path-timeline`} aria-label="Development path">
                  {pathSteps.length
                    ? pathSteps.slice(0, 3).map((step, i) => (
                        <React.Fragment key={`${step.stage}-${i}`}>
                          {i > 0 ? (
                            <span className={`${prefix}-path-arrow`} aria-hidden>
                              →
                            </span>
                          ) : null}
                          <div className={`${prefix}-path-step is-${step.status || "future"}`}>
                            <span className={`${prefix}-path-stage`}>{step.stage}</span>
                            <span className={`${prefix}-path-detail`}>{step.detail}</span>
                          </div>
                        </React.Fragment>
                      ))
                    : null}
                </div>
                {plan.alternate_path ? (
                  <p className={`${prefix}-alt-path`}>
                    <span>Alternate</span> {plan.alternate_path}
                  </p>
                ) : null}
              </section>

              <section className={`${prefix}-pane`}>
                <p className={`${prefix}-pane-label`}>
                  {isScouting ? "Scouting profile" : "Production"}
                </p>
                {stats.length ? (
                  <div className={`${prefix}-stat-row`}>
                    {stats.map(([label, value]) => (
                      <p key={label} className={`${prefix}-stat`}>
                        <span>{label}</span>
                        <strong>{value}</strong>
                      </p>
                    ))}
                  </div>
                ) : null}
                {isScouting ? (
                  <>
                    <p className={`${prefix}-scout-head`}>
                      {production.headline || "Scouting profile"}
                    </p>
                    <div className={`${prefix}-stat-row`}>
                      {production.floor_label ? (
                        <p className={`${prefix}-stat`}>
                          <span>Floor</span>
                          <strong>{production.floor_label}</strong>
                        </p>
                      ) : null}
                      {production.ceiling_label || production.potential_label ? (
                        <p className={`${prefix}-stat`}>
                          <span>Ceiling</span>
                          <strong>{production.ceiling_label || production.potential_label}</strong>
                        </p>
                      ) : null}
                      {production.scouting_confidence_label ? (
                        <p className={`${prefix}-stat`}>
                          <span>Confidence</span>
                          <strong>{production.scouting_confidence_label}</strong>
                        </p>
                      ) : null}
                      {production.risk_level ? (
                        <p className={`${prefix}-stat`}>
                          <span>Risk</span>
                          <strong>{production.risk_level}</strong>
                        </p>
                      ) : null}
                    </div>
                    {production.board_context ? (
                      <p className={`${prefix}-context`}>{production.board_context}</p>
                    ) : null}
                    {scoutNotes.length ? (
                      <ul className={`${prefix}-note-list`}>
                        {scoutNotes.slice(0, 3).map((note) => (
                          <li key={note}>{note}</li>
                        ))}
                      </ul>
                    ) : null}
                  </>
                ) : (
                  <p className={`${prefix}-context`}>
                    {[
                      production.league_context,
                      production.production_trend,
                      production.league,
                      production.potential_label ? `Ceiling ${production.potential_label}` : null,
                      production.data_confidence
                        ? `${production.data_confidence} confidence`
                        : null,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </p>
                )}
              </section>

              <section className={`${prefix}-pane`}>
                <p className={`${prefix}-pane-label`}>Org fit · Rights preview</p>
                <div className={`${prefix}-fit-grid`}>
                  <p className={`${prefix}-plan-kv`}>
                    <span>Need filled</span>
                    <strong>{fit.need_filled || "Organizational depth"}</strong>
                  </p>
                  <p className={`${prefix}-plan-kv`}>
                    <span>Fit</span>
                    <strong>
                      {[fit.fit_grade, fit.fit_label].filter(Boolean).join(" · ") || "—"}
                    </strong>
                  </p>
                  <p className={`${prefix}-plan-kv`}>
                    <span>Pipeline</span>
                    <strong>{fit.pipeline_label || `Rank #${fit.expected_pipeline_rank ?? "—"}`}</strong>
                  </p>
                  <p className={`${prefix}-plan-kv`}>
                    <span>Depth ahead</span>
                    <strong>
                      NHL {fit.nhl_players_ahead ?? 0} · AHL {fit.ahl_players_ahead ?? 0} · Pros{" "}
                      {fit.prospects_ahead ?? 0}
                    </strong>
                  </p>
                  <p className={`${prefix}-plan-kv`}>
                    <span>Environment</span>
                    <strong>{fit.environment_grade || "Acceptable environment"}</strong>
                  </p>
                  <p className={`${prefix}-plan-kv`}>
                    <span>Rights</span>
                    <strong>
                      {rights.rights_status_label || "Exclusive rights"}
                      {rights.signing_recommendation ? ` · ${rights.signing_recommendation}` : ""}
                    </strong>
                  </p>
                </div>
                {fit.fit_tension_note ? (
                  <p className={`${prefix}-context`}>{fit.fit_tension_note}</p>
                ) : fit.environment_reason ? (
                  <p className={`${prefix}-context`}>{fit.environment_reason}</p>
                ) : null}
                {rights.signing_reason ? (
                  <p className={`${prefix}-context`}>{rights.signing_reason}</p>
                ) : null}
                {rights.elc_can_slide || plan.elc_can_slide ? (
                  <p className={`${prefix}-context`}>ELC can slide · {rights.rights_deadline_label || "Rights window open"}</p>
                ) : (
                  <p className={`${prefix}-context`}>{rights.rights_deadline_label || "Rights window open"}</p>
                )}
              </section>
            </div>
          </div>
        ) : (
          <p className={`${prefix}-empty`}>{raw.headline || "No picks recorded for your club"}</p>
        )
      }
      railTitle="Your Haul"
      railContent={
        picks.length ? (
          picks.map((p) => {
            const id = String(p.prospect_id || "");
            const active = id && id === String(selectedId);
            const dest =
              p.development_plan?.next_club ||
              p.development_plan?.next_destination ||
              (typeof p.recommended_path === "string" &&
              !/sign|elc|rights|expire|camp/i.test(p.recommended_path)
                ? p.recommended_path
                : null) ||
              p.league ||
              "Development path";
            const eta = p.development_plan?.eta_range || null;
            const role = p.development_plan?.recommended_role || p.selection_verdict || null;
            return (
              <button
                key={id || p.overall_pick}
                type="button"
                className={`${prefix}-rail-btn`}
                onClick={() => setSelectedId(id)}
                onKeyDown={(e) => onRailKeyDown(e, id)}
                aria-pressed={active}
              >
                <Card
                  prefix={prefix}
                  rank={p.overall_pick}
                  title={p.prospect_name || getPlayerName(p)}
                  details={[
                    [
                      p.position || getPlayerPosition(p),
                      p.selection_grade,
                      p.selection_verdict,
                    ]
                      .filter(Boolean)
                      .join(" · "),
                    [dest, role].filter(Boolean).join(" · "),
                    [eta, p.risk_level ? `${p.risk_level} risk` : null].filter(Boolean).join(" · "),
                  ].filter(Boolean)}
                  active={active}
                />
              </button>
            );
          })
        ) : (
          <p className={`${prefix}-empty`}>No picks recorded for your club</p>
        )
      }
    />
  );
}

export function ProspectRightsEventMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const prefix = "prospectrights";
  const raw = pickFranchiseData(franchiseState, eventData, ["prospect_rights", "offseason.prospect_rights"]) || {};
  const prospects = safeArray(raw.prospects);
  const priority = safeArray(raw.recommended_signing_priority);
  const notes = safeArray(raw.notifications);
  const warnings = safeArray(raw.warning_reasons);
  const [selected, setSelected] = React.useState(0);
  const focus = prospects[selected] || prospects[0] || null;
  const actions = safeArray(focus?.available_actions).filter((a) => a?.enabled !== false);

  return (
    <CinematicEventShell
      prefix={prefix}
      phaseLabel="PROSPECT RIGHTS"
      seasonLabel={seasonLabel(franchiseState)}
      title="PROSPECT RIGHTS"
      eyebrow="Unsigned pipeline"
      ctaLabel="Continue to Re-Sign"
      tickerItems={["RIGHTS REVIEW", "ELC DECISIONS", "RESERVE LIST", "DEVELOPMENT PATHS"]}
      onContinue={onContinue}
      onBack={onBack}
      heroContent={
        focus ? (
          <>
            <p className={`${prefix}-hero-name`}>{focus.name || getPlayerName(focus)}</p>
            <div className={`${prefix}-meta`}>
              <span className={`${prefix}-chip`}>Contracts {raw.contracts || "—"}</span>
              <span className={`${prefix}-chip`}>ELC slots {raw.elc_slots_available ?? "—"}</span>
              <span className={`${prefix}-chip`}>{focus.rights_status || "rights"}</span>
              {focus.eta != null ? <span className={`${prefix}-chip`}>ETA {focus.eta}y</span> : null}
              {focus.development_environment?.grade ? (
                <span className={`${prefix}-chip`}>Env {focus.development_environment.grade}</span>
              ) : null}
            </div>
            <p className={`${prefix}-empty`}>
              {[
                focus.recommended_label || focus.recommended_action,
                focus.returning_to ? `Return ${focus.returning_to}` : null,
                focus.elc_slide_eligible ? "Slide eligible" : null,
                ...(focus.development_environment?.reasons || []).slice(0, 1),
              ].filter(Boolean).join(" · ")}
            </p>
            <div className={`${prefix}-meta`} style={{ marginTop: 8 }}>
              {actions.slice(0, 5).map((a) => (
                <span key={a.id} className={`${prefix}-chip`}>{a.label}</span>
              ))}
            </div>
            <p className={`${prefix}-empty`}>Sign ELCs in Cap Ledger · Contracts. Leaving does not auto-complete.</p>
            {warnings[0] ? <p className={`${prefix}-empty`}>{warnings[0]}</p> : null}
          </>
        ) : (
          <>
            <p className={`${prefix}-hero-name`}>Contracts {raw.contracts || "—"}</p>
            <div className={`${prefix}-meta`}>
              <span className={`${prefix}-chip`}>Reserve rights {raw.reserve_rights ?? "—"}</span>
              <span className={`${prefix}-chip`}>ELC slots {raw.elc_slots_available ?? "—"}</span>
            </div>
          </>
        )
      }
      railTitle="Org Prospects"
      railContent={
        <>
          {notes.slice(0, 2).map((n, i) => (
            <Card
              key={`n-${i}`}
              prefix={prefix}
              title={n.player_name || n.type || "Notice"}
              details={[n.message].filter(Boolean)}
              active
            />
          ))}
          {priority.length ? (
            <Card prefix={prefix} title={`${priority.length} priority signs`} details={["Rights nearing expiry"]} active />
          ) : null}
          {prospects.length ? (
            prospects.slice(0, 16).map((p, i) => (
              <button
                key={p.player_id || i}
                type="button"
                onClick={() => setSelected(i)}
                style={{ all: "unset", display: "block", width: "100%", cursor: "pointer" }}
              >
                <Card
                  prefix={prefix}
                  title={p.name || getPlayerName(p)}
                  details={[
                    p.position || getPlayerPosition(p),
                    p.rights_through != null ? `Through ${p.rights_through}` : p.rights_status,
                    p.returning_to ? `→ ${p.returning_to}` : null,
                    p.development_environment?.grade,
                    p.eta != null ? `ETA ${p.eta}y` : null,
                  ].filter(Boolean)}
                  active={i === selected}
                />
              </button>
            ))
          ) : (
            <p className={`${prefix}-empty`}>No unsigned org prospects on file</p>
          )}
        </>
      }
    />
  );
}
