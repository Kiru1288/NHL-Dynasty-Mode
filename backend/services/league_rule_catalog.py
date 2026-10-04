"""
Board of Governors rule catalog — 200 proposals a franchise GM can vote on.

Every effect key is consumed by a live system (see league_governance.EFFECT_SPECS for
the consumer of each key). Nothing here is decorative: a passed rule changes revenue,
costs, the cap, contracts, trades, scouting, injuries, franchise values, relocation or
expansion odds.

Rule fields
  id        stable id (never reuse)
  cat       category key (CATEGORIES)
  title     short name shown on the ballot
  summary   one sentence: what changes
  effects   {effect_key: value}; absolute keys (max_term_*) set a value, others add
  lean      {trait: weight} — how each kind of club leans (positive = toward YES)
  base      league-wide lean before traits (positive = popular with owners)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

CATEGORIES: Dict[str, Dict[str, Any]] = {
    "cap": {"label": "Salary Cap", "threshold": "two_thirds"},
    "contracts": {"label": "Contracts", "threshold": "two_thirds"},
    "revenue": {"label": "Revenue & Markets", "threshold": "two_thirds"},
    "trades": {"label": "Trades & Waivers", "threshold": "two_thirds"},
    "draft": {"label": "Draft & Development", "threshold": "two_thirds"},
    "player": {"label": "Player Rules", "threshold": "majority"},
    "game": {"label": "Game & Schedule", "threshold": "majority"},
    "uniform": {"label": "Uniforms & Presentation", "threshold": "majority"},
    "structure": {"label": "League Structure", "threshold": "three_quarters"},
}


def R(
    rid: str,
    cat: str,
    title: str,
    summary: str,
    effects: Dict[str, float],
    lean: Optional[Dict[str, float]] = None,
    base: float = 0.0,
) -> Dict[str, Any]:
    return {
        "id": rid,
        "cat": cat,
        "title": title,
        "summary": summary,
        "effects": dict(effects),
        "lean": dict(lean or {}),
        "base": float(base),
    }


RULES: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------ CAP (25)
    R("CAP01", "cap", "Cap Smoothing Mechanism", "Limits year-to-year cap swings by trimming model growth.", {"cap_growth": -0.5}, {"cap_tight": -0.15, "cap_room": 0.1, "small": 0.1}, 0.05),
    R("CAP02", "cap", "Accelerated Revenue Pass-Through", "Passes record revenue into the cap faster: +$2M next season and faster growth.", {"cap_growth": 0.8, "cap_adjust_m": 2.0}, {"cap_tight": 0.25, "poor": -0.2, "small": -0.1}, 0.0),
    R("CAP03", "cap", "Raise the Payroll Floor", "Lower limit rises by 2% of the upper limit.", {"floor_ratio": 0.02}, {"small": -0.25, "poor": -0.25, "large": 0.12, "rich": 0.1}, -0.05),
    R("CAP04", "cap", "Lower the Payroll Floor", "Lower limit drops by 3% of the upper limit.", {"floor_ratio": -0.03}, {"small": 0.25, "poor": 0.2, "large": -0.12}, 0.0),
    R("CAP05", "cap", "Max Salary to 22% of Cap", "Stars can earn up to 22% of the upper limit.", {"max_salary_pct": 0.02}, {"contender": 0.08, "rich": 0.05, "small": -0.1}, -0.08),
    R("CAP06", "cap", "Max Salary to 18% of Cap", "Caps any single player at 18% of the upper limit.", {"max_salary_pct": -0.02}, {"small": 0.15, "poor": 0.1, "large": -0.05}, 0.04),
    R("CAP07", "cap", "Media Deal Cap Bump", "One-time $3M bump to next season's upper limit.", {"cap_adjust_m": 3.0}, {"cap_tight": 0.28, "poor": -0.15}, 0.05),
    R("CAP08", "cap", "Escrow Buffer Hold-Back", "Holds $1.5M of next season's cap back as an escrow buffer.", {"cap_adjust_m": -1.5}, {"poor": 0.2, "small": 0.1, "cap_tight": -0.28}, 0.0),
    R("CAP09", "cap", "Full HRR Indexing", "Cap growth tracks hockey-related revenue one-for-one.", {"cap_growth": 0.5}, {"rich": 0.1, "poor": -0.1, "cap_tight": 0.1}, 0.05),
    R("CAP10", "cap", "Cap Freeze Year", "Freezes growth: $2.5M off next season and slower growth after.", {"cap_adjust_m": -2.5, "cap_growth": -0.3}, {"poor": 0.25, "cap_tight": -0.3, "contender": -0.1}, -0.15),
    R("CAP11", "cap", "Floor Relief for Money-Losing Clubs", "Lowers the floor slightly and adds 0.5% to the revenue-sharing pool.", {"floor_ratio": -0.015, "revenue_share": 0.5}, {"poor": 0.25, "rich": -0.1}, 0.0),
    R("CAP12", "cap", "Payroll Luxury Levy", "Top-revenue clubs pay a levy into revenue sharing.", {"revenue_share": 1.0, "rev_large": -0.5}, {"large": -0.32, "small": 0.25}, -0.05),
    R("CAP13", "cap", "Dead-Cap Relief Pool", "Adds $0.5M of cap room league-wide to offset buyouts.", {"cap_adjust_m": 0.5}, {"cap_tight": 0.2}, 0.08),
    R("CAP14", "cap", "Bonus Cushion Restored", "Bonus overage cushion returns: +$1M cap and more bonus room.", {"cap_adjust_m": 1.0, "bonus_pct": 0.02}, {"cap_tight": 0.2, "rich": 0.1}, 0.0),
    R("CAP15", "cap", "Narrow the Payroll Range", "Floor up 3%, max salary down 1% — tighter competitive balance.", {"floor_ratio": 0.03, "max_salary_pct": -0.01}, {"small": -0.2, "large": 0.05, "poor": -0.15}, -0.1),
    R("CAP16", "cap", "Two-Year Cap Guarantee", "League guarantees cap projections two years out.", {"cap_growth": 0.2, "fan": 0.2}, {}, 0.15),
    R("CAP17", "cap", "Floor Penalty Waiver", "Clubs under the floor get a one-season grace period.", {"floor_ratio": -0.01}, {"poor": 0.15, "small": 0.08}, 0.05),
    R("CAP18", "cap", "Windfall Pass-Through", "New media money hits the cap: +$2M and faster growth.", {"cap_adjust_m": 2.0, "cap_growth": 0.3}, {"cap_tight": 0.2, "poor": -0.1}, 0.05),
    R("CAP19", "cap", "6% Cap Growth Ceiling", "Caps annual cap growth to protect small-market budgets.", {"cap_growth": -0.4}, {"poor": 0.15, "small": 0.1, "cap_tight": -0.15}, 0.0),
    R("CAP20", "cap", "Rookie Bonus Cap Exclusion", "Entry-level bonuses no longer count: +$0.75M effective cap.", {"cap_adjust_m": 0.75}, {"rebuilder": 0.2}, 0.05),
    R("CAP21", "cap", "Bonus Overage Forgiveness", "Performance-bonus overages are forgiven: +$0.5M cap.", {"cap_adjust_m": 0.5}, {"contender": 0.1, "rebuilder": 0.1}, 0.1),
    R("CAP22", "cap", "New Owner Floor Phase-In", "New owners phase into the floor; fewer forced sales.", {"floor_ratio": -0.005, "relocation_ease": -0.03}, {"poor": 0.1}, 0.08),
    R("CAP23", "cap", "Median-Linked Floor", "Floor tracks median payroll; sharing pool grows 0.3%.", {"floor_ratio": 0.01, "revenue_share": 0.3}, {"small": 0.05, "poor": -0.1}, 0.0),
    R("CAP24", "cap", "Round Upper Limit Up", "Upper limit rounds up to the next $0.5M.", {"cap_adjust_m": 0.4}, {}, 0.2),
    R("CAP25", "cap", "Escrow Ceiling at 6%", "Owners absorb escrow above 6%: higher operating costs.", {"opex": 0.5}, {"all": -0.1, "rich": 0.05}, -0.05),

    # ------------------------------------------------------------ CONTRACTS (30)
    R("CON01", "contracts", "8-Year Max for Own Players", "Re-signing your own players allows 8-year terms again.", {"max_term_own": 8}, {"contender": 0.1, "rebuilder": 0.05, "large": 0.05}, 0.0),
    R("CON02", "contracts", "6-Year Max for Own Players", "Re-signing terms capped at 6 years.", {"max_term_own": 6}, {"poor": 0.1, "small": 0.05}, -0.05),
    R("CON03", "contracts", "7-Year Max for Free Agents", "UFAs may sign 7-year deals.", {"max_term_ufa": 7}, {"rich": 0.1, "large": 0.15, "small": -0.1}, -0.05),
    R("CON04", "contracts", "5-Year Max for Free Agents", "UFA terms capped at 5 years.", {"max_term_ufa": 5}, {"small": 0.15, "poor": 0.1, "large": -0.1}, 0.0),
    R("CON05", "contracts", "Signing Bonus Limit", "Signing bonuses capped well below current revenue-based room.", {"bonus_pct": -0.06}, {"small": 0.25, "poor": 0.2, "large": -0.2, "rich": -0.15}, 0.0),
    R("CON06", "contracts", "Uncapped Signing Bonuses", "Rich clubs can front-load far more cash as signing bonuses.", {"bonus_pct": 0.10}, {"large": 0.25, "rich": 0.2, "small": -0.3, "poor": -0.2}, -0.1),
    R("CON07", "contracts", "Lower Bonus Revenue Floor", "Clubs need $15M less revenue to offer signing bonuses.", {"bonus_floor_m": -15.0}, {"small": 0.25, "medium": 0.1, "large": -0.05}, 0.0),
    R("CON08", "contracts", "Raise Bonus Revenue Floor", "Clubs need $15M more revenue to offer signing bonuses.", {"bonus_floor_m": 15.0}, {"large": 0.2, "small": -0.25}, -0.05),
    R("CON09", "contracts", "League Minimum to $1.0M", "League minimum salary rises by $150K.", {"min_salary_m": 0.15}, {"all": -0.12, "cap_room": 0.05}, 0.0),
    R("CON10", "contracts", "Minimum Salary Freeze", "League minimum held $50K lower for now.", {"min_salary_m": -0.05}, {"poor": 0.1}, 0.05),
    R("CON11", "contracts", "Star Bonus Market", "More star free agents insist on signing bonuses (+10%).", {"fa_bonus_demand": 10.0}, {"large": 0.2, "rich": 0.15, "small": -0.2}, -0.05),
    R("CON12", "contracts", "Agent Bonus Limits", "Fewer free agents can demand signing bonuses (-10%).", {"fa_bonus_demand": -10.0}, {"small": 0.2, "poor": 0.15, "large": -0.05}, 0.0),
    R("CON13", "contracts", "15% Salary Variance Rule", "Year-to-year salary swings limited to 15%: less bonus room.", {"bonus_pct": -0.03}, {"small": 0.1}, 0.05),
    R("CON14", "contracts", "Back-Diving Enforcement", "Investigates back-loaded deals; bonus room trimmed.", {"bonus_pct": -0.02, "fan": 0.2}, {}, 0.1),
    R("CON15", "contracts", "Entry-Level Bonus Cap", "Caps ELC bonuses: $0.25M more usable cap for everyone.", {"cap_adjust_m": 0.25}, {"rebuilder": 0.1}, 0.05),
    R("CON16", "contracts", "Veteran Minimum Exception", "35+ veterans can sign below the league minimum.", {"min_salary_m": -0.03}, {"contender": 0.1}, 0.05),
    R("CON17", "contracts", "Seven-Day Interview Window", "Pending UFAs talk to all clubs a week early; more bonus asks.", {"fa_bonus_demand": 3.0, "fan": 0.3}, {"large": 0.1, "small": -0.05}, 0.05),
    R("CON18", "contracts", "Extended Exclusive Re-Sign Window", "Clubs get longer to re-sign their own; fewer bonus asks.", {"fa_bonus_demand": -3.0}, {"small": 0.1}, 0.08),
    R("CON19", "contracts", "Restore 8/7 Term Limits", "Returns to 8 years for own players and 7 for UFAs.", {"max_term_own": 8, "max_term_ufa": 7}, {"large": 0.1, "rich": 0.05, "small": -0.05}, -0.05),
    R("CON20", "contracts", "No-Move Clause Limits", "NMCs limited to 8-year veterans; fewer forced trade sagas.", {"trade_demand_rate": -10.0}, {"all": 0.05}, 0.05),
    R("CON21", "contracts", "No-Trade Clause Expansion", "More players earn NTCs; more trade standoffs.", {"trade_demand_rate": 10.0}, {"all": -0.1}, -0.05),
    R("CON22", "contracts", "Bonus Escrow Accounts", "Bonuses paid into escrow: less room, lower carrying costs.", {"bonus_pct": -0.02, "opex": -0.3}, {"poor": 0.1}, 0.05),
    R("CON23", "contracts", "Performance Bonus Reduction", "Smaller performance bonuses free $0.3M of cap.", {"cap_adjust_m": 0.3}, {}, 0.1),
    R("CON24", "contracts", "Long-Term Deal Insurance Pool", "League insures long deals; owners pay premiums.", {"opex": 0.4}, {"small": 0.1, "rich": -0.05}, -0.02),
    R("CON25", "contracts", "Concussion Salary Guarantee", "Salaries fully guaranteed after concussions.", {"opex": 0.6, "fan": 0.4}, {"all": -0.08}, 0.0),
    R("CON26", "contracts", "Ten-Day Bonus Payment Rule", "Bonuses paid within 10 days: $5M higher revenue bar.", {"bonus_floor_m": 5.0}, {"large": 0.1, "small": -0.1}, 0.0),
    R("CON27", "contracts", "21% Max for Re-Signed Stars", "Re-signed stars can earn 21% of the cap.", {"max_salary_pct": 0.01}, {"contender": 0.1, "small": -0.05}, 0.0),
    R("CON28", "contracts", "Small-Market Bonus Allowance", "Small markets can offer bonuses at $8M less revenue.", {"bonus_floor_m": -8.0, "rev_small": 0.5}, {"small": 0.28, "large": -0.15}, 0.0),
    R("CON29", "contracts", "Mandatory Bonus Disclosure", "All bonuses published; fewer agents chase them.", {"fa_bonus_demand": -4.0, "fan": 0.2}, {}, 0.08),
    R("CON30", "contracts", "Six-Year Term for Everyone", "All contracts capped at 6 years.", {"max_term_own": 6, "max_term_ufa": 6}, {"small": 0.1, "poor": 0.05, "large": -0.08}, -0.1),

    # -------------------------------------------------------------- REVENUE (25)
    R("REV01", "revenue", "Expanded Revenue Sharing", "Two more points of top-market revenue go to the sharing pool.", {"revenue_share": 2.0}, {"small": 0.35, "large": -0.35, "poor": 0.2, "rich": -0.15}, 0.0),
    R("REV02", "revenue", "Reduced Revenue Sharing", "Sharing pool shrinks by 1.5 points.", {"revenue_share": -1.5}, {"large": 0.35, "small": -0.35, "rich": 0.1}, -0.05),
    R("REV03", "revenue", "Jersey Advertisement Patches", "Sponsor patches on game jerseys: +1.5% revenue, fans grumble.", {"rev_all": 1.5, "fan": -0.8}, {"poor": 0.15}, 0.15),
    R("REV04", "revenue", "Helmet Sponsor Decals", "Sponsor decals on helmets.", {"rev_all": 0.8, "fan": -0.4}, {}, 0.15),
    R("REV05", "revenue", "League-Wide Dynamic Pricing", "Ticket prices float with demand.", {"rev_all": 1.2, "fan": -1.0}, {"large": 0.1}, 0.05),
    R("REV06", "revenue", "Ticket Price Freeze", "Ticket prices frozen for a season.", {"rev_all": -1.0, "fan": 1.5}, {"poor": -0.2, "rich": 0.05}, -0.12),
    R("REV07", "revenue", "Star Marketing Fund", "League markets its stars: star revenue +15%.", {"star_rev": 15.0}, {"large": 0.1, "small": 0.05}, 0.05),
    R("REV08", "revenue", "Equal National TV Split", "National TV split evenly: small markets +2%, large -1%.", {"rev_small": 2.0, "rev_large": -1.0}, {"small": 0.3, "large": -0.25}, 0.0),
    R("REV09", "revenue", "Local Media Rights Pooling", "Local TV partly pooled into revenue sharing.", {"revenue_share": 1.0, "rev_large": -0.5}, {"small": 0.2, "large": -0.25}, 0.0),
    R("REV10", "revenue", "Sports Betting Partnerships", "League-wide betting partners: +2% revenue.", {"rev_all": 2.0, "fan": -0.5}, {}, 0.1),
    R("REV11", "revenue", "Arena Naming Rights Deregulation", "No cap on naming-rights deals: big markets cash in.", {"rev_large": 1.5}, {"large": 0.2, "small": -0.05}, 0.0),
    R("REV12", "revenue", "Playoff Gate Sharing", "20% of playoff gate goes into revenue sharing.", {"playoff_rev": -20.0, "revenue_share": 0.5}, {"contender": -0.25, "rebuilder": 0.15, "small": 0.1}, 0.0),
    R("REV13", "revenue", "Playoff Revenue Bonus Pool", "Playoff revenue +25% for teams that advance.", {"playoff_rev": 25.0}, {"contender": 0.2, "rebuilder": -0.1}, 0.0),
    R("REV14", "revenue", "Streaming Package Expansion", "Out-of-market streaming bundle: +1% revenue.", {"rev_all": 1.0, "fan": 0.3}, {}, 0.2),
    R("REV15", "revenue", "International Merchandise Push", "Global merch stores: star revenue +8%.", {"star_rev": 8.0, "rev_all": 0.5}, {}, 0.15),
    R("REV16", "revenue", "Small-Market Growth Fund", "Small markets +3% revenue and faster value growth.", {"rev_small": 3.0, "value_growth": 0.5}, {"small": 0.3, "large": -0.2}, 0.0),
    R("REV17", "revenue", "Franchise Sale Tax", "Tax on franchise sales slows value growth.", {"value_growth": -0.8}, {"poor": 0.1, "all": -0.1}, -0.05),
    R("REV18", "revenue", "Local Broadcast Deregulation", "Big-market local TV deals uncapped.", {"rev_large": 2.0, "rev_small": -0.5}, {"large": 0.3, "small": -0.3}, 0.0),
    R("REV19", "revenue", "Concession Price Caps", "Arena food and drink prices capped.", {"rev_all": -0.6, "fan": 1.0}, {"rich": 0.05, "poor": -0.1}, 0.0),
    R("REV20", "revenue", "Arena Investment Credit", "League credit for arena upgrades: lower costs, faster value growth.", {"opex": -1.0, "value_growth": 0.4}, {"poor": 0.1}, 0.1),
    R("REV21", "revenue", "Travel Cost Pooling", "Travel costs pooled across clubs.", {"opex": -0.8}, {"small": 0.15, "medium": 0.05, "large": -0.05}, 0.05),
    R("REV22", "revenue", "Charter Flight Standards", "Upgraded charters: higher costs, fewer injuries.", {"opex": 0.6, "injury_rate": -2.0, "fan": 0.1}, {"rich": 0.1, "poor": -0.15}, 0.0),
    R("REV23", "revenue", "Star Appearance Requirements", "Stars must appear at league events: star revenue +10%.", {"star_rev": 10.0, "fan": 0.5}, {"all": 0.05}, 0.0),
    R("REV24", "revenue", "Digital Rights Bundle", "Esports and digital rights sold centrally.", {"rev_all": 0.8}, {}, 0.2),
    R("REV25", "revenue", "Season Ticket Rebates", "Rebates for season-ticket holders after losing seasons.", {"rev_all": -0.5, "fan": 1.2}, {"contender": 0.05, "poor": -0.1}, 0.0),

    # --------------------------------------------------------------- TRADES (20)
    R("TRD01", "trades", "Four Retained-Salary Slots", "Clubs may carry four retained contracts.", {"retention_slots": 1}, {"contender": 0.15, "rebuilder": 0.15}, 0.05),
    R("TRD02", "trades", "Two Retained-Salary Slots", "Clubs limited to two retained contracts.", {"retention_slots": -1}, {"small": 0.05}, -0.05),
    R("TRD03", "trades", "40% Retention Cap", "Maximum salary retention drops to 40%.", {"retention_max_pct": -10.0}, {"poor": 0.05, "contender": -0.1}, 0.0),
    R("TRD04", "trades", "60% Retention Cap", "Maximum salary retention rises to 60%.", {"retention_max_pct": 10.0}, {"contender": 0.15, "rebuilder": 0.05}, -0.05),
    R("TRD05", "trades", "Trade Request Transparency", "Trade requests become public; more players ask out.", {"trade_demand_rate": 15.0, "fan": 0.3}, {"all": -0.08}, 0.0),
    R("TRD06", "trades", "Trade Demand Cooling-Off", "Players must wait 30 days before re-requesting a trade.", {"trade_demand_rate": -20.0}, {"all": 0.08}, 0.05),
    R("TRD07", "trades", "Earlier Trade Deadline", "Deadline moves two weeks earlier.", {"trade_volume": -8.0, "fan": -0.2}, {"rebuilder": -0.05}, 0.0),
    R("TRD08", "trades", "Deadline Day Broadcast Special", "Deadline day becomes a national broadcast.", {"rev_all": 0.3, "fan": 0.6, "trade_volume": 5.0}, {}, 0.12),
    R("TRD09", "trades", "Trade Call Window Expansion", "Trades can be filed overnight and on holidays.", {"trade_volume": 15.0}, {"contender": 0.05}, 0.05),
    R("TRD10", "trades", "Holiday Trade Freeze Extended", "Longer holiday freeze on trades.", {"trade_volume": -10.0, "fan": -0.1}, {}, 0.0),
    R("TRD11", "trades", "Waiver Priority by Points %", "Waiver order follows points percentage daily.", {"trade_volume": 3.0, "fan": 0.2}, {"rebuilder": 0.1}, 0.08),
    R("TRD12", "trades", "Conditional Pick Simplification", "Conditional picks limited to two conditions.", {"trade_volume": 5.0}, {}, 0.08),
    R("TRD13", "trades", "Three-Team Trade Fast Lane", "Central registry approves three-way deals same day.", {"trade_volume": 8.0, "fan": 0.2}, {}, 0.05),
    R("TRD14", "trades", "Re-Acquisition Ban Extended", "Traded players can't return to a club for a full year.", {"trade_volume": -5.0}, {}, 0.0),
    R("TRD15", "trades", "Cap-Dump Pick Tax", "Clubs dumping salary must attach a pick; fewer dumps.", {"trade_volume": -6.0}, {"cap_tight": -0.2, "cap_room": 0.1}, 0.0),
    R("TRD16", "trades", "Player Consent for Retained Deals", "Players can veto retained-salary trades; fewer slots used.", {"retention_slots": -1, "trade_demand_rate": -5.0}, {"all": -0.05}, -0.05),
    R("TRD17", "trades", "Open Trade Market Week", "One week each December where any player can be shopped.", {"trade_volume": 10.0, "trade_demand_rate": 5.0}, {}, 0.0),
    R("TRD18", "trades", "Trade Value Disclosure", "Trade terms published in full.", {"fan": 0.4, "trade_volume": -3.0}, {}, 0.05),
    R("TRD19", "trades", "Waiver Exemption Extended", "Young players waiver-exempt one more season.", {"trade_volume": 2.0}, {"rebuilder": 0.15}, 0.05),
    R("TRD20", "trades", "Mid-Season Re-Entry Waivers", "Re-entry waivers return; teams claim half the salary.", {"trade_volume": 4.0, "opex": -0.2}, {"poor": 0.1}, 0.0),

    # ---------------------------------------------------------------- DRAFT (20)
    R("DRF01", "draft", "Scouting Budget Floor", "Every club must spend at least the league scouting floor (+15%).", {"scouting_budget": 15.0, "opex": 0.2}, {"rebuilder": 0.15, "poor": -0.05}, 0.05),
    R("DRF02", "draft", "Scouting Budget Cap", "Scouting spend capped (-15%) to level the field.", {"scouting_budget": -15.0, "opex": -0.2}, {"small": 0.15, "large": -0.15}, 0.0),
    R("DRF03", "draft", "Central Scouting Data Share", "League shares combine data; teams can scout further (+8%).", {"scouting_budget": 8.0}, {"small": 0.1}, 0.1),
    R("DRF04", "draft", "Draft Combine Broadcast", "Combine televised nationally.", {"rev_all": 0.3, "fan": 0.5}, {}, 0.15),
    R("DRF05", "draft", "Draft Weekend Fan Festival", "Draft becomes a fan festival in the host city.", {"rev_all": 0.4, "fan": 0.6}, {}, 0.12),
    R("DRF06", "draft", "European Scouting Exchange", "Shared European scouting network: +10% scouting reach.", {"scouting_budget": 10.0, "star_rev": 2.0}, {"small": 0.05}, 0.05),
    R("DRF07", "draft", "Prospect Development Fund", "League funds AHL development; small costs, deeper scouting.", {"scouting_budget": 5.0, "opex": 0.3}, {"rebuilder": 0.1}, 0.0),
    R("DRF08", "draft", "Junior Scouting Restrictions", "Scouts limited at junior events (-10% scouting).", {"scouting_budget": -10.0}, {"poor": 0.1, "rich": -0.1}, -0.05),
    R("DRF09", "draft", "Draft Pick Trade Window", "Picks tradable during the draft broadcast.", {"trade_volume": 4.0, "fan": 0.4}, {}, 0.1),
    R("DRF10", "draft", "Prospect Showcase Tour", "Top prospects tour league cities.", {"fan": 0.6, "star_rev": 3.0}, {}, 0.1),
    R("DRF11", "draft", "Analytics Department Minimum", "Every club must staff analytics (+5% scouting, higher costs).", {"scouting_budget": 5.0, "opex": 0.4}, {"rich": 0.05, "poor": -0.1}, 0.0),
    R("DRF12", "draft", "Scouting Travel Subsidy", "League subsidises scouting travel for small markets.", {"scouting_budget": 6.0, "revenue_share": 0.2}, {"small": 0.15, "large": -0.05}, 0.05),
    R("DRF13", "draft", "Virtual Draft Format", "Teams draft remotely; lower event costs.", {"opex": -0.2, "fan": -0.3}, {"poor": 0.05}, 0.0),
    R("DRF14", "draft", "Draft Lottery Broadcast Upgrade", "Prime-time lottery show.", {"rev_all": 0.2, "fan": 0.4}, {}, 0.12),
    R("DRF15", "draft", "U.S. College Scouting Push", "NCAA partnership broadens scouting (+6%).", {"scouting_budget": 6.0}, {}, 0.08),
    R("DRF16", "draft", "Scouting Staff Limits", "Cap on scouting staff size (-8%).", {"scouting_budget": -8.0, "opex": -0.2}, {"small": 0.1, "rich": -0.1}, 0.0),
    R("DRF17", "draft", "Draft Day Season-Ticket Drive", "Clubs sell season tickets at draft parties.", {"rev_all": 0.3}, {}, 0.12),
    R("DRF18", "draft", "Women's Pro Partnership Showcase", "Joint draft showcase with the women's league.", {"fan": 0.5, "rev_all": 0.2}, {}, 0.1),
    R("DRF19", "draft", "Prospect Data Privacy Rules", "Medical data limited at the combine (-4% scouting).", {"scouting_budget": -4.0, "fan": 0.1}, {}, 0.0),
    R("DRF20", "draft", "Global Draft Showcase", "Draft held overseas every other year.", {"star_rev": 4.0, "fan": 0.2, "opex": 0.2}, {"large": 0.05}, 0.0),

    # --------------------------------------------------------------- PLAYER (25)
    R("PLY01", "player", "Mandatory Visors", "All players must wear visors.", {"injury_rate": -3.0, "fan": -0.2}, {}, 0.15),
    R("PLY02", "player", "Neck Laceration Protection", "Cut-resistant neck guards required.", {"injury_rate": -2.0}, {}, 0.2),
    R("PLY03", "player", "Blindside Hit Ban Expanded", "Larger suspensions for blindside hits.", {"injury_rate": -4.0, "fan": 0.1}, {}, 0.12),
    R("PLY04", "player", "Fighting Major Becomes Ejection", "Fighting earns a game misconduct.", {"injury_rate": -3.0, "fan": -0.8}, {"all": -0.05}, 0.0),
    R("PLY05", "player", "Hybrid Icing Upgrade", "Faster hybrid icing calls.", {"injury_rate": -1.5}, {}, 0.18),
    R("PLY06", "player", "Mandatory Rest Days", "Players get one guaranteed rest day per week.", {"injury_rate": -2.5, "opex": 0.3}, {"contender": 0.05}, 0.0),
    R("PLY07", "player", "Load Management Allowed", "Clubs may rest healthy stars; fewer injuries, fan backlash.", {"injury_rate": -3.0, "fan": -0.8, "star_rev": -3.0}, {"contender": 0.15, "rebuilder": -0.1}, -0.05),
    R("PLY08", "player", "Concussion Spotter Authority", "Spotters can pull players mid-game.", {"injury_rate": -2.0, "fan": 0.2}, {}, 0.15),
    R("PLY09", "player", "Player Social Media Freedom", "Players may post freely, even during games.", {"star_rev": 5.0, "fan": 0.4}, {}, 0.08),
    R("PLY10", "player", "Player Media Availability Rule", "Stars must speak after every game.", {"star_rev": 3.0, "fan": 0.3}, {}, 0.1),
    R("PLY11", "player", "Mental Health Leave Program", "Paid mental-health leave without cap penalty.", {"opex": 0.3, "fan": 0.4, "trade_demand_rate": -5.0}, {}, 0.1),
    R("PLY12", "player", "Olympic Participation Guaranteed", "League breaks for every Olympics.", {"star_rev": 6.0, "injury_rate": 1.0, "fan": 0.5}, {"rich": -0.05}, 0.05),
    R("PLY13", "player", "World Cup of Hockey Every 4 Years", "Regular best-on-best tournament.", {"star_rev": 5.0, "rev_all": 0.6}, {}, 0.1),
    R("PLY14", "player", "Player Equipment Weight Limits", "Lighter equipment standards.", {"injury_rate": 1.5, "fan": 0.2}, {}, 0.0),
    R("PLY15", "player", "Stick Flex Regulation", "Limits on ultra-flex sticks.", {"fan": -0.1}, {}, 0.05),
    R("PLY16", "player", "Goalie Pad Reduction", "Smaller goalie pads.", {"fan": 0.6, "injury_rate": 0.5}, {}, 0.08),
    R("PLY17", "player", "Player Agent Fee Cap", "Agent fees capped; fewer agent-driven demands.", {"trade_demand_rate": -8.0, "fa_bonus_demand": -2.0}, {"all": 0.05}, 0.0),
    R("PLY18", "player", "Pre-Season Fitness Testing Returns", "Mandatory fitness testing comes back.", {"injury_rate": -1.0, "trade_demand_rate": 3.0}, {"all": -0.05}, 0.0),
    R("PLY19", "player", "Travel Fatigue Rule", "No back-to-backs involving cross-country flights.", {"injury_rate": -2.0, "opex": 0.2}, {"medium": 0.05}, 0.05),
    R("PLY20", "player", "Star Player Fan Vote", "Fans vote stars onto league marketing campaigns.", {"star_rev": 4.0, "fan": 0.4}, {}, 0.1),
    R("PLY21", "player", "Diving Fines Doubled", "Embellishment fines doubled.", {"fan": 0.3}, {}, 0.18),
    R("PLY22", "player", "Player Tracking Data Shared", "Chip data shared with players' agents.", {"fa_bonus_demand": 2.0, "star_rev": 2.0}, {}, 0.0),
    R("PLY23", "player", "Off-Ice Conduct Policy Tightened", "Harsher suspensions for off-ice conduct.", {"fan": 0.4, "trade_demand_rate": 2.0}, {}, 0.1),
    R("PLY24", "player", "Veteran Leadership Bonus", "League funds veteran mentorship stipends.", {"opex": 0.2, "trade_demand_rate": -4.0}, {"rebuilder": 0.05}, 0.05),
    R("PLY25", "player", "Injury Reserve Roster Spot", "Extra IR roster spot league-wide.", {"injury_rate": -1.0, "opex": 0.2}, {"contender": 0.08}, 0.08),

    # ----------------------------------------------------------------- GAME (25)
    R("GAM01", "game", "10-Minute 3-on-3 Overtime", "Overtime doubles to 10 minutes; fewer shootouts.", {"fan": 0.8, "injury_rate": 0.5}, {}, 0.12),
    R("GAM02", "game", "Abolish the Shootout", "Ties after overtime stand.", {"fan": -0.4}, {"all": -0.05}, -0.02),
    R("GAM03", "game", "Three-Point Regulation Win", "3 points for a regulation win, 2 for OT/SO, 1 for an OT loss.", {"fan": 0.6, "rev_all": 0.2}, {"contender": 0.05}, 0.05),
    R("GAM04", "game", "Coach's Challenge for Any Penalty", "Coaches can challenge any penalty once a game.", {"fan": -0.2}, {}, 0.0),
    R("GAM05", "game", "Major Penalty Served in Full", "Power plays don't end on a goal for majors.", {"fan": 0.4}, {}, 0.08),
    R("GAM06", "game", "Bigger Ice Surface Trial", "Two arenas trial a wider sheet.", {"opex": 0.3, "fan": 0.2}, {"poor": -0.1}, -0.05),
    R("GAM07", "game", "Play-In Round", "Seeds 7-10 in each conference play in.", {"playoff_rev": 10.0, "fan": 0.6, "rev_all": 0.4}, {"rebuilder": 0.08}, 0.05),
    R("GAM08", "game", "Shorten Preseason to Three Games", "Preseason cut to three games.", {"rev_all": -0.2, "injury_rate": -0.5}, {}, 0.05),
    R("GAM09", "game", "Outdoor Game Expansion", "Four outdoor games every season.", {"rev_all": 0.8, "fan": 0.6}, {"large": 0.05}, 0.1),
    R("GAM10", "game", "International Regular-Season Series", "Six regular-season games in Europe.", {"star_rev": 4.0, "rev_all": 0.4, "injury_rate": 0.5}, {}, 0.05),
    R("GAM11", "game", "Video Review in Booth Only", "All reviews handled centrally in Toronto.", {"fan": 0.3}, {}, 0.12),
    R("GAM12", "game", "Delay-of-Game Puck-Over-Glass Relaxed", "Puck-over-glass becomes a faceoff, not a penalty.", {"fan": 0.2}, {}, 0.08),
    R("GAM13", "game", "Faceoff Violation Penalty", "Second faceoff violation is a minor.", {"fan": -0.1}, {}, 0.0),
    R("GAM14", "game", "Goalie Interference Standard", "Clearer goalie interference rule.", {"fan": 0.4}, {}, 0.15),
    R("GAM15", "game", "Matinee Game Push", "More afternoon games for families.", {"rev_all": 0.3, "fan": 0.3}, {}, 0.1),
    R("GAM16", "game", "Back-to-Back Limit", "No team plays more than 12 back-to-backs.", {"injury_rate": -1.5, "opex": 0.2}, {}, 0.05),
    R("GAM17", "game", "Rivalry Week", "Division rivals meet in a marketed week.", {"rev_all": 0.4, "fan": 0.5}, {}, 0.12),
    R("GAM18", "game", "Two Referees + Video Ref", "A third official watches video live.", {"opex": 0.2, "fan": 0.3}, {}, 0.05),
    R("GAM19", "game", "Bye Week Doubled", "Two bye weeks per team.", {"injury_rate": -1.5, "rev_all": -0.3}, {"contender": 0.05}, 0.0),
    R("GAM20", "game", "Conference-Based Playoffs", "Playoffs reseed by conference standings.", {"playoff_rev": 5.0, "fan": 0.3}, {}, 0.05),
    R("GAM21", "game", "Overtime Power Play Cap", "OT minors become 4-on-3 for 60 seconds.", {"fan": 0.2}, {}, 0.05),
    R("GAM22", "game", "Shootout Shooter Order Free", "Coaches can reuse shooters in the shootout.", {"fan": 0.2, "star_rev": 1.0}, {}, 0.05),
    R("GAM23", "game", "Neutral-Site Showcase Games", "Two neutral-site games in non-NHL cities.", {"rev_all": 0.3, "expansion_pressure": 5.0}, {}, 0.05),
    R("GAM24", "game", "Shot Clock Trial", "30-second offensive-zone shot clock trial.", {"fan": -0.3}, {"all": -0.05}, -0.05),
    R("GAM25", "game", "Empty-Net Delay Rule", "Pulled goalie can't return until a stoppage.", {"fan": 0.1}, {}, 0.0),

    # -------------------------------------------------------------- UNIFORM (15)
    R("UNI01", "uniform", "Jersey Length Standard", "Jerseys must hang below the pants line; no more tucks.", {"fan": 0.2}, {}, 0.08),
    R("UNI02", "uniform", "Shorter Jersey Cut", "Tighter, shorter jerseys league-wide.", {"fan": -0.3, "rev_all": 0.2}, {}, 0.0),
    R("UNI03", "uniform", "Heritage Third Jerseys Every Year", "Every club releases a heritage third jersey.", {"rev_all": 0.6, "fan": 0.4}, {}, 0.15),
    R("UNI04", "uniform", "Colour vs Colour Games", "Both teams wear colour jerseys in rivalry games.", {"fan": 0.4, "rev_all": 0.2}, {}, 0.1),
    R("UNI05", "uniform", "Name Bars in Native Script", "Players may wear name bars in their native alphabet.", {"star_rev": 2.0, "fan": 0.2}, {}, 0.08),
    R("UNI06", "uniform", "Gold Captain Letters", "Captain letters in gold for the anniversary season.", {"fan": 0.1}, {}, 0.12),
    R("UNI07", "uniform", "Retro Logo Week", "Clubs wear retro logos for one week.", {"rev_all": 0.3, "fan": 0.3}, {}, 0.12),
    R("UNI08", "uniform", "Sock Stripe Regulations Lifted", "Clubs may change sock stripes freely.", {"rev_all": 0.1}, {}, 0.05),
    R("UNI09", "uniform", "Glove Colour Freedom", "Gloves no longer have to match the jersey.", {"fan": 0.1, "star_rev": 1.0}, {}, 0.05),
    R("UNI10", "uniform", "Helmet Number Removal", "Numbers removed from helmets for ad space.", {"rev_all": 0.3, "fan": -0.3}, {"poor": 0.05}, 0.0),
    R("UNI11", "uniform", "Pride & Heritage Night Freedom", "Clubs choose their own warm-up jerseys.", {"fan": 0.2}, {}, 0.05),
    R("UNI12", "uniform", "Ice Logo Sponsor Boards", "Sponsor logos painted at centre ice.", {"rev_all": 0.5, "fan": -0.4}, {"poor": 0.08}, 0.05),
    R("UNI13", "uniform", "Black Ice Night", "Dark ice for one marquee game a season.", {"fan": 0.3, "rev_all": 0.1}, {}, 0.05),
    R("UNI14", "uniform", "Uniform Supplier Tender", "New uniform supplier deal: more merch revenue.", {"rev_all": 0.6, "star_rev": 2.0}, {}, 0.12),
    R("UNI15", "uniform", "Player-Designed Jerseys", "Stars co-design an alternate jersey each season.", {"star_rev": 4.0, "fan": 0.4}, {}, 0.08),

    # ------------------------------------------------------------ STRUCTURE (15)
    R("STR01", "structure", "Relocation Standards Tightened", "Clubs must show three straight losing years before moving.", {"relocation_ease": -0.08}, {"poor": -0.2, "rich": 0.1}, 0.05),
    R("STR02", "structure", "Relocation Standards Eased", "Struggling clubs can apply to move after one bad season.", {"relocation_ease": 0.08}, {"poor": 0.2, "small": 0.1, "rich": -0.05}, -0.05),
    R("STR03", "structure", "Expansion Committee Formed", "League opens a formal expansion review.", {"expansion_pressure": 25.0}, {"rich": 0.1, "poor": 0.05}, 0.05),
    R("STR04", "structure", "Expansion Moratorium", "No expansion bids considered for now.", {"expansion_pressure": -30.0}, {"small": 0.05}, -0.05),
    R("STR05", "structure", "Arena Standards Act", "All arenas must meet new standards; costs rise, values grow.", {"opex": 0.6, "value_growth": 0.6, "relocation_ease": 0.03}, {"rich": 0.1, "poor": -0.2}, -0.05),
    R("STR06", "structure", "Ownership Debt Limits", "Limits on franchise debt; slower value growth.", {"value_growth": -0.4, "relocation_ease": -0.03}, {"rich": 0.05, "poor": -0.05}, 0.0),
    R("STR07", "structure", "Private Equity Ownership Allowed", "Funds can buy minority stakes; values climb.", {"value_growth": 1.0}, {"all": 0.08}, 0.05),
    R("STR08", "structure", "Market Protection Radius", "Bigger territorial rights around each club.", {"rev_large": 0.4, "expansion_pressure": -10.0}, {"large": 0.15, "small": 0.05}, 0.0),
    R("STR09", "structure", "Small-Market Stabilisation Fund", "Fund for clubs at relocation risk.", {"revenue_share": 0.6, "relocation_ease": -0.05}, {"small": 0.2, "poor": 0.2, "large": -0.15}, 0.0),
    R("STR10", "structure", "Neutral-Site Seasons for Bid Cities", "Bid cities host neutral-site games to prove demand.", {"expansion_pressure": 15.0, "rev_all": 0.2}, {}, 0.05),
    R("STR11", "structure", "Two-Conference Realignment Review", "Review of division alignment; travel costs drop.", {"opex": -0.3}, {"medium": 0.05}, 0.05),
    R("STR12", "structure", "Franchise Valuation Floor", "League backstops franchise values in sales.", {"value_growth": 0.5, "relocation_ease": -0.04}, {"poor": 0.15}, 0.05),
    R("STR13", "structure", "Arena Lease Bargaining Support", "League negotiates leases alongside clubs.", {"opex": -0.5, "relocation_ease": -0.03}, {"poor": 0.15, "small": 0.1}, 0.05),
    R("STR14", "structure", "Relocation Fee Raised", "Moving costs a fee shared by the other clubs.", {"relocation_ease": -0.05, "revenue_share": 0.2}, {"rich": 0.1, "poor": -0.15}, 0.0),
    R("STR15", "structure", "Expansion Fee Indexation", "Expansion fees index to franchise values.", {"expansion_pressure": 10.0, "value_growth": 0.3}, {"all": 0.05}, 0.05),
]

RULE_BY_ID: Dict[str, Dict[str, Any]] = {r["id"]: r for r in RULES}

assert len(RULES) == 200, f"rule catalog must hold 200 rules, has {len(RULES)}"
assert len(RULE_BY_ID) == len(RULES), "duplicate rule id in catalog"

#: Effects that SET a value (latest passed rule wins) rather than add.
ABSOLUTE_EFFECTS = frozenset({"max_term_own", "max_term_ufa"})

#: Candidate cities for relocation / expansion: (city, abbr, conference, division hint, value_b, tier)
CANDIDATE_MARKETS: List[Dict[str, Any]] = [
    {"city": "Houston", "abbr": "HOU", "conference": "Western", "value_b": 2.6, "tier": "large"},
    {"city": "Atlanta", "abbr": "ATL", "conference": "Eastern", "value_b": 2.2, "tier": "medium"},
    {"city": "Quebec City", "abbr": "QUE", "conference": "Eastern", "value_b": 1.7, "tier": "small"},
    {"city": "Kansas City", "abbr": "KCY", "conference": "Western", "value_b": 1.8, "tier": "medium"},
    {"city": "Portland", "abbr": "POR", "conference": "Western", "value_b": 1.9, "tier": "medium"},
    {"city": "Milwaukee", "abbr": "MIL", "conference": "Western", "value_b": 1.6, "tier": "small"},
    {"city": "Hamilton", "abbr": "HAM", "conference": "Eastern", "value_b": 1.8, "tier": "medium"},
    {"city": "Phoenix", "abbr": "PHX", "conference": "Western", "value_b": 1.9, "tier": "medium"},
    {"city": "Austin", "abbr": "AUS", "conference": "Western", "value_b": 2.1, "tier": "medium"},
    {"city": "Cleveland", "abbr": "CLE", "conference": "Eastern", "value_b": 1.7, "tier": "medium"},
    {"city": "San Diego", "abbr": "SDG", "conference": "Western", "value_b": 2.0, "tier": "medium"},
    {"city": "Baltimore", "abbr": "BAL", "conference": "Eastern", "value_b": 1.7, "tier": "medium"},
    {"city": "Oklahoma City", "abbr": "OKC", "conference": "Western", "value_b": 1.5, "tier": "small"},
    {"city": "Hartford", "abbr": "HFD", "conference": "Eastern", "value_b": 1.5, "tier": "small"},
    {"city": "Saskatoon", "abbr": "SAS", "conference": "Western", "value_b": 1.3, "tier": "small"},
    {"city": "Halifax", "abbr": "HFX", "conference": "Eastern", "value_b": 1.3, "tier": "small"},
]

#: Nicknames for expansion clubs (picked per save).
EXPANSION_NICKNAMES: List[str] = [
    "Aviators", "Outlaws", "Comets", "Mammoths", "Wolves", "Ironmen", "Thunder", "Blizzard",
    "Mustangs", "Nordiques", "Whalers", "Monarchs", "Pioneers", "Express", "Storm", "Huskies",
]
