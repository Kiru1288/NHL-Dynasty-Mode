"""
NHL Dynasty Mode — Puckr / IceHole template pools.

Expansion v2: 150 tweet templates, 30 meme tweets, 100 Reddit threads.
Full angle taxonomy, tag system, placeholder registry, render + filter utils.
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, TypedDict


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------

class TweetTemplate(TypedDict, total=False):
    text: str
    angles: List[str]
    tags: List[str]
    requires: List[str]
    tone: str          # "hot_take" | "analytical" | "narrative" | "chaos" | "cold"


class RedditTemplate(TypedDict, total=False):
    title: str
    body: str
    flair: str
    angles: List[str]
    tags: List[str]
    requires: List[str]
    format: str        # "discussion" | "breakdown" | "postmortem" | "guide" | "rant" | "showoff"


# ---------------------------------------------------------------------------
# Placeholder registry — every key that can appear in a template
# ---------------------------------------------------------------------------

PLACEHOLDER_REGISTRY: Dict[str, str] = {
    # Player
    "name":              "Player full name",
    "first_name":        "Player first name",
    "overall":           "Player current OVR rating (int)",
    "prior_overall":     "Player OVR at start of season (int)",
    "delta_overall":     "OVR change this season (±int)",
    "cap_hit":           "Player cap hit in $M (float)",
    "term":              "Contract term in years (int)",
    "years_remaining":   "Years left on current deal (int)",
    "points":            "Player season points (int)",
    "goals":             "Player season goals (int)",
    "assists":           "Player season assists (int)",
    "ppg":               "Points per game (float, 2dp)",
    "position":          "Player position abbreviation",
    "age":               "Player age (int)",
    "morale":            "Player morale score 0-100 (int)",
    "confidence":        "Player confidence score 0-100 (int)",
    "heat":              "Trade request heat level 0-100 (int)",
    "ntc_type":          "No-trade clause type string",
    "injury_type":       "Injury description string",
    "games_missed":      "Games missed due to injury (int)",
    "games_played":      "Games played (int)",
    "save_pct":          "Goalie save percentage (.3dp)",
    "gaa":               "Goalie goals-against average (float)",
    "shutouts":          "Goalie shutouts (int)",
    # Prospect
    "prospect_name":     "Prospect full name",
    "prospect_overall":  "Prospect projected OVR (int)",
    "prospect_floor":    "Prospect floor OVR (int)",
    "prospect_ceiling":  "Prospect ceiling OVR (int)",
    "prospect_conf":     "Scouting confidence 0-100 (int)",
    "draft_pick":        "Draft pick number (int)",
    "draft_round":       "Draft round (int)",
    "draft_year":        "Draft year (int)",
    "league_of_origin":  "Prospect's current league string",
    # Team
    "team":              "User team name",
    "team_record":       "Team W-L-OT record string",
    "team_rank":         "Team rank in conference (int)",
    "league_rank":       "Team rank league-wide (int)",
    "cap_space":         "Available cap space in $M (float)",
    "salary_cap":        "League salary cap in $M (float)",
    "locker_unity":      "Locker room unity score 0-100 (int)",
    "franchise_rep":     "Org reputation label string",
    "wins":              "Team season wins (int)",
    "losses":            "Team season losses (int)",
    "goal_diff":         "Team goal differential (±int)",
    "pp_pct":            "Power play % (float)",
    "pk_pct":            "Penalty kill % (float)",
    "xgf_pct":          "Expected goals for % (float)",
    # Rival / CPU
    "rival_team":        "CPU rival team name",
    "rival_player":      "Rival team's player name",
    "rival_cap_hit":     "Rival player cap hit in $M (float)",
    "rival_term":        "Rival player contract term (int)",
    "rival_record":      "Rival team W-L-OT record string",
    # Media / Storyline
    "headline":          "Media headline string",
    "reporter":          "Reporter name/handle string",
    "rumor_source":      "Rumor credibility label string",
    "storyline":         "Active storyline label string",
    # Context
    "season":            "Season label e.g. '2027-28'",
    "trade_return":      "Summary of trade return string",
    "trade_asset":       "Asset given up in trade string",
    "offer_sheet_amt":   "Offer sheet AAV in $M (float)",
    "draft_class_rank":  "Draft class quality label string",
    "coach_name":        "Head coach name string",
    "agent_name":        "Player agent name string",
}


# ---------------------------------------------------------------------------
# Angle taxonomy
# ---------------------------------------------------------------------------

ANGLES = [
    "trade_market",       # active trade discussions, deadline, assets
    "contract_battle",    # extensions, FA, arbitration, buyouts
    "locker_room",        # chemistry, conflict, morale, meetings
    "draft_board",        # prospects, scouting, combine, lottery
    "slump_watch",        # player development, underperformance, breakout
    "league_wire",        # standings, rival news, simulation events
    "goaltending",        # goalie performance, tandem decisions
    "injury_watch",       # injuries, recovery, lineup impact
    "rebuild_mode",       # long-term organizational direction
    "win_now",            # contender decisions, deadline buys
    "cap_crisis",         # salary crunch, buyouts, buried contracts
    "media_spin",         # press conferences, reporter bias, narratives
    "ownership_pressure", # mandate changes, fan reaction, payroll
    "dynasty_legacy",     # multi-season history, franchise records
]


# ---------------------------------------------------------------------------
# Tag taxonomy
# ---------------------------------------------------------------------------

TAGS = [
    "trade", "trade_demand", "ntc", "offer_sheet",
    "contract", "extension", "free_agency", "arbitration", "buyout",
    "draft", "prospect", "scouting", "lottery", "development", "callup",
    "rebuild", "tank", "retool", "win_now",
    "cap", "cap_crisis", "buried",
    "goalie", "injury", "lineup",
    "culture", "chemistry", "locker_room", "leadership", "meeting",
    "media", "rumor", "reputation", "agent",
    "rival", "cpu", "owner", "coaching",
    "playoffs", "deadline", "breakout", "slump",
    "analytics", "scouting", "character",
    "life", "retirement", "milestone",
    "meta", "guide", "postmortem", "showoff",
    "meme", "rant", "chaos",
]


# ---------------------------------------------------------------------------
# 150 DYNASTY TWEET TEMPLATES
# ---------------------------------------------------------------------------

DYNASTY_TWEET_TEMPLATES: List[TweetTemplate] = [

    # ── TRADE MARKET ──────────────────────────────────────────────────────

    {
        "text": "just traded {name} for cap space and futures. {cap_space}M opened up. my cap sheet thinks im insane. my timeline thinks it might work. we'll see who's right in 3 years",
        "angles": ["trade_market"], "tags": ["trade", "rebuild"], "requires": ["name", "cap_space"], "tone": "narrative",
    },
    {
        "text": "cpu team just offered me a 1st for {name} at {ppg} PPG. that number should excite me. instead im suddenly unsure if he's worth more than i think",
        "angles": ["trade_market"], "tags": ["trade"], "requires": ["name", "ppg"], "tone": "analytical",
    },
    {
        "text": "made a trade nobody understands. gave up {trade_asset}, got {trade_return}. in 4 years, check back",
        "angles": ["trade_market"], "tags": ["trade"], "requires": ["trade_asset", "trade_return"], "tone": "cold",
    },
    {
        "text": "deadline passed and {team} stood pat. hardest gm call when {team_record} is right on the bubble",
        "angles": ["trade_market"], "tags": ["trade", "deadline"], "requires": ["team", "team_record"], "tone": "cold",
    },
    {
        "text": "rival just gutted their core. division just opened up and im the only one paying attention",
        "angles": ["trade_market", "league_wire"], "tags": ["rival", "win_now"], "requires": ["rival_team"], "tone": "hot_take",
    },
    {
        "text": "{name} has a {ntc_type} and he's not waiving it for a contender. that's a problem. now we negotiate differently",
        "angles": ["trade_market"], "tags": ["trade", "ntc"], "requires": ["name", "ntc_type"], "tone": "analytical",
    },
    {
        "text": "pulled off the deadline buy. {trade_return} in exchange for {cap_space}M and a mid-round pick. win-now mode activated",
        "angles": ["trade_market", "win_now"], "tags": ["trade", "deadline", "win_now"], "requires": ["trade_return", "cap_space"], "tone": "narrative",
    },
    {
        "text": "got lowballed in a trade offer for {name}. they really tried it. the market just told me exactly what not to accept",
        "angles": ["trade_market"], "tags": ["trade"], "requires": ["name"], "tone": "hot_take",
    },
    {
        "text": "three-team trade just went through and none of the gms are satisfied. perfect",
        "angles": ["trade_market"], "tags": ["trade"], "requires": [], "tone": "cold",
    },
    {
        "text": "{name} has a preferred destination list and {team} isn't on it. so now we're doing this the hard way",
        "angles": ["trade_market"], "tags": ["trade", "trade_demand"], "requires": ["name", "team"], "tone": "narrative",
    },
    {
        "text": "retained {cap_hit}M of {name}'s salary to get the deal done. painful now, right later",
        "angles": ["trade_market", "cap_crisis"], "tags": ["trade", "cap"], "requires": ["name", "cap_hit"], "tone": "analytical",
    },
    {
        "text": "just acquired {name} and the locker room reacted immediately. unity at {locker_unity}. the room has opinions",
        "angles": ["trade_market", "locker_room"], "tags": ["trade", "chemistry"], "requires": ["name", "locker_unity"], "tone": "narrative",
    },

    # ── CONTRACT BATTLE ───────────────────────────────────────────────────

    {
        "text": "{name}'s agent called. {cap_hit}M ask. {cap_space}M available. doing the math in real time and it doesn't work. and we're gonna do it anyway probably",
        "angles": ["contract_battle"], "tags": ["contract", "extension"], "requires": ["name", "cap_hit", "cap_space"], "tone": "chaos",
    },
    {
        "text": "extension talks with {name} broke down at {cap_hit}M AAV. he's testing free agency. {rival_team} already circling",
        "angles": ["contract_battle"], "tags": ["extension", "free_agency"], "requires": ["name", "cap_hit", "rival_team"], "tone": "cold",
    },
    {
        "text": "just bought out {name}'s deal. {cap_hit}M dead cap for {term} years. ugly but necessary",
        "angles": ["contract_battle", "cap_crisis"], "tags": ["buyout", "cap"], "requires": ["name", "cap_hit", "term"], "tone": "cold",
    },
    {
        "text": "offer sheet on {rival_player} at {offer_sheet_amt}M. {rival_team} has 7 days. im not bluffing",
        "angles": ["contract_battle"], "tags": ["offer_sheet"], "requires": ["rival_player", "offer_sheet_amt", "rival_team"], "tone": "hot_take",
    },
    {
        "text": "{rival_team} matched our offer sheet. good — now they're attached at {rival_cap_hit}M while im flexible",
        "angles": ["contract_battle"], "tags": ["offer_sheet", "cap"], "requires": ["rival_team", "rival_cap_hit"], "tone": "analytical",
    },
    {
        "text": "signed {name} to {cap_hit}M x {term}. market said he was worth it. ownership said do it. done",
        "angles": ["contract_battle"], "tags": ["contract", "free_agency"], "requires": ["name", "cap_hit", "term"], "tone": "cold",
    },
    {
        "text": "rookie ELC on {name} is a cheat code for {term} more years. use the window while it exists",
        "angles": ["contract_battle", "win_now"], "tags": ["contract", "win_now"], "requires": ["name", "term"], "tone": "analytical",
    },
    {
        "text": "going to arbitration with {name}. relationship might not survive this. {cap_hit}M vs his {cap_hit}M ask. here we go",
        "angles": ["contract_battle"], "tags": ["arbitration", "contract"], "requires": ["name", "cap_hit"], "tone": "narrative",
    },
    {
        "text": "free agency day 1. lost {name} to {rival_team} at {rival_cap_hit}M. knew it was coming. still stings",
        "angles": ["contract_battle"], "tags": ["free_agency"], "requires": ["name", "rival_team", "rival_cap_hit"], "tone": "narrative",
    },
    {
        "text": "the contract i gave {name} looked fine at the time. two years later its buried at {cap_hit}M and he's {overall} OVR. lesson learned",
        "angles": ["contract_battle", "cap_crisis"], "tags": ["contract", "buried", "cap_crisis"], "requires": ["name", "cap_hit", "overall"], "tone": "postmortem",
    },
    {
        "text": "qualifying offer sent to {name}. he accepts or hits RFA market. either way, {team} has leverage",
        "angles": ["contract_battle"], "tags": ["contract", "free_agency"], "requires": ["name", "team"], "tone": "cold",
    },
    {
        "text": "{name} agreed to less term for more AAV. he's betting on himself. I respect it even if it complicates year 3",
        "angles": ["contract_battle"], "tags": ["contract", "extension"], "requires": ["name"], "tone": "analytical",
    },

    # ── LOCKER ROOM ───────────────────────────────────────────────────────

    {
        "text": "locker room unity at {locker_unity}. two guys need a mediator. this is how windows close before they open",
        "angles": ["locker_room"], "tags": ["chemistry", "locker_room"], "requires": ["locker_unity"], "tone": "cold",
    },
    {
        "text": "had a player meeting with {name} and promised more ice time. then roster math happened. morale at {morale}. im incompetent and he knows it",
        "angles": ["locker_room"], "tags": ["meeting", "locker_room"], "requires": ["name", "morale"], "tone": "chaos",
    },
    {
        "text": "{name} just asked for a private meeting. this is either a contract conversation or an NTC conversation. neither is comfortable",
        "angles": ["locker_room", "contract_battle"], "tags": ["meeting", "ntc"], "requires": ["name"], "tone": "narrative",
    },
    {
        "text": "veteran {name} is mentoring our top prospect. the influence on character traits is real. this is how culture builds",
        "angles": ["locker_room", "draft_board"], "tags": ["leadership", "development", "culture"], "requires": ["name", "prospect_name"], "tone": "analytical",
    },
    {
        "text": "promised {name} the captaincy and now three other guys think they earned it too. room feels different this week",
        "angles": ["locker_room"], "tags": ["leadership", "culture"], "requires": ["name"], "tone": "narrative",
    },
    {
        "text": "scratch {name} once and his agent calls. scratch him twice and the media runs with it. this is the job",
        "angles": ["locker_room", "media_spin"], "tags": ["locker_room", "media"], "requires": ["name"], "tone": "cold",
    },
    {
        "text": "locker room is split because we traded {name}. his closest friend is still on the roster and feeling it. unity {locker_unity}",
        "angles": ["locker_room"], "tags": ["chemistry", "trade"], "requires": ["name", "locker_unity"], "tone": "narrative",
    },
    {
        "text": "morale at {morale} after the losing streak. everyone pointing fingers at different things. first good gm instinct is to stay calm",
        "angles": ["locker_room"], "tags": ["locker_room", "slump"], "requires": ["morale"], "tone": "cold",
    },
    {
        "text": "conflict between {name} and a teammate surfaced publicly. didn't catch it early enough. that's on me",
        "angles": ["locker_room"], "tags": ["locker_room", "chemistry"], "requires": ["name"], "tone": "narrative",
    },
    {
        "text": "offered the captaincy to {name}. he thought about it for 48 hours. that pause told me something",
        "angles": ["locker_room"], "tags": ["leadership", "meeting"], "requires": ["name"], "tone": "cold",
    },
    {
        "text": "{name}'s confidence at {confidence}. not a slump — a mental thing. the meeting this week will matter more than the lineup card",
        "angles": ["locker_room", "slump_watch"], "tags": ["meeting", "slump"], "requires": ["name", "confidence"], "tone": "analytical",
    },
    {
        "text": "veteran buying in to a smaller role for the good of the room. that character trait is worth {cap_hit}M in ways that don't show in the box score",
        "angles": ["locker_room"], "tags": ["leadership", "culture"], "requires": ["cap_hit"], "tone": "analytical",
    },

    # ── DRAFT BOARD / PROSPECTS ───────────────────────────────────────────

    {
        "text": "people sleeping on {prospect_name}. {prospect_overall} OVR ceiling, {points} points in {games_played} GP in {league_of_origin}. nobody's watching that league. we are",
        "angles": ["draft_board"], "tags": ["prospect", "scouting"], "requires": ["prospect_name", "prospect_overall"], "tone": "hot_take",
    },
    {
        "text": "just spent 6 hours going through draft board. took {prospect_name} at {draft_pick}. every scouting report contradicts itself. I went with the tape",
        "angles": ["draft_board"], "tags": ["draft", "scouting"], "requires": ["prospect_name", "draft_pick"], "tone": "narrative",
    },
    {
        "text": "late-round dart at {prospect_name} in round {draft_round}. floor is a bust. ceiling is {prospect_ceiling} OVR. that's the bet",
        "angles": ["draft_board"], "tags": ["draft", "prospect"], "requires": ["prospect_name", "draft_round", "prospect_ceiling"], "tone": "analytical",
    },
    {
        "text": "prospect just jumped to {prospect_overall} OVR because of better linemates and real deployment. environment beats upside every time",
        "angles": ["draft_board"], "tags": ["development", "prospect"], "requires": ["prospect_overall"], "tone": "analytical",
    },
    {
        "text": "scouting confidence on {prospect_name} went from {prospect_floor} to {prospect_ceiling} OVR range in one weekend at the combine. now im paying attention",
        "angles": ["draft_board"], "tags": ["prospect", "scouting"], "requires": ["prospect_name", "prospect_floor", "prospect_ceiling"], "tone": "narrative",
    },
    {
        "text": "development camp report on {prospect_name}: everything we hoped. scouting confidence spiked. stash or push him?",
        "angles": ["draft_board"], "tags": ["prospect", "development"], "requires": ["prospect_name"], "tone": "analytical",
    },
    {
        "text": "moved up in the lottery and landed the {draft_pick}th pick. changed the {season} draft strategy entirely. big week",
        "angles": ["draft_board"], "tags": ["lottery", "draft"], "requires": ["draft_pick", "season"], "tone": "narrative",
    },
    {
        "text": "traded back in the draft, stacked picks. someone else just took who i wanted at {draft_pick}. rebuilds require patience i apparently don't have",
        "angles": ["draft_board"], "tags": ["draft", "rebuild"], "requires": ["draft_pick"], "tone": "chaos",
    },
    {
        "text": "lost the lottery at {team_record}. league is rigged. best odds. worst outcome. stacking assets anyway",
        "angles": ["draft_board"], "tags": ["lottery", "tank"], "requires": ["team_record"], "tone": "rant",
    },
    {
        "text": "called up {prospect_name} and immediately his confidence spiked. opportunity does more than a +3 OVR rating ever will",
        "angles": ["draft_board", "slump_watch"], "tags": ["callup", "development"], "requires": ["prospect_name"], "tone": "analytical",
    },
    {
        "text": "buried {prospect_name} in the AHL for 40 games. now the analytics look different. {prospect_overall} OVR on the way up",
        "angles": ["draft_board"], "tags": ["development", "prospect"], "requires": ["prospect_name", "prospect_overall"], "tone": "cold",
    },
    {
        "text": "this draft class is elite. {draft_class_rank} grade across the board. {team} has two first-rounders. this is the year",
        "angles": ["draft_board"], "tags": ["draft", "rebuild"], "requires": ["draft_class_rank", "team"], "tone": "hot_take",
    },

    # ── SLUMP WATCH / DEVELOPMENT ─────────────────────────────────────────

    {
        "text": "{name} went from {prior_overall} to {overall} OVR with nothing but more ice time and better deployment. the sim rewards opportunity",
        "angles": ["slump_watch"], "tags": ["development", "breakout"], "requires": ["name", "prior_overall", "overall"], "tone": "analytical",
    },
    {
        "text": "{name} in a slump. {points} points in {games_played} GP isn't the player we paid {cap_hit}M for. meeting is scheduled",
        "angles": ["slump_watch"], "tags": ["slump", "meeting"], "requires": ["name", "points", "games_played", "cap_hit"], "tone": "cold",
    },
    {
        "text": "analytics say {name} is fine. eye test says something is off. {xgf_pct}% xGF and .{ppg} PPG and im still worried",
        "angles": ["slump_watch"], "tags": ["analytics", "slump"], "requires": ["name", "xgf_pct", "ppg"], "tone": "analytical",
    },
    {
        "text": "{name} just had a career year at {points} points. {overall} OVR and a {ppg} PPG clip. extension is now a different conversation",
        "angles": ["slump_watch", "contract_battle"], "tags": ["breakout", "extension"], "requires": ["name", "points", "overall", "ppg"], "tone": "narrative",
    },
    {
        "text": "challenged {name} in a meeting after the performance dip. he pushed back. i respect it. lineup card doesn't change though",
        "angles": ["slump_watch", "locker_room"], "tags": ["meeting", "slump"], "requires": ["name"], "tone": "cold",
    },
    {
        "text": "{name} breakout started the moment we gave him a real role. {prior_overall} OVR in a bubble wrap, now {overall} OVR with real minutes",
        "angles": ["slump_watch", "draft_board"], "tags": ["breakout", "development"], "requires": ["name", "prior_overall", "overall"], "tone": "narrative",
    },
    {
        "text": "dropped {name} a line change. two games later he's producing. sometimes management is just getting out of the way",
        "angles": ["slump_watch"], "tags": ["slump", "coaching"], "requires": ["name"], "tone": "cold",
    },
    {
        "text": "{name} hasn't scored in {games_missed} games. the narrative is building. we protect him publicly, address it privately",
        "angles": ["slump_watch", "media_spin"], "tags": ["slump", "media"], "requires": ["name", "games_missed"], "tone": "narrative",
    },

    # ── GOALTENDING ───────────────────────────────────────────────────────

    {
        "text": "tandem is cooked. {name} at {save_pct} SV% and .{gaa} GAA through {games_played} GP. pulling the starter isn't an option but it's a conversation",
        "angles": ["goaltending"], "tags": ["goalie", "slump"], "requires": ["name", "save_pct", "gaa", "games_played"], "tone": "cold",
    },
    {
        "text": "goalie controversy starting. backup numbers better than starter. {name} at {save_pct} vs backup at slightly better. this week is important",
        "angles": ["goaltending", "locker_room"], "tags": ["goalie", "lineup"], "requires": ["name", "save_pct"], "tone": "narrative",
    },
    {
        "text": "{name} just posted a {save_pct} SV% in a back-to-back. {shutouts} shutouts this season. he's the reason we're in this race",
        "angles": ["goaltending", "win_now"], "tags": ["goalie", "breakout"], "requires": ["name", "save_pct", "shutouts"], "tone": "narrative",
    },
    {
        "text": "just signed a new goalie at {cap_hit}M. either genius or panic. history will decide. save percentage won't matter if we don't score",
        "angles": ["goaltending", "contract_battle"], "tags": ["goalie", "contract"], "requires": ["cap_hit"], "tone": "chaos",
    },

    # ── LEAGUE WIRE ───────────────────────────────────────────────────────

    {
        "text": "{team} at {team_record} and rank {league_rank}. this isn't the plan but the plan might still work. adjusting",
        "angles": ["league_wire"], "tags": ["rebuild", "rebuild"], "requires": ["team", "team_record", "league_rank"], "tone": "cold",
    },
    {
        "text": "{rival_team} just gave {rival_player} {rival_cap_hit}M x {rival_term}. absolute disaster of a contract. and it helps {team} immensely",
        "angles": ["league_wire"], "tags": ["rival", "contract", "cap"], "requires": ["rival_team", "rival_player", "rival_cap_hit", "rival_term", "team"], "tone": "hot_take",
    },
    {
        "text": "media is reporting {headline}. {reporter} dropped this with zero source. {team_record} record tells a different story",
        "angles": ["league_wire", "media_spin"], "tags": ["media", "rumor"], "requires": ["headline", "reporter", "team_record"], "tone": "rant",
    },
    {
        "text": "standings right now: {team} at {team_record} with {games_played} games left. playoff math is uncomfortable but doable",
        "angles": ["league_wire"], "tags": ["playoffs"], "requires": ["team", "team_record", "games_played"], "tone": "analytical",
    },
    {
        "text": "ownership isn't happy with {team_record}. conversation happened. the message: playoffs or else. timeline just accelerated",
        "angles": ["league_wire", "ownership_pressure"], "tags": ["owner", "rebuild"], "requires": ["team_record", "team"], "tone": "cold",
    },
    {
        "text": "all-star break. {team} at {team_record}. honest assessment: we need to be {wins} wins better in the second half. doable? yes. easy? no",
        "angles": ["league_wire"], "tags": ["playoffs", "rebuild"], "requires": ["team", "team_record", "wins"], "tone": "analytical",
    },
    {
        "text": "power play at {pp_pct}%. penalty kill at {pk_pct}%. special teams are costing {team} 3-5 wins this season. that's the whole story",
        "angles": ["league_wire"], "tags": ["coaching", "analytics"], "requires": ["pp_pct", "pk_pct", "team"], "tone": "analytical",
    },
    {
        "text": "rival just poached our coaching idea and it's working for them. flattering and infuriating simultaneously",
        "angles": ["league_wire"], "tags": ["rival", "coaching"], "requires": ["rival_team"], "tone": "chaos",
    },
    {
        "text": "second-half schedule is brutal. {games_played} games in 45 days with a {team_record} record to protect. find out who this team really is",
        "angles": ["league_wire"], "tags": ["playoffs", "win_now"], "requires": ["games_played", "team_record"], "tone": "narrative",
    },

    # ── INJURY WATCH ──────────────────────────────────────────────────────

    {
        "text": "{name} out with {injury_type}. {games_missed} games minimum. no timeline beyond that. this changes everything about the next 6 weeks",
        "angles": ["injury_watch"], "tags": ["injury", "lineup"], "requires": ["name", "injury_type", "games_missed"], "tone": "cold",
    },
    {
        "text": "injury to {name} during contract year. {cap_hit}M ask was already spicy. now it's complicated for everyone",
        "angles": ["injury_watch", "contract_battle"], "tags": ["injury", "contract"], "requires": ["name", "cap_hit"], "tone": "narrative",
    },
    {
        "text": "{name} cleared for return. we managed minutes through recovery, confidence at {confidence}. cautiously optimistic",
        "angles": ["injury_watch"], "tags": ["injury", "lineup"], "requires": ["name", "confidence"], "tone": "cold",
    },
    {
        "text": "prospect {prospect_name} went down in {league_of_origin}. development clock just paused for {games_missed} games minimum",
        "angles": ["injury_watch", "draft_board"], "tags": ["injury", "prospect"], "requires": ["prospect_name", "league_of_origin", "games_missed"], "tone": "cold",
    },

    # ── REBUILD MODE ──────────────────────────────────────────────────────

    {
        "text": "commit to the rebuild or don't. half-measures at {team_record} is how you end up mediocre for a decade",
        "angles": ["rebuild_mode"], "tags": ["rebuild", "tank"], "requires": ["team_record"], "tone": "hot_take",
    },
    {
        "text": "year {season} of the rebuild. {cap_space}M in space, {draft_pick} first-rounders incoming. the foundation is real",
        "angles": ["rebuild_mode"], "tags": ["rebuild"], "requires": ["season", "cap_space", "draft_pick"], "tone": "narrative",
    },
    {
        "text": "keeping {name} through the rebuild because culture matters. he understands the timeline. {cap_hit}M for that signal is a deal",
        "angles": ["rebuild_mode", "contract_battle"], "tags": ["rebuild", "culture", "contract"], "requires": ["name", "cap_hit"], "tone": "analytical",
    },
    {
        "text": "first full rebuild season complete. {team_record} as expected. the pipeline is real even if the W column isn't",
        "angles": ["rebuild_mode", "draft_board"], "tags": ["rebuild", "tank"], "requires": ["team_record", "team"], "tone": "cold",
    },
    {
        "text": "{team}'s rebuild is ahead of schedule. {team_record} looks better than we projected. ownership is quiet. that's the goal",
        "angles": ["rebuild_mode", "league_wire"], "tags": ["rebuild"], "requires": ["team", "team_record"], "tone": "narrative",
    },

    # ── WIN NOW ───────────────────────────────────────────────────────────

    {
        "text": "the window is open. {name} at {overall} OVR, core on reasonable deals, {cap_space}M to add. if not now, when",
        "angles": ["win_now"], "tags": ["win_now", "deadline"], "requires": ["name", "overall", "cap_space"], "tone": "hot_take",
    },
    {
        "text": "bought at the deadline. {trade_return} in, {trade_asset} and a pick out. this better work because we just leveraged the future",
        "angles": ["win_now", "trade_market"], "tags": ["deadline", "win_now", "trade"], "requires": ["trade_return", "trade_asset"], "tone": "narrative",
    },
    {
        "text": "cup run or cap disaster. no in between at {team_record} with this payroll. we're going for it",
        "angles": ["win_now"], "tags": ["win_now", "playoffs"], "requires": ["team_record"], "tone": "hot_take",
    },
    {
        "text": "veteran signed as a rental at {cap_hit}M for {term} years. no NMC, plays any role. exactly the personality we needed in this room",
        "angles": ["win_now", "contract_battle"], "tags": ["win_now", "free_agency"], "requires": ["cap_hit", "term"], "tone": "cold",
    },

    # ── CAP CRISIS ────────────────────────────────────────────────────────

    {
        "text": "cap situation for {team} in {season}: {cap_space}M in space and {name} still needs a new deal. this is how bad decisions compound",
        "angles": ["cap_crisis"], "tags": ["cap", "cap_crisis"], "requires": ["team", "season", "cap_space", "name"], "tone": "analytical",
    },
    {
        "text": "buried {name} at {cap_hit}M in the minors. dead cap is eating {cap_space}M this season. everything hurts",
        "angles": ["cap_crisis"], "tags": ["buried", "cap_crisis", "cap"], "requires": ["name", "cap_hit", "cap_space"], "tone": "rant",
    },
    {
        "text": "buyout on {name} costs {cap_hit}M over {term} years. keeping him costs {cap_hit}M now. neither is good. that's a cap crisis",
        "angles": ["cap_crisis"], "tags": ["buyout", "cap_crisis"], "requires": ["name", "cap_hit", "term"], "tone": "analytical",
    },

    # ── MEDIA SPIN ────────────────────────────────────────────────────────

    {
        "text": "{reporter} just dropped {headline} with 'sources close to the situation.' my sources say this is speculation dressed as fact",
        "angles": ["media_spin"], "tags": ["media", "rumor"], "requires": ["reporter", "headline"], "tone": "rant",
    },
    {
        "text": "media narrative on {team} right now: {headline}. {team_record} record. one of those is real information",
        "angles": ["media_spin"], "tags": ["media"], "requires": ["team", "headline", "team_record"], "tone": "cold",
    },
    {
        "text": "press conference went sideways because of one honest answer. now {headline} is the story. this is why GMs say nothing",
        "angles": ["media_spin"], "tags": ["media", "reputation"], "requires": ["headline"], "tone": "narrative",
    },
    {
        "text": "anonymous source just dropped a rumor about {name}. credibility: {rumor_source}. acting accordingly",
        "angles": ["media_spin", "trade_market"], "tags": ["rumor", "media"], "requires": ["name", "rumor_source"], "tone": "cold",
    },
    {
        "text": "{reporter}'s timeline on {name} is completely wrong. the paperwork on that move happened 3 weeks ago. media is always last",
        "angles": ["media_spin"], "tags": ["media", "trade"], "requires": ["reporter", "name"], "tone": "hot_take",
    },

    # ── OWNERSHIP PRESSURE ────────────────────────────────────────────────

    {
        "text": "ownership wants a cup in {term} years. my model says {term}+2 years. one of us is right and the other signs my checks",
        "angles": ["ownership_pressure"], "tags": ["owner", "rebuild"], "requires": ["term"], "tone": "chaos",
    },
    {
        "text": "payroll meeting with ownership: {cap_hit}M max spend. the player i want costs {cap_hit}M+. bridging that gap is the actual job",
        "angles": ["ownership_pressure", "contract_battle"], "tags": ["owner", "cap"], "requires": ["cap_hit"], "tone": "narrative",
    },
    {
        "text": "ownership approved the rebuild. that means {term} years of patience before the mandate shifts. clock starts now",
        "angles": ["ownership_pressure", "rebuild_mode"], "tags": ["owner", "rebuild"], "requires": ["term"], "tone": "cold",
    },

    # ── DYNASTY LEGACY ────────────────────────────────────────────────────

    {
        "text": "just looked at the legacy wall. {team} has made {wins} playoff appearances in {term} seasons under this staff. the work shows",
        "angles": ["dynasty_legacy"], "tags": ["milestone", "meta"], "requires": ["team", "wins", "term"], "tone": "narrative",
    },
    {
        "text": "franchise record for {name} tonight. {points} career points in a {team} jersey. this is what building looks like",
        "angles": ["dynasty_legacy"], "tags": ["milestone"], "requires": ["name", "points", "team"], "tone": "narrative",
    },
    {
        "text": "retired {name}'s number. drafted him, developed him, won with him. the full arc. this is the real franchise mode payoff",
        "angles": ["dynasty_legacy"], "tags": ["retirement", "milestone"], "requires": ["name"], "tone": "narrative",
    },
    {
        "text": "season {season} recap: {team} went {team_record}. not the result we wanted. history remembers the decision-making, not just the wins",
        "angles": ["dynasty_legacy"], "tags": ["meta", "postmortem"], "requires": ["season", "team", "team_record"], "tone": "cold",
    },
    {
        "text": "gm career stat: {wins} wins, {losses} losses, {wins} playoff appearances, {wins} cup runs. the record is what it is",
        "angles": ["dynasty_legacy"], "tags": ["meta"], "requires": ["wins", "losses"], "tone": "cold",
    },

    # ── ADDITIONAL HOT TAKES / NARRATIVE ──────────────────────────────────

    {
        "text": "three scouts, three completely different reads on {prospect_name}. I'm the tiebreaker. love this job",
        "angles": ["draft_board"], "tags": ["scouting", "draft"], "requires": ["prospect_name"], "tone": "narrative",
    },
    {
        "text": "just realized my entire franchise philosophy emerged from decisions i made under pressure. turns out i'm an analytics-first org. didn't choose it",
        "angles": ["dynasty_legacy", "rebuild_mode"], "tags": ["meta", "culture"], "requires": [], "tone": "narrative",
    },
    {
        "text": "{name} and {rival_player} got into it publicly. both are on my radar now for different reasons",
        "angles": ["locker_room", "trade_market"], "tags": ["chemistry", "rival"], "requires": ["name", "rival_player"], "tone": "cold",
    },
    {
        "text": "just hit the {salary_cap}M cap ceiling for the first time. this is what winning looks like. it's uncomfortable",
        "angles": ["cap_crisis", "win_now"], "tags": ["cap", "win_now"], "requires": ["salary_cap"], "tone": "narrative",
    },
    {
        "text": "{name}'s life event changed his priorities heading into free agency. money still talks but location now matters more. adjusted the pitch",
        "angles": ["contract_battle", "locker_room"], "tags": ["life", "free_agency"], "requires": ["name"], "tone": "analytical",
    },
    {
        "text": "agent for {name} wants a no-movement clause. I want term. we're negotiating two different contracts right now",
        "angles": ["contract_battle"], "tags": ["contract", "ntc", "agent"], "requires": ["name", "agent_name"], "tone": "cold",
    },
    {
        "text": "{coach_name}'s system has transformed {name}'s defensive numbers. {overall} OVR is underselling what he brings. extension makes sense",
        "angles": ["slump_watch", "contract_battle"], "tags": ["coaching", "extension", "analytics"], "requires": ["coach_name", "name", "overall"], "tone": "analytical",
    },
    {
        "text": "organization reputation is '{franchise_rep}'. that took {term} years to build. agents know it. prospects know it. it moves free agency",
        "angles": ["dynasty_legacy"], "tags": ["reputation", "culture"], "requires": ["franchise_rep", "term"], "tone": "cold",
    },
]


# ---------------------------------------------------------------------------
# 30 MEME / SHITPOST TWEET TEMPLATES
# ---------------------------------------------------------------------------

MEME_TWEET_TEMPLATES: List[TweetTemplate] = [
    {
        "text": "me: *has 1 good deadline*\nmedia: genius gm, untouchable\nme: *one bad move*\nmedia: {team} is cooked, fire everyone",
        "tags": ["meme"], "requires": ["team"], "tone": "chaos",
    },
    {
        "text": "scouting report: 'limited NHL upside'\nalso {name}: {overall} OVR in year 3\nme: …did we read the same document",
        "tags": ["meme"], "requires": ["name", "overall"], "tone": "chaos",
    },
    {
        "text": "cpu gm just gave a {age}yo {cap_hit}M to play depth minutes\nim not saying the ai is bad at contracts but\n*takes notes aggressively*",
        "tags": ["meme", "cpu"], "requires": ["age", "cap_hit"], "tone": "chaos",
    },
    {
        "text": "agent: {name} wants {cap_hit}M\nme: thats insane\nalso me: *checks {cap_space}M cap space*\nalso also me: he's right though",
        "tags": ["meme", "contract"], "requires": ["name", "cap_hit", "cap_space"], "tone": "chaos",
    },
    {
        "text": "my plan was rebuild\nmarket said contend\nownership said stanley cup\n{season}: {team} at {team_record} and im cooked",
        "tags": ["meme", "rebuild"], "requires": ["season", "team", "team_record"], "tone": "chaos",
    },
    {
        "text": "promised {name} more ice time in a meeting\nhockey gods: *immediately injure three guys*\nme: the simulation is conscious and hates me",
        "tags": ["meme", "injury"], "requires": ["name"], "tone": "chaos",
    },
    {
        "text": "'trust the process' i whisper as {team} sits {team_record}\n{term} years later: actually worked perfectly\nprocess vindicated",
        "tags": ["meme", "rebuild"], "requires": ["team", "team_record", "term"], "tone": "chaos",
    },
    {
        "text": "{rival_team} gave up their franchise piece for futures\neveryone: that's cooked\nme: 👀 checking back in 4 years",
        "tags": ["meme", "rival"], "requires": ["rival_team"], "tone": "chaos",
    },
    {
        "text": "draft board:\nround 1: 'elite potential, franchise upside'\nround 5: 'limited ceiling, project'\nalso round 5 guy: {prospect_overall} OVR in the show 👁️👄👁️",
        "tags": ["meme", "draft"], "requires": ["prospect_overall"], "tone": "chaos",
    },
    {
        "text": "entire franchise built on hope that {name} hits {overall} OVR ceiling\nif he doesn't im literally done\nits fine\nits fine\neverything is fine",
        "tags": ["meme", "development"], "requires": ["name", "overall"], "tone": "chaos",
    },
    {
        "text": "gm brain at 2am:\n- should i trade {name}?\n- what's his actual ceiling?\n- is {cap_hit}M too much?\n- what if he's actually underrated?\n[no sleep]",
        "tags": ["meme", "trade"], "requires": ["name", "cap_hit"], "tone": "chaos",
    },
    {
        "text": "agent says {name} wants 'winning culture'\nalso the market: {rival_cap_hit}M AAV from a 15th-place team\nalright fair",
        "tags": ["meme", "agent", "contract"], "requires": ["name", "rival_cap_hit"], "tone": "chaos",
    },
    {
        "text": "me before deadline: disciplined, patient, building the right way\nme 10 minutes before deadline: yeah okay fine let's trade for {name}",
        "tags": ["meme", "deadline", "trade"], "requires": ["name"], "tone": "chaos",
    },
    {
        "text": "scouting department: 'we love {prospect_name} at this value'\nalso scouting department, week later: 'we're less confident now'\nme: ???",
        "tags": ["meme", "scouting"], "requires": ["prospect_name"], "tone": "chaos",
    },
    {
        "text": "cpu gm: *builds perfect roster on a {cap_space}M budget*\nalso cpu gm: *gives {rival_player} {rival_cap_hit}M in the same week*\nhow",
        "tags": ["meme", "cpu", "cap"], "requires": ["cap_space", "rival_player", "rival_cap_hit"], "tone": "chaos",
    },
    {
        "text": "locker room unity: 94\nmorale: 91\nalso the media: '{team} is fractured, sources say'\nsources are lying",
        "tags": ["meme", "media", "locker_room"], "requires": ["team"], "tone": "chaos",
    },
    {
        "text": "analyst: {name} is clearly in decline\nme: he's {age}. he's fine\nalso {name}: immediately drops {delta_overall} OVR\n\n...okay",
        "tags": ["meme", "analytics", "slump"], "requires": ["name", "age", "delta_overall"], "tone": "chaos",
    },
    {
        "text": "ownership: 'make the playoffs or else'\nme: it's game 14\nowner: 'yes'\n\n[franchise anxiety intensifies]",
        "tags": ["meme", "owner"], "requires": [], "tone": "chaos",
    },
    {
        "text": "prospect ranked {draft_pick}th overall\nmy scouts: 'elite, can't miss'\ntheir scouts: 'project, developmental'\nme: *takes him, shakes*",
        "tags": ["meme", "draft", "scouting"], "requires": ["draft_pick"], "tone": "chaos",
    },
    {
        "text": "'{name} demands a trade'\nbro has {morale} morale and hasn't been scratched once\nthe character trait simulation is working as intended",
        "tags": ["meme", "trade_demand", "character"], "requires": ["name", "morale"], "tone": "chaos",
    },
    {
        "text": "my cap projection for year 5:\nme: it'll work out\nalso me: *stares at {cap_space}M of space disappearing in real time*\nit won't work out",
        "tags": ["meme", "cap", "cap_crisis"], "requires": ["cap_space"], "tone": "chaos",
    },
    {
        "text": "three-team trade on the table\nteam A: wants picks\nteam B: wants a player\nme: wants both and giving neither\ndeadline in 4 hours",
        "tags": ["meme", "trade", "deadline"], "requires": [], "tone": "chaos",
    },
    {
        "text": "coach {coach_name}: 'i need depth scoring'\nme: i have {cap_space}M\ncoach: 'so no depth scoring'\nme: correct",
        "tags": ["meme", "cap", "coaching"], "requires": ["coach_name", "cap_space"], "tone": "chaos",
    },
    {
        "text": "the era where i gave {name} a {cap_hit}M extension and immediately watched him decline is called the {season} era\nwe don't talk about {season}",
        "tags": ["meme", "contract", "slump"], "requires": ["name", "cap_hit", "season"], "tone": "chaos",
    },
    {
        "text": "rebuild status update:\nyear 1: terrible\nyear 2: still terrible\nyear 3: aggressively terrible\nyear 4: wait, {team_record}? oh. oh no we're good",
        "tags": ["meme", "rebuild"], "requires": ["team_record"], "tone": "chaos",
    },
    {
        "text": "cpu team simulating trades at 3am while im asleep\nwoke up to: {rival_team} acquires {name}\nIM WHAT",
        "tags": ["meme", "cpu", "trade"], "requires": ["rival_team", "name"], "tone": "chaos",
    },
    {
        "text": "journalism:\n{reporter}: 'sources say {team} close to {headline}'\nalso {reporter}: wrong\nalso also {reporter}: 'new sources say—'\n\nblocked",
        "tags": ["meme", "media"], "requires": ["reporter", "team", "headline"], "tone": "chaos",
    },
    {
        "text": "asked {name} to waive his {ntc_type}\nhim: 'i'll think about it'\nhim, 3 weeks later: 'no'\nme: [entire trade collapses]",
        "tags": ["meme", "ntc", "trade"], "requires": ["name", "ntc_type"], "tone": "chaos",
    },
    {
        "text": "honestly the {season} draft class saved this franchise\nme, who drafted {prospect_name} in round {draft_round} and expected nothing: smart",
        "tags": ["meme", "draft", "prospect"], "requires": ["season", "prospect_name", "draft_round"], "tone": "chaos",
    },
    {
        "text": "my gm philosophy:\nyear 1-3: patience, culture, development\nyear 4: *one bad week*\nyear 4 still: okay fine let's win now",
        "tags": ["meme", "rebuild", "win_now"], "requires": [], "tone": "chaos",
    },
]


# ---------------------------------------------------------------------------
# 100 REDDIT THREAD TEMPLATES
# ---------------------------------------------------------------------------

REDDIT_THREAD_TEMPLATES: List[RedditTemplate] = [

    # ── SCOUTING / DRAFT ──────────────────────────────────────────────────

    {
        "title": "Why your read on {prospect_name} is probably wrong — and how to fix it",
        "body": "Scouting confidence on {prospect_name} was sitting in the {prospect_floor}-{prospect_ceiling} OVR range with total disagreement between my staff. Through {games_played} GP in {league_of_origin} he posted {points} points. At what point does production override projection?",
        "flair": "Scouting", "angles": ["draft_board"], "tags": ["scouting", "prospect"],
        "requires": ["prospect_name", "prospect_floor", "prospect_ceiling"], "format": "discussion",
    },
    {
        "title": "Draft day breakdown: round-by-round decisions at pick {draft_pick}",
        "body": "Full walkthrough of my {season} draft logic. Why I moved up, who I passed on at {draft_pick}, and which late-round swings at round {draft_round} already have me nervous/excited.",
        "flair": "Draft", "angles": ["draft_board"], "tags": ["draft"],
        "requires": ["draft_pick", "season"], "format": "breakdown",
    },
    {
        "title": "{draft_class_rank} draft class eval: what I got right, what I missed",
        "body": "Season-end audit of every pick. Scouting grade at draft vs current OVR. Which prospects exceeded their ceiling, which ones capped out early. What I'd change.",
        "flair": "Draft", "angles": ["draft_board"], "tags": ["draft", "postmortem"],
        "requires": ["draft_class_rank"], "format": "postmortem",
    },
    {
        "title": "Lottery loss at {team_record} — the probabilistic acceptance guide",
        "body": "Best odds. Worst outcome. {team} had {draft_pick}% lottery odds and got nothing. How do you rebuild your draft capital after this? Asset accumulation thread.",
        "flair": "Rebuild", "angles": ["draft_board"], "tags": ["lottery", "tank", "rebuild"],
        "requires": ["team_record", "draft_pick", "team"], "format": "guide",
    },
    {
        "title": "{prospect_name} tape breakdown: what the OVR isn't telling you",
        "body": "Projected at {prospect_overall} OVR with {prospect_conf}% scouting confidence. The tape suggests something different. Breaking down the evaluation gap and when to trust the model vs what you see.",
        "flair": "Scouting", "angles": ["draft_board"], "tags": ["scouting", "prospect"],
        "requires": ["prospect_name", "prospect_overall", "prospect_conf"], "format": "breakdown",
    },
    {
        "title": "Hidden gems: late-round prospects that outperformed projections",
        "body": "Started with a {draft_round}rd round pick and {prospect_floor} OVR floor. {prospect_name} is now at {prospect_overall} OVR. What did we see that others didn't, and how do you find the next one?",
        "flair": "Scouting", "angles": ["draft_board"], "tags": ["prospect", "scouting"],
        "requires": ["draft_round", "prospect_name", "prospect_overall"], "format": "breakdown",
    },
    {
        "title": "Development acceleration: what actually moves the OVR needle",
        "body": "{name} went from {prior_overall} to {overall} OVR in one season. Nothing changed in his potential — everything changed in his deployment, linemates, and minutes. Full environment breakdown.",
        "flair": "Development", "angles": ["draft_board", "slump_watch"], "tags": ["development", "breakout"],
        "requires": ["name", "prior_overall", "overall"], "format": "breakdown",
    },
    {
        "title": "Stash vs push: when to keep prospects in the minors",
        "body": "{prospect_name} at {prospect_overall} OVR is ready by every metric. But the NHL roster at {team_record} doesn't have room. How long do you stash before it damages development?",
        "flair": "Development", "angles": ["draft_board"], "tags": ["prospect", "development"],
        "requires": ["prospect_name", "prospect_overall", "team_record"], "format": "discussion",
    },
    {
        "title": "Scouting staff accuracy audit — mine is broken and here's the evidence",
        "body": "Tracked every projection vs reality over {term} seasons. My scouts were right {wins}% of the time in round 1. Round 4+? Coin flip. Should we trust late reports at all?",
        "flair": "Scouting", "angles": ["draft_board"], "tags": ["scouting", "meta"],
        "requires": ["term", "wins"], "format": "breakdown",
    },
    {
        "title": "Is {prospect_name} actually worth two firsts? The case for and against",
        "body": "Ceiling {prospect_ceiling} OVR, {prospect_conf}% confidence. Position scarcity premium is real. But trading two firsts for a projection is terrifying. Make the case either way.",
        "flair": "Trade", "angles": ["draft_board", "trade_market"], "tags": ["prospect", "trade"],
        "requires": ["prospect_name", "prospect_ceiling", "prospect_conf"], "format": "discussion",
    },

    # ── CONTRACT / CAP ────────────────────────────────────────────────────

    {
        "title": "The {name} extension I almost signed — and why walking away was right",
        "body": "He wanted {cap_hit}M x {term}. I offered {cap_hit}M x 3. Personality, morale ({morale}), and locker room dynamics made me think twice. The cap math that saved the franchise.",
        "flair": "Contracts", "angles": ["contract_battle"], "tags": ["contract", "extension"],
        "requires": ["name", "cap_hit", "term", "morale"], "format": "postmortem",
    },
    {
        "title": "Year-5 cap explosion survival guide for {team}",
        "body": "Starting {season} with {cap_space}M space and {wins} guys needing new deals. Front-load vs back-load, retained salary math, internal raises — full cap planning thread.",
        "flair": "Contracts", "angles": ["contract_battle", "cap_crisis"], "tags": ["cap", "extension"],
        "requires": ["team", "season", "cap_space", "wins"], "format": "guide",
    },
    {
        "title": "Offer sheet strategy: when to pull the trigger on {rival_player}",
        "body": "RFA at {prospect_overall} OVR on {rival_team}'s books. Our {offer_sheet_amt}M offer sheet would force their hand. The risk/reward of burning a rival relationship for one player.",
        "flair": "Contracts", "angles": ["contract_battle"], "tags": ["offer_sheet"],
        "requires": ["rival_player", "rival_team", "offer_sheet_amt"], "format": "discussion",
    },
    {
        "title": "Arbitration with {name}: the aftermath no one talks about",
        "body": "We won at {cap_hit}M. He wanted more. The relationship damage after an arbitration ruling — how long it takes to recover, what it costs in morale, future negotiations.",
        "flair": "Contracts", "angles": ["contract_battle"], "tags": ["arbitration", "contract"],
        "requires": ["name", "cap_hit"], "format": "postmortem",
    },
    {
        "title": "Buried contract on {name} at {cap_hit}M: escape routes ranked",
        "body": "{name} is {overall} OVR and {cap_hit}M against the cap with {years_remaining} years remaining. Buyout math, trade potential, waiver risk — what's actually available to you?",
        "flair": "Contracts", "angles": ["cap_crisis"], "tags": ["buried", "buyout", "cap_crisis"],
        "requires": ["name", "cap_hit", "overall", "years_remaining"], "format": "guide",
    },
    {
        "title": "ELC exploitation: maximizing your window with {name} on entry-level",
        "body": "{name} at {overall} OVR for {cap_hit}M is the most important player on the cap sheet. How do you build around cheap production before the raise hits in {season}?",
        "flair": "Contracts", "angles": ["contract_battle", "win_now"], "tags": ["contract", "win_now"],
        "requires": ["name", "overall", "cap_hit", "season"], "format": "guide",
    },
    {
        "title": "Free agency day 1 losers — and why it's okay",
        "body": "Lost {name} to {rival_team} at {rival_cap_hit}M. {cap_space}M still available. Sometimes the best move is letting the market overpay and pivoting to the next target.",
        "flair": "Contracts", "angles": ["contract_battle"], "tags": ["free_agency"],
        "requires": ["name", "rival_team", "rival_cap_hit", "cap_space"], "format": "postmortem",
    },
    {
        "title": "Agent relationships: the long-term cost of a bad negotiation",
        "body": "Lowballed {name} at {cap_hit}M when market was clearly higher. Agent {agent_name} has {wins} clients in the league. The compound cost of one bad deal across future negotiations.",
        "flair": "Contracts", "angles": ["contract_battle"], "tags": ["agent", "reputation"],
        "requires": ["name", "cap_hit", "agent_name", "wins"], "format": "breakdown",
    },
    {
        "title": "Cap structure by franchise phase: what to prioritize and when",
        "body": "Tank phase vs build phase vs contend phase contract strategy. How {team}'s {cap_space}M space changes meaning at {team_record}. Thread for long-term cap thinking.",
        "flair": "Contracts", "angles": ["contract_battle", "rebuild_mode"], "tags": ["cap", "meta"],
        "requires": ["team", "cap_space", "team_record"], "format": "guide",
    },
    {
        "title": "No-trade and no-movement clauses: the negotiation most GMs lose",
        "body": "{name} wanted a {ntc_type}. I gave a limited NMC instead. The difference matters when you're trying to move him at the deadline {years_remaining} years later.",
        "flair": "Contracts", "angles": ["contract_battle", "trade_market"], "tags": ["ntc", "contract"],
        "requires": ["name", "ntc_type", "years_remaining"], "format": "guide",
    },

    # ── TRADE / DEADLINE ──────────────────────────────────────────────────

    {
        "title": "Trade deadline strategy: the full decision tree at {team_record}",
        "body": "Sell vs hold vs buy framework at {league_rank} in conference with {cap_space}M in space. Rental value calculations, draft capital math, competitive window math.",
        "flair": "Trade", "angles": ["trade_market"], "tags": ["deadline", "trade"],
        "requires": ["team_record", "league_rank", "cap_space"], "format": "guide",
    },
    {
        "title": "{name} trade postmortem: what went wrong",
        "body": "Moved {name} at {ppg} PPG and {overall} OVR for {trade_return}. Locker room unity dropped to {locker_unity}. Six months later — did I read the return correctly?",
        "flair": "Trade", "angles": ["trade_market", "locker_room"], "tags": ["trade", "postmortem"],
        "requires": ["name", "ppg", "overall", "trade_return", "locker_unity"], "format": "postmortem",
    },
    {
        "title": "Three trade scenarios for {name} — which one do you take?",
        "body": "Option A: prospect + picks. Option B: NHL player + cap relief. Option C: wait and trade deadline premium. Modeling the probability-weighted outcome of each.",
        "flair": "Trade", "angles": ["trade_market"], "tags": ["trade"],
        "requires": ["name"], "format": "discussion",
    },
    {
        "title": "The trade that looked terrible and aged incredibly well",
        "body": "Everyone called it a panic move when I gave up {trade_asset} for {trade_return}. {season} later — full accounting of why the analytics said yes when the optics said no.",
        "flair": "Trade", "angles": ["trade_market"], "tags": ["trade"],
        "requires": ["trade_asset", "trade_return", "season"], "format": "breakdown",
    },
    {
        "title": "CPU trading behavior: patterns, tells, and how to exploit them",
        "body": "{rival_team} trades the same profile of player every deadline. {rival_player} getting moved at {rival_cap_hit}M proves the pattern. What CPU GMs consistently miss.",
        "flair": "Meta", "angles": ["trade_market"], "tags": ["cpu", "trade"],
        "requires": ["rival_team", "rival_player", "rival_cap_hit"], "format": "breakdown",
    },
    {
        "title": "When a player's NTC makes your trade impossible — solutions thread",
        "body": "{name} has a {ntc_type} and won't waive for the destination we need. Meeting went nowhere at morale {morale}. What are the real options when the player holds all the cards?",
        "flair": "Trade", "angles": ["trade_market", "locker_room"], "tags": ["ntc", "trade"],
        "requires": ["name", "ntc_type", "morale"], "format": "discussion",
    },
    {
        "title": "Retained salary trades: when to pay the premium and when to walk",
        "body": "Retained {cap_hit}M to move {name}. Dead cap math over {years_remaining} years vs the prospect haul we got. Was the premium worth the asset return?",
        "flair": "Trade", "angles": ["trade_market", "cap_crisis"], "tags": ["trade", "cap"],
        "requires": ["cap_hit", "name", "years_remaining"], "format": "breakdown",
    },
    {
        "title": "How I almost ruined my franchise at the trade deadline — cautionary tale",
        "body": "Was {term} hours from trading {trade_asset} for a rental at {cap_hit}M. What stopped me, what the analytics said, and why deadline panic is a real psychological trap.",
        "flair": "Trade", "angles": ["trade_market"], "tags": ["deadline", "trade", "postmortem"],
        "requires": ["term", "trade_asset", "cap_hit"], "format": "postmortem",
    },

    # ── LOCKER ROOM / CULTURE ─────────────────────────────────────────────

    {
        "title": "Locker room conflict cost me a Stanley Cup window — full timeline",
        "body": "Unity was {locker_unity} going into the stretch run. A move involving {name} broke two relationships I didn't know were load-bearing. How one trade can undo three years of culture.",
        "flair": "Culture", "angles": ["locker_room"], "tags": ["chemistry", "culture"],
        "requires": ["locker_unity", "name"], "format": "postmortem",
    },
    {
        "title": "Player meetings guide: what conversations actually change outcomes",
        "body": "Tested every meeting type across multiple seasons. Promised {name} more ice time (morale {morale}). Which commitments move the needle, which ones backfire, and what you can't unsay.",
        "flair": "Culture", "angles": ["locker_room"], "tags": ["meeting", "culture"],
        "requires": ["name", "morale"], "format": "guide",
    },
    {
        "title": "The captaincy conversation that went sideways",
        "body": "Offered {name} the C. He asked for 48 hours. Room found out before he answered. Three different players now have opinions. How to recover when the selection process becomes a storyline.",
        "flair": "Culture", "angles": ["locker_room"], "tags": ["leadership", "culture"],
        "requires": ["name"], "format": "postmortem",
    },
    {
        "title": "How I accidentally built elite chemistry at {team}",
        "body": "Unity at {locker_unity}, morale {morale}. I didn't plan this — it emerged from specific player acquisitions and coaching philosophy over {term} seasons. Breaking down what actually compounds.",
        "flair": "Culture", "angles": ["locker_room"], "tags": ["culture", "chemistry"],
        "requires": ["team", "locker_unity", "morale", "term"], "format": "breakdown",
    },
    {
        "title": "Mentor-prospect pairing: which veteran traits actually transfer",
        "body": "Put {name} ({overall} OVR) in a mentor role for {prospect_name}. Character interaction logged measurable changes over {games_played} games. Thread on what transfers and what doesn't.",
        "flair": "Development", "angles": ["locker_room", "draft_board"], "tags": ["leadership", "development"],
        "requires": ["name", "overall", "prospect_name", "games_played"], "format": "breakdown",
    },
    {
        "title": "Character system deep dive: why two {overall} OVR players react completely differently",
        "body": "Same rating, opposite reactions to scratches, lineup changes, trade rumors. {name} vs another {overall} OVR forward — the hidden attributes that make the simulation believable.",
        "flair": "Culture", "angles": ["locker_room"], "tags": ["character"],
        "requires": ["overall", "name"], "format": "breakdown",
    },
    {
        "title": "The trade demand I should have seen coming — dissatisfaction system explained",
        "body": "{name}'s heat reached {heat} before I even noticed. Escalation stages, subtle signals, what the meeting history was telling me. Full postmortem on a preventable crisis.",
        "flair": "Culture", "angles": ["locker_room", "trade_market"], "tags": ["trade_demand", "character"],
        "requires": ["name", "heat"], "format": "postmortem",
    },
    {
        "title": "Off-ice events that changed my franchise — and how to manage them",
        "body": "Three life events this season affected player priorities in negotiations. {name}'s situation shifted the {cap_hit}M conversation entirely. The simulation respects personhood and it matters.",
        "flair": "Culture", "angles": ["locker_room", "contract_battle"], "tags": ["life", "culture"],
        "requires": ["name", "cap_hit"], "format": "breakdown",
    },
    {
        "title": "Relationship map of {team}'s org — vulnerability analysis",
        "body": "Mapped every significant player relationship. Two pairs are load-bearing for team unity. One departure could collapse it. What does your org's dependency graph look like?",
        "flair": "Culture", "angles": ["locker_room"], "tags": ["culture", "chemistry"],
        "requires": ["team"], "format": "discussion",
    },
    {
        "title": "Promise tracking: what broke my franchise reputation with players",
        "body": "Made {wins} promises across {term} seasons. Kept {losses}% of them. The specific broken commitments that damaged trust — and how reputation follows you across negotiations for years.",
        "flair": "Culture", "angles": ["locker_room", "dynasty_legacy"], "tags": ["reputation", "meeting"],
        "requires": ["wins", "term", "losses"], "format": "postmortem",
    },
    {
        "title": "Coaching decisions that split the locker room — what I'd change",
        "body": "Playing time allocation controversy involving {name} dropped unity to {locker_unity}. Which coaching calls create chemistry damage, and when is performance worth the locker room cost?",
        "flair": "Culture", "angles": ["locker_room"], "tags": ["coaching", "chemistry"],
        "requires": ["name", "locker_unity"], "format": "postmortem",
    },

    # ── REBUILD / TANKING ─────────────────────────────────────────────────

    {
        "title": "8-season rebuild timeline for {team} — the full arc",
        "body": "Year-by-year breakdown from {team_record} at the bottom to {wins} playoff appearances. What the early pain was for, when the prospects arrived, which draft classes carried the load.",
        "flair": "Rebuild", "angles": ["rebuild_mode", "draft_board"], "tags": ["rebuild", "meta"],
        "requires": ["team", "team_record", "wins"], "format": "breakdown",
    },
    {
        "title": "Tank or retool? The decision framework nobody explains clearly",
        "body": "At {team_record} with {cap_space}M and {league_rank} in the league — is the talent gap actually too large to bridge, or are we one piece away? Expected value math for both paths.",
        "flair": "Rebuild", "angles": ["rebuild_mode"], "tags": ["tank", "rebuild"],
        "requires": ["team_record", "cap_space", "league_rank"], "format": "guide",
    },
    {
        "title": "The rebuild I blew by contending too early — postmortem",
        "body": "Had the assets and the patience. Then ownership pressure at {team_record} pushed a premature win-now pivot. Three years of cap inefficiency later, explaining how it collapsed.",
        "flair": "Rebuild", "angles": ["rebuild_mode", "ownership_pressure"], "tags": ["rebuild", "owner"],
        "requires": ["team_record"], "format": "postmortem",
    },
    {
        "title": "Keeping a star through the rebuild: {name} case study",
        "body": "{name} at {overall} OVR agreed to stay through {term} years of rebuild. The messaging, the promises, the morale management at {morale}. How to keep your anchor without lying to him.",
        "flair": "Rebuild", "angles": ["rebuild_mode", "locker_room"], "tags": ["rebuild", "culture"],
        "requires": ["name", "overall", "term", "morale"], "format": "breakdown",
    },
    {
        "title": "Why my second rebuild worked when the first didn't — {team}",
        "body": "First attempt failed because of timeline acceleration. Second attempt: different patience threshold, better asset management, same ownership. What structurally changed at {team_record}.",
        "flair": "Rebuild", "angles": ["rebuild_mode"], "tags": ["rebuild", "postmortem"],
        "requires": ["team", "team_record"], "format": "postmortem",
    },
    {
        "title": "Asset accumulation without full tanking — is it possible?",
        "body": "{team} at {team_record} — not bad enough to tank, not good enough to compete. How to build draft capital, develop prospects, and stay competitive simultaneously.",
        "flair": "Rebuild", "angles": ["rebuild_mode", "trade_market"], "tags": ["rebuild", "trade"],
        "requires": ["team", "team_record"], "format": "discussion",
    },

    # ── WIN NOW / CONTENDER ───────────────────────────────────────────────

    {
        "title": "The contender window: how long is {team} actually good?",
        "body": "{name} at {age} with {years_remaining} on his deal. {team_record} now. When does the window close, and what's the maximum leverage point before it does?",
        "flair": "Win Now", "angles": ["win_now"], "tags": ["win_now"],
        "requires": ["name", "age", "years_remaining", "team_record", "team"], "format": "analytical",
    },
    {
        "title": "Deadline buy autopsy: what {trade_return} cost us long-term",
        "body": "Gave up {trade_asset} and a pick for a rental at {cap_hit}M. Didn't win the cup. Now accounting for the long-term cost to the pipeline. Was the swing worth taking?",
        "flair": "Win Now", "angles": ["win_now", "trade_market"], "tags": ["deadline", "win_now"],
        "requires": ["trade_return", "trade_asset", "cap_hit"], "format": "postmortem",
    },
    {
        "title": "Going for it: how to build a genuine cup contender at {team_record}",
        "body": "{cap_space}M in space, {name} at {overall} OVR, and a legitimate shot. The buy-now strategy, rental market approach, and what a real push looks like without blowing up the future.",
        "flair": "Win Now", "angles": ["win_now"], "tags": ["win_now", "deadline"],
        "requires": ["cap_space", "name", "overall", "team_record"], "format": "guide",
    },

    # ── GOALTENDING ───────────────────────────────────────────────────────

    {
        "title": "Goalie controversy: pulling {name} at {save_pct} SV% — right or wrong?",
        "body": "{gaa} GAA and {save_pct} SV% through {games_played} GP. Confidence at {confidence}. Backup numbers slightly better. When do you pull the trigger on a starter vs protect their confidence?",
        "flair": "Roster", "angles": ["goaltending"], "tags": ["goalie", "lineup"],
        "requires": ["name", "save_pct", "gaa", "games_played", "confidence"], "format": "discussion",
    },
    {
        "title": "Goaltending tandem strategy: the {name} decision tree",
        "body": "{name} at {cap_hit}M is the starter. {save_pct} SV% says starter, {shutouts} shutouts confirm it — but backup is cheaper and comparable. The tandem math nobody likes to do.",
        "flair": "Contracts", "angles": ["goaltending", "contract_battle"], "tags": ["goalie", "contract"],
        "requires": ["name", "cap_hit", "save_pct", "shutouts"], "format": "breakdown",
    },
    {
        "title": "Hot goalie playoff run: how much does {name}'s {save_pct} SV% actually matter?",
        "body": "In a real playoff run, goaltending variance peaks. {name} posted {save_pct} in the regular season. What's the realistic expectation, and how much is your window dependent on goaltending?",
        "flair": "Playoffs", "angles": ["goaltending", "win_now"], "tags": ["goalie", "playoffs"],
        "requires": ["name", "save_pct"], "format": "discussion",
    },

    # ── INJURY ────────────────────────────────────────────────────────────

    {
        "title": "{name}'s injury recovery: what the timeline actually looks like",
        "body": "{injury_type} with {games_missed} games projected. Confidence at {confidence} post-injury. How injury affects development trajectory, contract negotiations, and trade value simultaneously.",
        "flair": "Roster", "angles": ["injury_watch"], "tags": ["injury", "development"],
        "requires": ["name", "injury_type", "games_missed", "confidence"], "format": "breakdown",
    },
    {
        "title": "Injury in a contract year — the negotiation nightmare nobody prepares for",
        "body": "{name} going down with {injury_type} during his walk year at {cap_hit}M ask. Both sides now navigating incomplete information. How do you value a player you haven't seen healthy?",
        "flair": "Contracts", "angles": ["injury_watch", "contract_battle"], "tags": ["injury", "contract"],
        "requires": ["name", "injury_type", "cap_hit"], "format": "discussion",
    },
    {
        "title": "Lineup resilience: how deep does your roster need to be?",
        "body": "Lost {name} for {games_missed} games and {team_record} held. Or collapsed — share your data. What's the minimum organizational depth before one injury changes a season?",
        "flair": "Roster", "angles": ["injury_watch"], "tags": ["injury", "lineup"],
        "requires": ["name", "games_missed", "team_record"], "format": "discussion",
    },

    # ── MEDIA / REPUTATION ────────────────────────────────────────────────

    {
        "title": "Media manipulation 101: how {reporter}'s coverage shaped my season",
        "body": "{reporter} dropped {headline} mid-season with zero accuracy. Credibility: {rumor_source}. How rumor cycles actually move morale, trade value, and locker room dynamics — even when false.",
        "flair": "Media", "angles": ["media_spin"], "tags": ["media", "rumor"],
        "requires": ["reporter", "headline", "rumor_source"], "format": "breakdown",
    },
    {
        "title": "Franchise reputation: how '{franchise_rep}' label affects every future deal",
        "body": "Built a '{franchise_rep}' reputation over {term} seasons. Now measuring the compound cost in free agent pitches, agent relationships, and player willingness to extend. Real data.",
        "flair": "Culture", "angles": ["dynasty_legacy", "media_spin"], "tags": ["reputation", "meta"],
        "requires": ["franchise_rep", "term"], "format": "breakdown",
    },
    {
        "title": "The press conference that created a media narrative I'm still fighting",
        "body": "One honest answer about {name}'s role became {headline}. How media framing compounds across weeks when you can't course-correct without feeding more coverage.",
        "flair": "Media", "angles": ["media_spin"], "tags": ["media", "reputation"],
        "requires": ["name", "headline"], "format": "postmortem",
    },
    {
        "title": "Rumor credibility tiers: which sources to actually trust",
        "body": "Ranked every reporter and source by accuracy over {term} seasons. {reporter} is '{rumor_source}' credibility. How to weight information when you're operating on incomplete data.",
        "flair": "Meta", "angles": ["media_spin"], "tags": ["media", "rumor", "meta"],
        "requires": ["term", "reporter", "rumor_source"], "format": "guide",
    },

    # ── ANALYTICS ─────────────────────────────────────────────────────────

    {
        "title": "Analytics vs eye test: when I trusted the wrong signal on {name}",
        "body": "xGF at {xgf_pct}%, {ppg} PPG, but the tape said something was wrong. Or the tape said fine and the analytics said decline. Which metric saved the decision, which one lied?",
        "flair": "Analytics", "angles": ["slump_watch"], "tags": ["analytics", "slump"],
        "requires": ["name", "xgf_pct", "ppg"], "format": "postmortem",
    },
    {
        "title": "Expected goals model vs goals: the {name} over/underperformance case",
        "body": "{name} has {goals} goals on xG of {xgf_pct} expected. Sustainable or noise? How the model should change your extension offer at {cap_hit}M.",
        "flair": "Analytics", "angles": ["slump_watch", "contract_battle"], "tags": ["analytics", "contract"],
        "requires": ["name", "goals", "xgf_pct", "cap_hit"], "format": "breakdown",
    },
    {
        "title": "Special teams analytics: the {pp_pct}% power play problem",
        "body": "PP at {pp_pct}%, PK at {pk_pct}%. {team} is losing {wins} games per season purely to special teams. The personnel and system changes that actually move these numbers.",
        "flair": "Analytics", "angles": ["league_wire"], "tags": ["analytics", "coaching"],
        "requires": ["pp_pct", "pk_pct", "team", "wins"], "format": "breakdown",
    },
    {
        "title": "Usage metrics and ice time: are you deploying {name} correctly?",
        "body": "{name} at {overall} OVR in a {ppg} PPG role — but zone starts and deployment context say we're misusing him. What the usage data suggests about lineup optimization.",
        "flair": "Analytics", "angles": ["slump_watch"], "tags": ["analytics", "coaching"],
        "requires": ["name", "overall", "ppg"], "format": "breakdown",
    },

    # ── OWNERSHIP / LEGACY ────────────────────────────────────────────────

    {
        "title": "Ownership mandate: when to push back and when to comply",
        "body": "Owner wants playoffs in {term} years. My model says {term}+2. At {team_record}, who's right? The conversation, the compromise, and what happens when the mandate and the hockey decision conflict.",
        "flair": "Management", "angles": ["ownership_pressure"], "tags": ["owner", "rebuild"],
        "requires": ["term", "team_record"], "format": "discussion",
    },
    {
        "title": "{term}-season legacy audit: what this franchise actually built",
        "body": "{wins} playoff appearances, {losses} first-round exits, {wins} deep runs, {wins} cup finals. The honest accounting of {team}'s franchise arc and what the legacy wall actually says.",
        "flair": "Legacy", "angles": ["dynasty_legacy"], "tags": ["meta", "legacy"],
        "requires": ["term", "wins", "losses", "team"], "format": "breakdown",
    },
    {
        "title": "Retiring {name}'s number: what that moment actually means in dynasty mode",
        "body": "Drafted him. Developed him at {prior_overall} OVR. Won with him at {overall}. Now the conversation about legacy, retired numbers, and what franchise mode is actually trying to do.",
        "flair": "Legacy", "angles": ["dynasty_legacy"], "tags": ["retirement", "milestone"],
        "requires": ["name", "prior_overall", "overall"], "format": "narrative",
    },
    {
        "title": "Franchise identity formation: how {team} became what it is without choosing it",
        "body": "Never selected 'analytics-first org' from a menu. But {wins} decision patterns over {term} seasons produced exactly that identity. What your decisions say about your franchise philosophy.",
        "flair": "Legacy", "angles": ["dynasty_legacy"], "tags": ["identity", "culture", "meta"],
        "requires": ["team", "wins", "term"], "format": "breakdown",
    },
    {
        "title": "Stanley Cup run I didn't see coming — complete breakdown",
        "body": "{team} entered the playoffs at {league_rank} in the league with {team_record}. Everything that had to go right: goaltending, health, matchup luck, player performances. Full account.",
        "flair": "Playoffs", "angles": ["dynasty_legacy", "win_now"], "tags": ["playoffs", "milestone"],
        "requires": ["team", "league_rank", "team_record"], "format": "breakdown",
    },
    {
        "title": "The season everything collapsed: {team} {season} postmortem",
        "body": "Started {season} at {team_record} with real expectations. Ended with a rebuild. Injury, contract fallout, locker room split — cataloging every contributing factor.",
        "flair": "Legacy", "angles": ["dynasty_legacy"], "tags": ["postmortem", "meta"],
        "requires": ["team", "season", "team_record"], "format": "postmortem",
    },

    # ── PLAYOFFS ──────────────────────────────────────────────────────────

    {
        "title": "Playoff predictors: what actually separates winners at {team_record}",
        "body": "Regular season analytics vs playoff performance correlation. {name}'s character traits, goaltending variance from {name}'s {save_pct} SV%, experience factor — which variables actually predict success.",
        "flair": "Playoffs", "angles": ["win_now", "goaltending"], "tags": ["playoffs", "analytics"],
        "requires": ["team_record", "name", "save_pct"], "format": "breakdown",
    },
    {
        "title": "First-round exit autopsy: what cost {team} the series",
        "body": "{team_record} regular season, eliminated in {draft_round} games. Which injuries, matchup decisions, and lineup errors created the collapse. Full series breakdown.",
        "flair": "Playoffs", "angles": ["win_now", "league_wire"], "tags": ["playoffs", "postmortem"],
        "requires": ["team_record", "team", "draft_round"], "format": "postmortem",
    },
    {
        "title": "Playoff roster construction: what changes for the second season",
        "body": "{name} at {overall} OVR is excellent regular season. Playoff deployment is different. How do you build a 23-man roster for a 2-month war when regular season logic breaks down?",
        "flair": "Playoffs", "angles": ["win_now"], "tags": ["playoffs", "lineup"],
        "requires": ["name", "overall"], "format": "guide",
    },

    # ── META / PHILOSOPHY ─────────────────────────────────────────────────

    {
        "title": "10-year franchise patterns: what I learned playing {team} long-term",
        "body": "Tracked every major decision across {term} seasons at {team_record} peaks and valleys. Draft hit rate, contract efficiency, trade return accuracy, development environment ROI.",
        "flair": "Meta", "angles": ["dynasty_legacy"], "tags": ["meta", "guide"],
        "requires": ["team", "term", "team_record"], "format": "breakdown",
    },
    {
        "title": "Decision frameworks: how to handle incomplete information as a GM",
        "body": "You never have full information on {name}'s true potential, {prospect_name}'s ceiling, or whether {team_record} is real or noise. The mental models for making high-stakes decisions with uncertainty.",
        "flair": "Meta", "angles": ["rebuild_mode", "draft_board"], "tags": ["meta", "guide"],
        "requires": ["name", "prospect_name", "team_record"], "format": "guide",
    },
    {
        "title": "Risk tolerance calibration: how aggressive should you actually be?",
        "body": "Gave up {trade_asset} for {trade_return} at {team_record}. Was that appropriate risk for the timeline? Building a personal framework for knowing when to be bold vs conservative.",
        "flair": "Meta", "angles": ["trade_market", "rebuild_mode"], "tags": ["meta", "trade"],
        "requires": ["trade_asset", "trade_return", "team_record"], "format": "guide",
    },
    {
        "title": "The most underrated GM moves in dynasty mode",
        "body": "Not the blockbusters. The quiet depth signings at {cap_hit}M, the scout trust on {prospect_name}, the manager hire that changed development OVR curves. What doesn't get posted but matters.",
        "flair": "Meta", "angles": ["dynasty_legacy"], "tags": ["meta", "culture"],
        "requires": ["cap_hit", "prospect_name"], "format": "discussion",
    },
    {
        "title": "Share your biggest mistake and what it cost you long-term",
        "body": "Mine: gave {name} a {cap_hit}M x {term} deal based on one big season. {delta_overall} OVR drop next year. Dead cap for {years_remaining}. What's yours and what did you learn?",
        "flair": "Meta", "angles": ["dynasty_legacy"], "tags": ["meta", "postmortem"],
        "requires": ["name", "cap_hit", "term", "delta_overall", "years_remaining"], "format": "discussion",
    },
    {
        "title": "The dynasty mode decision that still haunts me — community confessional",
        "body": "I passed on {prospect_name} at {draft_pick}. He went {draft_round} picks later to {rival_team} and is now {prospect_overall} OVR. Community thread: what's your biggest draft regret?",
        "flair": "Meta", "angles": ["draft_board"], "tags": ["draft", "postmortem", "meta"],
        "requires": ["prospect_name", "draft_pick", "draft_round", "rival_team", "prospect_overall"], "format": "discussion",
    },
    {
        "title": "Full dynasty mode retrospective after {term} seasons at {team}",
        "body": "Started at {team_record}. {wins} wins later. The full arc — tank phase, rebuild phase, contend phase, win-now phase. Everything worth remembering about {team}'s franchise history.",
        "flair": "Legacy", "angles": ["dynasty_legacy"], "tags": ["meta", "legacy"],
        "requires": ["term", "team", "team_record", "wins"], "format": "breakdown",
    },
]


# ---------------------------------------------------------------------------
# Utility: render a template with context
# ---------------------------------------------------------------------------

def render_template(template: Dict[str, Any], ctx: Dict[str, Any]) -> Optional[str]:
    """
    Fill all {placeholder} tokens using ctx.
    Returns None if any required key is missing or empty.
    """
    requires: List[str] = list(template.get("requires") or [])
    for key in requires:
        if not ctx.get(key):
            return None

    source = template.get("text") or template.get("title") or ""
    body   = template.get("body", "")

    def _fill(s: str) -> str:
        def replacer(m: re.Match) -> str:
            return str(ctx.get(m.group(1), m.group(0)))
        return re.sub(r"\{(\w+)\}", replacer, s)

    if body:
        return f"{_fill(source)}\n\n{_fill(body)}"
    return _fill(source)


# ---------------------------------------------------------------------------
# Utility: filter template pool
# ---------------------------------------------------------------------------

def _angle_matches(template: Dict[str, Any], angle: str) -> bool:
    angles: List[str] = list(template.get("angles") or [])
    if not angles:
        return True
    return angle in angles or "league_wire" in angles


def filter_templates(
    pool: List[Dict[str, Any]],
    *,
    angle: str = "",
    ctx: Dict[str, Any],
    tag: str = "",
    tone: str = "",
    fmt: str = "",
    max_results: int = 0,
    shuffle: bool = False,
) -> List[Dict[str, Any]]:
    """
    Filter a pool by angle, tag, tone/format, and available context keys.
    Optionally shuffle and cap the result list.
    """
    out: List[Dict[str, Any]] = []
    for row in pool:
        if tag and tag not in list(row.get("tags") or []):
            continue
        if tone and row.get("tone") != tone:
            continue
        if fmt and row.get("format") != fmt:
            continue
        if angle and not _angle_matches(row, angle):
            continue
        requires: List[str] = list(row.get("requires") or [])
        if any(not ctx.get(k) for k in requires):
            continue
        out.append(row)

    if shuffle:
        random.shuffle(out)
    if max_results:
        out = out[:max_results]
    return out


# ---------------------------------------------------------------------------
# Utility: pick one random renderable template
# ---------------------------------------------------------------------------

def pick_one(
    pool: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    *,
    angle: str = "",
    tag: str = "",
    tone: str = "",
) -> Optional[str]:
    """Return a single rendered template string, or None if nothing qualifies."""
    candidates = filter_templates(
        pool, angle=angle, ctx=ctx, tag=tag, tone=tone, shuffle=True, max_results=20
    )
    for candidate in candidates:
        rendered = render_template(candidate, ctx)
        if rendered:
            return rendered
    return None


# ---------------------------------------------------------------------------
# Utility: bulk render for a feed
# ---------------------------------------------------------------------------

def build_feed(
    pool: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    *,
    angle: str = "",
    tag: str = "",
    max_results: int = 10,
) -> List[str]:
    """
    Return up to max_results rendered strings from a shuffled filtered pool.
    Skips templates that fail to render.
    """
    candidates = filter_templates(
        pool, angle=angle, ctx=ctx, tag=tag, shuffle=True
    )
    feed: List[str] = []
    for candidate in candidates:
        if len(feed) >= max_results:
            break
        rendered = render_template(candidate, ctx)
        if rendered:
            feed.append(rendered)
    return feed


# ---------------------------------------------------------------------------
# Utility: list all angles / tags that exist in a pool
# ---------------------------------------------------------------------------

def pool_angles(pool: List[Dict[str, Any]]) -> List[str]:
    seen: set = set()
    for row in pool:
        for a in list(row.get("angles") or []):
            seen.add(a)
    return sorted(seen)


def pool_tags(pool: List[Dict[str, Any]]) -> List[str]:
    seen: set = set()
    for row in pool:
        for t in list(row.get("tags") or []):
            seen.add(t)
    return sorted(seen)


# ---------------------------------------------------------------------------
# Utility: validate pool against placeholder registry
# ---------------------------------------------------------------------------

def validate_pool(pool: List[Dict[str, Any]]) -> List[str]:
    """
    Return a list of warnings for any {placeholder} tokens that are
    not registered in PLACEHOLDER_REGISTRY.
    """
    warnings: List[str] = []
    for i, row in enumerate(pool):
        source = (row.get("text") or row.get("title") or "") + (row.get("body") or "")
        for token in re.findall(r"\{(\w+)\}", source):
            if token not in PLACEHOLDER_REGISTRY:
                warnings.append(
                    f"Template #{i}: unknown placeholder {{{token!r}}} — add to PLACEHOLDER_REGISTRY"
                )
    return warnings


# ---------------------------------------------------------------------------
# Convenience exports
# ---------------------------------------------------------------------------

ALL_TWEET_TEMPLATES: List[TweetTemplate] = DYNASTY_TWEET_TEMPLATES + MEME_TWEET_TEMPLATES

ALL_POOLS: Dict[str, List[Dict[str, Any]]] = {
    "tweets":  ALL_TWEET_TEMPLATES,
    "dynasty_tweets": DYNASTY_TWEET_TEMPLATES,
    "meme_tweets": MEME_TWEET_TEMPLATES,
    "reddit":  REDDIT_THREAD_TEMPLATES,
}

__all__ = [
    "PLACEHOLDER_REGISTRY",
    "ANGLES",
    "TAGS",
    "DYNASTY_TWEET_TEMPLATES",
    "MEME_TWEET_TEMPLATES",
    "ALL_TWEET_TEMPLATES",
    "REDDIT_THREAD_TEMPLATES",
    "ALL_POOLS",
    "render_template",
    "filter_templates",
    "pick_one",
    "build_feed",
    "pool_angles",
    "pool_tags",
    "validate_pool",
]