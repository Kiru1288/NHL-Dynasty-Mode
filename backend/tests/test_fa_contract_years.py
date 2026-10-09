"""Free agency, contract term, and year-slide behavior."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

from services.contract_economy import (  # noqa: E402
    activate_pending_extension,
    advance_dict_contract_one_season,
    forward_years_from_reported,
    get_contract_cap_hit,
    is_current_ufa_class,
    prune_owned_from_fa_pools,
    release_unresigned_free_agents,
    resolved_rights_status,
    stamp_forward_contract_term,
    _sync_term_from_season_cap_hits,
)
from services.elc_offer_engine import process_elc_slides  # noqa: E402


def _player(pid, *, years, expiry, aav=4.0, age=30, pending=False, name=None, extra=None):
    contract = {
        "years_remaining": years,
        "years": years,
        "expiry_year": expiry,
        "aav_m": aav,
        "cap_hit_m": aav,
        "rights_status": "UFA",
        "pending_july1_expiry": pending,
        "contract_type": "STANDARD",
        "type": "STANDARD",
    }
    if extra:
        contract.update(extra)
    return SimpleNamespace(
        id=pid,
        name=name or pid,
        age=age,
        retired=False,
        pending_july1_expiry=pending,
        user_signed=False,
        rights_status=contract["rights_status"],
        position="C",
        contract=contract,
        cap_hit_m=aav,
        aav_m=aav,
    )


def _session(roster, *, season_year=2026, outcomes=None, free_agents=None):
    team = SimpleNamespace(
        team_id="OTT",
        id="OTT",
        roster=list(roster),
        ahl_roster=[],
        echl_roster=[],
        prospect_pool=[],
        rfa_rights=[],
    )
    league = SimpleNamespace(teams=[team], free_agents=list(free_agents or []), overseas_free_agents=[])
    session = SimpleNamespace(
        sim=SimpleNamespace(league=league, rng=None),
        season_calendar_year=season_year,
        july1_contracts_expired=True,
        user_team_id="OTT",
        resign_phase_outcomes=outcomes or {},
        phase="offseason",
        free_agency_open=True,
    )
    return session, team, league


class UfaClassTests(unittest.TestCase):
    def test_this_july_only(self):
        self.assertTrue(is_current_ufa_class(
            years_remaining=1, expiry_year=2027, season_year=2026,
            pending_july1=True, extension_signed=False,
        ))
        self.assertTrue(is_current_ufa_class(
            years_remaining=1, expiry_year=2027, season_year=2026,
            pending_july1=False, extension_signed=False,
        ))
        self.assertFalse(is_current_ufa_class(
            years_remaining=3, expiry_year=2030, season_year=2026,
            pending_july1=False, extension_signed=False,
        ))
        self.assertFalse(is_current_ufa_class(
            years_remaining=1, expiry_year=2028, season_year=2026,
            pending_july1=False, extension_signed=False,
        ))
        self.assertFalse(is_current_ufa_class(
            years_remaining=1, expiry_year=2027, season_year=2026,
            pending_july1=True, extension_signed=True,
        ))

    def test_zero_years_is_already_done(self):
        self.assertTrue(is_current_ufa_class(
            years_remaining=0, expiry_year=2027, season_year=2026,
            pending_july1=False, extension_signed=False,
        ))


class ReportedYearsTests(unittest.TestCase):
    def test_capwages_years_move_forward_after_the_season(self):
        # 2 years reported for 2026-27. After that season is played, 1 remains.
        self.assertEqual(forward_years_from_reported(2, 2026, season_already_played=False), 2)
        self.assertEqual(forward_years_from_reported(2, 2026, season_already_played=True), 1)
        self.assertEqual(forward_years_from_reported(1, 2026, season_already_played=True), 0)
        self.assertEqual(forward_years_from_reported(3, 2028, season_already_played=False), 1)

    def test_stamp_forward_does_not_touch_a_user_signing(self):
        deal = {"years_remaining": 1, "aav_m": 6.5, "source": "signed", "user_signed": True, "expiry_year": 2028}
        self.assertFalse(stamp_forward_contract_term(
            deal, None, 4, 2026, season_already_played=True,
        ))
        self.assertEqual(deal["years_remaining"], 1)
        self.assertEqual(deal["expiry_year"], 2028)

    def test_stamp_forward_clears_a_false_july_flag(self):
        player = SimpleNamespace(pending_july1_expiry=True)
        deal = {
            "years_remaining": 1,
            "aav_m": 8.0,
            "cap_hit_m": 8.0,
            "pending_july1_expiry": True,
            "source": "spotrac",
        }
        self.assertTrue(stamp_forward_contract_term(
            deal, player, 2, 2026, season_already_played=True,
        ))
        self.assertEqual(deal["years_remaining"], 2)
        self.assertEqual(deal["effective_season"], 2027)
        self.assertEqual(deal["expiry_year"], 2029)
        self.assertNotIn("pending_july1_expiry", deal)
        self.assertFalse(player.pending_july1_expiry)
        self.assertFalse(is_current_ufa_class(
            years_remaining=deal["years_remaining"],
            expiry_year=deal["expiry_year"],
            season_year=2026,
            pending_july1=False,
            extension_signed=False,
        ))


class YearSlideTests(unittest.TestCase):
    def test_one_season_burns_a_year_and_keeps_the_unsigned_year(self):
        deal = {
            "years_remaining": 3,
            "years": 3,
            "aav_m": 8.0,
            "cap_hit_m": 8.0,
            "effective_season": 2026,
            "expiry_year": 2029,
            "season_cap_hits": [8.0, 8.0, 8.0],
            "clause_by_year": ["", "", "NTC"],
        }
        advance_dict_contract_one_season(deal, 2026)
        self.assertEqual(deal["years_remaining"], 2)
        self.assertEqual(deal["effective_season"], 2027)
        self.assertEqual(deal["expiry_year"], 2029)
        self.assertEqual(deal["season_cap_hits"], [8.0, 8.0])
        self.assertFalse(deal.get("no_trade_clause"))
        self.assertEqual(deal.get("clause_future_type"), "NTC")

    def test_second_slide_turns_the_scheduled_clause_on(self):
        deal = {
            "years_remaining": 2,
            "years": 3,
            "aav_m": 8.0,
            "cap_hit_m": 8.0,
            "effective_season": 2027,
            "expiry_year": 2029,
            "season_cap_hits": [8.0, 8.0],
            "clause_by_year": ["", "NTC"],
        }
        advance_dict_contract_one_season(deal, 2027)
        self.assertEqual(deal["years_remaining"], 1)
        self.assertEqual(deal["expiry_year"], 2029)
        self.assertTrue(deal.get("no_trade_clause"))
        self.assertEqual(deal.get("clause_display"), "NTC")

    def test_future_deal_is_not_pulled_onto_the_season_that_just_ended(self):
        deal = {
            "years_remaining": 4,
            "years": 4,
            "aav_m": 6.5,
            "cap_hit_m": 6.5,
            "effective_season": 2027,
            "expiry_year": 2031,
            "season_cap_hits": [6.5, 6.5, 6.5, 6.5],
            "source": "signed",
        }
        _sync_term_from_season_cap_hits(deal, 2026)
        self.assertEqual(deal["effective_season"], 2027)
        self.assertEqual(deal["expiry_year"], 2031)
        self.assertEqual(deal["years_remaining"], 4)

    def test_stacked_signing_bonus_does_not_replace_the_aav(self):
        self.assertEqual(get_contract_cap_hit({
            "aav_m": 5.0,
            "cap_hit_m": 13.0,
            "signing_bonus_m": 8.0,
            "years_remaining": 1,
            "contract_type": "STANDARD",
        }), 5.0)


class ExtensionAndRightsTests(unittest.TestCase):
    def test_extension_replaces_the_expiring_deal(self):
        player = _player("ext", years=1, expiry=2027, aav=3.0, pending=True, extra={
            "pending_extension": {"years": 5, "aav_m": 7.25, "cap_hit_m": 7.25, "expiry_year": 2032},
        })
        self.assertTrue(activate_pending_extension(player, 2026))
        self.assertEqual(player.contract["years_remaining"], 5)
        self.assertEqual(player.contract["aav_m"], 7.25)
        self.assertEqual(player.contract["expiry_year"], 2032)
        self.assertNotIn("pending_extension", player.contract)
        self.assertFalse(player.pending_july1_expiry)

    def test_rights_follow_age_unless_a_real_source_said_ufa(self):
        young = _player("y", years=2, expiry=2029, age=24)
        self.assertEqual(resolved_rights_status(young), "RFA")
        sourced = _player("s", years=1, expiry=2027, age=25, extra={
            "rights_status": "UFA", "rights_source": "capwages",
        })
        self.assertEqual(resolved_rights_status(sourced), "UFA")
        veteran = _player("v", years=2, expiry=2029, age=33)
        self.assertEqual(resolved_rights_status(veteran), "UFA")
        tagged = _player("r", years=1, expiry=2027, age=31, extra={"rights_status": "RFA"})
        self.assertEqual(resolved_rights_status(tagged), "RFA")


class FreeAgencyRosterTests(unittest.TestCase):
    def test_unsigned_leaves_and_a_multi_year_deal_stays(self):
        walk = _player("walk", years=1, expiry=2027, aav=6.5, age=32, name="Unsigned Vet")
        stay = _player("stay", years=4, expiry=2030, aav=8.0, age=28, name="Core")
        session, team, league = _session([walk, stay])
        out = release_unresigned_free_agents(session)
        self.assertEqual([p.id for p in team.roster], ["stay"])
        self.assertIn(walk, league.free_agents)
        self.assertEqual(walk.cap_hit_m, 0)
        self.assertEqual(out["cap_freed_m"], 6.5)
        self.assertIn("Unsigned Vet", session.fa_cap_release_note["text"])

    def test_extension_is_activated_instead_of_walking_him(self):
        player = _player("ext", years=1, expiry=2027, aav=3.0, age=29, pending=True, extra={
            "pending_extension": {"years": 4, "aav_m": 6.0, "cap_hit_m": 6.0, "expiry_year": 2031},
        })
        session, team, league = _session([player])
        out = release_unresigned_free_agents(session)
        self.assertEqual([p.id for p in team.roster], ["ext"])
        self.assertEqual(league.free_agents, [])
        self.assertEqual(player.contract["years_remaining"], 4)
        self.assertEqual(player.contract["aav_m"], 6.0)
        self.assertEqual(out["released"], 0)

    def test_desk_release_removes_a_deal_that_still_shows_years(self):
        player = _player("gone", years=3, expiry=2029, aav=4.25, age=26, name="Let Go")
        session, team, league = _session(
            [player],
            outcomes={"gone": {"phase_status": "released"}},
        )
        release_unresigned_free_agents(session)
        self.assertEqual(team.roster, [])
        self.assertEqual(player.cap_hit_m, 0)
        self.assertTrue(league.free_agents or team.rfa_rights)

    def test_accepted_resign_is_not_released(self):
        player = _player("kept", years=1, expiry=2027, aav=5.0, age=29, pending=True)
        session, team, league = _session(
            [player],
            outcomes={"kept": {"phase_status": "accepted"}},
        )
        out = release_unresigned_free_agents(session)
        self.assertEqual([p.id for p in team.roster], ["kept"])
        self.assertEqual(out["released"], 0)
        self.assertEqual(player.cap_hit_m, 5.0)

    def test_young_expiry_becomes_rfa_rights_not_an_open_ufa(self):
        player = _player("kid", years=1, expiry=2027, aav=1.2, age=23, name="Young")
        session, team, league = _session([player])
        release_unresigned_free_agents(session)
        self.assertEqual(team.roster, [])
        self.assertFalse(any(getattr(p, "id", None) == "kid" for p in league.free_agents))
        self.assertTrue(any(str(row.get("player_id")) == "kid" for row in team.rfa_rights))

    def test_signed_player_is_pulled_off_the_wire(self):
        rostered = _player("both", years=4, expiry=2030, aav=5.0, age=28)
        ghost = SimpleNamespace(id="both", name="Ghost Copy", retired=False)
        other = SimpleNamespace(id="real-fa", name="Real FA", retired=False)
        league = SimpleNamespace(
            teams=[SimpleNamespace(
                team_id="OTT", roster=[rostered], ahl_roster=[], echl_roster=[], prospect_pool=[], rfa_rights=[],
            )],
            free_agents=[ghost, other, ghost],
            overseas_free_agents=[],
        )
        removed = prune_owned_from_fa_pools(league)
        ids = [p.id for p in league.free_agents]
        self.assertEqual(ids, ["real-fa"])
        self.assertGreaterEqual(removed, 1)


class FivePlayerFreeAgencyScenario(unittest.TestCase):
    """One July, five players, five different statuses."""

    def test_five_statuses_on_one_roster(self):
        open_ufa = _player(
            "ufa", years=1, expiry=2027, aav=6.5, age=32, pending=True, name="Open UFA",
        )
        restricted = _player(
            "rfa", years=1, expiry=2027, aav=1.2, age=23, pending=True, name="Restricted",
            extra={"rights_status": "RFA"},
        )
        resigned = _player(
            "signed", years=1, expiry=2027, aav=5.0, age=29, pending=True, name="Re-Signed",
        )
        extended = _player(
            "ext", years=1, expiry=2027, aav=3.0, age=27, pending=True, name="Extended",
            extra={"pending_extension": {
                "years": 4, "aav_m": 6.0, "cap_hit_m": 6.0, "expiry_year": 2031,
            }},
        )
        let_go = _player(
            "gone", years=3, expiry=2029, aav=4.25, age=28, name="Let Go",
        )
        session, team, league = _session(
            [open_ufa, restricted, resigned, extended, let_go],
            outcomes={
                "signed": {"phase_status": "accepted", "player_id": "signed"},
                "gone": {"phase_status": "released", "player_id": "gone"},
            },
        )

        out = release_unresigned_free_agents(session)

        self.assertEqual([p.id for p in team.roster], ["signed", "ext"])
        self.assertEqual(resigned.cap_hit_m, 5.0)
        self.assertEqual(resigned.contract["years_remaining"], 1)

        self.assertEqual(extended.contract["years_remaining"], 4)
        self.assertEqual(extended.contract["aav_m"], 6.0)
        self.assertEqual(extended.contract["expiry_year"], 2031)
        self.assertNotIn("pending_extension", extended.contract)

        market_ids = [p.id for p in league.free_agents]
        self.assertIn("ufa", market_ids)
        self.assertIn("gone", market_ids)
        self.assertNotIn("rfa", market_ids)
        self.assertNotIn("signed", market_ids)
        self.assertNotIn("ext", market_ids)
        self.assertEqual(open_ufa.cap_hit_m, 0)
        self.assertEqual(let_go.cap_hit_m, 0)
        self.assertEqual(resolved_rights_status(open_ufa), "UFA")

        rights_ids = [str(row.get("player_id")) for row in team.rfa_rights]
        self.assertEqual(rights_ids, ["rfa"])
        self.assertEqual(restricted.cap_hit_m, 0)
        self.assertEqual(resolved_rights_status(restricted), "RFA")

        freed_ids = [row["player_id"] for row in out["user_released"]]
        self.assertEqual(freed_ids, ["ufa", "rfa", "gone"])
        self.assertEqual(out["cap_freed_m"], 11.95)
        self.assertEqual(out["released"], 3)
        note = session.fa_cap_release_note["text"]
        self.assertIn("Open UFA", note)
        self.assertIn("Restricted", note)
        self.assertIn("Let Go", note)
        self.assertNotIn("Re-Signed", note)
        self.assertNotIn("Extended", note)


class ResignFiveStylesScenario(unittest.TestCase):
    """Re-sign five players: different overalls, ages, and agents."""

    def test_five_styles_price_and_sign_differently(self):
        from app.sim_engine.franchise.player_agent_engine import (
            PLAYER_AGENTS,
            agent_public_view,
        )
        from services.contract_economy import (
            compute_player_demand,
            evaluate_contract_offer,
            sign_player_to_team,
        )
        from services.negotiation_meetings import _do_agent

        specs = [
            ("depth", "Depth Center", 74, 26, "walsh"),
            ("twoway", "Two-Way Wing", 80, 24, "rossi"),
            ("top6", "Top-Six Scorer", 86, 25, "kim"),
            ("star", "Franchise Star", 91, 27, "carter"),
            ("vet", "Aging Veteran", 84, 34, "blake"),
        ]
        agents = {a["id"]: a for a in PLAYER_AGENTS}
        players = []
        for pid, name, ovr, age, agent_id in specs:
            agent = agents[agent_id]
            players.append(SimpleNamespace(
                id=pid,
                name=name,
                age=age,
                retired=False,
                position="C",
                identity=SimpleNamespace(name=name, age=age, position="C"),
                ovr=ovr,
                ratings={"dev_potential": 88 if ovr >= 86 else 76},
                season_stats={"gp": 70, "pts": 20 if ovr < 80 else 55},
                morale=70,
                agent_profile={
                    "id": agent_id,
                    "style": agent["style"],
                    "style_label": agent["style_label"],
                },
                contract={
                    "years_remaining": 1,
                    "years": 1,
                    "expiry_year": 2027,
                    "aav_m": 1.0,
                    "cap_hit_m": 1.0,
                    "rights_status": "UFA" if age >= 27 else "RFA",
                    "contract_type": "STANDARD",
                    "pending_july1_expiry": True,
                },
                cap_hit_m=1.0,
                aav_m=1.0,
            ))

        team = SimpleNamespace(
            team_id="OTT",
            id="OTT",
            roster=list(players),
            ahl_roster=[],
            echl_roster=[],
            prospect_pool=[],
            rfa_rights=[],
            buyout_cap_hits=[],
            salary_cap_m=104.0,
        )
        league = SimpleNamespace(
            teams=[team],
            free_agents=[],
            salary_cap_m=104.0,
            cap_floor_m=76.0,
        )
        session = SimpleNamespace(
            user_team_id="OTT",
            season_calendar_year=2026,
            phase="offseason",
            negotiation_meetings={},
            agent_relationships={},
            sim=SimpleNamespace(league=league),
            team_by_id={"OTT": team},
        )

        rows = []
        for player in players:
            before = compute_player_demand(player, team, league, context="re_sign")
            meeting = _do_agent(session, player, "own", "firm_number")
            self.assertTrue(meeting.get("ok"), meeting)
            after = compute_player_demand(player, team, league, context="re_sign")
            view = agent_public_view(player, session)
            offer = {
                "aav_m": after["want_aav_m"],
                "years": after["want_years"],
                "context": "re_sign",
            }
            verdict = evaluate_contract_offer(player, team, offer, league, context="re_sign")
            if not verdict["accepted"] and verdict.get("preferred_clause") not in (None, "", "None"):
                clause = verdict["preferred_clause"]
                offer = dict(offer)
                if clause == "NMC":
                    offer["nmc"] = True
                elif clause == "M-NTC":
                    offer["ntc"] = True
                    offer["ntc_mode"] = "MODIFIED"
                    offer["m_ntc"] = True
                else:
                    offer["ntc"] = True
                    offer["ntc_mode"] = "FULL"
                verdict = evaluate_contract_offer(player, team, offer, league, context="re_sign")
            low = evaluate_contract_offer(
                player, team,
                {"aav_m": round(after["want_aav_m"] * 0.55, 3), "years": 1, "context": "re_sign"},
                league, context="re_sign",
            )
            signed = None
            if verdict["accepted"]:
                signed = sign_player_to_team(
                    player, team, league, 2026,
                    {**offer, "context": "re_sign", "force": True, "rights": "UFA" if player.age >= 27 else "RFA"},
                )
            rows.append({
                "id": player.id,
                "name": player.name,
                "ovr": player.ovr,
                "age": player.age,
                "agent": view["name"],
                "style": view["style_label"],
                "difficulty": view["deal_difficulty"],
                "market": before["market_value_m"],
                "ask_before": before["want_aav_m"],
                "ask_after": after["want_aav_m"],
                "years": after["want_years"],
                "accepted": verdict["accepted"],
                "interest": verdict["interest"],
                "low_interest": low["interest"],
                "low_accepted": low["accepted"],
                "reason": verdict["reason"],
                "signed_ok": None if signed is None else bool(signed.get("ok")),
                "signed_aav": None if not signed or not signed.get("ok") else player.contract.get("aav_m"),
                "signed_years": None if not signed or not signed.get("ok") else player.contract.get("years_remaining"),
            })

        by_id = {row["id"]: row for row in rows}
        player_by_id = {p.id: p for p in players}
        self.assertEqual(
            [row["difficulty"] for row in rows],
            ["easy", "easy", "medium", "hard", "hard"],
        )
        # Overall sets the market. Agent style only nudges the ask a couple percent.
        markets = [by_id[k]["market"] for k in ("depth", "twoway", "vet", "top6", "star")]
        self.assertEqual(markets, sorted(markets))
        self.assertGreater(by_id["star"]["ask_after"], by_id["top6"]["ask_after"])
        self.assertGreater(by_id["depth"]["ask_before"], 0.7)

        # Firm number moves real dollars, not a percent or two.
        # Patient and leverage agents come down. A leaker and a disruptor dig in.
        self.assertGreaterEqual(by_id["depth"]["ask_before"] - by_id["depth"]["ask_after"], 0.25)
        self.assertGreaterEqual(by_id["twoway"]["ask_before"] - by_id["twoway"]["ask_after"], 0.25)
        self.assertGreaterEqual(by_id["top6"]["ask_before"] - by_id["top6"]["ask_after"], 0.80)
        self.assertGreaterEqual(by_id["star"]["ask_after"] - by_id["star"]["ask_before"], 0.80)
        self.assertGreaterEqual(by_id["vet"]["ask_after"] - by_id["vet"]["ask_before"], 0.70)

        for row in rows:
            self.assertTrue(row["accepted"], row)
            self.assertTrue(row["signed_ok"], row)
            self.assertAlmostEqual(row["signed_aav"], row["ask_after"], places=2)
            self.assertEqual(row["signed_years"], row["years"])
            self.assertGreater(row["interest"], row["low_interest"])
            self.assertFalse(row["low_accepted"], row)
            self.assertIsNone((player_by_id[row["id"]].contract or {}).get("pending_extension"))

        # A 34-year-old's term ask stays short (1–3 years, plus a small security bump).
        self.assertLessEqual(by_id["vet"]["years"], 5)


class NegotiationExtremeTests(unittest.TestCase):
    """Hometown yes/no and a lowball have to move the ask by real money."""

    def _player(self, pid, *, ovr, age, name):
        return SimpleNamespace(
            id=pid,
            name=name,
            age=age,
            retired=False,
            position="C",
            identity=SimpleNamespace(name=name, age=age, position="C"),
            ovr=ovr,
            ratings={"dev_potential": 90 if ovr >= 88 else 78},
            season_stats={"gp": 75, "pts": 90 if ovr >= 88 else 40},
            morale=70,
            contract={
                "years_remaining": 1,
                "years": 1,
                "expiry_year": 2027,
                "aav_m": 4.0,
                "cap_hit_m": 4.0,
                "rights_status": "UFA" if age >= 27 else "RFA",
                "contract_type": "STANDARD",
                "pending_july1_expiry": True,
            },
        )

    def _session(self, player, personality):
        team = SimpleNamespace(
            team_id="OTT", id="OTT", roster=[player], ahl_roster=[], echl_roster=[],
            prospect_pool=[], rfa_rights=[], buyout_cap_hits=[], salary_cap_m=104.0,
        )
        league = SimpleNamespace(teams=[team], free_agents=[], salary_cap_m=104.0, cap_floor_m=76.0)
        return SimpleNamespace(
            user_team_id="OTT",
            season_calendar_year=2026,
            phase="offseason",
            negotiation_meetings={},
            agent_relationships={},
            contract_talks={},
            universe_players={player.id: {"personality": personality, "state": {"gm_trust": 80, "morale": 75},
                                          "gm_relationship": {"negotiation_goodwill": 80},
                                          "life": {"community_connection": 85}}},
            sim=SimpleNamespace(league=league),
            team_by_id={"OTT": team},
        ), team, league

    def test_loyal_hometown_yes_is_millions_not_a_token(self):
        from services.contract_economy import compute_player_demand
        from services.negotiation_meetings import ask_hometown_discount

        accepted = None
        for n in range(30):
            player = self._player(f"loyal{n}", ovr=91, age=29, name="Loyal Star")
            session, team, league = self._session(player, {"loyalty": 95, "money_focus": 5})
            before = compute_player_demand(player, team, league, context="re_sign")["want_aav_m"]
            res = ask_hometown_discount(session, player.id)
            self.assertTrue(res.get("ok"), res)
            if res.get("accepted"):
                after = compute_player_demand(player, team, league, context="re_sign")["want_aav_m"]
                accepted = (before, after, res)
                break
        self.assertIsNotNone(accepted, "a loyal star should agree to a hometown deal inside 30 seeds")
        before, after, res = accepted
        self.assertGreaterEqual(before - after, 2.0, res)

    def test_refusing_a_hometown_ask_raises_the_number(self):
        from services.contract_economy import compute_player_demand
        from services.negotiation_meetings import ask_hometown_discount

        refused = None
        for n in range(20):
            player = self._player(f"money{n}", ovr=91, age=27, name="Money Star")
            session, team, league = self._session(player, {"loyalty": 5, "money_focus": 95})
            before = compute_player_demand(player, team, league, context="re_sign")["want_aav_m"]
            res = ask_hometown_discount(session, player.id)
            self.assertTrue(res.get("ok"), res)
            if not res.get("accepted"):
                after = compute_player_demand(player, team, league, context="re_sign")["want_aav_m"]
                refused = (before, after, res)
                break
        self.assertIsNotNone(refused, "a money-first star should turn down a hometown ask")
        before, after, res = refused
        self.assertGreaterEqual(after - before, 1.5, res)

    def test_one_lowball_reprices_the_ask(self):
        from services.contract_economy import _record_talks_round, compute_player_demand

        player = self._player("lowball", ovr=91, age=27, name="Insulted Star")
        session, team, league = self._session(player, {"loyalty": 40, "money_focus": 70})
        before = compute_player_demand(player, team, league, context="re_sign")["want_aav_m"]
        _record_talks_round(session, player, team, before * 0.50, before)
        after = compute_player_demand(player, team, league, context="re_sign")["want_aav_m"]
        self.assertGreaterEqual(after - before, 1.0, (before, after))


class ElcSlideTests(unittest.TestCase):
    def _elc(self, pid, gp, *, on_nhl=False, used=0):
        player = SimpleNamespace(
            id=pid,
            name=pid,
            age=19,
            retired=False,
            nhl_games_played_this_season=gp,
            season_stats={"gp": gp} if on_nhl else {},
            elc_slide_eligible=True,
            contract={
                "contract_type": "ELC",
                "type": "ELC",
                "years": 3,
                "years_remaining": 3,
                "expiry_year": 2029,
                "aav_m": 0.95,
                "cap_hit_m": 0.95,
                "slide_eligible": True,
                "slide_years_used": used,
                "effective_season": 2026,
            },
        )
        return player

    def test_low_games_slide_the_year_instead_of_burning_it(self):
        player = self._elc("slide", gp=4, on_nhl=False)
        team = SimpleNamespace(
            roster=[], ahl_roster=[player], prospect_pool=[],
        )
        session = SimpleNamespace(sim=SimpleNamespace(league=SimpleNamespace(teams=[team])))
        out = process_elc_slides(session, 2026)
        self.assertEqual(len(out["slid"]), 1)
        self.assertEqual(player.contract["years_remaining"], 3)
        self.assertEqual(player.contract["expiry_year"], 2030)
        self.assertEqual(player.contract["slide_years_used"], 1)
        self.assertTrue(player.contract["slide_triggered"])

    def test_nhl_games_burn_the_year(self):
        player = self._elc("burn", gp=40, on_nhl=True)
        team = SimpleNamespace(roster=[player], ahl_roster=[], prospect_pool=[])
        session = SimpleNamespace(sim=SimpleNamespace(league=SimpleNamespace(teams=[team])))
        out = process_elc_slides(session, 2026)
        self.assertEqual(len(out["burned"]), 1)
        self.assertEqual(out["slid"], [])
        self.assertFalse(player.contract["slide_eligible"])
        self.assertEqual(player.contract["expiry_year"], 2029)
        self.assertTrue(player.contract_burned)

    def test_a_second_slide_is_not_granted(self):
        player = self._elc("once", gp=0, used=1)
        team = SimpleNamespace(roster=[], ahl_roster=[player], prospect_pool=[])
        session = SimpleNamespace(sim=SimpleNamespace(league=SimpleNamespace(teams=[team])))
        out = process_elc_slides(session, 2026)
        self.assertEqual(out["count"], 0)
        self.assertEqual(player.contract["expiry_year"], 2029)

    def test_standard_deal_does_not_slide(self):
        player = self._elc("std", gp=0)
        player.contract["contract_type"] = "STANDARD"
        player.contract["type"] = "STANDARD"
        team = SimpleNamespace(roster=[], ahl_roster=[player], prospect_pool=[])
        session = SimpleNamespace(sim=SimpleNamespace(league=SimpleNamespace(teams=[team])))
        out = process_elc_slides(session, 2026)
        self.assertEqual(out["count"], 0)
        self.assertEqual(player.contract["expiry_year"], 2029)


class CapSpaceLogicTests(unittest.TestCase):
    """One cap number from the hub, the salary-cap card, re-sign, and free agency."""

    def _skater(self, pid, *, years, aav, pending=False, expiry=None, extra=None, buried=False):
        contract = {
            "years_remaining": years,
            "years": years,
            "expiry_year": expiry if expiry is not None else 2026 + years,
            "aav_m": aav,
            "cap_hit_m": aav,
            "rights_status": "UFA",
            "pending_july1_expiry": pending,
            "contract_type": "STANDARD",
            "type": "STANDARD",
        }
        if extra:
            contract.update(extra)
        return SimpleNamespace(
            id=pid,
            player_id=pid,
            name=pid,
            age=29,
            overall=84,
            ovr=84,
            retired=False,
            pending_july1_expiry=pending,
            is_buried=buried,
            buried=buried,
            in_minors=buried,
            on_ir=False,
            on_ltir=False,
            contract=contract,
            position="D",
            identity=SimpleNamespace(name=pid, age=29, position="D"),
        )

    def _club(self, roster, **fields):
        base = dict(
            team_id="OTT",
            id="OTT",
            roster=list(roster),
            ahl_roster=[],
            echl_roster=[],
            prospect_pool=[],
            rfa_rights=[],
            buyout_cap_hits=[],
            retained_salary=[],
            bonus_overage=[],
            performance_bonus_reserve_m=0,
        )
        base.update(fields)
        return SimpleNamespace(**base)

    def _snap(self, team, *, year=2026, opening=False, count_expiring=False, cursor=0, league=None):
        from app.sim_engine.economy.cap_engine import calculate_team_cap_snapshot

        if league is None:
            league = SimpleNamespace(
                salary_cap_m=104.0,
                cap_floor_m=76.9,
                season_year=year,
                cap_schedule_m={year: 104.0},
                teams=[team],
            )
        return calculate_team_cap_snapshot(
            team,
            league,
            season_label=f"{year}-{year + 1 - 2000:02d}",
            calendar_cursor=cursor,
            regular_season_last_index=192,
            include_expiring=count_expiring,
            opening_day=opening,
        )

    def test_hydrate_keeps_july_flag_and_does_not_restore_old_term(self):
        from services.contract_economy import hydrate_player_contract

        player = self._skater("zub", years=1, aav=4.5, pending=True, expiry=2027)
        player.pending_july1_expiry = True
        player.contract.pop("pending_july1_expiry")
        player.contract["season_cap_hits"] = [4.5, 4.5, 4.5]
        hydrate_player_contract(player, 2026)
        self.assertTrue(player.contract.get("pending_july1_expiry"))
        self.assertTrue(player.pending_july1_expiry)
        self.assertEqual(int(player.contract["years_remaining"]), 1)

    def test_sync_does_not_stretch_a_july_deal_from_the_old_grid(self):
        deal = {
            "pending_july1_expiry": True,
            "years_remaining": 1,
            "aav_m": 4.5,
            "cap_hit_m": 4.5,
            "season_cap_hits": [4.5, 4.5, 4.5],
            "effective_season": 2024,
        }
        _sync_term_from_season_cap_hits(deal, 2026)
        self.assertEqual(deal["years_remaining"], 1)

    def test_old_season_snapshot_does_not_roll_the_announced_cap_backward(self):
        team = self._club([self._skater("core", years=4, aav=5.0, expiry=2030)])
        league = SimpleNamespace(
            salary_cap_m=113.5,
            cap_floor_m=83.9,
            season_year=2027,
            cap_schedule_m={2026: 104.0, 2027: 113.5},
            teams=[team],
        )
        snap = self._snap(team, year=2026, opening=True, league=league)
        self.assertAlmostEqual(snap["upperLimit"], 104.0, places=2)
        self.assertAlmostEqual(league.salary_cap_m, 113.5, places=2)
        self.assertEqual(int(league.season_year), 2027)

    def test_opening_day_drops_unsigned_final_year_and_keeps_an_extension(self):
        expiring = self._skater("zub", years=1, aav=13.0, pending=True, expiry=2027)
        extended = self._skater("vet", years=1, aav=8.0, pending=True, expiry=2027, extra={
            "pending_extension": {"aav_m": 2.0, "cap_hit_m": 2.0, "years": 3, "years_remaining": 3},
        })
        staying = self._skater("core", years=4, aav=5.0, expiry=2030)
        team = self._club([expiring, extended, staying])
        opening = self._snap(team, opening=True)
        in_season = self._snap(team, opening=False, count_expiring=True)
        self.assertAlmostEqual(opening["activeRosterCapHit"], 7.0, places=2)
        self.assertAlmostEqual(in_season["activeRosterCapHit"], 26.0, places=2)
        self.assertAlmostEqual(
            in_season["usableCapSpace"] - opening["usableCapSpace"],
            -19.0,
            places=2,
        )

    def test_in_season_hub_and_office_count_the_same_expiring_money(self):
        from services.contract_economy import cap_books_for_session, get_team_cap_snapshot_full

        expiring = self._skater("zub", years=1, aav=13.0, pending=True, expiry=2027)
        team = self._club([expiring])
        league = SimpleNamespace(
            salary_cap_m=104.0,
            cap_floor_m=76.9,
            season_year=2026,
            cap_schedule_m={2026: 104.0},
            teams=[team],
        )
        regular = SimpleNamespace(
            phase="regular",
            season_calendar_year=2026,
            calendar_cursor=40,
            nhl_regular_season_last_index=192,
            sim=SimpleNamespace(league=league),
        )
        resign = SimpleNamespace(
            phase="re_sign",
            season_calendar_year=2026,
            calendar_cursor=192,
            nhl_regular_season_last_index=192,
            sim=SimpleNamespace(league=league),
        )
        regular_books = cap_books_for_session(regular)
        resign_books = cap_books_for_session(resign)
        self.assertTrue(regular_books["count_expiring"])
        self.assertFalse(regular_books["opening_day"])
        self.assertTrue(resign_books["opening_day"])
        self.assertFalse(resign_books["count_expiring"])
        regular_full = get_team_cap_snapshot_full(team, league, **{
            k: regular_books[k]
            for k in ("season_year", "calendar_cursor", "regular_season_last_index", "count_expiring", "opening_day")
        })
        resign_full = get_team_cap_snapshot_full(team, league, **{
            k: resign_books[k]
            for k in ("season_year", "calendar_cursor", "regular_season_last_index", "count_expiring", "opening_day")
        })
        self.assertAlmostEqual(regular_full["active_roster_cap_hit_m"], 13.0, places=2)
        self.assertAlmostEqual(resign_full["active_roster_cap_hit_m"], 0.0, places=2)
        self.assertGreater(resign_full["usable_cap_space_m"], regular_full["usable_cap_space_m"] + 10)

    def test_finished_calendar_does_not_inflate_deadline_space(self):
        team = self._club([self._skater("core", years=4, aav=103.0, expiry=2030)])
        snap = self._snap(team, opening=False, count_expiring=True, cursor=192)
        self.assertAlmostEqual(snap["usableCapSpace"], snap["projectedDeadlineSpace"], places=2)
        self.assertLess(snap["usableCapSpace"], 5.0)

    def test_multi_year_buyout_dict_counts_one_season(self):
        from app.sim_engine.economy.cap_engine import _sum_money_records_millions

        records = {"2026-27": 1.5, "2027-28": 1.5, "2028-29": 1.5}
        self.assertAlmostEqual(_sum_money_records_millions(records, None), 1.5, places=2)
        team = self._club([], buyout_cap_hits=records)
        snap = self._snap(team, opening=True)
        self.assertAlmostEqual(snap["buyoutCapHit"], 1.5, places=2)

    def test_retained_share_is_not_the_full_aav(self):
        from app.sim_engine.economy.cap_engine import player_cap_hit_millions

        player = self._skater("kept", years=3, aav=8.0, expiry=2029)
        player.retained_share_pct = 50
        player.retained_share_expiry = player.contract["expiry_year"]
        self.assertAlmostEqual(player_cap_hit_millions(player), 4.0, places=2)

    def test_stacked_cap_hit_above_aav_stays_at_aav(self):
        from app.sim_engine.economy.cap_engine import player_full_cap_hit_millions

        player = self._skater("bonus", years=3, aav=6.0, expiry=2029)
        player.contract["cap_hit_m"] = 14.0
        self.assertAlmostEqual(player_full_cap_hit_millions(player), 6.0, places=2)

    def test_duplicate_roster_rows_are_charged_once_on_opening_day(self):
        one = self._skater("twin", years=4, aav=6.0, expiry=2030)
        team = self._club([one, one])
        snap = self._snap(team, opening=True)
        self.assertAlmostEqual(snap["activeRosterCapHit"], 6.0, places=2)

    def test_buried_final_year_deal_is_off_next_years_books(self):
        buried = self._skater("ahl", years=1, aav=4.0, pending=True, expiry=2027, buried=True)
        team = self._club([], ahl_roster=[buried])
        opening = self._snap(team, opening=True)
        in_season = self._snap(team, opening=False, count_expiring=True)
        self.assertAlmostEqual(opening["buriedCapHit"], 0.0, places=2)
        self.assertGreater(in_season["buriedCapHit"], 1.0)

    def test_unstamped_bonus_reserve_does_not_tax_opening_day(self):
        team = self._club(
            [self._skater("core", years=4, aav=5.0, expiry=2030)],
            performance_bonus_reserve_m=12.0,
        )
        opening = self._snap(team, opening=True)
        in_season = self._snap(team, opening=False, count_expiring=True)
        self.assertAlmostEqual(opening["performanceBonusReserve"], 0.0, places=2)
        self.assertAlmostEqual(in_season["performanceBonusReserve"], 12.0, places=2)
        team.performance_bonus_reserve_season = 2026
        stamped = self._snap(team, opening=True)
        self.assertAlmostEqual(stamped["performanceBonusReserve"], 12.0, places=2)

    def test_final_year_resign_replaces_the_deal_instead_of_staying_pending(self):
        from services.contract_economy import sign_player_to_team

        player = self._skater("zub", years=1, aav=4.5, pending=False, expiry=2027)
        team = self._club([player])
        league = SimpleNamespace(
            salary_cap_m=104.0,
            cap_floor_m=76.9,
            season_year=2026,
            cap_schedule_m={2026: 104.0, 2027: 113.5},
            teams=[team],
            free_agents=[],
        )
        result = sign_player_to_team(
            player,
            team,
            league,
            2026,
            {"aav_m": 6.0, "years": 4, "context": "re_sign", "force": True},
        )
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(result.get("status"), "accepted")
        self.assertFalse(result.get("pending_extension"))
        self.assertIsNone(player.contract.get("pending_extension"))
        self.assertEqual(int(player.contract["years_remaining"]), 4)
        self.assertEqual(int(player.contract["expiry_year"]), 2031)
        self.assertAlmostEqual(float(player.contract["aav_m"]), 6.0, places=2)

    def test_multi_year_extension_still_waits_until_the_current_deal_ends(self):
        from services.contract_economy import sign_player_to_team

        player = self._skater("stay", years=3, aav=5.0, pending=False, expiry=2029)
        team = self._club([player])
        league = SimpleNamespace(
            salary_cap_m=104.0,
            cap_floor_m=76.9,
            season_year=2026,
            cap_schedule_m={2026: 104.0},
            teams=[team],
            free_agents=[],
        )
        result = sign_player_to_team(
            player,
            team,
            league,
            2026,
            {"aav_m": 7.0, "years": 4, "context": "re_sign", "force": True},
        )
        self.assertTrue(result.get("ok"), result)
        self.assertTrue(result.get("pending_extension"))
        self.assertEqual(int(player.contract["years_remaining"]), 3)
        self.assertAlmostEqual(float(player.contract["aav_m"]), 5.0, places=2)
        self.assertAlmostEqual(float(player.contract["pending_extension"]["aav_m"]), 7.0, places=2)


if __name__ == "__main__":
    unittest.main()
