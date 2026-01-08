import random
from contextlib import redirect_stdout
from datetime import datetime
import os
import time

from app.sim_engine.engine import SimEngine
from app.sim_engine.entities.player import (
    Player,
    IdentityBio,
    BackstoryUpbringing,
    PersonalityTraits,
    CareerArcSeeds,
    Position,
    Shoots,
    BackstoryType,
    UpbringingType,
    SupportLevel,
    PressureLevel,
    DevResources,
)
from app.sim_engine.entities.team import Team, TeamArchetype


# ==================================================
# RANDOM PLAYER FACTORY
# ==================================================

def create_random_player(rng: random.Random) -> Player:
    identity = IdentityBio(
        name=f"SimPlayer_{rng.randint(1000, 9999)}",
        age=18,
        birth_year=2007,
        birth_country="CAN",
        birth_city="Unknown",
        height_cm=rng.randint(170, 200),
        weight_kg=rng.randint(70, 100),
        position=rng.choice(list(Position)),
        shoots=rng.choice(list(Shoots)),
        draft_year=2025,
        draft_round=rng.randint(1, 7),
        draft_pick=rng.randint(1, 217),
    )

    backstory = BackstoryUpbringing(
        backstory=rng.choice(list(BackstoryType)),
        upbringing=rng.choice(list(UpbringingType)),
        family_support=rng.choice(list(SupportLevel)),
        early_pressure=rng.choice(list(PressureLevel)),
        dev_resources=rng.choice(list(DevResources)),
    )

    traits = PersonalityTraits(
        loyalty=rng.random(),
        ambition=rng.random(),
        money_focus=rng.random(),
        family_priority=rng.random(),
        legacy_drive=rng.random(),
        confidence=rng.random(),
        volatility=rng.random(),
        work_ethic=rng.random(),
        mental_toughness=rng.random(),
        coachability=rng.random(),
        leadership=rng.random(),
    )

    career = CareerArcSeeds(
        expected_peak_age=rng.randint(24, 30),
        decline_rate=rng.random(),
        breakout_probability=rng.random() * 0.25,
        bust_probability=rng.random() * 0.20,
        prime_duration=rng.random(),
        season_consistency=rng.random(),
        regression_resistance=rng.random(),
        ceiling_floor_gap=rng.random(),
        dev_curve_seed=rng.randint(1, 2_000_000_000),
    )

    return Player(
        identity=identity,
        backstory=backstory,
        ratings={},
        traits=traits,
        career=career,
        rng_seed=rng.randint(1, 2_000_000_000),
    )


# ==================================================
# TEAM DEBUG SNAPSHOT
# ==================================================

def dump_team_snapshot(team: Team):
    print("\n================ TEAM SNAPSHOT ================")
    print(f"Team: {team.city} {team.name}")
    print(f"Archetype: {team.archetype}")
    print(f"Status: {team.state.status}")

    print("\n[MARKET]")
    for k, v in vars(team.market).items():
        print(f"{k:22s}: {v}")

    print("\n[OWNERSHIP]")
    for k, v in vars(team.ownership).items():
        print(f"{k:22s}: {v:.3f}")

    print("\n[REPUTATION]")
    for k, v in vars(team.reputation).items():
        print(f"{k:22s}: {v:.3f}")

    print("\n[ORG PHILOSOPHY]")
    print(f"Development Quality    : {team.development_quality:.3f}")
    print(f"Prospect Patience      : {team.prospect_patience:.3f}")
    print(f"Risk Tolerance         : {team.risk_tolerance:.3f}")

    print("\n[DYNAMIC STATE]")
    print(f"Competitive Score      : {team.state.competitive_score:.3f}")
    print(f"Team Morale            : {team.state.team_morale:.3f}")
    print(f"Org Pressure           : {team.state.organizational_pressure:.3f}")
    print(f"Stability              : {team.state.stability:.3f}")
    print(f"Ownership Stability    : {team.state.ownership_stability:.3f}")
    print(f"Arena Security         : {team.state.arena_security:.3f}")
    print(f"Financial Health       : {team.state.financial_health:.3f}")

    print("\n[ROSTER QUALITY PROXY]")
    print(f"Star Count             : {team.roster_quality.star_count}")
    print(f"Core Count             : {team.roster_quality.core_count}")
    print(f"Depth Quality          : {team.roster_quality.depth_quality:.3f}")

    if team.state.triggered_events:
        print("\n[TRIGGERED EVENTS]")
        for e in team.state.triggered_events:
            print(f"- {e}")

    print("==============================================")


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    # --------------------------------------------------
    # TRUE RANDOM SEED (system entropy)
    # --------------------------------------------------
    master_seed = int(time.time_ns())
    rng = random.Random(master_seed)

    # --------------------------------------------------
    # Create RANDOM entities
    # --------------------------------------------------
    player = create_random_player(rng)

    team = Team(
        team_id=rng.randint(1, 32),
        city=rng.choice(
            ["Ottawa", "Toronto", "Montreal", "Boston", "New York", "Chicago", "Detroit"]
        ),
        name=rng.choice(
            ["Senators", "Maple Leafs", "Canadiens", "Bruins", "Rangers", "Blackhawks", "Red Wings"]
        ),
        division=rng.choice(["Atlantic", "Metropolitan", "Central", "Pacific"]),
        conference=rng.choice(["East", "West"]),
        archetype=rng.choice(
            [
                TeamArchetype.PATIENT_BUILDER,
                TeamArchetype.WIN_NOW,
                TeamArchetype.MEDIOCRE,
                TeamArchetype.DRAFT_AND_DEVELOP,
                TeamArchetype.CHAOTIC,
            ]
        ),
        rng=rng,
    )

    sim = SimEngine(seed=rng.randint(1, 2_000_000_000))
    sim.set_player(player)
    sim.set_team(team)

    # --------------------------------------------------
    # Terminal header (minimal)
    # --------------------------------------------------
    print("\n=================================================")
    print(f"CAREER SIM RUN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=================================================")
    print(f"Seed: {master_seed}")
    print(f"Player: {player.name}")
    print(f"Team: {team.city} {team.name} ({team.archetype})")
    print(f"Initial OVR: {round(player.ovr(), 3)}")
    print("-------------------------------------------------")
    print("Running sim… full output → app/sim_results.txt")
    print("-------------------------------------------------\n")

    # --------------------------------------------------
    # Ensure output directory exists
    # --------------------------------------------------
    os.makedirs("app", exist_ok=True)

    # --------------------------------------------------
    # WRITE FULL SIM OUTPUT TO FILE
    # --------------------------------------------------
    with open("app/sim_results.txt", "a", encoding="utf-8") as f:
        with redirect_stdout(f):

            print("\n=================================================")
            print(f"CAREER SIM RUN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=================================================")
            print(f"Seed: {master_seed}")
            print(f"Player: {player.name}")
            print(f"Team: {team.city} {team.name} ({team.archetype})")
            print(f"Initial OVR: {round(player.ovr(), 3)}")
            print("-------------------------------------------------\n")

            dump_team_snapshot(team)

            sim.sim_years(40)

            print("\n================ CAREER SUMMARY ================")
            print(f"Final Age              : {player.age}")
            print(f"Final OVR              : {round(player.ovr(), 3)}")
            print(f"Retired                : {player.retired}")
            if player.retired:
                print(f"Retirement Reason      : {player.retirement_reason}")
            print("===============================================\n")

    print("Simulation complete. Results saved to app/sim_results.txt\n")
