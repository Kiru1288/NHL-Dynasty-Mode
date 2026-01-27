import random
from contextlib import redirect_stdout
from datetime import datetime
import os
import time

from app.sim_engine.engine import SimEngine
from app.sim_engine.entities.coach import Coach
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

    ratings = {
        "skating": rng.uniform(0.45, 0.60),
        "offense": rng.uniform(0.45, 0.60),
        "passing": rng.uniform(0.45, 0.60),
        "defense": rng.uniform(0.45, 0.60),
        "physical": rng.uniform(0.45, 0.60),
        "iq": rng.uniform(0.45, 0.60),
    }

    return Player(
        identity=identity,
        backstory=backstory,
        ratings=ratings,
        traits=traits,
        career=career,
        rng_seed=rng.randint(1, 2_000_000_000),
    )


# ==================================================
# COACH FACTORY (MATCHES Coach.__init__)
# ==================================================
def create_random_coach(rng: random.Random) -> Coach:
    coach = Coach(
        coach_id=f"COACH_{rng.randint(1000, 9999)}",
        name=f"Coach_{rng.randint(100, 999)}",
        age=rng.randint(38, 65),
        nationality=rng.choice(["CAN", "USA", "SWE", "FIN", "CZE"]),
        experience_years=rng.randint(5, 35),
        career_stage=rng.choice(["rookie", "established", "veteran", "late_career"]),
        role=rng.choice(["head_coach", "assistant", "associate", "skills_coach"]),
        specialization=rng.choice([
            "player_development",
            "systems",
            "motivation",
            "tactics",
            "discipline",
        ]),
    )
    coach.rng = rng  # optional, but safe
    return coach


# ==================================================
# TEAM SNAPSHOT
# ==================================================
def dump_team_snapshot(team: Team):
    print("\n================ TEAM SNAPSHOT ================")
    print(f"Team: {team.city} {team.name}")
    print(f"Archetype: {team.archetype}")
    print(f"Status: {team.state.status}")
    print(f"Competitive Score : {team.state.competitive_score:.3f}")
    print(f"Team Morale       : {team.state.team_morale:.3f}")
    print(f"Stability         : {team.state.stability:.3f}")
    print("===============================================")


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    print(">>> run_sim.py STARTED")

    master_seed = int(time.time_ns())
    rng = random.Random(master_seed)

    player = create_random_player(rng)

    team = Team(
        team_id=rng.randint(1, 32),
        city=rng.choice(["Ottawa", "Toronto", "Montreal", "Boston", "New York", "Chicago", "Detroit"]),
        name=rng.choice(["Senators", "Maple Leafs", "Canadiens", "Bruins", "Rangers", "Blackhawks", "Red Wings"]),
        division=rng.choice(["Atlantic", "Metropolitan", "Central", "Pacific"]),
        conference=rng.choice(["East", "West"]),
        archetype=rng.choice([
            TeamArchetype.PATIENT_BUILDER,
            TeamArchetype.WIN_NOW,
            TeamArchetype.MEDIOCRE,
            TeamArchetype.DRAFT_AND_DEVELOP,
            TeamArchetype.CHAOTIC,
        ]),
        rng=rng,
    )

    # 🔒 REQUIRED: attach coach BEFORE engine sees team
    team.coach = create_random_coach(rng)

    sim = SimEngine(seed=master_seed)
    sim.set_player(player)
    sim.set_team(team)

    os.makedirs("app", exist_ok=True)

    print(">>> Simulation wiring complete. Writing output to app/sim_results.txt")

    with open("app/sim_results.txt", "a", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("\n=================================================")
            print(f"CAREER SIM RUN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=================================================")
            print(f"Seed   : {master_seed}")
            print(f"Player : {player.name}")
            print(f"Team   : {team.city} {team.name} ({team.archetype})")
            print(f"Coach  : {team.coach.name} | {team.coach.role} | {team.coach.specialization}")
            print(f"Initial OVR: {player.ovr():.3f}")
            print("-------------------------------------------------")

            dump_team_snapshot(team)

            sim.sim_years(years=40, debug_dump=True, sleep_s=0.0)

            print("\n================ CAREER SUMMARY ================")
            print(f"Final Age : {player.age}")
            print(f"Final OVR : {player.ovr():.3f}")
            print(f"Retired   : {player.retired}")
            if player.retired:
                print(f"Reason    : {player.retirement_reason}")
            print("===============================================")

    print(">>> Simulation finished successfully.")
