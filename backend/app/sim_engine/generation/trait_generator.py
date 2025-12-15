def generate_traits(rng):
    traits = {}

    if rng.rand() < 0.15:
        traits["tough_upbringing"] = True
        traits["determination_bonus"] = 0.1

    if rng.rand() < 0.05:
        traits["chronic_condition"] = "diabetes"
        traits["injury_risk_bonus"] = 0.15

    return traits
