FIRST_NAMES = [
    "Alex", "Connor", "Elias", "Leo", "Mikko", "Ryan", "Jack", "Noah"
]

LAST_NAMES = [
    "Smith", "Hughes", "Pettersson", "Dubois", "Kuznetsov", "Miller"
]

def generate_name(rng):
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
