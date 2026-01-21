def generate_attributes(rng):
    return {
        "shooting": rng.uniform(0.3, 0.8),
        "passing": rng.uniform(0.3, 0.8),
        "defense": rng.uniform(0.3, 0.8),
        "skating": rng.uniform(0.3, 0.8),
        "durability": rng.uniform(0.3, 0.8),
        "determination": rng.uniform(0.3, 0.8),
    }
