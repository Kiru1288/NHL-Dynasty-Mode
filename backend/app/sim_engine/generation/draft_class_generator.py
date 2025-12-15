from app.sim_engine.entities.player import Player
from app.sim_engine.generation.name_generator import generate_name
from app.sim_engine.generation.attribute_generator import generate_attributes
from app.sim_engine.generation.trait_generator import generate_traits


def generate_draft_class(rng, size=210):
    players = []

    for _ in range(size):
        player = Player(
            name=generate_name(rng),
            age=18,
            attributes=generate_attributes(rng),
            traits=generate_traits(rng),
        )
        players.append(player)

    return players
