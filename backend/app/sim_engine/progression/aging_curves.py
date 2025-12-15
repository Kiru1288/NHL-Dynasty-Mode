def age_player(player):
    player.age_one_year()

    # Simple decline after 30 (placeholder)
    if player.age > 30:
        for k in player.attributes:
            player.attributes[k] *= 0.995
