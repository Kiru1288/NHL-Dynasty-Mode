def check_retirement(player, rng):
    if player.age < 35:
        return False

    chance = 0.05 + (player.age - 35) * 0.03
    if rng.rand() < chance:
        player.retired = True
        return True

    return False
