import random

class RandomManager:
    def __init__(self, seed=None):
        self.seed = seed
        self.random = random.Random(seed)

    def rand(self):
        return self.random.random()

    def randint(self, a, b):
        return self.random.randint(a, b)

    def choice(self, seq):
        return self.random.choice(seq)

    def uniform(self, a, b):
        return self.random.uniform(a, b)
