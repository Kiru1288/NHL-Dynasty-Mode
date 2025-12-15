class Player:
    def __init__(self, name, age, attributes, traits=None, backstory=None):
        self.name = name
        self.age = age

        self.attributes = attributes          # dict
        self.traits = traits or {}             # determination, durability, etc.
        self.backstory = backstory or {}       # tags, not prose

        self.energy = 1.0
        self.morale = 0.5
        self.retired = False

    def age_one_year(self):
        self.age += 1

    def __repr__(self):
        return f"{self.name} (Age {self.age})"
