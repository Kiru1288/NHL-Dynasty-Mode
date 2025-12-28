from app.sim_engine.engine import SimEngine

if __name__ == "__main__":
    sim = SimEngine(seed=42)
    sim.sim_years(40)
