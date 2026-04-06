"""
Player generation system (human + global).

NOTE: This project currently only contains a subset of the originally planned
generation modules. Per repository constraints, the full implementation lives
inside the existing files in this folder.
"""

from __future__ import annotations

from app.sim_engine.generation.name_generator import generate_human_identity
from app.sim_engine.generation.attribute_generator import (
    Archetype,
    generate_backstory,
    generate_attributes,
    generate_health,
    compute_development_rate,
)
from app.sim_engine.generation.trait_generator import generate_traits
from app.sim_engine.generation.draft_class_generator import generate_draft_class, generate_player_profile

__all__ = [
    "Archetype",
    "generate_human_identity",
    "generate_backstory",
    "generate_attributes",
    "generate_health",
    "compute_development_rate",
    "generate_traits",
    "generate_player_profile",
    "generate_draft_class",
]

