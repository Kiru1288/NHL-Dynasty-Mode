"""
decision_weights.py

Defines how different LIFE DOMAINS influence player decisions.

This file does NOT:
- Roll randomness
- Trigger outcomes
- Evaluate game state

It ONLY defines:
- Which domains matter for which decisions
- How strongly each domain pulls
- How traits inside each domain are weighted

This is the soul of the simulation.
"""

# -------------------------------------------------------------------
# DOMAIN DEFINITIONS
# -------------------------------------------------------------------
# Domains represent major areas of a human life.
# Traits and events push or pull pressure inside these domains.

DOMAINS = [
    "career_identity",     # ambition, legacy, confidence, meaning
    "health",              # injuries, pain, mental health
    "family",              # relationships, stability, time
    "psychological",       # burnout, anxiety, joy, stress
    "security",            # money, safety, long-term comfort
    "environment",         # media pressure, city, culture, law
]

# -------------------------------------------------------------------
# DECISION WEIGHTS
# -------------------------------------------------------------------
# Positive weights PUSH toward the decision
# Negative weights PULL away from the decision

DECISION_WEIGHTS = {

    # ===============================================================
    # RETIREMENT
    # ===============================================================
    "retirement": {
        "domains": {
            "health": {
                "injury_risk": 0.9,
                "chronic_pain": 0.8,
                "mental_fatigue": 0.7,
            },
            "family": {
                "family_priority": 0.8,
                "relationship_strain": 0.6,
                "desire_for_normal_life": 0.5,
            },
            "psychological": {
                "burnout": 0.9,
                "anxiety": 0.6,
                "loss_of_joy": 0.8,
            },
            "career_identity": {
                "legacy_drive": -0.7,
                "ambition": -0.6,
                "confidence": -0.4,
            },
            "security": {
                "financial_security": 0.4,
                "money_focus": -0.3,
            },
        }
    },

    # ===============================================================
    # TRADE REQUEST
    # ===============================================================
    "trade_request": {
        "domains": {
            "career_identity": {
                "competitiveness": 0.9,
                "ego": 0.6,
                "confidence": 0.4,
            },
            "environment": {
                "media_pressure": 0.6,
                "market_fit": 0.5,
                "fan_hostility": 0.4,
            },
            "family": {
                "family_priority": -0.6,
                "stability_need": -0.5,
            },
            "psychological": {
                "frustration": 0.7,
                "patience": -0.6,
            },
            "career_identity_negative": {
                "loyalty": -0.8,
            },
        }
    },

    # ===============================================================
    # CONTRACT ACCEPTANCE / HOLDOUT
    # ===============================================================
    "contract_decision": {
        "domains": {
            "security": {
                "money_focus": 0.9,
                "financial_anxiety": 0.7,
            },
            "career_identity": {
                "ambition": 0.5,
                "legacy_drive": 0.4,
            },
            "family": {
                "family_priority": 0.4,
                "stability_need": 0.6,
            },
            "psychological": {
                "confidence": 0.3,
                "fear_of_decline": 0.6,
            },
        }
    },

    # ===============================================================
    # PLAY THROUGH INJURY
    # ===============================================================
    "play_through_injury": {
        "domains": {
            "career_identity": {
                "competitiveness": 0.9,
                "ego": 0.6,
                "leadership": 0.5,
            },
            "health": {
                "injury_risk": -0.9,
                "pain_tolerance": 0.6,
            },
            "psychological": {
                "fear_of_replacement": 0.7,
                "confidence": 0.4,
            },
            "family": {
                "family_priority": -0.5,
            },
        }
    },

    # ===============================================================
    # LEAVE OF ABSENCE (MENTAL / PERSONAL)
    # ===============================================================
    "leave_of_absence": {
        "domains": {
            "psychological": {
                "burnout": 1.0,
                "anxiety": 0.8,
                "emotional_distress": 0.7,
            },
            "health": {
                "mental_health_risk": 0.9,
            },
            "career_identity": {
                "legacy_drive": -0.5,
                "confidence": -0.4,
            },
            "environment": {
                "media_pressure": 0.6,
            },
        }
    },
}

# -------------------------------------------------------------------
# EVENT PRESSURE PRESETS (OFF-ICE LIFE EVENTS)
# -------------------------------------------------------------------
# These do NOT cause outcomes.
# They inject domain pressure that influences decisions over time.

EVENT_PRESSURES = {

    "mental_health_crisis": {
        "psychological": 0.8,
        "health": 0.4,
        "career_identity": -0.3,
    },

    "legal_trouble_minor": {
        "environment": 0.5,
        "psychological": 0.4,
        "security": 0.3,
    },

    "legal_trouble_major": {
        "environment": 0.9,
        "psychological": 0.7,
        "security": 0.6,
        "career_identity": -0.6,
    },

    "family_illness": {
        "family": 0.9,
        "psychological": 0.6,
        "career_identity": -0.4,
    },

    "media_scandal": {
        "environment": 0.8,
        "psychological": 0.6,
        "career_identity": -0.5,
    },

    "loss_of_love_for_game": {
        "psychological": 0.9,
        "career_identity": -0.8,
    },
}

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------

def get_decision_weights(decision_name: str) -> dict:
    """
    Returns domain -> trait weight mapping for a decision.
    """
    return DECISION_WEIGHTS.get(decision_name, {}).get("domains", {})


def get_event_pressure(event_name: str) -> dict:
    """
    Returns domain pressure deltas for an off-ice event.
    """
    return EVENT_PRESSURES.get(event_name, {})
