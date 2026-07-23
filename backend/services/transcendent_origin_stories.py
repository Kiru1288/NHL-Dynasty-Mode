"""Origin story templates for transcendent and rare prospect flavor."""
from __future__ import annotations

from typing import Any, Dict

from services.draft_ranking_logic import (
    ORIGIN_STORY_BY_KEY,
    TRANSCENDENT_BACKSTORIES,
    pick_transcendent_backstory,
)


def pick_origin_story(rng: Any, *, transcendent: bool = False, seed_hint: int = 0, key: str = "") -> Dict[str, Any]:
    if key and key in ORIGIN_STORY_BY_KEY:
        tpl = dict(ORIGIN_STORY_BY_KEY[key])
    elif rng is not None:
        picked = pick_transcendent_backstory(rng)
        tpl = dict(picked.get("origin_story") or picked)
    else:
        idx = abs(seed_hint) % len(TRANSCENDENT_BACKSTORIES)
        tpl = dict(TRANSCENDENT_BACKSTORIES[idx % len(TRANSCENDENT_BACKSTORIES)])
    if transcendent and tpl.get("key") == "undersized_skill_wizard":
        tpl = dict(TRANSCENDENT_BACKSTORIES[0])
    return {
        "origin_story": {
            "key": tpl["key"],
            "title": tpl["title"],
            "summary": tpl["summary"],
            "traits": list(tpl.get("traits") or []),
            "full_text": tpl.get("full_text") or tpl["summary"],
        },
        **{k: tpl[k] for k in ("key", "title", "summary", "traits", "full_text") if k in tpl},
    }


def attach_origin_story_to_player(player: Any, story: Dict[str, Any]) -> None:
    block = story.get("origin_story") or story
    setattr(player, "origin_story", block)
    setattr(player, "backstory_key", str(block.get("key") or ""))
