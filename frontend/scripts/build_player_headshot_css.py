"""One-off builder: merges base playerHeadshot.css with extended engine styles."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = (ROOT / "src" / "styles" / "playerHeadshot.css").read_text(encoding="utf-8")
if "@import" in BASE and len(BASE.strip()) < 120:
    # already migrated — read components file as base if styles is import-only
    comp = ROOT / "src" / "components" / "PlayerHeadshot.css"
    if comp.exists():
        BASE = comp.read_text(encoding="utf-8")

# If base still has import only, read from backup by reconstructing from git isn't available;
# use components if it has headshot-1
if ".headshot-1" not in BASE:
    raise SystemExit("Base CSS with headshot-1..15 not found")

# Strip prior extension marker if re-running
marker = "/* =========================================================\n   Extended Headshots 16–60"
if marker in BASE:
    BASE = BASE.split(marker)[0].rstrip()

palettes = [
    ("#e8b892", "#b06f48", "#1a120c", "#090605", "#0f4c81", "#08243f"),
    ("#c99672", "#875b42", "#8a9299", "#46515f", "#334155", "#0f172a"),
    ("#b87b52", "#77462e", "#241610", "#120a08", "#7f1d1d", "#450a0a"),
    ("#f0bc91", "#b9784f", "#2a2018", "#120d0a", "#1e3a8a", "#172554"),
    ("#d2a078", "#91603e", "#2a1b12", "#090504", "#ffffff", "#9da8b1"),
    ("#d7a47c", "#8e5c3c", "#151515", "#050505", "#2b4c7e", "#101f3d"),
    ("#c89267", "#875538", "#111111", "#030303", "#14532d", "#052e16"),
    ("#c89267", "#875538", "transparent", "transparent", "#27272a", "#09090b"),
    ("#5e3a29", "#392117", "#080808", "#020202", "#7c2d12", "#431407"),
    ("#7b4d36", "#44291d", "#0b0b0b", "#000000", "#854d0e", "#422006"),
    ("#e0b084", "#a66f48", "#d9b45a", "#8f6327", "#1d4ed8", "#172554"),
    ("#c98962", "#8a563b", "#1d2939", "#020617", "#111827", "#020617"),
    ("#f0bc91", "#b9784f", "#a8421e", "#5f1d0b", "#991b1b", "#450a0a"),
    ("#db9d6f", "#9c623c", "#312014", "#100907", "#006d77", "#00343a"),
    ("#b9855e", "#815237", "#122032", "#07101c", "#44403c", "#1c1917"),
    ("#f0bc91", "#c58b5f", "#c4a574", "#8a6a3d", "#2563eb", "#1e3a8a"),
    ("#e8c090", "#b07040", "#ffd700", "#b8860b", "#111827", "#030712"),
    ("#a96e4b", "#70422c", "#14100e", "#040302", "#57534e", "#292524"),
    ("#d7a47c", "#8e5c3c", "#101820", "#020608", "#0369a1", "#0c4a6e"),
    ("#c99672", "#875b42", "#4a4034", "#1c1917", "#374151", "#111827"),
    ("#8f5f41", "#5d3828", "#111111", "#030303", "#166534", "#14532d"),
    ("#c58b5f", "#8d5b3d", "#2b1810", "#120a06", "#7c3aed", "#4c1d95"),
    ("#d39a71", "#875538", "#50331e", "#1d1008", "#dc2626", "#7f1d1d"),
    ("#7b4d36", "#44291d", "#090909", "#000000", "#0891b2", "#164e63"),
    ("#e0b084", "#a66f48", "#8f6327", "#5c4018", "#ca8a04", "#713f12"),
    ("#c99672", "#875b42", "#9aa2ad", "#46515f", "#64748b", "#0f172a"),
    ("#5e3a29", "#392117", "#1b1614", "#070504", "#be123c", "#881337"),
    ("#b9855e", "#815237", "#122032", "#07101c", "#0284c7", "#075985"),
    ("#f0bc91", "#b9784f", "#d9b45a", "#8f6327", "#059669", "#064e3b"),
    ("#c89267", "#875538", "#2a2420", "#110e0c", "#9333ea", "#581c87"),
    ("#db9d6f", "#9c623c", "#312014", "#100907", "#ea580c", "#7c2d12"),
    ("#a96e4b", "#70422c", "#14100e", "#040302", "#0d9488", "#134e4a"),
    ("#c98962", "#8a563b", "#1d2939", "#020617", "#4338ca", "#312e81"),
    ("#d2a078", "#91603e", "#2a1b12", "#090504", "#f59e0b", "#78350f"),
    ("#e8b892", "#b06f48", "#1a120c", "#090605", "#14b8a6", "#115e59"),
    ("#7b4d36", "#44291d", "#0b0b0b", "#000000", "#ef4444", "#991b1b"),
    ("#d7a47c", "#8e5c3c", "#101820", "#020608", "#38bdf8", "#0c4a6e"),
    ("#c58b5f", "#8d5b3d", "#241610", "#120a08", "#84cc16", "#365314"),
    ("#b87b52", "#77462e", "#19110d", "#050302", "#f97316", "#7c2d12"),
    ("#c89267", "#875538", "#151515", "#050505", "#6366f1", "#312e81"),
    ("#5e3a29", "#392117", "#080808", "#020202", "#78716c", "#292524"),
    ("#e0b084", "#a66f48", "#d9b45a", "#8f6327", "#0ea5e9", "#0369a1"),
    ("#c99672", "#875b42", "#4a4034", "#1c1917", "#a855f7", "#6b21a8"),
    ("#f0bc91", "#b9784f", "#a8421e", "#5f1d0b", "#22c55e", "#14532d"),
    ("#b9855e", "#815237", "#122032", "#07101c", "#eab308", "#713f12"),
]

EXTENSIONS = (ROOT / "src" / "components" / "_player_headshot_extensions.css").read_text(encoding="utf-8")

parts = [
    BASE.rstrip(),
    "\n\n/* =========================================================\n   Extended Headshots 16–60\n   ========================================================= */\n",
]

for idx, pal in enumerate(palettes, start=16):
    skin, skin_shadow, hair, hair2, jersey, jersey2 = pal
    parts.append(
        f".headshot-{idx} {{ --skin:{skin}; --skin-shadow:{skin_shadow}; --hair:{hair}; --hair2:{hair2}; --jersey:{jersey}; --jersey2:{jersey2}; }}\n"
    )

parts.append(EXTENSIONS)
out = "".join(parts)
(ROOT / "src" / "components" / "PlayerHeadshot.css").write_text(out, encoding="utf-8")
(ROOT / "src" / "styles" / "playerHeadshot.css").write_text('@import "../components/PlayerHeadshot.css";\n', encoding="utf-8")
print(f"Wrote {len(out)} bytes")
