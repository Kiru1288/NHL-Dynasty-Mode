/** Fixed logical resolution (scaled to fit viewport). */
export const GAME_W = 1280;
export const GAME_H = 720;

export const SCREENS = {
  SETUP: "setup",
  HUB: "hub",
  ROSTER: "roster",
  SETTINGS: "settings",
};

/** Hub left-rail entries (controller order). */
export const HUB_MENU = [
  { id: "roster", label: "ROSTERS" },
  { id: "ops", label: "FRANCHISE OPS" },
  { id: "settings", label: "ADVANCED" },
  { id: "new", label: "NEW FRANCHISE" },
];

/** Discrete slider rows (local UI rules — not wired to sim yet). */
export const SETTINGS_ROWS = [
  { key: "roughing", label: "Roughing" },
  { key: "hooking", label: "Hooking" },
  { key: "slashing", label: "Slashing" },
  { key: "interference", label: "Interference" },
];
