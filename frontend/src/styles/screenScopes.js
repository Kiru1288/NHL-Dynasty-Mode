// Keeps the <html> scope classes from gameScopes.js in step with what is on screen.
//
// game.css used ":root:has(.trade-hub-root, …)" to turn each screen's styles on.
// That made every DOM insert re-test ~9,000 selectors (about 110 ms on a busy
// screen). Now one MutationObserver checks the 33 scope lists after each batch
// of DOM changes. The callback runs as a microtask, before the browser paints,
// so a screen never shows a frame without its styles.

import GAME_SCOPES from "./gameScopes";

// A kept-alive screen (the hub stays mounted, hidden, while other screens show) must not
// switch its scope on.
function visibleMatch(selector) {
  const first = document.querySelector(selector);
  if (!first) return false;
  if (!first.closest("[data-keepalive-hidden]")) return true;
  for (const el of document.querySelectorAll(selector)) {
    if (!el.closest("[data-keepalive-hidden]")) return true;
  }
  return false;
}

function sync(root) {
  const cls = root.classList;
  for (const [name, selector] of GAME_SCOPES) {
    const on = visibleMatch(selector);
    if (on !== cls.contains(name)) cls.toggle(name, on);
  }
}

export function installScreenScopes() {
  if (typeof document === "undefined" || typeof MutationObserver === "undefined") return;
  const root = document.documentElement;
  sync(root);
  // Only structural changes and class changes can turn a scope on or off.
  new MutationObserver(() => sync(root)).observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["class", "data-keepalive-hidden"],
  });
}
