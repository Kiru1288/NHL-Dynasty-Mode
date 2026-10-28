/**
 * In-game confirm dialog (replaces window.confirm, which breaks immersion and is
 * blocked in some embedded views). Returns a Promise<boolean>.
 *
 *   if (!(await gameConfirm("Clear every lineup slot?"))) return;
 */
// Confirm dialog styles live in styles/game.css (.gcf-*).
function ensureStyle() {}

export function gameConfirm(message, options = {}) {
  if (typeof document === "undefined") return Promise.resolve(false);
  const {
    title = "Confirm",
    confirmLabel = "Confirm",
    cancelLabel = "Cancel",
    danger = false,
  } = options;
  ensureStyle();
  return new Promise((resolve) => {
    const previous = document.activeElement;
    const backdrop = document.createElement("div");
    backdrop.className = "gcf-backdrop";
    backdrop.setAttribute("role", "presentation");
    const card = document.createElement("div");
    card.className = "gcf-card";
    card.setAttribute("role", "alertdialog");
    card.setAttribute("aria-modal", "true");
    const h = document.createElement("p");
    h.className = "gcf-title";
    h.id = `gcf-title-${Date.now()}`;
    h.textContent = title;
    const p = document.createElement("p");
    p.className = "gcf-msg";
    p.textContent = String(message || "");
    card.setAttribute("aria-labelledby", h.id);
    const row = document.createElement("div");
    row.className = "gcf-row";
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "gcf-btn";
    cancel.textContent = cancelLabel;
    const ok = document.createElement("button");
    ok.type = "button";
    ok.className = `gcf-btn ${danger ? "gcf-btn--danger" : "gcf-btn--ok"}`;
    ok.textContent = confirmLabel;
    row.append(cancel, ok);
    card.append(h, p, row);
    backdrop.appendChild(card);

    const close = (value) => {
      document.removeEventListener("keydown", onKey, true);
      backdrop.remove();
      if (previous && typeof previous.focus === "function") previous.focus();
      resolve(value);
    };
    const onKey = (e) => {
      if (e.key === "Escape") { e.preventDefault(); close(false); }
      else if (e.key === "Enter") { e.preventDefault(); close(true); }
      else if (e.key === "Tab") {
        e.preventDefault();
        (document.activeElement === ok ? cancel : ok).focus();
      }
    };
    cancel.addEventListener("click", () => close(false));
    ok.addEventListener("click", () => close(true));
    backdrop.addEventListener("mousedown", (e) => { if (e.target === backdrop) close(false); });
    document.addEventListener("keydown", onKey, true);
    document.body.appendChild(backdrop);
    ok.focus();
  });
}

export default gameConfirm;
