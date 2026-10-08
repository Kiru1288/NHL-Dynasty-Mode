/**
 * In-game confirm dialog (replaces window.confirm, which breaks immersion and is
 * blocked in some embedded views). Returns a Promise<boolean>.
 *
 *   if (!(await gameConfirm("Clear every lineup slot?"))) return;
 */
const STYLE_ID = "game-confirm-style";

function ensureStyle() {
  if (typeof document === "undefined" || document.getElementById(STYLE_ID)) return;
  const el = document.createElement("style");
  el.id = STYLE_ID;
  el.textContent = `
.gcf-backdrop{position:fixed;inset:0;z-index:4000;display:flex;align-items:center;justify-content:center;
  background:rgba(4,10,18,.62);padding:16px}
.gcf-card{width:min(420px,100%);background:#0f1b29;color:#e6eef6;border:1px solid rgba(255,255,255,.14);
  border-radius:10px;box-shadow:0 18px 48px rgba(0,0,0,.45);padding:20px 20px 16px;font:500 15px/1.45 inherit}
.gcf-title{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#8fb2d6;margin:0 0 8px}
.gcf-msg{margin:0 0 18px;white-space:pre-wrap}
.gcf-row{display:flex;gap:10px;justify-content:flex-end;flex-wrap:wrap}
.gcf-btn{font:600 14px/1 inherit;padding:10px 16px;border-radius:6px;cursor:pointer;border:1px solid rgba(255,255,255,.2);
  background:transparent;color:inherit}
.gcf-btn--ok{background:#2f6fd6;border-color:#2f6fd6;color:#fff}
.gcf-btn--danger{background:#c8323f;border-color:#c8323f;color:#fff}
.gcf-btn:focus-visible{outline:2px solid #9cc4ff;outline-offset:2px}`;
  document.head.appendChild(el);
}

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
