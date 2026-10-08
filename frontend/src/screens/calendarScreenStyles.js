/**
 * Calendar screen stylesheet (split out of CalendarScreen.js).
 * Rendered by <CalendarStyles /> at mount so it keeps winning over shared .nhlcal-* shell rules.
 */
export const CALENDAR_CSS = `
      .nhlcal-root {
        --bg: #04101a;
        --bg-2: #061522;
        --panel: rgba(9, 25, 38, 0.94);
        --panel-2: rgba(12, 35, 52, 0.94);
        --panel-3: rgba(15, 46, 66, 0.78);
        --line: rgba(156, 218, 236, 0.14);
        --line-2: rgba(115, 229, 241, 0.25);
        --line-strong: rgba(73, 231, 240, 0.5);
        --text: #e9f7fb;
        --muted: #8096a8;
        --muted-2: #607789;
        --cyan: #13d8e7;
        --cyan-soft: rgba(19, 216, 231, 0.13);
        --gold: #e9a83c;
        --gold-soft: rgba(233, 168, 60, 0.14);
        --green: #52df94;
        --green-soft: rgba(82, 223, 148, 0.13);
        --red: #ff606d;
        --red-soft: rgba(255, 96, 109, 0.13);
        --orange: #ff8a4c;
        --orange-soft: rgba(255, 138, 76, 0.13);
        --blue: #8ab4ff;
        --blue-soft: rgba(138, 180, 255, 0.13);
        --purple: var(--ops-info, #8ab4ff);
        --purple-soft: rgba(138, 180, 255, 0.14);
        --shadow: 0 24px 70px rgba(0, 0, 0, 0.42);

        min-height: 100dvh;
        height: 100dvh;
        width: 100%;
        background:
          radial-gradient(circle at 24% 0%, rgba(19, 216, 231, 0.12), transparent 30%),
          radial-gradient(circle at 92% 18%, rgba(233, 168, 60, 0.08), transparent 26%),
          linear-gradient(180deg, #06131f 0%, #020a11 100%);
        color: var(--text);
        display: grid;
        grid-template-columns: 94px minmax(0, 1fr);
        overflow: hidden;
        font-family:
          Inter,
          ui-sans-serif,
          system-ui,
          -apple-system,
          BlinkMacSystemFont,
          "Segoe UI",
          sans-serif;
      }

      .nhlcal-root *,
      .nhlcal-root *::before,
      .nhlcal-root *::after {
        box-sizing: border-box;
      }

      .nhlcal-root button {
        font-family: inherit;
      }

      .nhlcal-sidebar {
        min-height: 100vh;
        background:
          linear-gradient(180deg, rgba(5, 16, 26, 0.98), rgba(3, 10, 17, 0.98)),
          radial-gradient(circle at 100% 14%, rgba(19, 216, 231, 0.14), transparent 34%);
        border-right: 1px solid var(--line);
        display: flex;
        flex-direction: column;
        align-items: stretch;
        position: relative;
        z-index: 4;
      }

      .nhlcal-brand-button {
        height: 112px;
        border: 0;
        background: transparent;
        color: var(--text);
        display: grid;
        place-items: center;
        border-bottom: 1px solid var(--line);
        cursor: pointer;
      }

      .nhlcal-shield-icon {
        width: 30px;
        height: 34px;
        border: 2px solid rgba(223, 245, 250, 0.52);
        display: grid;
        place-items: center;
        color: rgba(223, 245, 250, 0.75);
        clip-path: polygon(50% 0, 92% 16%, 92% 72%, 50% 100%, 8% 72%, 8% 16%);
        font-size: 15px;
      }

      .nhlcal-side-nav {
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding: 18px 0;
      }

      .nhlcal-side-button {
        width: 100%;
        min-height: 66px;
        border: 0;
        background: transparent;
        color: var(--muted);
        display: grid;
        place-items: center;
        gap: 4px;
        cursor: pointer;
        position: relative;
        transition:
          color 0.2s ease,
          background 0.2s ease,
          transform 0.2s ease;
      }

      .nhlcal-side-button:hover {
        color: var(--text);
        background: rgba(255, 255, 255, 0.035);
      }

      /* Franchise command rail: the active department is registered on a hard
         broadcast rail and notched out of the rail plate. No glow. */
      .nhlcal-side-button.is-active {
        color: var(--cyan);
        background: linear-gradient(90deg, rgba(19, 216, 231, 0.16), rgba(19, 216, 231, 0.02));
        clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%);
      }

      .nhlcal-side-button.is-active::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 3px;
        border-radius: 0;
        background: var(--cyan);
        box-shadow: none;
      }

      .nhlcal-side-icon {
        font-size: 20px;
        line-height: 1;
      }

      /* Department callsign sits above the symbol like a control-room label. */
      .nhlcal-side-code {
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 0.12em;
        color: var(--muted-2);
      }

      .nhlcal-side-button.is-active .nhlcal-side-code {
        color: var(--cyan);
      }

      .nhlcal-side-label {
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.02em;
      }

      /* Pending count is a numbered plate, not a notification bubble. */
      .nhlcal-side-button em {
        position: absolute;
        right: 12px;
        top: 12px;
        min-width: 18px;
        height: 16px;
        padding: 0 3px;
        border-radius: var(--radius-ops, 2px);
        background: var(--cyan);
        color: #021016;
        font-size: 11px;
        display: grid;
        place-items: center;
        font-style: normal;
        font-weight: 900;
        font-variant-numeric: tabular-nums;
      }

      .nhlcal-settings-button {
        margin-top: auto;
        height: 88px;
        border: 0;
        border-top: 1px solid var(--line);
        background: transparent;
        color: var(--muted);
        display: grid;
        place-items: center;
        gap: 4px;
        cursor: pointer;
      }

      .nhlcal-settings-button span {
        font-size: 22px;
      }

      .nhlcal-settings-button small {
        font-size: 11px;
        font-weight: 800;
      }

      .nhlcal-main {
        min-width: 0;
        height: 100dvh;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        padding: 10px 14px 10px;
      }

      .nhlcal-main::-webkit-scrollbar {
        width: 10px;
      }

      .nhlcal-main::-webkit-scrollbar-track {
        background: rgba(4, 16, 26, 0.72);
        border-radius: 999px;
      }

      .nhlcal-main::-webkit-scrollbar-thumb {
        background: rgba(110, 173, 191, 0.25);
        border-radius: 999px;
        border: 2px solid rgba(4, 16, 26, 0.72);
      }

      .nhlcal-main::-webkit-scrollbar-thumb:hover {
        background: rgba(19, 216, 231, 0.38);
      }

      .nhlcal-scroll-surface {
        scrollbar-width: thin;
        scrollbar-color: rgba(110, 173, 191, 0.35) rgba(4, 16, 26, 0.72);
      }

      .nhlcal-scroll-surface::-webkit-scrollbar {
        width: 8px;
        height: 8px;
      }

      .nhlcal-scroll-surface::-webkit-scrollbar-track {
        background: rgba(4, 16, 26, 0.72);
        border-radius: 999px;
      }

      .nhlcal-scroll-surface::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, rgba(19, 216, 231, 0.34), rgba(110, 173, 191, 0.28));
        border-radius: 999px;
        border: 2px solid rgba(4, 16, 26, 0.72);
      }

      .nhlcal-scroll-surface::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, rgba(19, 216, 231, 0.52), rgba(110, 173, 191, 0.42));
      }

      .nhlcal-topbar {
        min-height: 56px;
        flex: 0 0 auto;
        display: grid;
        grid-template-columns: minmax(180px, 0.9fr) minmax(240px, 1fr) minmax(260px, 0.95fr);
        align-items: center;
        gap: 10px;
      }

      .nhlcal-team-identity {
        display: flex;
        align-items: center;
        gap: 12px;
        min-width: 0;
      }

      .nhlcal-team-city {
        margin: 0 0 1px;
        color: rgba(233, 247, 251, 0.78);
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      .nhlcal-team-identity h1 {
  margin: 0;
  line-height: 0.95;
  text-transform: uppercase;
  color: var(--text);
  text-shadow: 0 0 24px rgba(19, 216, 231, 0.12);
}

      .nhlcal-month-control {
        text-align: center;
        min-width: 0;
      }

      .nhlcal-month-control p,
      .nhlcal-season-phase {
        margin: 0 0 4px;
        color: var(--cyan);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: var(--type-phase-label-size, 0.68rem);
        font-weight: 900;
      }

      .nhlcal-month-title-block {
        display: grid;
        justify-items: center;
        gap: 2px;
        min-width: 0;
      }

      .nhlcal-month-year {
        color: var(--muted);
        font-size: 0.6875rem;
        font-weight: 900;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        line-height: 1;
      }

      .nhlcal-month-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        flex-wrap: nowrap;
      }

      .nhlcal-month-row h2 {
        margin: 0;
        font-size: clamp(22px, 2.2vw, 32px);
        line-height: 0.95;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        white-space: nowrap;
        font-family: var(--font-broadcast-display, inherit);
      }

      .nhlcal-month-row h2::first-letter {
        color: var(--cyan);
      }

      /* Month stepping uses squared broadcast keys, not circular bubbles. */
      .nhlcal-month-row button {
        width: 32px;
        height: 32px;
        border-radius: var(--radius-hud, 4px);
        border: 1px solid var(--line);
        background: rgba(12, 31, 47, 0.72);
        color: var(--text);
        font-size: 24px;
        line-height: 1;
        cursor: pointer;
        transition:
          border-color 0.2s ease,
          background 0.2s ease,
          transform 0.2s ease;
      }

      .nhlcal-month-row button:hover {
        border-color: var(--line-strong);
        background: rgba(19, 216, 231, 0.12);
        transform: translateY(-1px);
      }

      .nhlcal-action-cluster {
        justify-self: end;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 8px;
        min-width: 0;
      }

      .nhlcal-lineup-block {
        justify-self: end;
        max-width: 420px;
        margin: 0;
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid rgba(232, 92, 92, 0.45);
        background: rgba(70, 16, 20, 0.72);
        color: #ffd4d4;
        font-size: 12px;
        font-weight: 700;
        line-height: 1.35;
        text-align: right;
      }

      .nhlcal-action-primary,
      .nhlcal-action-secondary {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 8px;
        flex-wrap: wrap;
      }

      /* Broadcast control: returns the schedule to the live date. */
      /* Scoped to the month row so the arrow-button font-size does not win on
         specificity and blow the label past the chip. */
      .nhlcal-month-row button.nhlcal-today-chip {
  height: 32px;
  width: auto;
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  overflow: visible;
  border: 1px solid rgba(233, 168, 60, 0.35);
  border-radius: var(--radius-ops, 2px);
  background: rgba(233, 168, 60, 0.08);
  color: #ffd88d;
  padding: 0 14px 0 12px;
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  cursor: pointer;
  align-self: center;
}

      .nhlcal-month-row button.nhlcal-today-chip:hover {
        border-color: rgba(233, 168, 60, 0.55);
        background: rgba(233, 168, 60, 0.14);
      }

      .nhlcal-date-chip.compact {
        min-width: 0;
        height: auto;
        border-left: 0;
        padding-left: 0;
      }

      .nhlcal-advance-error-banner {
        flex: 0 0 auto;
        margin-top: 6px;
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid rgba(255, 96, 109, 0.28);
        background: rgba(255, 96, 109, 0.08);
        color: #ffc4c9;
        font-size: 11px;
        font-weight: 700;
      }

      .nhlcal-advance-error-banner.is-blocked {
        border-color: rgba(233, 168, 60, 0.32);
        background: rgba(233, 168, 60, 0.1);
        color: #ffd88d;
      }

      .nhlcal-schedule-alert {
        flex: 0 0 auto;
        margin-top: 8px;
        padding: 10px 12px;
        border-radius: 8px;
        border: 1px solid rgba(255, 96, 109, 0.28);
        background: rgba(255, 96, 109, 0.08);
        color: #ffc4c9;
        font-size: 12px;
        font-weight: 700;
      }

      .nhlcal-menu-toggle {
        width: 46px;
        height: 46px;
        border-radius: 10px;
        border: 1px solid var(--line);
        background: rgba(12, 31, 47, 0.72);
        display: grid;
        place-items: center;
        gap: 4px;
        padding: 12px;
        cursor: pointer;
      }

      .nhlcal-menu-toggle span {
        display: block;
        width: 19px;
        height: 2px;
        background: var(--text);
        border-radius: 999px;
      }

      .nhlcal-quick-link {
        border: 1px solid var(--line);
        border-radius: 4px;
        background: rgba(12, 31, 47, 0.72);
        color: var(--text);
        padding: 9px 14px;
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        cursor: pointer;
        transition:
          border-color 0.2s ease,
          transform 0.2s ease,
          background 0.2s ease;
      }

      .nhlcal-quick-link:hover {
        border-color: var(--line-strong);
        background: rgba(19, 216, 231, 0.12);
        transform: translateY(-1px);
      }

      .nhlcal-online-chip {
        display: grid;
        gap: 3px;
        padding-right: 6px;
      }

      .nhlcal-online-chip strong {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.14em;
      }

      .nhlcal-online-chip span {
        color: #56dc75;
        font-size: 11px;
        text-transform: uppercase;
        font-weight: 900;
        letter-spacing: 0.1em;
      }

      .nhlcal-date-chip {
        min-width: 158px;
        height: 58px;
        border-left: 1px solid var(--line);
        padding-left: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
      }

      .nhlcal-date-icon {
        width: 42px;
        height: 42px;
        display: grid;
        place-items: center;
        border-radius: 12px;
        background: rgba(136, 180, 255, 0.12);
        border: 1px solid rgba(136, 180, 255, 0.14);
        color: #b8ceff;
      }

      .nhlcal-date-chip strong,
      .nhlcal-date-chip span {
        display: block;
      }

      .nhlcal-date-chip strong {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-date-chip span {
        margin-top: 3px;
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
      }

      .nhlcal-wjc-hub-button {
        min-height: 40px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border-radius: var(--radius-ops, 2px);
        border: 1px solid rgba(0, 216, 223, 0.28);
        background: rgba(7, 22, 35, 0.88);
        color: #dffcff;
        cursor: pointer;
        padding: 0 12px;
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        transition:
          border-color 150ms ease,
          background 150ms ease;
      }

      /* Tournament code plate replaces the decorative globe symbol. */
      .nhlcal-wjc-hub-button > span {
        padding: 1px 4px;
        border: 1px solid rgba(0, 216, 223, 0.38);
        border-radius: var(--radius-ops, 2px);
        font-size: 0.6875rem;
        letter-spacing: 0.08em;
        color: #7fe6ef;
      }

      .nhlcal-wjc-hub-button:hover {
        border-color: rgba(232, 165, 54, 0.38);
        background: rgba(19, 216, 231, 0.08);
      }

      .nhlcal-wjc-hub-button.is-modal {
        width: 100%;
        justify-content: center;
        min-height: 42px;
        margin-top: 12px;
      }

      .nhlcal-event-modal-actions {
        margin-top: 4px;
      }

      .nhlcal-wjc-menu-host {
        position: fixed;
        inset: 0;
        z-index: 12050;
        background: #08090c;
        overflow: hidden;
        display: flex;
        flex-direction: column;
      }

      .nhlcal-wjc-menu-host .wjc-event-shell,
      .nhlcal-wjc-menu-host .wjc-stage-root,
      .nhlcal-wjc-menu-host .wjc-page-root {
        width: 100%;
        height: 100%;
        min-height: 0;
        max-height: 100%;
        flex: 1 1 auto;
      }

      .nhlcal-wjc-menu-host .wjc-event-shell {
        display: flex;
        flex-direction: column;
        min-height: 0;
        overflow: hidden;
      }

      /* Broadcast action: advancing league time. Rink-cut, flat deadline gold,
         and a 1px press instead of a lift. */
      .nhlcal-advance-button {
        height: 40px;
        min-width: 132px;
        border: 0;
        border-radius: 0;
        clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%);
        background: #e9a83c;
        color: #1b1002;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 11px;
        font-weight: 1000;
        cursor: pointer;
        box-shadow: none;
        transition:
          background 0.15s ease,
          transform 0.11s ease;
      }

      .nhlcal-advance-button:hover:not(:disabled) {
        background: #f4c66e;
      }

      .nhlcal-advance-button:active:not(:disabled) {
        transform: translateY(1px);
      }

      .nhlcal-advance-button-secondary {
        min-width: 92px;
        background: rgba(7, 22, 35, 0.92);
        color: #dffcff;
        border: 1px solid rgba(19, 216, 231, 0.32);
        box-shadow: none;
      }

      

      .nhlcal-advance-button-secondary:hover:not(:disabled) {
  background: rgba(19, 216, 231, 0.12);
        border-color: rgba(19, 216, 231, 0.52);
        filter: brightness(1.06);
      }

      .nhlcal-advance-button:disabled {
        cursor: not-allowed;
        opacity: 0.72;
        transform: none;
      }

      .nhlcal-advance-button.is-busy {
        filter: saturate(0.75) brightness(0.9);
      }

      .nhlcal-advance-alert {
        margin: 0 0 18px;
        border: 1px solid var(--line2);
        background: rgba(8, 23, 35, 0.92);
        border-radius: 6px;
        padding: 16px 18px;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 18px;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.24);
      }

      .nhlcal-advance-alert strong {
        display: block;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        font-size: 12px;
        margin-bottom: 6px;
      }

      .nhlcal-advance-alert p {
        margin: 0;
        color: var(--muted);
        line-height: 1.45;
      }

      .nhlcal-advance-alert ul {
        margin: 10px 0 0;
        padding-left: 18px;
        color: var(--text);
      }

      .nhlcal-advance-alert li {
        margin: 4px 0;
        color: rgba(232, 244, 251, 0.88);
      }

      .nhlcal-advance-alert button {
        border: 1px solid rgba(255, 255, 255, 0.16);
        background: rgba(255, 255, 255, 0.08);
        color: var(--text);
        border-radius: 12px;
        padding: 10px 14px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        cursor: pointer;
        white-space: nowrap;
      }

      .nhlcal-advance-alert.is-blocked {
        border-color: rgba(244, 189, 82, 0.42);
        background:
          radial-gradient(circle at 0% 0%, rgba(244, 189, 82, 0.16), transparent 38%),
          rgba(8, 23, 35, 0.94);
      }

      .nhlcal-advance-alert.is-blocked strong {
        color: #f4bd52;
      }

      .nhlcal-advance-alert.is-error {
        border-color: rgba(255, 100, 100, 0.42);
        background:
          radial-gradient(circle at 0% 0%, rgba(255, 100, 100, 0.16), transparent 38%),
          rgba(8, 23, 35, 0.94);
      }

      .nhlcal-advance-alert.is-error strong {
        color: #ff6464;
      }

      .nhlcal-advance-button:hover {
        transform: translateY(-1px);
        filter: brightness(1.04);
      }

      .nhlcal-advance-button span {
        margin-right: 10px;
      }

      .nhlcal-stat-strip {
        flex: 0 0 auto;
        margin-top: 6px;
        display: flex;
        flex-wrap: wrap;
        gap: 0;
        border-top: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        background: rgba(6, 21, 34, 0.72);
        border-radius: var(--radius-control, 6px);
        overflow: hidden;
        box-shadow: none;
      }

      .nhlcal-stat-pill {
        flex: 1 1 0;
        min-width: 88px;
        min-height: 44px;
        padding: 6px 10px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-right: 1px solid rgba(156, 218, 236, 0.08);
        background: transparent;
        border-radius: 0;
      }

      .nhlcal-stat-pill:last-child {
        border-right: 0;
      }

      .nhlcal-stat-icon {
        width: 28px;
        height: 28px;
        flex: 0 0 auto;
        display: grid;
        place-items: center;
        border-radius: 8px;
        background: rgba(148, 185, 205, 0.12);
        border: 1px solid rgba(148, 185, 205, 0.12);
        color: rgba(233, 247, 251, 0.8);
        font-size: 13px;
      }

      .nhlcal-stat-pill span,
      .nhlcal-stat-pill small {
        display: block;
      }

      .nhlcal-stat-pill span {
        color: var(--muted);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 1000;
      }

      .nhlcal-stat-pill strong {
        display: block;
        margin-top: 2px;
        color: var(--text);
        font-size: 16px;
        line-height: 1;
        font-weight: 1000;
        letter-spacing: 0.02em;
        min-height: 16px;
      }

      .nhlcal-stat-pill strong.is-missing {
        color: rgba(233, 247, 251, 0.45);
      }

      .nhlcal-stat-pill.is-skeleton .nhlcal-stat-icon,
      .nhlcal-skeleton-bar {
        background: linear-gradient(90deg, rgba(255,255,255,0.05), rgba(255,255,255,0.12), rgba(255,255,255,0.05));
        background-size: 200% 100%;
        animation: nhlcal-skeleton 1.2s ease infinite;
        border-radius: 8px;
      }

      .nhlcal-stat-pill.is-skeleton .nhlcal-stat-icon {
        width: 34px;
        height: 34px;
      }

      .nhlcal-skeleton-bar {
        display: block;
        height: 18px;
        margin-top: 4px;
      }

      .nhlcal-skeleton-bar.short {
        height: 10px;
        width: 70%;
        margin-top: 8px;
      }

      @keyframes nhlcal-skeleton {
        0% { background-position: 100% 0; }
        100% { background-position: -100% 0; }
      }

      .nhlcal-stat-pill small {
        margin-top: 4px;
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlcal-stat-pill.tone-cyan .nhlcal-stat-icon {
        color: var(--cyan);
        background: var(--cyan-soft);
      }

      .nhlcal-stat-pill.tone-green .nhlcal-stat-icon {
        color: var(--green);
        background: var(--green-soft);
      }

      .nhlcal-stat-pill.tone-danger .nhlcal-stat-icon {
        color: var(--red);
        background: var(--red-soft);
      }

      .nhlcal-stat-pill.tone-gold .nhlcal-stat-icon {
        color: var(--gold);
        background: var(--gold-soft);
      }

      .nhlcal-stat-pill.tone-blue .nhlcal-stat-icon {
        color: var(--blue);
        background: var(--blue-soft);
      }

      .nhlcal-content-grid {
        margin-top: 6px;
        flex: 1 1 auto;
        min-height: 0;
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(272px, 318px);
        gap: 8px;
        overflow: hidden;
      }

      .nhlcal-calendar-panel {
        min-width: 0;
        min-height: 0;
        display: flex;
        flex-direction: column;
        border: 1px solid var(--line);
        border-radius: var(--radius-card, 8px);
        background: rgba(6, 21, 34, 0.82);
        overflow: hidden;
        box-shadow: var(--depth-registered, inset 0 1px 0 rgba(255, 255, 255, 0.04));
      }

      .nhlcal-calendar-toolbar {
        flex: 0 0 auto;
        padding: 8px 10px;
        border-bottom: 1px solid var(--line);
        background: rgba(5, 17, 27, 0.72);
      }

      .nhlcal-calendar-toolbar-group {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }

      .nhlcal-calendar-toolbar-group button {
        height: 30px;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: rgba(14, 35, 50, 0.9);
        color: rgba(233, 247, 251, 0.82);
        padding: 0 10px;
        font-size: 11px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        cursor: pointer;
      }

      .nhlcal-calendar-toolbar-group button.is-active,
      .nhlcal-calendar-toolbar-group button:hover {
        border-color: var(--line-strong);
        color: var(--text);
        background: rgba(19, 216, 231, 0.11);
      }

      .nhlcal-week-header {
        display: grid;
        grid-template-columns: repeat(7, 1fr);
        height: 34px;
        border-bottom: 1px solid var(--line);
        background: rgba(5, 17, 27, 0.62);
        border-radius: 12px 12px 0 0;
        overflow: hidden;
      }

      .nhlcal-week-header div {
        display: grid;
        place-items: center;
        color: rgba(233, 247, 251, 0.88);
        text-transform: uppercase;
        font-size: 12px;
        font-weight: 900;
        letter-spacing: 0.12em;
        border-right: 1px solid rgba(156, 218, 236, 0.08);
      }

      .nhlcal-week-header div:last-child {
        border-right: 0;
      }

      .nhlcal-month-grid {
        display: grid;
        grid-template-columns: repeat(7, 1fr);
        flex: 1 1 auto;
        min-height: 0;
        overflow: auto;
      }

      .nhlcal-month-grid.nhlcal-scroll-surface {
        scrollbar-gutter: stable;
      }

      .nhlcal-day-cell {
  position: relative;
  border: 0;
  border-right: 1px solid rgba(156, 218, 236, 0.11);
  border-bottom: 1px solid rgba(156, 218, 236, 0.11);
  background: linear-gradient(180deg, rgba(11, 31, 45, 0.72), rgba(7, 22, 34, 0.72));
  color: var(--text);
  text-align: left;
  padding: 10px 11px;
  overflow: visible;
  cursor: pointer;
  transition: background 0.2s ease,
          box-shadow 0.2s ease,
          border-color 0.2s ease,
          transform 0.2s ease;
}

      .nhlcal-month-grid.is-dense .nhlcal-day-cell {
  padding: 7px;
}

      /* Selection is a hard broadcast frame, not a glow. */
      .nhlcal-day-cell.is-selected {
        box-shadow: inset 0 0 0 2px rgba(19, 216, 231, 0.95);
        z-index: 5;
      }

      .nhlcal-day-cell.is-today:not(.is-selected) {
        background:
          linear-gradient(180deg, rgba(233, 168, 60, 0.1), rgba(7, 22, 34, 0.82));
      }

      /* Department signature: the timeline notch. Today is registered on the
         schedule rail the way a broadcast marks the live position. */
      .nhlcal-day-cell.is-today:not(.is-selected)::before {
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        top: 0;
        height: 3px;
        background: var(--gold);
        pointer-events: none;
        z-index: 3;
      }

      .nhlcal-day-cell.is-today:not(.is-selected)::after {
        content: "";
        position: absolute;
        left: 50%;
        top: 3px;
        transform: translateX(-50%);
        width: 12px;
        height: 6px;
        background: var(--gold);
        clip-path: polygon(0 0, 100% 0, 50% 100%);
        pointer-events: none;
        z-index: 3;
      }

      .nhlcal-day-cell.is-today .nhlcal-day-number {
        color: var(--gold);
      }

      

      .nhlcal-empty-day-line.is-muted {
        opacity: 0.35;
      }

      .nhlcal-day-cell:nth-child(7n) {
        border-right: 0;
      }

      .nhlcal-day-cell:nth-last-child(-n + 7) {
        border-bottom: 0;
      }

      .nhlcal-day-cell:hover {
        background:
          linear-gradient(180deg, rgba(16, 44, 62, 0.84), rgba(7, 25, 38, 0.82));
        z-index: 4;
      }

      .nhlcal-day-cell.is-muted {
        color: rgba(233, 247, 251, 0.38);
        background:
          linear-gradient(180deg, rgba(8, 18, 28, 0.72), rgba(5, 12, 19, 0.72));
      }

      .nhlcal-day-cell.has-team-game {
        background:
          radial-gradient(circle at 50% 50%, rgba(19, 216, 231, 0.11), transparent 72%),
          linear-gradient(180deg, rgba(9, 37, 51, 0.86), rgba(5, 22, 33, 0.86));
      }

      .nhlcal-day-cell.has-special-events {
        background:
          radial-gradient(circle at 12% 6%, rgba(233, 168, 60, 0.12), transparent 34%),
          linear-gradient(180deg, rgba(13, 34, 48, 0.82), rgba(6, 22, 34, 0.82));
      }

      .nhlcal-day-cell.has-critical-event {
        background:
          radial-gradient(circle at 12% 6%, rgba(255, 96, 109, 0.18), transparent 36%),
          radial-gradient(circle at 88% 12%, rgba(233, 168, 60, 0.13), transparent 34%),
          linear-gradient(180deg, rgba(38, 18, 29, 0.86), rgba(8, 22, 34, 0.86));
      }

      .nhlcal-day-cell.has-high-event:not(.has-critical-event) {
        background:
          radial-gradient(circle at 12% 6%, rgba(233, 168, 60, 0.17), transparent 36%),
          linear-gradient(180deg, rgba(31, 35, 37, 0.86), rgba(7, 22, 34, 0.86));
      }

      .nhlcal-day-number-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 22px;
        position: relative;
        z-index: 2;
      }

      .nhlcal-day-number {
  color: rgba(233, 247, 251, 0.9);
}

      .nhlcal-day-marker-row {
        display: inline-flex;
        align-items: center;
        justify-content: flex-end;
        gap: 5px;
      }

      /* Event counts are deadline-gold count plates. */
      .nhlcal-event-corner-badge {
        min-width: 20px;
        height: 18px;
        padding: 0 5px;
        display: inline-grid;
        place-items: center;
        border-radius: var(--radius-ops, 2px);
        background: rgba(233, 168, 60, 0.95);
        border: 1px solid rgba(255, 214, 135, 0.45);
        color: #1b1002;
        font-size: 11px;
        font-weight: 1000;
        font-variant-numeric: tabular-nums;
        box-shadow: none;
      }
.nhlcal-event-corner-badge img {
  width: 18px;
  height: 18px;
  object-fit: contain;
  display: block;
  filter: drop-shadow(0 0 5px rgba(255, 255, 255, 0.18));
}

.nhlcal-special-event-tile.has-logo {
  grid-template-columns: 34px minmax(0, 1fr);
}

.nhlcal-special-event-icon img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  filter: drop-shadow(0 0 6px rgba(255, 255, 255, 0.16));
}

.nhlcal-special-event-tile.has-logo .nhlcal-special-event-icon {
  width: 34px;
  height: 30px;
  padding: 3px;
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(255, 255, 255, 0.13);
}

.nhlcal-event-modal-icon.has-logo {
  padding: 8px;
  background: rgba(255, 255, 255, 0.08);
}

.nhlcal-event-modal-icon.has-logo img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  filter: drop-shadow(0 0 12px rgba(255, 255, 255, 0.18));
}

      .nhlcal-corner-cut {
        width: 0;
        height: 0;
        border-top: 13px solid rgba(19, 216, 231, 0.62);
        border-left: 13px solid transparent;
        position: absolute;
        top: -10px;
        right: -11px;
        filter: drop-shadow(0 0 8px rgba(19, 216, 231, 0.28));
        opacity: 0.85;
      }

      .nhlcal-day-content {
        margin-top: 8px;
        display: grid;
        gap: 8px;
        width: 100%;
      }

      .nhlcal-day-special-events,
      .nhlcal-day-games {
        display: grid;
        gap: 7px;
        width: 100%;
      }

      .nhlcal-empty-day-line {
  text-align: center;
        color: rgba(128, 150, 168, 0.42);
        font-size: 11px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding-top: 4px;
      }

      .nhlcal-special-event-tile {
  width: 100%;
  display: grid;
  align-items: center;
  text-align: left;
  border: 1px solid rgba(233, 168, 60, 0.24);
  cursor: pointer;
  transition: border-color 0.18s ease,
          transform 0.18s ease,
          background 0.18s ease;
}

      .nhlcal-special-event-tile:hover {
        transform: translateY(-1px);
        border-color: rgba(255, 214, 135, 0.48);
        background:
          radial-gradient(circle at 0% 0%, rgba(233, 168, 60, 0.22), transparent 58%),
          linear-gradient(180deg, rgba(64, 43, 20, 0.88), rgba(23, 25, 29, 0.82));
      }

      .nhlcal-special-event-icon {
        width: 26px;
        height: 26px;
        display: grid;
        place-items: center;
        border-radius: 8px;
        background: rgba(233, 168, 60, 0.18);
        border: 1px solid rgba(233, 168, 60, 0.2);
        color: #ffd88d;
        font-size: 13px;
        font-weight: 1000;
      }

      .nhlcal-special-event-copy {
        min-width: 0;
        display: grid;
        gap: 2px;
      }

      .nhlcal-special-event-copy strong {
  min-width: 0;
  color: rgba(255, 239, 211, 0.96);
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.01em;
  text-transform: none;
  overflow: hidden;
  display: -webkit-box;
  -webkit-box-orient: vertical;
  line-height: 1.2;
  word-break: break-word;
}

      .nhlcal-special-event-copy span {
  min-width: 0;
  color: rgba(232, 203, 160, 0.72);
  font-size: 11px;
  font-weight: 700;
  overflow: hidden;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 1;
  line-height: 1.2;
}

      .nhlcal-special-event-tile.priority-critical {
  border-color: rgba(255, 96, 109, 0.42);
}

      .nhlcal-special-event-tile.priority-critical .nhlcal-special-event-icon {
        background: rgba(255, 96, 109, 0.16);
        border-color: rgba(255, 96, 109, 0.3);
        color: #ffc4ca;
      }

      .nhlcal-special-event-tile.priority-high:not(.priority-critical) {
        border-color: rgba(233, 168, 60, 0.36);
      }

      .nhlcal-special-event-tile.tone-medical {
        border-color: rgba(255, 96, 109, 0.34);
        background:
          radial-gradient(circle at 0% 0%, rgba(255, 96, 109, 0.16), transparent 56%),
          linear-gradient(180deg, rgba(60, 22, 31, 0.78), rgba(18, 20, 25, 0.74));
      }

      .nhlcal-special-event-tile.tone-medical .nhlcal-special-event-icon {
        color: #ffcbd1;
        background: rgba(255, 96, 109, 0.14);
        border-color: rgba(255, 96, 109, 0.28);
      }

      .nhlcal-special-event-tile.tone-trade {
        border-color: rgba(19, 216, 231, 0.32);
        background:
          radial-gradient(circle at 0% 0%, rgba(19, 216, 231, 0.16), transparent 56%),
          linear-gradient(180deg, rgba(16, 48, 58, 0.78), rgba(18, 23, 28, 0.74));
      }

      .nhlcal-special-event-tile.tone-trade .nhlcal-special-event-icon {
        color: #baf9ff;
        background: rgba(19, 216, 231, 0.13);
        border-color: rgba(19, 216, 231, 0.28);
      }

      .nhlcal-special-event-tile.tone-draft {
        border-color: rgba(138, 180, 255, 0.35);
        background:
          radial-gradient(circle at 0% 0%, rgba(138, 180, 255, 0.18), transparent 56%),
          linear-gradient(180deg, rgba(42, 28, 66, 0.78), rgba(18, 20, 28, 0.74));
      }

      .nhlcal-special-event-tile.tone-draft .nhlcal-special-event-icon {
        color: #ead7ff;
        background: rgba(138, 180, 255, 0.14);
        border-color: rgba(138, 180, 255, 0.28);
      }

      .nhlcal-special-event-tile.tone-playoff {
        border-color: rgba(82, 223, 148, 0.34);
        background:
          radial-gradient(circle at 0% 0%, rgba(82, 223, 148, 0.16), transparent 56%),
          linear-gradient(180deg, rgba(20, 58, 40, 0.78), rgba(16, 23, 23, 0.74));
      }

      .nhlcal-special-event-tile.tone-playoff .nhlcal-special-event-icon {
        color: #caffdf;
        background: rgba(82, 223, 148, 0.13);
        border-color: rgba(82, 223, 148, 0.28);
      }

      .nhlcal-special-event-tile.tone-showcase,
      .nhlcal-special-event-tile.tone-star,
      .nhlcal-special-event-tile.tone-international {
        border-color: rgba(138, 180, 255, 0.34);
        background:
          radial-gradient(circle at 0% 0%, rgba(138, 180, 255, 0.17), transparent 56%),
          linear-gradient(180deg, rgba(25, 39, 68, 0.78), rgba(16, 21, 29, 0.74));
      }

      .nhlcal-special-event-tile.tone-showcase .nhlcal-special-event-icon,
      .nhlcal-special-event-tile.tone-star .nhlcal-special-event-icon,
      .nhlcal-special-event-tile.tone-international .nhlcal-special-event-icon {
        color: #d6e3ff;
        background: rgba(138, 180, 255, 0.13);
        border-color: rgba(138, 180, 255, 0.28);
      }

      .nhlcal-more-events {
        min-height: 24px;
        border: 1px dashed rgba(233, 168, 60, 0.3);
        border-radius: 8px;
        background: rgba(233, 168, 60, 0.06);
        color: rgba(255, 220, 160, 0.86);
        display: grid;
        place-items: center;
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        cursor: pointer;
      }

      .nhlcal-game-tile {
        display: grid;
        grid-template-columns: 4px auto minmax(0, 1fr) auto;
        align-items: center;
        gap: 8px;
        width: 100%;
        min-height: 58px;
        min-width: 0;
        position: relative;
        border-radius: 12px;
        padding: 8px 9px;
        cursor: pointer;
        text-align: left;
        overflow: hidden;
        transition:
          background 120ms ease,
          border-color 120ms ease,
          box-shadow 120ms ease;
        background:
          radial-gradient(circle at 0% 0%, rgba(19, 216, 231, 0.11), transparent 55%),
          linear-gradient(180deg, rgba(16, 44, 62, 0.82), rgba(7, 24, 36, 0.84));
        border: 1px solid rgba(156, 218, 236, 0.16);
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.05),
          0 2px 10px rgba(0, 0, 0, 0.18);
      }

      .nhlcal-game-tile:hover {
        border-color: rgba(19, 216, 231, 0.34);
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.08),
          0 6px 14px rgba(0, 0, 0, 0.25);
      }

      .nhlcal-game-tile.is-home .nhlcal-game-tile-accent {
        background: linear-gradient(180deg, rgba(21, 238, 255, 0.92), rgba(38, 160, 214, 0.72));
      }

      .nhlcal-game-tile.is-away .nhlcal-game-tile-accent {
        background: linear-gradient(180deg, rgba(129, 174, 214, 0.9), rgba(88, 122, 166, 0.68));
      }

      .nhlcal-game-tile.is-final {
        background:
          radial-gradient(circle at 0% 0%, rgba(76, 130, 158, 0.08), transparent 60%),
          linear-gradient(180deg, rgba(11, 31, 46, 0.84), rgba(5, 19, 29, 0.88));
      }

      .nhlcal-game-tile.is-final.result-win {
        background:
          radial-gradient(circle at 0% 0%, rgba(68, 204, 128, 0.16), transparent 58%),
          linear-gradient(180deg, rgba(17, 64, 44, 0.7), rgba(8, 35, 24, 0.72));
        border-color: rgba(95, 226, 155, 0.34);
      }

      .nhlcal-game-tile.is-final.result-loss {
        background:
          radial-gradient(circle at 0% 0%, rgba(238, 86, 86, 0.16), transparent 58%),
          linear-gradient(180deg, rgba(72, 24, 28, 0.7), rgba(36, 11, 14, 0.72));
        border-color: rgba(241, 120, 120, 0.34);
      }

      .nhlcal-game-tile.is-final.result-otl {
        background:
          radial-gradient(circle at 0% 0%, rgba(233, 168, 60, 0.15), transparent 58%),
          linear-gradient(180deg, rgba(64, 45, 20, 0.68), rgba(35, 24, 12, 0.72));
        border-color: rgba(233, 168, 60, 0.3);
      }

      .nhlcal-game-tile.is-upcoming {
        border-color: rgba(19, 216, 231, 0.2);
      }

      .nhlcal-game-tile.is-expanded {
        border-color: rgba(19, 216, 231, 0.52);
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.09),
          0 0 0 1px rgba(19, 216, 231, 0.18),
          0 10px 24px rgba(5, 20, 30, 0.35);
      }

      .nhlcal-game-tile-accent {
        width: 4px;
        height: 100%;
        min-height: 42px;
        border-radius: 0;
        box-shadow: 0 0 8px rgba(19, 216, 231, 0.45);
      }

      .nhlcal-game-tile-logo {
        display: inline-flex;
        align-items: center;
        justify-content: center;
      }

      .nhlcal-game-tile-main {
        min-width: 0;
        display: grid;
        gap: 3px;
      }

      .nhlcal-game-match-line {
        display: flex;
        align-items: center;
        gap: 5px;
        min-width: 0;
      }

      .nhlcal-game-match-line strong {
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        color: rgba(233, 247, 251, 0.94);
        font-size: 12px;
        font-weight: 1000;
        letter-spacing: 0.04em;
      }

      .nhlcal-game-relation {
        flex: 0 0 auto;
        min-width: 26px;
        height: 18px;
        padding: 0 5px;
        border-radius: var(--radius-ops, 2px);
        display: inline-grid;
        place-items: center;
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(233, 247, 251, 0.88);
        background: rgba(109, 153, 178, 0.26);
        border: 1px solid rgba(156, 218, 236, 0.2);
      }

      .nhlcal-game-relation.home {
        color: rgba(152, 246, 255, 0.95);
        background: rgba(19, 216, 231, 0.18);
        border-color: rgba(19, 216, 231, 0.38);
      }

      .nhlcal-game-relation.away {
        color: rgba(191, 214, 241, 0.94);
        background: rgba(105, 137, 175, 0.2);
        border-color: rgba(117, 156, 201, 0.35);
      }

      .nhlcal-game-meta-line {
        display: flex;
        align-items: center;
        gap: 5px;
        min-width: 0;
        color: var(--muted);
        font-size: 11px;
        font-weight: 850;
      }

      .nhlcal-game-meta-line span,
      .nhlcal-game-meta-line em {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .nhlcal-game-meta-line em {
        color: rgba(174, 206, 223, 0.68);
        font-style: normal;
      }

      .nhlcal-game-tile-side {
        display: grid;
        justify-items: end;
        gap: 3px;
      }

      /* Month cells are ~120px wide: keep the matchup text readable instead of
         letting the crest + status pill squeeze it to zero width. */
      .nhlcal-month-grid .nhlcal-game-tile {
        gap: 6px;
        padding-left: 7px;
        padding-right: 7px;
      }

      

      .nhlcal-month-grid .nhlcal-game-match-line strong {
  letter-spacing: 0;
}

      /* The whole tile toggles; the chevron only eats matchup width here. */
      .nhlcal-month-grid .nhlcal-game-chevron {
        display: none;
      }

      .nhlcal-month-grid .nhlcal-game-match-line {
        gap: 3px;
      }

      .nhlcal-month-grid .nhlcal-game-relation {
        min-width: 0;
        padding: 0 3px;
        letter-spacing: 0.02em;
      }

      .nhlcal-game-score-mini {
        min-width: 38px;
        height: 22px;
        padding: 0 6px;
        border-radius: 7px;
        background: rgba(0, 0, 0, 0.24);
        border: 1px solid rgba(255, 255, 255, 0.08);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 3px;
        color: rgba(233, 247, 251, 0.9);
        font-size: 12px;
        font-weight: 1000;
      }

      .nhlcal-game-score-mini em {
        color: rgba(176, 205, 219, 0.78);
        font-style: normal;
      }

      .nhlcal-game-status-pill {
        min-width: 42px;
        height: 22px;
        padding: 0 6px;
        border-radius: 7px;
        border: 1px solid rgba(19, 216, 231, 0.24);
        background: rgba(19, 216, 231, 0.08);
        color: rgba(180, 238, 245, 0.95);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 1000;
        letter-spacing: 0.06em;
        text-transform: uppercase;
      }

      .nhlcal-game-chevron {
        color: rgba(170, 206, 223, 0.78);
        font-size: 11px;
        line-height: 1;
        font-weight: 1000;
      }

      .nhlcal-game-expand-details {
        grid-column: 1 / -1;
        margin-top: 6px;
        padding: 8px;
        border-radius: 9px;
        background: rgba(2, 10, 16, 0.32);
        border-top: 1px solid rgba(19, 216, 231, 0.18);
        display: grid;
        gap: 8px;
      }

      .nhlcal-game-expand-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
      }

      .nhlcal-game-expand-header strong {
        color: rgba(233, 247, 251, 0.95);
        font-size: 11px;
        letter-spacing: 0.09em;
      }

      .nhlcal-game-expand-header span {
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
      }

      .nhlcal-game-expand-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
      }

      .nhlcal-game-expand-row > div {
        min-width: 0;
        display: grid;
        grid-template-columns: auto 1fr;
        grid-template-rows: auto auto;
        align-items: center;
        column-gap: 6px;
        row-gap: 1px;
      }

      .nhlcal-game-expand-row > div span {
        color: rgba(233, 247, 251, 0.92);
        font-size: 11px;
        font-weight: 900;
      }

      .nhlcal-game-expand-row > div small {
        color: var(--muted);
        font-size: 11px;
        grid-column: 2;
      }

      .nhlcal-month-grid.is-dense .nhlcal-game-tile {
        min-height: 46px;
        padding: 6px 7px;
        grid-template-columns: 3px auto minmax(0, 1fr) auto;
      }

      .nhlcal-month-grid.is-dense .nhlcal-game-meta-line em {
        display: none;
      }

      .nhlcal-month-grid.is-dense .nhlcal-game-match-line strong {
        font-size: 11px;
      }

      .nhlcal-month-grid.is-dense .nhlcal-special-event-tile {
        min-height: 34px;
        grid-template-columns: 24px minmax(0, 1fr);
        padding: 5px 6px;
      }

      .nhlcal-month-grid.is-dense .nhlcal-special-event-copy span {
        display: none;
      }

      .nhlcal-more-games {
        color: var(--muted);
        font-size: 11px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-day-hover-card {
        position: absolute;
        left: 10px;
        bottom: calc(100% - 10px);
        width: 245px;
        border: 1px solid var(--line-2);
        border-radius: 12px;
        background: rgba(4, 15, 24, 0.98);
        padding: 12px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.42);
        z-index: 20;
        pointer-events: none;
      }

      .nhlcal-day-hover-card strong,
      .nhlcal-day-hover-card span {
        display: block;
      }

      .nhlcal-day-hover-card strong {
        font-size: 12px;
        color: var(--text);
      }

      .nhlcal-day-hover-card span {
        margin-top: 5px;
        color: var(--muted);
        font-size: 11px;
        line-height: 1.35;
      }

      .nhlcal-calendar-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  border-top: 1px solid var(--line);
  background: rgba(5, 16, 25, 0.72);
  border-radius: 0 0 12px 12px;
}

      .nhlcal-week-header {
        flex: 0 0 auto;
      }

      .nhlcal-legend {
        display: flex;
        align-items: center;
        gap: 18px;
        flex-wrap: wrap;
      }

      .nhlcal-legend span {
        display: flex;
        align-items: center;
        gap: 8px;
        color: rgba(233, 247, 251, 0.74);
        font-size: 11px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-legend-filter {
        display: flex;
        align-items: center;
        gap: 8px;
        border: 0;
        background: transparent;
        color: rgba(233, 247, 251, 0.74);
        font-size: 11px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        cursor: pointer;
        padding: 0;
      }

      .nhlcal-legend-filter.is-active {
        color: var(--text);
      }

      .nhlcal-day-cell.is-legend-dimmed {
        opacity: 0.35;
      }

      .dot {
        width: 12px;
        height: 12px;
        border-radius: 999px;
        background: var(--muted-2);
      }

      .dot.home {
        background: var(--cyan);
      }

      .dot.away {
        background: #6b8090;
      }

      .dot.team-game {
        background: #0b707c;
      }

      .dot.win {
        background: #2fd67b;
      }

      .dot.loss {
        background: #ff6070;
      }

      .dot.otl {
        background: #e9a83c;
      }

      .dot.special {
        background: var(--gold);
      }

      .dot.critical {
        background: var(--red);
      }

      .nhlcal-calendar-actions {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        justify-content: flex-end;
      }

      .nhlcal-calendar-actions button {
        height: 34px;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: rgba(14, 35, 50, 0.9);
        color: rgba(233, 247, 251, 0.82);
        padding: 0 12px;
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        cursor: pointer;
      }

      .nhlcal-calendar-actions button:hover,
      .nhlcal-calendar-actions button.is-active {
        border-color: var(--line-strong);
        color: var(--text);
        background: rgba(19, 216, 231, 0.11);
      }
      .nhlcal-calendar-footer .nhlcal-calendar-actions {
        display: none;
      }

      .nhlcal-calendar-footer {
        flex: 0 0 auto;
        min-height: 36px;
        padding: 6px 10px;
      }

              .nhlcal-right-rail {
        min-width: 0;
        min-height: 0;
        /* Scroll rather than crush the standings table to a 1px body. */
        overflow-x: hidden;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      .nhlcal-rail-preview-wrap {
        flex: 0 0 auto;
        min-height: 0;
        max-height: 32%;
        overflow: hidden;
        display: flex;
        flex-direction: column;
      }

      .nhlcal-rail-preview-wrap .nhlcal-preview-card {
        flex: 1 1 auto;
        min-height: 0;
        overflow-x: hidden;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
      }

      .nhlcal-rail-preview-wrap .nhlcal-preview-card.nhlcal-scroll-surface {
        scrollbar-gutter: stable;
      }

      .nhlcal-stretch-row {
  display: grid;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  border-bottom: 1px solid rgba(156, 218, 236, 0.1);
  font-size: 0.8125rem;
}

      .nhlcal-stretch-list {
        display: flex;
        flex-direction: column;
        gap: 0;
      }

      .nhlcal-selected-summary-strip span {
        padding: 4px 8px;
        border-radius: var(--radius-pill, 999px);
        border: 1px solid rgba(156, 218, 236, 0.12);
        background: rgba(255, 255, 255, 0.03);
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
      }

      .nhlcal-preview-mode-banner,
      .nhlcal-preview-missing {
        margin: 0 14px 10px;
        padding: 8px 10px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 700;
      }

      .nhlcal-preview-mode-banner {
        border: 1px solid rgba(136, 180, 255, 0.24);
        background: rgba(136, 180, 255, 0.08);
        color: #cfe0ff;
      }

      .nhlcal-preview-missing {
        border: 1px solid rgba(255, 96, 109, 0.24);
        background: rgba(255, 96, 109, 0.08);
        color: #ffc4c9;
      }

      .nhlcal-tab-row button.is-active {
  border-color: var(--line-strong);
  background: rgba(19, 216, 231, 0.14);
  box-shadow: inset 0 -2px 0 var(--cyan);
}

      .nhlcal-broadcast-strip,
      .nhlcal-card {
        border: 1px solid var(--line);
        border-radius: var(--radius-card, 8px);
        background: rgba(6, 21, 34, 0.78);
        box-shadow: none;
      }

      .nhlcal-broadcast-strip {
        display: flex;
        flex-direction: column;
        min-height: 0;
      }

      .nhlcal-selected-summary-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        padding: 0 12px 8px;
        border-bottom: 1px solid rgba(156, 218, 236, 0.1);
      }

      .nhlcal-card-header {
        min-height: 58px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 14px;
        padding: 16px 18px 10px;
      }

      .nhlcal-card-header.compact {
        padding-bottom: 10px;
      }

      .nhlcal-card-header p,
      .nhlcal-card-header h3 {
        margin: 0;
      }

      .nhlcal-card-header p {
        color: var(--cyan);
        font-size: 11px;
        font-weight: 1000;
        letter-spacing: 0.13em;
        text-transform: uppercase;
      }

      .nhlcal-card-header h3 {
        margin-top: 4px;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-card-header button,
      .nhlcal-mini-header button {
        border: 0;
        background: transparent;
        color: var(--muted);
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
        cursor: pointer;
      }

      .nhlcal-card-header button:hover,
      .nhlcal-mini-header button:hover {
        color: var(--cyan);
      }

      .nhlcal-header-pill {
        border: 1px solid var(--line);
        border-radius: var(--radius-ops, 2px);
        background: rgba(255, 255, 255, 0.035);
        color: var(--muted);
        padding: 4px 8px;
        letter-spacing: 0.08em;
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
        white-space: nowrap;
      }

      .nhlcal-preview-card {
        overflow: hidden;
      }

      .nhlcal-preview-card .nhlcal-card-header {
        min-height: 46px;
        padding: 10px 12px 6px;
        /* Narrow rail: let the count pill drop under the date instead of
           crushing the title column into the strip below. */
        flex-wrap: wrap;
        row-gap: 6px;
        flex-shrink: 0;
      }

      .nhlcal-preview-card .nhlcal-card-header > div {
        flex: 1 1 140px;
        min-width: 0;
      }

      .nhlcal-preview-card .nhlcal-card-header h3 {
        font-size: 12px;
      }

      .nhlcal-preview-card .nhlcal-header-pill {
        font-size: 11px;
      }

      .nhlcal-matchup-stage {
        min-height: 118px;
        display: grid;
        grid-template-columns: 1fr 0.72fr 1fr;
        align-items: center;
        gap: 6px;
        padding: 6px 12px 12px;
      }

      .nhlcal-matchup-team {
        display: grid;
        place-items: center;
        text-align: center;
        gap: 6px;
      }

      .nhlcal-matchup-team strong {
        font-size: 20px;
        line-height: 1;
        font-weight: 1000;
        letter-spacing: 0.06em;
      }

      .nhlcal-matchup-team span {
        color: var(--muted);
        font-size: 12px;
        font-weight: 900;
      }

      .nhlcal-versus {
        display: grid;
        place-items: center;
        text-align: center;
      }

      .nhlcal-versus strong {
        width: 44px;
        height: 44px;
        border-radius: var(--radius-ops, 2px);
        display: grid;
        place-items: center;
        background: rgba(255, 255, 255, 0.08);
        color: rgba(233, 247, 251, 0.72);
        font-size: 16px;
        font-weight: 1000;
      }

      .nhlcal-versus span {
        margin-top: 8px;
        color: var(--text);
        font-size: 12px;
        font-weight: 1000;
      }

      .nhlcal-versus small {
        margin-top: 4px;
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
      }

      .nhlcal-tab-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        padding: 0 14px;
        border-top: 1px solid var(--line);
      }

      .nhlcal-tab-row-three {
        grid-template-columns: repeat(3, 1fr);
      }

      .nhlcal-tab-row button {
        height: 42px;
        border: 0;
        border-bottom: 2px solid transparent;
        background: transparent;
        color: var(--muted);
        text-transform: uppercase;
        font-size: 11px;
        font-weight: 1000;
        letter-spacing: 0.08em;
        cursor: pointer;
      }

      .nhlcal-tab-row button.is-active {
        color: var(--text);
        border-bottom-color: var(--cyan);
      }

      .nhlcal-preview-lines {
        padding: 10px 14px 14px;
        display: grid;
      }

      .nhlcal-preview-lines div {
        min-height: 31px;
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: center;
        gap: 12px;
        border-bottom: 1px solid rgba(156, 218, 236, 0.09);
      }

      .nhlcal-preview-lines div:last-child {
        border-bottom: 0;
      }

      .nhlcal-preview-lines span {
        color: var(--muted);
        font-size: 11px;
        font-weight: 900;
      }

      .nhlcal-preview-lines strong {
        color: rgba(233, 247, 251, 0.9);
        font-size: 12px;
        font-weight: 1000;
        text-align: right;
      }

      .nhlcal-wide-action {
        width: calc(100% - 28px);
        min-height: 36px;
        margin: 0 14px 14px;
        border: 1px solid var(--line);
        border-radius: 7px;
        background: rgba(15, 38, 55, 0.9);
        color: rgba(233, 247, 251, 0.88);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 11px;
        font-weight: 1000;
        cursor: pointer;
        padding: 0 12px;
      }

      .nhlcal-wide-action:hover {
        border-color: var(--line-strong);
      }

      .nhlcal-wide-action.muted {
        margin-top: 12px;
        margin-bottom: 14px;
      }

      /* Schedule standby, not a decorative orb: department code over a hard
         rule, then the operational explanation. */
      .nhlcal-empty-preview {
        min-height: 0;
        display: grid;
        justify-items: center;
        align-content: start;
        text-align: center;
        padding: 16px 20px 18px;
      }

      .nhlcal-empty-orb {
        padding: 0 0 7px;
        border-bottom: 2px solid var(--ops-cyan, #13d8e7);
        color: var(--ops-cyan, #13d8e7);
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 0.16em;
        text-transform: uppercase;
      }

      .nhlcal-empty-preview h4 {
        margin: 10px 0 0;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-empty-preview p {
        margin: 6px 0 0;
        max-width: 300px;
        color: var(--muted);
        font-size: 12px;
        line-height: 1.45;
      }

      .nhlcal-empty-subnote {
        color: rgba(233, 247, 251, 0.7) !important;
        font-weight: 800;
      }

      .nhlcal-selected-event-strip {
        padding: 0 14px 2px;
        display: grid;
        gap: 7px;
      }

      .nhlcal-selected-event-chip {
        min-height: 34px;
        border-radius: 8px;
        border: 1px solid rgba(233, 168, 60, 0.24);
        background: rgba(233, 168, 60, 0.07);
        color: var(--text);
        display: grid;
        grid-template-columns: 28px minmax(0, 1fr);
        align-items: center;
        gap: 7px;
        padding: 5px 8px;
        text-align: left;
        cursor: pointer;
      }

      .nhlcal-selected-event-chip span {
        width: 24px;
        height: 24px;
        border-radius: 7px;
        display: grid;
        place-items: center;
        background: rgba(233, 168, 60, 0.14);
      }

      .nhlcal-selected-event-chip strong {
        min-width: 0;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
      }

      .nhlcal-selected-day-panel {
        padding: 12px 14px 16px;
        display: grid;
        gap: 12px;
      }

      .nhlcal-selected-section {
        border: 1px solid rgba(156, 218, 236, 0.1);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.025);
        padding: 12px;
      }

      .nhlcal-selected-section-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
      }

      .nhlcal-selected-section-head span {
        color: var(--cyan);
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.1em;
      }

      .nhlcal-selected-section-head strong {
        color: var(--text);
        font-size: 12px;
        font-weight: 1000;
      }

      .nhlcal-selected-section-head button {
        border: 1px solid var(--line);
        border-radius: var(--radius-ops, 2px);
        background: rgba(19, 216, 231, 0.07);
        color: var(--cyan);
        padding: 5px 9px;
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
        cursor: pointer;
      }

      .nhlcal-selected-event-list,
      .nhlcal-selected-injury-list,
      .nhlcal-selected-slate-list {
        display: grid;
        gap: 8px;
      }

      .nhlcal-selected-event-row {
        width: 100%;
        min-height: 50px;
        border: 1px solid rgba(233, 168, 60, 0.2);
        border-radius: 10px;
        background:
          radial-gradient(circle at 0% 0%, rgba(233, 168, 60, 0.1), transparent 50%),
          rgba(255, 255, 255, 0.025);
        color: var(--text);
        display: grid;
        grid-template-columns: 34px minmax(0, 1fr);
        align-items: center;
        gap: 10px;
        text-align: left;
        padding: 8px;
        cursor: pointer;
      }

      .nhlcal-selected-event-row:hover {
        border-color: rgba(233, 168, 60, 0.42);
      }

      .nhlcal-selected-event-icon {
        width: 32px;
        height: 32px;
        border-radius: 9px;
        display: grid;
        place-items: center;
        background: rgba(233, 168, 60, 0.12);
        color: #ffd88d;
      }

      .nhlcal-selected-event-row strong,
      .nhlcal-selected-event-row small {
        display: block;
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlcal-selected-event-row strong {
        color: var(--text);
        font-size: 12px;
        font-weight: 1000;
        white-space: nowrap;
      }

      .nhlcal-selected-event-row small {
        margin-top: 3px;
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
        line-height: 1.3;
      }

      .nhlcal-selected-injury-list article {
        min-height: 42px;
        border: 1px solid rgba(255, 96, 109, 0.14);
        border-radius: 8px;
        background: rgba(255, 96, 109, 0.055);
        padding: 8px 10px;
      }

      .nhlcal-selected-injury-list strong,
      .nhlcal-selected-injury-list span {
        display: block;
      }

      .nhlcal-selected-injury-list strong {
        color: rgba(255, 213, 217, 0.95);
        font-size: 12px;
        font-weight: 1000;
      }

      .nhlcal-selected-injury-list span {
        margin-top: 3px;
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
      }

      .nhlcal-selected-slate-list article {
        min-height: 32px;
        display: grid;
        grid-template-columns: 1fr 16px 1fr auto;
        gap: 8px;
        align-items: center;
        border-bottom: 1px solid rgba(156, 218, 236, 0.07);
        color: rgba(233, 247, 251, 0.78);
        font-size: 11px;
        font-weight: 900;
      }

      .nhlcal-selected-slate-list article:last-child {
        border-bottom: 0;
      }

      .nhlcal-selected-slate-list article.is-user-game {
        color: var(--cyan);
      }

      .nhlcal-selected-slate-list article span:first-child {
        text-align: right;
      }

      .nhlcal-selected-slate-list article em {
        color: var(--muted);
        font-style: normal;
        text-align: center;
      }

      .nhlcal-selected-slate-list article strong {
        color: inherit;
        font-size: 11px;
        text-align: right;
      }

      .nhlcal-selected-more {
        margin: 4px 0 0;
        color: var(--muted);
        font-size: 11px;
        font-weight: 900;
        text-align: center;
      }

      .nhlcal-standings-card {
        flex: 1 1 0;
        min-height: 0;
        max-height: none;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        border-color: rgba(19, 216, 231, 0.22);
        box-shadow:
          0 12px 28px rgba(0, 0, 0, 0.28),
          inset 0 1px 0 rgba(19, 216, 231, 0.08);
      }

      .nhlcal-standings-card--extended {
        flex: 1 0 auto;
        min-height: 280px;
      }

      .nhlcal-standings-header {
        flex: 0 0 auto;
        min-height: 46px;
        padding: 10px 12px 6px;
      }

      .nhlcal-standings-header-actions {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        justify-content: flex-end;
      }

      .nhlcal-standings-count {
        padding: 3px 7px;
        border-radius: var(--radius-ops, 2px);
        font-variant-numeric: tabular-nums;
        border: 1px solid rgba(156, 218, 236, 0.14);
        background: rgba(255, 255, 255, 0.03);
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        white-space: nowrap;
      }

      .nhlcal-standings-full-link {
        border: 1px solid rgba(19, 216, 231, 0.24);
        border-radius: 8px;
        background: rgba(19, 216, 231, 0.08);
        color: var(--cyan);
        padding: 6px 10px;
        font-size: 11px;
        font-weight: 1000;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        cursor: pointer;
      }

      .nhlcal-standings-table {
        flex: 1 1 auto;
        min-height: 0;
        display: flex;
        flex-direction: column;
        padding: 0 10px 6px;
      }

      .nhlcal-standings-body {
        flex: 1 1 auto;
        min-height: 0;
        overflow-x: hidden;
        overflow-y: auto;
        border-top: 1px solid rgba(156, 218, 236, 0.08);
        scrollbar-gutter: stable;
        padding-right: 2px;
      }

      .nhlcal-standings-footer {
        flex: 0 0 auto;
        padding: 0 10px 10px;
      }

      .nhlcal-standings-full-button {
        width: 100%;
        min-height: 34px;
        border: 1px solid rgba(156, 218, 236, 0.18);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.04);
        color: rgba(233, 247, 251, 0.88);
        font-size: 11px;
        font-weight: 1000;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        cursor: pointer;
      }

      .nhlcal-standings-full-button:hover,
      .nhlcal-standings-full-link:hover {
        border-color: rgba(19, 216, 231, 0.42);
        background: rgba(19, 216, 231, 0.1);
      }

      .nhlcal-standings-head,
      .nhlcal-standings-row {
        display: grid;
        grid-template-columns: 22px minmax(0, 1fr) 24px 22px 22px 26px 28px 34px;
        align-items: center;
        gap: 4px;
      }

      .nhlcal-standings-head {
        flex: 0 0 auto;
        height: 26px;
        color: rgba(233, 247, 251, 0.64);
        text-transform: uppercase;
        font-size: 11px;
        font-weight: 1000;
        letter-spacing: 0.06em;
        border-bottom: 1px solid rgba(156, 218, 236, 0.12);
      }

      .nhlcal-standings-row {
        min-height: 34px;
        color: rgba(233, 247, 251, 0.8);
        font-size: 11px;
        font-weight: 800;
        border-bottom: 1px solid rgba(156, 218, 236, 0.07);
      }

      .nhlcal-standings-row > span:nth-child(2) {
        display: flex;
        align-items: center;
        gap: 7px;
        min-width: 0;
      }

      .nhlcal-standings-row > span:nth-child(2) strong {
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .nhlcal-standings-row > span:not(:nth-child(2)),
      .nhlcal-standings-head > span:not(:nth-child(2)) {
        text-align: right;
      }

      .nhlcal-standings-row.is-user-team {
        color: var(--cyan);
        background: rgba(19, 216, 231, 0.1);
        box-shadow: inset 3px 0 0 var(--cyan);
      }

      .nhlcal-standings-row.is-user-team strong {
        color: var(--cyan);
      }

      .nhlcal-mini-card-row {
        flex: 0 0 auto;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        max-height: 96px;
        overflow: hidden;
      }

      .nhlcal-table-empty {
        padding: 14px 0;
        color: var(--muted);
        font-size: 11px;
        text-align: center;
      }

      .nhlcal-stretch-card,
      .nhlcal-league-card {
        min-height: 245px;
        overflow: hidden;
      }

      .nhlcal-mini-header {
        min-height: 54px;
        padding: 15px 15px 8px;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 10px;
      }

      .nhlcal-mini-header h3 {
        margin: 0;
        color: var(--cyan);
        font-size: 12px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.1em;
      }

      .nhlcal-mini-header span {
        color: var(--muted);
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
      }

      .nhlcal-stretch-list,
      .nhlcal-league-list {
        padding: 0 14px;
        display: grid;
        gap: 7px;
      }

      .nhlcal-stretch-row,
      .nhlcal-league-row {
        min-height: 25px;
        display: grid;
        align-items: center;
        gap: 7px;
        color: rgba(233, 247, 251, 0.82);
        font-size: 11px;
        font-weight: 900;
      }

      .nhlcal-stretch-row {
        grid-template-columns: 42px 22px minmax(0, 1fr) auto;
      }

      .nhlcal-stretch-row > span {
        color: var(--muted);
      }

      .nhlcal-stretch-row strong {
        min-width: 0;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
      }

      .nhlcal-stretch-row em {
        min-width: 45px;
        padding: 3px 6px;
        border-radius: 5px;
        background: rgba(255, 255, 255, 0.045);
        color: var(--muted);
        font-size: 11px;
        font-style: normal;
        text-align: center;
        text-transform: uppercase;
      }

      .nhlcal-stretch-row em.home {
        color: var(--cyan);
        background: var(--cyan-soft);
      }

      .nhlcal-stretch-row em.away {
        color: var(--blue);
        background: var(--blue-soft);
      }

      .nhlcal-league-row {
        grid-template-columns: 1fr 14px 1fr auto;
      }

      .nhlcal-league-row span:first-child {
        text-align: right;
      }

      .nhlcal-league-row em {
        color: var(--muted);
        font-style: normal;
        text-align: center;
      }

      .nhlcal-league-row strong {
        color: var(--muted);
        font-size: 11px;
        text-align: right;
      }

      .nhlcal-league-row.is-highlight span,
      .nhlcal-league-row.is-highlight strong {
        color: var(--cyan);
      }

      .nhlcal-storyline-list {
        gap: 9px;
      }

      .nhlcal-storyline-row {
        border: 1px solid rgba(156, 218, 236, 0.09);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.025);
        padding: 9px;
      }

      .nhlcal-storyline-topline {
        display: grid;
        grid-template-columns: 44px minmax(0, 1fr) auto;
        gap: 8px;
        align-items: center;
      }

      .nhlcal-storyline-topline span {
        color: var(--muted);
        font-size: 11px;
        font-weight: 900;
      }

      .nhlcal-storyline-topline strong {
        min-width: 0;
        color: var(--text);
        font-size: 11px;
        font-weight: 1000;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .nhlcal-storyline-topline em {
        border-radius: var(--radius-ops, 2px);
        padding: 2px 6px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background: rgba(255, 255, 255, 0.04);
        color: var(--muted);
        font-style: normal;
        font-size: 11px;
        font-weight: 1000;
      }

      .nhlcal-storyline-topline em.critical {
        color: var(--red);
        background: var(--red-soft);
      }

      .nhlcal-storyline-topline em.high {
        color: var(--gold);
        background: var(--gold-soft);
      }

      .nhlcal-subtext {
        margin-top: 5px;
        color: var(--muted);
        font-size: 11px;
        line-height: 1.35;
        font-weight: 800;
      }

      .nhlcal-storyline-choice-row {
        margin-top: 7px;
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
      }

      .nhlcal-storyline-choice-button {
        border: 1px solid var(--line);
        border-radius: var(--radius-hud, 4px);
        background: rgba(19, 216, 231, 0.07);
        color: var(--cyan);
        min-height: 26px;
        padding: 0 9px;
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
        cursor: pointer;
      }

      .nhlcal-storyline-choice-button:disabled {
        cursor: not-allowed;
        opacity: 0.58;
        filter: saturate(0.65);
      }

      .nhlcal-injury-mini-list {
        gap: 8px;
      }

      .nhlcal-injury-mini-row {
        min-height: 38px;
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto auto;
        grid-template-rows: auto auto;
        gap: 3px 8px;
        align-items: center;
        border: 1px solid rgba(255, 96, 109, 0.1);
        border-radius: 9px;
        background: rgba(255, 96, 109, 0.04);
        padding: 8px;
      }

      .nhlcal-injury-mini-row span {
        min-width: 0;
        color: var(--text);
        font-size: 11px;
        font-weight: 1000;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .nhlcal-injury-mini-row em {
        color: rgba(255, 198, 205, 0.9);
        font-size: 11px;
        font-style: normal;
        font-weight: 900;
      }

      .nhlcal-injury-mini-row strong {
        color: var(--red);
        font-size: 11px;
        font-weight: 1000;
      }

      .nhlcal-injury-mini-row small {
        grid-column: 1 / -1;
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
      }

      /* Standby notice, not a sentence floating in a void: a hairline registry
         line carrying the department state. */
      .nhlcal-small-empty {
        margin: 14px 0 0;
        padding: 10px 12px;
        border-left: 2px solid var(--line-strong);
        background: rgba(255, 255, 255, 0.02);
        color: var(--muted);
        font-size: 12px;
        line-height: 1.4;
      }

      .nhlcal-small-empty b {
        display: block;
        margin-bottom: 3px;
        color: var(--ops-cyan, #13d8e7);
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 0.16em;
        text-transform: uppercase;
      }

      .nhlcal-mini-button {
        width: calc(100% - 28px);
        min-height: 34px;
        margin: 13px 14px 14px;
        border: 1px solid var(--line);
        border-radius: 7px;
        background: rgba(15, 38, 55, 0.9);
        color: rgba(233, 247, 251, 0.78);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 11px;
        font-weight: 1000;
        cursor: pointer;
      }

      .nhlcal-bottom-grid {
        margin-top: 16px;
        display: grid;
        grid-template-columns: minmax(0, 1fr) 400px;
        gap: 16px;
      }

      .nhlcal-bottom-grid--collapsed {
        display: none;
      }

      .nhlcal-diagnostics-panel,
      .nhlcal-month-snapshot {
        padding: 16px;
      }

      .nhlcal-section-title {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 13px;
      }

      .nhlcal-section-title p,
      .nhlcal-section-title h3 {
        margin: 0;
      }

      .nhlcal-section-title p {
        color: var(--cyan);
        font-size: 12px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.12em;
      }

      .nhlcal-section-title h3 {
        margin-top: 4px;
        color: rgba(233, 247, 251, 0.94);
        font-size: 14px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-section-title > span {
        border: 1px solid var(--line);
        border-radius: var(--radius-ops, 2px);
        padding: 4px 8px;
        letter-spacing: 0.08em;
        background: rgba(255, 255, 255, 0.035);
        color: var(--muted);
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
        white-space: nowrap;
      }

      .nhlcal-diagnostic-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
      }

      .nhlcal-diagnostic-tile {
        min-height: 92px;
        border: 1px solid rgba(156, 218, 236, 0.11);
        border-radius: 9px;
        background:
          linear-gradient(180deg, rgba(15, 38, 56, 0.84), rgba(8, 25, 38, 0.78)),
          radial-gradient(circle at 100% 30%, rgba(19, 216, 231, 0.1), transparent 52%);
        padding: 13px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
      }

      .nhlcal-diagnostic-tile span,
      .nhlcal-diagnostic-tile small {
        display: block;
      }

      .nhlcal-diagnostic-tile span {
        color: var(--muted);
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-diagnostic-tile strong {
        display: block;
        margin-top: 6px;
        font-size: 26px;
        line-height: 1;
        font-weight: 1000;
      }

      .nhlcal-diagnostic-tile small {
        margin-top: 8px;
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
      }

      .nhlcal-diagnostic-tile em {
        color: rgba(19, 216, 231, 0.56);
        font-size: 34px;
        font-style: normal;
      }

      .nhlcal-diagnostic-tile.is-danger strong {
        color: var(--red);
      }

      .nhlcal-diagnostic-tile.is-danger em {
        color: rgba(255, 96, 109, 0.62);
      }

      .nhlcal-insight-row {
        margin-top: 12px;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
      }

      .nhlcal-insight-pill {
        min-height: 52px;
        border: 1px solid rgba(156, 218, 236, 0.1);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.03);
        display: grid;
        grid-template-columns: 28px minmax(0, 1fr);
        align-items: center;
        gap: 10px;
        padding: 10px;
      }

      .nhlcal-insight-pill span {
        width: 22px;
        height: 22px;
        border-radius: var(--radius-ops, 2px);
        display: grid;
        place-items: center;
        background: var(--gold-soft);
        color: var(--gold);
        font-size: 12px;
        font-weight: 1000;
      }

      .nhlcal-insight-pill p {
        margin: 0;
        color: rgba(233, 247, 251, 0.78);
        font-size: 11px;
        line-height: 1.35;
        font-weight: 750;
      }

      .nhlcal-snapshot-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
      }

      .nhlcal-snapshot-grid div {
        min-height: 74px;
        border: 1px solid rgba(156, 218, 236, 0.1);
        border-radius: 9px;
        background: rgba(255, 255, 255, 0.035);
        display: grid;
        place-items: center;
        text-align: center;
      }

      .nhlcal-snapshot-grid span {
        color: var(--muted);
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
      }

      .nhlcal-snapshot-grid strong {
        color: var(--text);
        font-size: 24px;
        font-weight: 1000;
      }

      .nhlcal-opponent-strip {
        margin-top: 12px;
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
      }

      .nhlcal-opponent-strip div {
        min-height: 60px;
        border: 1px solid rgba(156, 218, 236, 0.1);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.025);
        display: grid;
        place-items: center;
        gap: 4px;
        padding: 8px;
      }

      .nhlcal-opponent-strip span {
        color: var(--muted);
        font-size: 11px;
        font-weight: 1000;
        text-align: center;
      }

      .nhlcal-opponent-strip p {
        grid-column: 1 / -1;
        margin: 0;
        color: var(--muted);
        font-size: 12px;
        text-align: center;
        align-self: center;
      }

      .nhlcal-opponent-event strong {
        color: var(--gold);
        font-size: 20px;
      }

      .nhlcal-team-badge {
        --team-seed: 185 78% 48%;
        flex: 0 0 auto;
        display: grid;
        place-items: center;
        border-radius: 999px;
        background:
          radial-gradient(circle at 30% 25%, rgba(255, 255, 255, 0.2), transparent 28%),
          linear-gradient(135deg, hsl(var(--team-seed) / 0.95), hsl(var(--team-seed) / 0.38));
        border: 1px solid hsl(var(--team-seed) / 0.58);
        color: #eaffff;
        font-weight: 1000;
        letter-spacing: 0.05em;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.34);
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.16),
          0 0 20px hsl(var(--team-seed) / 0.16);
      }

      .nhlcal-team-badge.size-large {
        width: 94px;
        height: 94px;
        font-size: 23px;
        border-radius: 8px;
      }

      .nhlcal-team-badge.size-matchup {
        width: 86px;
        height: 86px;
        font-size: 20px;
        border-radius: 8px;
      }

      .nhlcal-team-badge.size-small {
        width: 34px;
        height: 34px;
        font-size: 11px;
      }

      .nhlcal-team-badge.size-tiny {
        width: 24px;
        height: 24px;
        font-size: 11px;
      }

      .nhlcal-team-badge.size-mini {
        width: 18px;
        height: 18px;
        font-size: 11px;
      }

      .nhlcal-team-badge.size-tile-main {
        width: 44px;
        height: 44px;
        font-size: 12px;
        border-radius: 12px;
      }

      .nhlcal-team-badge.size-tile-compact {
        width: 34px;
        height: 34px;
        font-size: 11px;
        border-radius: 10px;
      }

      .nhlcal-team-logo {
        flex: 0 0 auto;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: rgba(2, 11, 18, 0.85);
        border: 1px solid rgba(103, 157, 183, 0.32);
        overflow: hidden;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
      }

      .nhlcal-team-logo img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
        filter: drop-shadow(0 2px 5px rgba(0, 0, 0, 0.35));
      }

      .nhlcal-team-logo.size-large {
        width: 76px;
        height: 76px;
        border-radius: 6px;
        padding: 7px;
      }

      .nhlcal-matchup-stage-compact {
        min-height: 88px;
        padding: 4px 10px 8px;
      }

      .nhlcal-matchup-stage-compact .nhlcal-matchup-team strong {
        font-size: 16px;
      }

      .nhlcal-matchup-stage-compact .nhlcal-versus strong {
        width: 34px;
        height: 34px;
        font-size: 12px;
      }

      .nhlcal-preview-lines-compact div {
        min-height: 28px;
        padding: 4px 12px;
      }

      .nhlcal-preview-empty-stats {
        margin: 0 12px 10px;
        padding: 8px 10px;
        border-radius: 8px;
        border: 1px dashed rgba(156, 218, 236, 0.18);
        color: var(--muted);
        font-size: 11px;
        font-weight: 700;
        text-align: center;
      }

      .nhlcal-tab-row-two {
        grid-template-columns: 1fr 1fr;
      }

      .nhlcal-team-logo.size-matchup {
        width: 44px;
        height: 44px;
        border-radius: 11px;
        padding: 4px;
      }

      .nhlcal-team-logo.size-small {
        width: 34px;
        height: 34px;
        padding: 3px;
      }

      .nhlcal-team-logo.size-tiny {
        width: 24px;
        height: 24px;
        padding: 2px;
      }

      .nhlcal-team-logo.size-mini {
        width: 20px;
        height: 20px;
        padding: 1px;
      }

      .nhlcal-team-logo.size-tile-main {
        width: 44px;
        height: 44px;
        padding: 3px;
        border-radius: 12px;
      }

      .nhlcal-team-logo.size-tile-compact {
        width: 34px;
        height: 34px;
        padding: 2px;
        border-radius: 10px;
      }

      .nhlcal-drawer-backdrop,
      .nhlcal-modal-backdrop,
      .nhlcal-event-backdrop {
        position: fixed;
        inset: 0;
        z-index: 100;
        background: rgba(0, 5, 10, 0.58);
        backdrop-filter: blur(7px);
        display: flex;
      }

      .nhlcal-drawer-backdrop {
        justify-content: flex-end;
      }

      .nhlcal-modal-backdrop,
      .nhlcal-event-backdrop {
        align-items: center;
        justify-content: center;
      }

      .nhlcal-injury-backdrop {
        position: fixed;
        inset: 0;
        z-index: 100;
        background: rgba(0, 5, 10, 0.58);
        backdrop-filter: blur(7px);
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .nhlcal-event-modal {
        width: min(620px, 94vw);
        max-height: 88vh;
        overflow: auto;
        border: 1px solid var(--line-strong);
        border-radius: 6px;
        background:
          radial-gradient(circle at 0% 0%, rgba(233, 168, 60, 0.13), transparent 38%),
          linear-gradient(180deg, rgba(7, 21, 34, 0.98), rgba(2, 9, 15, 0.98));
        box-shadow: 0 28px 90px rgba(0, 0, 0, 0.55);
      }

      .nhlcal-event-modal.tone-medical {
        border-color: rgba(255, 96, 109, 0.48);
        background:
          radial-gradient(circle at 0% 0%, rgba(255, 96, 109, 0.16), transparent 38%),
          linear-gradient(180deg, rgba(32, 12, 20, 0.98), rgba(2, 9, 15, 0.98));
      }

      .nhlcal-event-modal.tone-trade {
        border-color: rgba(19, 216, 231, 0.45);
        background:
          radial-gradient(circle at 0% 0%, rgba(19, 216, 231, 0.14), transparent 38%),
          linear-gradient(180deg, rgba(7, 21, 34, 0.98), rgba(2, 9, 15, 0.98));
      }

      .nhlcal-event-modal-head {
        display: grid;
        grid-template-columns: 54px minmax(0, 1fr) 42px;
        align-items: start;
        gap: 14px;
        padding: 22px;
        border-bottom: 1px solid var(--line);
      }

      .nhlcal-event-modal-icon {
        width: 52px;
        height: 52px;
        border-radius: 10px;
        display: grid;
        place-items: center;
        background: rgba(233, 168, 60, 0.13);
        border: 1px solid rgba(233, 168, 60, 0.25);
        color: #ffd88d;
        font-size: 24px;
      }

      .nhlcal-event-modal-head p,
      .nhlcal-event-modal-head h2 {
        margin: 0;
      }

      .nhlcal-event-modal-head p {
        color: var(--gold);
        font-size: 11px;
        font-weight: 1000;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }

      .nhlcal-event-modal-head h2 {
        margin-top: 6px;
        color: var(--text);
        font-size: 25px;
        line-height: 1.05;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }

      .nhlcal-event-modal-head button {
        width: 42px;
        height: 42px;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.04);
        color: var(--text);
        font-size: 26px;
        line-height: 1;
        cursor: pointer;
      }

      .nhlcal-event-modal-body {
        padding: 18px 22px 22px;
        display: grid;
        gap: 12px;
      }

      .nhlcal-event-modal-description {
        margin: 0;
        color: rgba(233, 247, 251, 0.82);
        font-size: 13px;
        line-height: 1.55;
        font-weight: 750;
      }

      .nhlcal-event-modal-callout {
        border: 1px solid rgba(156, 218, 236, 0.12);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.03);
        padding: 12px;
      }

      .nhlcal-event-modal-callout span,
      .nhlcal-event-modal-callout strong {
        display: block;
      }

      .nhlcal-event-modal-callout span {
        color: var(--muted);
        font-size: 11px;
        font-weight: 1000;
        letter-spacing: 0.1em;
        text-transform: uppercase;
      }

      .nhlcal-event-modal-callout strong {
        margin-top: 5px;
        color: var(--text);
        font-size: 13px;
        line-height: 1.35;
      }

      .nhlcal-event-effect-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
      }

      .nhlcal-event-effect-grid article {
        border: 1px solid rgba(156, 218, 236, 0.1);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.025);
        padding: 11px;
      }

      .nhlcal-event-effect-grid span,
      .nhlcal-event-effect-grid strong {
        display: block;
      }

      .nhlcal-event-effect-grid span {
        color: var(--muted);
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
      }

      .nhlcal-event-effect-grid strong {
        margin-top: 4px;
        color: var(--cyan);
        font-size: 18px;
        font-weight: 1000;
      }

      .nhlcal-injury-report-modal {
        width: min(760px, 96vw);
        max-height: 88vh;
        overflow: auto;
        z-index: 101;
        border: 1px solid var(--line-strong);
        border-top: 2px solid var(--ops-cyan, #13d8e7);
        border-radius: 6px;
        background: linear-gradient(
          180deg,
          rgba(7, 21, 34, 0.98),
          rgba(2, 9, 15, 0.98)
        );
        box-shadow: 0 28px 90px rgba(0, 0, 0, 0.55);
      }

      .nhlcal-injury-report-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
        padding: 22px 22px 16px;
        border-bottom: 1px solid var(--line);
      }

      .nhlcal-injury-report-kicker {
        margin: 0;
        color: var(--cyan);
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.14em;
      }

      .nhlcal-injury-report-head h2 {
        margin: 6px 0 0;
        font-size: 26px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }

      .nhlcal-injury-report-stats {
        margin: 10px 0 0;
        font-size: 13px;
        color: var(--muted);
      }

      .nhlcal-injury-report-close {
        width: 34px;
        height: 34px;
        border-radius: var(--radius-ops, 2px);
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.04);
        color: var(--muted);
        font-size: 20px;
        line-height: 1;
        cursor: pointer;
        flex-shrink: 0;
        transition: color 110ms ease, border-color 110ms ease,
          background 110ms ease;
      }

      .nhlcal-injury-report-close:hover {
        color: var(--text);
        border-color: var(--ops-cyan, #13d8e7);
        background: rgba(19, 216, 231, 0.1);
      }

      .nhlcal-injury-report-body {
        padding: 16px 18px 22px;
      }

      .nhlcal-injury-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }

      .nhlcal-injury-table th,
      .nhlcal-injury-table td {
        text-align: left;
        padding: 8px 6px;
        border-bottom: 1px solid rgba(156, 218, 236, 0.08);
      }

      .nhlcal-injury-table th {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
      }

      .nhlcal-drawer {
        width: min(560px, 94vw);
        height: 100vh;
        overflow: auto;
        border-left: 1px solid var(--line-strong);
        background:
          radial-gradient(circle at 0% 0%, rgba(19, 216, 231, 0.13), transparent 34%),
          linear-gradient(180deg, rgba(7, 21, 34, 0.98), rgba(2, 9, 15, 0.98));
        box-shadow: -30px 0 80px rgba(0, 0, 0, 0.55);
      }

      .nhlcal-drawer-header {
        min-height: 100px;
        padding: 25px 25px 18px;
        border-bottom: 1px solid var(--line);
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 20px;
      }

      .nhlcal-drawer-header p,
      .nhlcal-drawer-header h2 {
        margin: 0;
      }

      .nhlcal-drawer-header p {
        color: var(--cyan);
        font-size: 12px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.16em;
      }

      .nhlcal-drawer-header h2 {
        margin-top: 6px;
        font-size: 32px;
        line-height: 1;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-drawer-header button,
      .nhlcal-modal header button {
        width: 34px;
        height: 34px;
        border-radius: var(--radius-ops, 2px);
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.035);
        color: var(--muted);
        font-size: 20px;
        line-height: 1;
        cursor: pointer;
        transition: color 110ms ease, border-color 110ms ease,
          background 110ms ease;
      }

      .nhlcal-modal header button:hover {
        color: var(--text);
        border-color: var(--ops-cyan);
        background: rgba(19, 216, 231, 0.1);
      }

      .nhlcal-drawer-tabs {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        border-bottom: 1px solid var(--line);
      }

      .nhlcal-drawer-tabs button {
        height: 48px;
        border: 0;
        border-right: 1px solid rgba(156, 218, 236, 0.08);
        border-bottom: 2px solid transparent;
        background: rgba(255, 255, 255, 0.02);
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 11px;
        font-weight: 1000;
        cursor: pointer;
      }

      .nhlcal-drawer-tabs button:last-child {
        border-right: 0;
      }

      .nhlcal-drawer-tabs button.is-active {
        color: var(--text);
        border-bottom-color: var(--cyan);
        background: rgba(19, 216, 231, 0.08);
      }

      .nhlcal-drawer-body {
        padding: 20px;
        display: grid;
        gap: 18px;
      }

      .nhlcal-drawer-section {
        border: 1px solid var(--line);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.025);
        padding: 16px;
      }

      .nhlcal-drawer-section h3 {
        margin: 0 0 12px;
        color: var(--cyan);
        font-size: 12px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.12em;
      }

      .nhlcal-drawer-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
      }

      .nhlcal-drawer-nav-card {
        min-height: 74px;
        border: 1px solid rgba(156, 218, 236, 0.1);
        border-radius: 10px;
        background:
          linear-gradient(180deg, rgba(18, 42, 60, 0.72), rgba(7, 23, 34, 0.72));
        color: var(--text);
        text-align: left;
        padding: 13px;
        cursor: pointer;
        transition:
          border-color 0.2s ease,
          transform 0.2s ease,
          background 0.2s ease;
      }

      .nhlcal-drawer-nav-card:hover {
        border-color: var(--line-strong);
        background: rgba(19, 216, 231, 0.08);
        transform: translateY(-1px);
      }

      .nhlcal-drawer-nav-card strong,
      .nhlcal-drawer-nav-card span {
        display: block;
      }

      .nhlcal-drawer-nav-card strong {
        font-size: 14px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-drawer-nav-card span {
        margin-top: 6px;
        color: var(--muted);
        font-size: 12px;
        font-weight: 800;
      }

      .nhlcal-drawer-feed {
        display: grid;
        gap: 10px;
      }

      .nhlcal-drawer-feed article {
        border: 1px solid rgba(156, 218, 236, 0.09);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.025);
        padding: 12px;
      }

      .nhlcal-drawer-feed strong {
        display: block;
        color: var(--text);
        font-size: 13px;
        font-weight: 1000;
      }

      .nhlcal-drawer-feed strong span {
        margin-right: 6px;
      }

      .nhlcal-drawer-feed p {
        margin: 6px 0 0;
        color: var(--muted);
        font-size: 12px;
        line-height: 1.45;
      }

      .nhlcal-drawer-event {
        border-color: rgba(233, 168, 60, 0.14) !important;
      }

      .nhlcal-drawer-event.tone-medical {
        border-color: rgba(255, 96, 109, 0.22) !important;
        background: rgba(255, 96, 109, 0.04) !important;
      }

      .nhlcal-drawer-event.tone-trade {
        border-color: rgba(19, 216, 231, 0.22) !important;
        background: rgba(19, 216, 231, 0.035) !important;
      }

      .nhlcal-drawer-empty {
        margin: 0;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.45;
      }

      .nhlcal-player-list,
      .nhlcal-draft-board {
        display: grid;
        gap: 8px;
      }

      .nhlcal-player-mini-row,
      .nhlcal-draft-mini-row {
        min-height: 54px;
        display: grid;
        grid-template-columns: 32px minmax(0, 1fr) auto;
        align-items: center;
        gap: 10px;
        border: 1px solid rgba(156, 218, 236, 0.09);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.025);
        padding: 8px 10px;
      }

      /* Rank/number markers are equipment plates, not circular tokens. */
      .nhlcal-player-mini-row > span,
      .nhlcal-draft-mini-row > span {
        width: 26px;
        height: 20px;
        border-radius: var(--radius-ops, 2px);
        display: grid;
        place-items: center;
        background: rgba(19, 216, 231, 0.1);
        color: var(--cyan);
        font-size: 11px;
        font-weight: 1000;
        font-variant-numeric: tabular-nums;
      }

      .nhlcal-player-mini-row strong,
      .nhlcal-player-mini-row small,
      .nhlcal-draft-mini-row strong,
      .nhlcal-draft-mini-row small {
        display: block;
        min-width: 0;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
      }

      .nhlcal-player-mini-row strong,
      .nhlcal-draft-mini-row strong {
        color: var(--text);
        font-size: 13px;
        font-weight: 1000;
      }

      .nhlcal-player-mini-row small,
      .nhlcal-draft-mini-row small {
        margin-top: 3px;
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
      }

      .nhlcal-player-mini-row em,
      .nhlcal-draft-mini-row em {
        color: var(--cyan);
        font-style: normal;
        font-size: 14px;
        font-weight: 1000;
      }

      .nhlcal-draft-mini-row em.up {
        color: var(--green);
      }

      .nhlcal-draft-mini-row em.down {
        color: var(--red);
      }

      .nhlcal-office-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
      }

      .nhlcal-office-metric {
        min-height: 70px;
        border: 1px solid rgba(156, 218, 236, 0.1);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.025);
        display: grid;
        place-items: center;
        text-align: center;
        padding: 10px;
      }

      .nhlcal-office-metric span,
      .nhlcal-office-metric strong {
        display: block;
      }

      .nhlcal-office-metric span {
        color: var(--muted);
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.1em;
      }

      .nhlcal-office-metric strong {
        margin-top: 5px;
        color: var(--text);
        font-size: 18px;
        font-weight: 1000;
      }

      .nhlcal-modal {
        width: min(520px, 94vw);
        border: 1px solid var(--line-strong);
        border-top: 2px solid var(--ops-cyan, #13d8e7);
        border-radius: 6px;
        background: linear-gradient(
          180deg,
          rgba(9, 27, 42, 0.98),
          rgba(3, 11, 18, 0.98)
        );
        box-shadow: 0 30px 90px rgba(0, 0, 0, 0.58);
        overflow: hidden;
      }

      .nhlcal-modal header {
        min-height: 72px;
        padding: 16px 18px;
        border-bottom: 1px solid var(--line);
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
      }

      .nhlcal-modal header p,
      .nhlcal-modal header h2 {
        margin: 0;
      }

      .nhlcal-modal header p {
        color: var(--cyan);
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.14em;
      }

      .nhlcal-modal header h2 {
        margin-top: 5px;
        font-size: 25px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }

      .nhlcal-settings-list {
        padding: 14px 21px 16px;
        display: grid;
        gap: 0;
      }

      /* Registry lines: hairline rows, lamp on the left, state on the right. */
      .nhlcal-settings-list button {
        min-height: 44px;
        border: 0;
        border-bottom: 1px solid var(--line);
        border-radius: 0;
        background: transparent;
        color: var(--text);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        padding: 0 12px 0 22px;
        cursor: pointer;
        position: relative;
        transition: background 110ms ease, box-shadow 110ms ease;
      }

      .nhlcal-settings-list button::before {
        content: "";
        position: absolute;
        left: 8px;
        top: 50%;
        width: 6px;
        height: 6px;
        margin-top: -3px;
        border: 1px solid var(--line-strong);
        background: transparent;
      }

      .nhlcal-settings-list button:hover {
        background: rgba(255, 255, 255, 0.03);
      }

      .nhlcal-settings-list button.is-active {
        background: rgba(19, 216, 231, 0.06);
        box-shadow: inset 2px 0 0 var(--ops-cyan, #13d8e7);
      }

      .nhlcal-settings-list button.is-active::before {
        border-color: var(--ops-cyan, #13d8e7);
        background: var(--ops-cyan, #13d8e7);
      }

      .nhlcal-settings-list span {
        font-size: 13px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-settings-list strong {
        color: var(--cyan);
        font-size: 12px;
        font-weight: 1000;
        text-transform: uppercase;
      }

      .nhlcal-settings-summary {
        border-top: 1px solid var(--line);
        padding: 18px 21px 21px;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
      }

      .nhlcal-settings-summary article {
        min-height: 52px;
        border: 1px solid rgba(156, 218, 236, 0.1);
        border-radius: var(--radius-ops, 2px);
        background: rgba(255, 255, 255, 0.025);
        display: grid;
        place-items: center;
        text-align: center;
        padding: 8px 10px;
        font-variant-numeric: tabular-nums;
      }

      .nhlcal-settings-summary span,
      .nhlcal-settings-summary strong {
        display: block;
      }

      .nhlcal-settings-summary span {
        color: var(--muted);
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
      }

      .nhlcal-settings-summary strong {
        color: var(--text);
        font-size: 13px;
        font-weight: 1000;
      }

      /* 1280-1920 are primary desktop game resolutions and must keep the
         two-column board layout; stacking only below that. */
      @media (max-width: 1120px) {
        .nhlcal-topbar {
          grid-template-columns: 1fr;
          min-height: auto;
          gap: 16px;
        }

        .nhlcal-action-cluster {
          justify-self: stretch;
          justify-content: flex-start;
        }

        .nhlcal-month-control {
          text-align: left;
        }

        .nhlcal-month-row {
          justify-content: flex-start;
        }

        .nhlcal-stat-strip {
          grid-template-columns: repeat(4, minmax(0, 1fr));
        }

        .nhlcal-content-grid,
        .nhlcal-bottom-grid {
          grid-template-columns: 1fr;
        }

        .nhlcal-right-rail {
          grid-template-columns: repeat(2, minmax(0, 1fr));
          align-items: start;
        }

        .nhlcal-preview-card {
          grid-column: 1 / -1;
        }
      }

      @media (max-width: 1080px) {
        .nhlcal-root {
          grid-template-columns: 1fr;
        }

        .nhlcal-sidebar {
          min-height: auto;
          height: auto;
          flex-direction: row;
          border-right: 0;
          border-bottom: 1px solid var(--line);
          overflow-x: auto;
        }

        .nhlcal-brand-button,
        .nhlcal-settings-button {
          width: 86px;
          height: 76px;
          flex: 0 0 auto;
          border-bottom: 0;
          border-top: 0;
          border-right: 1px solid var(--line);
        }

        .nhlcal-side-nav {
          flex-direction: row;
          padding: 0;
        }

        .nhlcal-side-button {
          width: 86px;
          min-height: 76px;
          flex: 0 0 auto;
        }

        .nhlcal-side-button.is-active::before {
          top: auto;
          right: 12px;
          left: 12px;
          bottom: 0;
          width: auto;
          height: 3px;
        }

        .nhlcal-main {
          height: calc(100vh - 76px);
          padding: 18px;
        }

        .nhlcal-stat-strip {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .nhlcal-content-grid,
        .nhlcal-bottom-grid,
        .nhlcal-right-rail {
          grid-template-columns: 1fr;
        }

        .nhlcal-standings-card--extended {
          min-height: 280px;
        }

        .nhlcal-calendar-footer {
          align-items: flex-start;
          flex-direction: column;
        }

        .nhlcal-calendar-actions {
          justify-content: flex-start;
        }

        .nhlcal-month-grid {
          min-width: 920px;
        }

        .nhlcal-calendar-panel {
          overflow-x: auto;
        }

        .nhlcal-week-header {
          min-width: 920px;
        }
      }

      @media (max-width: 720px) {
        .nhlcal-main {
          padding: 14px;
        }

        .nhlcal-team-identity h1 {
          font-size: 30px;
        }

        .nhlcal-month-row h2 {
          font-size: 30px;
          letter-spacing: 0.12em;
        }

        .nhlcal-action-cluster {
          gap: 8px;
        }

        .nhlcal-date-chip,
        .nhlcal-online-chip {
          display: none;
        }

        .nhlcal-advance-button {
          min-width: 150px;
        }

        .nhlcal-stat-strip {
          grid-template-columns: 1fr;
        }

        .nhlcal-diagnostic-grid,
        .nhlcal-insight-row,
        .nhlcal-snapshot-grid,
        .nhlcal-opponent-strip,
        .nhlcal-event-effect-grid,
        .nhlcal-settings-summary {
          grid-template-columns: 1fr;
        }

        .nhlcal-matchup-stage {
          grid-template-columns: 1fr;
          gap: 18px;
        }

        .nhlcal-drawer {
          width: 100vw;
        }

        .nhlcal-drawer-tabs {
          grid-template-columns: repeat(5, minmax(88px, 1fr));
          overflow-x: auto;
        }

        .nhlcal-drawer-grid,
        .nhlcal-office-grid {
          grid-template-columns: 1fr;
        }

        .nhlcal-event-modal-head {
          grid-template-columns: 48px minmax(0, 1fr) 38px;
          padding: 18px;
        }

        .nhlcal-event-modal-head h2 {
          font-size: 20px;
        }

        .nhlcal-injury-report-modal {
          width: 96vw;
        }

        .nhlcal-injury-report-body {
          overflow-x: auto;
        }

        .nhlcal-injury-table {
          min-width: 720px;
        }
      }

      /* ─── Vertical budget ───────────────────────────────────────────
         The month grid owns the remaining height and is the only scroll
         owner inside the panel. Without this the last two week rows fall
         outside the clipped content grid and cannot be reached at all. */

      .nhlcal-content-grid {
        grid-auto-rows: minmax(0, 1fr);
        align-items: stretch;
      }

      .nhlcal-content-grid > * {
        min-height: 0;
        max-height: 100%;
      }

      .nhlcal-calendar-panel {
        max-height: 100%;
      }

      .nhlcal-month-grid {
        grid-auto-rows: minmax(78px, auto);
        overflow-y: auto;
        overscroll-behavior: contain;
      }

      /* Keep the action cluster on one line so the topbar cannot grow into
         a four-row block and push the grid off screen. */
      .nhlcal-action-cluster {
        flex-wrap: nowrap;
      }

      .nhlcal-month-row button.nhlcal-today-chip {
        min-width: fit-content;
        white-space: nowrap;
      }

      .nhlcal-stat-strip {
        min-height: 0;
      }

      /* Day contents stay inside their cell; only the hover card escapes. */
      .nhlcal-day-content {
        min-height: 0;
        overflow: hidden;
      }

      /* Events read as broadcast strips, not as glossy cards floating in a
         flat grid: hard left rail, ladder radius, no bloom. */
      .nhlcal-special-event-tile {
        border-radius: var(--radius-ops, 2px);
        border-width: 0 0 0 3px;
        border-left-color: var(--ops-gold);
        background: linear-gradient(90deg, rgba(233, 168, 60, 0.14), rgba(7, 22, 34, 0.55));
        box-shadow: none;
        min-height: 30px;
        padding: 4px 6px;
        gap: 6px;
        grid-template-columns: 18px minmax(0, 1fr);
      }

      .nhlcal-special-event-tile.priority-critical {
        border-left-color: var(--ops-signal-red);
        background: linear-gradient(90deg, rgba(200, 16, 46, 0.2), rgba(7, 22, 34, 0.55));
      }

      .nhlcal-special-event-tile.priority-high {
        border-left-color: var(--ops-cyan);
        background: linear-gradient(90deg, rgba(19, 216, 231, 0.16), rgba(7, 22, 34, 0.55));
      }

      .nhlcal-special-event-copy strong {
        -webkit-line-clamp: 1;
        max-height: 1.3em;
      }

      /* The event subtitle repeats the title in current data — hide the echo. */
      .nhlcal-special-event-copy span {
        display: none;
      }

      /* "OFF DAY" repeats up to 30 times a month; keep it as quiet ledger
         text so real events are what the eye lands on. */
      .nhlcal-day-cell .nhlcal-day-empty,
      .nhlcal-day-cell .nhlcal-off-day {
        opacity: 0.42;
      }

      @media (max-height: 900px) {
        .nhlcal-topbar {
          min-height: 0;
          gap: 8px;
        }

        .nhlcal-team-identity h1 {
          font-size: 26px;
          line-height: 1.05;
        }

        .nhlcal-stat-strip {
          margin-top: 4px;
        }

        .nhlcal-stat-pill {
          padding-block: 6px;
        }

        .nhlcal-day-cell {
          min-height: 84px;
          padding: 7px 8px;
        }

        .nhlcal-month-grid.is-dense .nhlcal-day-cell {
          min-height: 74px;
          padding: 5px 6px;
        }

        .nhlcal-week-header {
          min-height: 32px;
          padding: 5px 12px;
        }

        .nhlcal-calendar-toolbar {
          padding: 5px 10px;
        }

        .nhlcal-calendar-footer {
          min-height: 28px;
        }
      }

      @media (max-height: 780px) {
        .nhlcal-day-cell {
          min-height: 76px;
        }

        .nhlcal-team-identity h1 {
          font-size: 23px;
        }
      }

      /* ===================================================================
         CALENDAR OVERHAUL (Calendar Screen UI Bug Report, Oct 2026)
         Appended last so it wins over the legacy rules above.
         =================================================================== */

      /* C2 — type scale: labels 11.5, body 14, names 16-18, headlines 28-34. */
      .nhlcal-root {
        font-size: 14px;
      }
      .nhlcal-month-title-block h2 {
        font-family: "Archivo Black", "Barlow Condensed", "Arial Narrow", sans-serif;
        font-size: clamp(26px, 2.4vw, 34px);
        letter-spacing: 0.03em;
        line-height: 1.05;
      }
      .nhlcal-team-identity h1 {
        font-family: "Barlow Condensed", "Archivo Black", "Arial Narrow", sans-serif;
        font-size: clamp(22px, 2vw, 28px);
        letter-spacing: 0.04em;
      }
      .nhlcal-day-number {
        font-size: 15px;
        font-weight: 800;
      }
      .nhlcal-week-header > div {
        font-size: 11.5px;
        letter-spacing: 0.14em;
      }

      /* C1 / C5 / C7 — hero strip: next game + the one Advance control + schedule intel. */
      .nhlcal-hero {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        margin-top: 8px;
        padding: 14px 20px;
        border: 1px solid var(--line-strong);
        border-left: 4px solid var(--cyan);
        border-radius: 10px;
        background: linear-gradient(90deg, var(--cyan-soft), var(--panel) 55%);
      }
      .nhlcal-hero-next {
        display: flex;
        align-items: center;
        gap: 18px;
        min-width: 0;
      }
      .nhlcal-hero-copy {
        min-width: 0;
      }
      .nhlcal-hero-kicker {
        margin: 0 0 2px;
        color: var(--cyan);
        font-size: 11.5px;
        font-weight: 900;
        letter-spacing: 0.14em;
        text-transform: uppercase;
      }
      .nhlcal-hero-title {
        margin: 0;
        font-family: "Archivo Black", "Barlow Condensed", "Arial Narrow", sans-serif;
        font-size: clamp(24px, 2.3vw, 32px);
        line-height: 1.05;
        color: var(--text);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .nhlcal-hero-title span {
        color: var(--muted);
        font-size: 0.7em;
      }
      .nhlcal-hero-intel {
        display: flex;
        flex-wrap: wrap;
        gap: 6px 16px;
        margin-top: 6px;
        color: var(--muted);
        font-size: 12px;
      }
      .nhlcal-hero-intel b {
        color: var(--text);
        font-size: 15px;
        margin-right: 3px;
        font-variant-numeric: tabular-nums;
      }
      .nhlcal-hero-intel .is-hot b { color: var(--green); }
      .nhlcal-hero-intel .is-cold b { color: var(--red); }
      .nhlcal-hero-actions .nhlcal-action-primary {
        display: flex;
        align-items: center;
        gap: 10px;
      }
      .nhlcal-hero-actions .nhlcal-advance-button {
        min-height: 52px;
        padding: 0 26px;
        font-size: 16px;
        font-weight: 900;
        letter-spacing: 0.06em;
      }
      .nhlcal-hero-actions .nhlcal-advance-button-secondary {
        min-height: 52px;
        padding: 0 16px;
        font-size: 13px;
      }
      .nhlcal-team-logo.size-hero,
      .nhlcal-team-badge.size-hero {
        width: 76px;
        height: 76px;
        padding: 6px;
        border-radius: 16px;
        flex: 0 0 auto;
        font-size: 18px;
      }
      .nhlcal-team-logo.size-hero img {
        width: 100%;
        height: 100%;
        object-fit: contain;
      }

      /* C3 — bigger day cells and MUCH bigger team logos in the month grid. */
      .nhlcal-day-cell {
        min-height: 128px;
      }
      .nhlcal-month-grid.is-dense .nhlcal-day-cell {
        min-height: 112px;
      }
      .nhlcal-month-grid .nhlcal-team-logo.size-tile-main,
      .nhlcal-month-grid .nhlcal-team-badge.size-tile-main {
        width: 54px;
        height: 54px;
        padding: 3px;
        border-radius: 12px;
        font-size: 14px;
      }
      .nhlcal-month-grid .nhlcal-team-logo.size-tile-compact,
      .nhlcal-month-grid .nhlcal-team-badge.size-tile-compact {
        width: 42px;
        height: 42px;
        padding: 2px;
        border-radius: 10px;
        font-size: 12px;
      }
      .nhlcal-month-grid .nhlcal-team-logo img {
        width: 100%;
        height: 100%;
        object-fit: contain;
      }
      .nhlcal-month-grid .nhlcal-game-match-line strong {
        font-size: 12.5px;
      }

      /* C6 — day cell is now a focusable gridcell (no nested buttons). */
      .nhlcal-day-cell:focus-visible {
        outline: 2px solid var(--cyan);
        outline-offset: -2px;
      }

      /* C11 — give the grid room on laptop widths; stack the rail when narrow. */
      @media (max-width: 1440px) {
        .nhlcal-content-grid {
          grid-template-columns: minmax(0, 1fr) minmax(248px, 286px);
        }
      }
      @media (max-width: 1240px) {
        .nhlcal-content-grid {
          grid-template-columns: minmax(0, 1fr);
          overflow: visible;
        }
        .nhlcal-hero {
          flex-wrap: wrap;
        }
      }
    

/* C8: opponent colour stripe on month-grid game tiles (falls back to none). */
.nhlcal-game-tile[style*="--opp-color"] {
  border-left: 3px solid var(--opp-color);
}

/* C10: alert hierarchy — errors loudest, blocks next, lineup gaps as a quiet note. */
.nhlcal-advance-alert.is-error {
  border-left: 4px solid #ff5a6e;
}
.nhlcal-advance-alert.is-blocked {
  border-left: 4px solid #f2b84b;
}
.nhlcal-advance-alert + .nhlcal-advance-alert,
.nhlcal-advance-alert + .nhlcal-lineup-block {
  margin-top: 8px;
}
.nhlcal-lineup-block {
  opacity: 0.88;
}
`;

export function CalendarStyles() {
  return <style>{CALENDAR_CSS}</style>;
}
