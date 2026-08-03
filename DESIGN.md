---
name: NHL Franchise Mode
description: Dark GM war-room UI — office sanctum home, cyan broadcast ops boards, cinematic season nights
colors:
  rink-cyan: "#13d8e7"
  rink-cyan-soft: "rgba(19, 216, 231, 0.13)"
  deadline-gold: "#e9a83c"
  deadline-gold-soft: "rgba(233, 168, 60, 0.14)"
  office-brass: "#c9a86a"
  arena-orange: "#e07020"
  franchise-violet: "#7b3eb0"
  ice-neon: "#38bdf8"
  nhl-signal-red: "#c8102e"
  void-teal: "#04101a"
  void-teal-deep: "#020a11"
  void-navy: "#03050c"
  office-charcoal: "#101218"
  panel-ops: "rgba(9, 25, 38, 0.94)"
  panel-ops-2: "rgba(12, 35, 52, 0.94)"
  panel-shell: "rgba(26, 36, 61, 0.55)"
  line-ops: "rgba(156, 218, 236, 0.14)"
  line-ops-strong: "rgba(73, 231, 240, 0.5)"
  office-line: "rgba(255, 255, 255, 0.08)"
  text-ice: "#e9f7fb"
  text-shell: "#eef0f5"
  text-office: "#ece8e0"
  muted-ops: "#8096a8"
  muted-shell: "#8b93a8"
  muted-office: "rgba(220, 216, 208, 0.58)"
  status-green: "#52df94"
  status-red: "#ff606d"
  status-blue: "#8ab4ff"
typography:
  display:
    fontFamily: "Archivo Black, Rajdhani, Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "1.35rem"
    fontWeight: 400
    lineHeight: 1.1
    letterSpacing: "0.03em"
  headline:
    fontFamily: "Chakra Petch, Arial Black, sans-serif"
    fontSize: "1.1rem"
    fontWeight: 700
    lineHeight: 1.15
    letterSpacing: "0.04em"
  title:
    fontFamily: "Inter, ui-sans-serif, system-ui, Segoe UI, sans-serif"
    fontSize: "0.95rem"
    fontWeight: 800
    lineHeight: 1.25
    letterSpacing: "0.08em"
  body:
    fontFamily: "Inter, IBM Plex Sans, Segoe UI, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.45
    letterSpacing: "normal"
  label:
    fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.72rem"
    fontWeight: 900
    lineHeight: 1.2
    letterSpacing: "0.14em"
rounded:
  ops: "2px"
  hud: "4px"
  control: "6px"
  card: "8px"
  panel: "10px"
  panel-lg: "12px"
  pill: "999px"
spacing:
  1: "4px"
  2: "8px"
  3: "12px"
  4: "16px"
  5: "24px"
  6: "32px"
components:
  button-ops-primary:
    backgroundColor: "{colors.rink-cyan}"
    textColor: "{colors.void-teal}"
    rounded: "{rounded.control}"
    padding: "10px 16px"
    typography: "{typography.label}"
  button-ops-primary-hover:
    backgroundColor: "{colors.ice-neon}"
    textColor: "{colors.void-teal}"
  button-ops-ghost:
    backgroundColor: "rgba(12, 31, 47, 0.72)"
    textColor: "{colors.text-ice}"
    rounded: "{rounded.pill}"
    padding: "0.45rem 0.9rem"
  button-ops-ghost-hover:
    backgroundColor: "{colors.rink-cyan-soft}"
    textColor: "{colors.text-ice}"
  button-shell-accent:
    backgroundColor: "{colors.arena-orange}"
    textColor: "{colors.text-shell}"
    rounded: "{rounded.control}"
    padding: "8px 14px"
  button-legacy-primary:
    backgroundColor: "{colors.nhl-signal-red}"
    textColor: "#ffffff"
    rounded: "{rounded.ops}"
    padding: "0.55rem 1.1rem"
  panel-ops:
    backgroundColor: "{colors.panel-ops}"
    textColor: "{colors.text-ice}"
    rounded: "{rounded.card}"
    padding: "{spacing.4}"
  panel-office-hud:
    backgroundColor: "rgba(8, 10, 14, 0.94)"
    textColor: "{colors.text-office}"
    rounded: "{rounded.hud}"
    padding: "12px 16px"
  nav-ops-active:
    backgroundColor: "{colors.rink-cyan-soft}"
    textColor: "{colors.rink-cyan}"
    rounded: "{rounded.control}"
  chip-live:
    backgroundColor: "rgba(19, 216, 231, 0.17)"
    textColor: "{colors.rink-cyan}"
    rounded: "{rounded.pill}"
    padding: "4px 10px"
    typography: "{typography.label}"
---

# Design System: NHL Franchise Mode

## Overview

**Creative North Star: "The GM War Room"**

This product looks and feels like the private command center of an NHL general manager: a dark sanctum you return to, then luminous ops boards and broadcast nights when the league calendar demands attention. The visual system is intentionally multi-register—not one flat theme—because the season job itself switches between sitting in the office, working dense tools, and living cinematic events.

Home is the **Office Sanctum**: warm charcoal, brass gold, condensed display type, glass HUD cards over a 3D office. Tools and many events speak **Broadcast Ops**: deep teal void, rink cyan accents, deadline gold highlights, hairline cyan rules, Inter UI. A third **Franchise Shell** register (navy, violet, arena orange, ice neon; Chakra Petch + IBM Plex) still wraps the app canvas and some hub chrome. Legacy NHL signal red remains available for older menu primitives but is not the default voice of new work.

Atmosphere is **glass over the void**: depth comes from stacked dark tones, frosted panels, soft inset highlights, and structural drop shadows—not from bright flat cards. Dense tables stay flatter; HUD cards and event boards lift. Reject light “sports marketing” dashboards, purple-on-white AI defaults, and toy/cartoon UI that breaks season-job seriousness.

**Key Characteristics:**
- Multi-register dark system: Office Sanctum → Broadcast Ops → Franchise Shell
- Cyan + gold as the ops/event accent pair; brass gold for office home
- Glass panels, radial void glows, film grain / noise at very low opacity
- Condensed uppercase labels; heavy tracking on phase/meta text
- Sharp-to-soft radius ladder: 2px ops → 4px HUD → 8–12px cards → pills for chips

## Colors

A night-side palette: near-black voids, translucent teal panels, and scarce luminous accents that mark live status, money, and ceremony.

### Primary
- **Rink Cyan** (`colors.rink-cyan`): Default accent for Broadcast Ops navigation, focus rings, live chips, and primary actions on calendar/event boards. Soft wash (`rink-cyan-soft`) tints selected rows and hover fills.

### Secondary
- **Deadline Gold** (`colors.deadline-gold`): Ceremonial / high-stakes highlight on ops boards (lottery energy, awards, featured metrics). Soft wash for gold-tinted panels.
- **Office Brass** (`colors.office-brass`): Sanctum accent—HUD edge lines, selected office chrome, warm highlights over the 3D hub. Not interchangeable with Deadline Gold in ops screens.

### Tertiary
- **Arena Orange** (`colors.arena-orange`): Franchise Shell header underline and shell CTAs.
- **Franchise Violet** (`colors.franchise-violet`): Shell secondary accent / glow companion to orange.
- **Ice Neon** (`colors.ice-neon`): Shell hover/focus neon; also a brighter cyan sibling usable for emphasis.
- **NHL Signal Red** (`colors.nhl-signal-red`): Legacy primary button accent (`theme.css` / `retro.css`). Prefer cyan/brass for new GM War Room work unless matching an existing red control.

### Neutral
- **Void Teal / Void Teal Deep** (`void-teal`, `void-teal-deep`): Ops and cinematic event page grounds.
- **Void Navy** (`void-navy`): Franchise Shell body ground (`game-ui.css`).
- **Office Charcoal** (`office-charcoal`): Sanctum ground.
- **Panel Ops / Panel Ops 2** (`panel-ops`, `panel-ops-2`): Primary translucent board surfaces.
- **Panel Shell** (`panel-shell`): Glassier shell panels.
- **Line Ops / Line Ops Strong** (`line-ops`, `line-ops-strong`): Hairline structure and strong focus borders.
- **Office Line** (`office-line`): Quiet sanctum borders.
- **Text Ice / Text Shell / Text Office**: Register-specific primary text.
- **Muted Ops / Muted Shell / Muted Office**: Secondary labels and meta.
- **Status Green / Red / Blue**: Outcome and signal colors on boards (`status-green`, `status-red`, `status-blue`).

### Named Rules
**The Scarce Light Rule.** Luminous accents (cyan, gold, brass, orange) should read as signals on a dark field—never as large flooded backgrounds. Soft washes exist so fills stay translucent.

**The Register Rule.** Do not mash Office Brass chrome onto a Broadcast Ops board, or Franchise Violet into the 3D office HUD, without an intentional scene transition. Each register keeps its accent family.

## Typography

**Display Font:** Archivo Black (via `--font-motion-control`; fallbacks Rajdhani / Barlow Condensed / Arial Narrow) — office hub and condensed franchise titles  
**Headline Font:** Chakra Petch — Franchise Shell headers and game chrome  
**Body / Ops Font:** Inter (ops, calendar, many events) with IBM Plex Sans on the shell body  
**Label Font:** Inter at heavy weight, wide tracking, often uppercase  

**Character:** Condensed sports-broadcast authority meeting a serious GM desk. Display type is blocky and athletic; ops UI is tight Inter with uppercase phase labels; shell chrome is angular Chakra Petch.

### Hierarchy
- **Display** (Archivo Black, ~1.1–1.4rem+, tracking ~0.03em): Office hub titles, sanctum identity.
- **Headline** (Chakra Petch 600–700): Shell brands, section chrome in `game-ui`.
- **Title** (Inter 800, tracked uppercase ~0.08em): Board section headers, event titles.
- **Body** (Inter / IBM Plex 400–600, ~0.875rem): Tables, descriptions, feed copy. Prefer dense readable lines on dark panels.
- **Label** (Inter 900, ~0.72rem, letter-spacing ~0.14em, uppercase): Phase chips, column headers, meta kicker text.

### Named Rules
**The Phase Label Rule.** Calendar phase, live status, and event kickers are uppercase tracked labels—not sentence-case chips. Their typography is part of the broadcast language.

**The Office Type Rule.** Inside `.office-hub`, inherit Archivo Black / motion-control; do not silently switch to Inter for primary HUD chrome.

## Layout

Full-viewport app shell (`100vw` / `100dvh`, `overflow: hidden` on root). The office hub is a single immersive stage with absolute HUD overlays. Broadcast Ops screens commonly use a **narrow left rail + main board** (about `94px` sidebar in `nhlcalShell`) with dense grid workspaces. Spacing rhythm clusters on **4 / 8 / 12 / 16 / 24 / 32px** (`spacing.1`–`6`; shell also uses 12 / 16 / 24 as `--g-space-*`).

Density is high by default: tables, strips, and multi-column boards are the norm for Operate surfaces. Cinematic events may open up for ceremony but still sit on the same void gradients. Responsive behavior is mostly collapse/stack breakpoints in the **900–1100px** band (plus tighter phone collapses ~560–680px); design for desktop war-room first.

### Named Rules
**The Home Base Rule.** After tools and events, return visual gravity to the office sanctum—not to a generic dashboard home.

## Elevation & Depth

**Glass over the void.** Depth is a hybrid of tonal stacking and structural shadow. Page grounds use layered linear + radial gradients (cyan/gold glows at low opacity). Panels are translucent dark glass with hairline borders; office HUD cards add backdrop blur (~18px) and deep drops. Franchise shell glass uses blur (~10–12px) plus inset top highlights. Dense data rows stay flatter; interactive cards and event boards lift.

### Shadow Vocabulary
- **Board Lift** (`box-shadow: 0 24px 70px rgba(0, 0, 0, 0.42)`): Ops / cinematic panels (`--shadow` in nhlcal / events).
- **HUD Lift** (`0 18px 42px rgba(0, 0, 0, 0.48)` plus inset `0 1px 0 rgba(255,255,255,0.04)`): Office HUD cards.
- **Shell Glass** (`0 12px 40px rgba(0, 0, 0, 0.45)` plus inset highlight): `.hub-glass`.
- **Cyan Active Glow** (`0 0 22px rgba(19, 216, 231, 0.8)`): Active nav rail indicator.

### Named Rules
**The Flat-Table Rule.** Data grids and long lists stay relatively flat; reserve heavy shadows for HUD cards, modal boards, and featured event panels.

**The Grain Whisper Rule.** Film-grain / noise overlays stay around ~3–5% opacity. Atmosphere, not texture wallpaper.

## Shapes

Form language is **rectilinear broadcast geometry** with a short radius ladder: near-sharp ops controls (`2px`), office HUD (`4px`), shell controls (`6px`), schedule/snap cards (`8–10px`), larger panels (`10–12px`), and true pills (`999px`) for live chips and some ghost buttons. Borders are hairline cyan/white at low alpha—not thick ornamental frames. Circular marks appear for team logos and schedule tile avatars. Avoid large soft “app card” radii (16px+) on primary boards unless a specific event skin already owns that silhouette.

### Named Rules
**The Radius Ladder Rule.** Pick radius from the register’s step; don’t invent a new corner language per screen.

## Components

Controls read **condensed, uppercase, confident**—sharp on ops boards, slightly softer on hub cards, with restrained motion (border/background shifts; occasional 1px lift or scale on interactive shell elements).

### Buttons
- **Shape:** Ops primary ~6px; ghost often pill; legacy retro buttons near-square with bright border.
- **Ops Primary:** Rink Cyan fill, void-teal text, heavy tracked label.
- **Ops Ghost:** Translucent panel fill, hairline border; hover adds cyan wash and slight lift.
- **Shell Accent:** Arena Orange on Franchise Shell chrome.
- **Legacy Primary:** NHL Signal Red (`ui-btn--primary`) for older menu paths.
- **Hover / Focus:** Brighten fill or strengthen cyan border; focus-visible often `2px` cyan outline. Prefer focus rings over vague glow-only states.

### Chips
- **Style:** Pill live/status chips with cyan soft fill and cyan text; gold soft variants for ceremonial tags.
- **State:** Active nav uses cyan wash + 3px cyan rail glow; inactive stays muted.

### Cards / Containers
- **Corner Style:** 8–12px on hub/ops cards; 4px on office HUD.
- **Background:** `panel-ops` / glass blacks; never light paper.
- **Shadow Strategy:** Board Lift / HUD Lift per Elevation.
- **Border:** `line-ops` or `office-line`; strong cyan only for focus/active.
- **Internal Padding:** commonly 12–16px (`spacing.3`–`4`).

### Inputs / Fields
- **Style:** Dark translucent fields, hairline borders, Inter text; match surrounding register.
- **Focus:** Cyan outline / border shift to `line-ops-strong`.
- **Error / Disabled:** Status red soft wash for errors; reduced opacity (~0.45) when disabled (legacy buttons).

### Navigation
- **Office:** Spatial / object-driven hub (3D office) with HUD cards—not a dense left rail.
- **Ops:** Narrow icon rail; active item cyan text + left glow bar.
- **Shell:** Sidebar/header links with 6px radius, selected violet/orange-tinted states in `game-ui`.

### Signature: Cinematic Event Shell
Shared event language (via `nhlcalShell` / `buildCinematicCss`): void-teal gradient stage, topbar with uppercase phase text, ghost/leave actions, spotlight vignette, and board lift panels. New season events should extend this shell rather than inventing a fourth global palette.

### Signature: Office HUD Card
Frosted dark card, brass hairline accent bar, deep shadow, Archivo Black labels—overlaid on the 3D office without turning the viewport into a dashboard grid.

## Do's and Don'ts

### Do:
- **Do** treat the Office Sanctum as home and Broadcast Ops as the default language for calendar/tools/events.
- **Do** keep accents scarce on dark voids; use soft washes for fills.
- **Do** use uppercase tracked labels for phase, live, and meta chrome.
- **Do** climb the radius ladder (2 → 4 → 6 → 8–12 → pill) instead of inventing corners.
- **Do** lift HUD cards and event boards; keep dense tables flatter.
- **Do** preserve existing team logos, real NHL abbreviations, and sim terminology in UI chrome.

### Don't:
- **Don't** flood screens with purple/orange shell glow inside the office sanctum or cyan boards.
- **Don't** introduce light-mode paper backgrounds or large white cards as the default surface.
- **Don't** replace Archivo Black office HUD type with Inter (or Inter ops boards with Cookie/display novelty fonts) without a scene reason.
- **Don't** use heavy grain, neon everything, or emoji as decoration.
- **Don't** fabricate broadcast network branding, NHL/EA logos as product identity, or fake endorsements in chrome.
- **Don't** flatten cinematic events into ordinary CRUD pages—ceremony is part of the system.
