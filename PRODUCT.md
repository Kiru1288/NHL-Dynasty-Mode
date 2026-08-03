# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary users are NHL / franchise-management fans who want a richer season calendar, storylines, and systems sandbox than console franchise modes. They play as a general manager: building a contender, living the league calendar, and making roster, cap, draft, and scouting decisions that feel like a real season job.

## Product Purpose

NHL Franchise Mode is a browser GM simulation. Players run an NHL franchise day-by-day through seasons—trades, salary cap, entry draft, scouting, chemistry, storylines, and cinematic league events—backed by a custom SimEngine.

Success means the season rhythm feels credible: advance the calendar, manage the club, and react to events and storylines so planning and immersion are rewarded.

## Positioning

A custom web franchise sim with systems depth and cinematic calendar moments—not a licensed EA NHL clone. The differentiator is owning the full GM loop (sim fidelity + event presentation + office hub) in one playable web product.

## Operating Context

- Solo browser play: React frontend + local/backend franchise API + SimEngine.
- Home base is the GM office / hub; players navigate into roster, trades, draft, scouting, cap, league ops, calendar, and stats tools.
- Season play advances through NHL calendar phases (regular season, playoffs, offseason stages).
- Notable rituals: draft lottery, entry draft, World Juniors, trade deadline, free agency, awards, training camp, and other cinematic event menus.

## Capabilities and Constraints

Confirmed functionality includes franchise setup, day/season advance, trades, contracts/cap, draft lottery and entry draft, scouting, chemistry/lines, storylines/decisions, league operations, and cinematic offseason/in-season event surfaces.

Technical shape: web app (`frontend/` CRA React), Python franchise API (`backend/`), simulation rules in SimEngine. Live playable franchise API runs through `backend/services/franchise_sim.py` and related services.

Terminology to preserve: GM, franchise, salary cap / cap hit / cap space, ELC, draft picks/rights, AHL vs NHL roster context, waivers, storylines, scouting, chemistry, calendar phases.

Open / undecided: commercial distribution, licensing posture beyond “unofficial fan project,” multiplayer, and any formal accessibility standard beyond ordinary web usability.

## Brand Commitments

Product name: **NHL Franchise Mode** (unofficial fan project; not an NHL/EA licensed product). Voice should read as a serious GM sim with seasonal spectacle—not toy UI or generic sports-marketing copy. Binding visual direction is not set here; incumbent UI in `frontend/` is evidence for later design work.

## Evidence on Hand

- Playable frontend screens and cinematic event menus under `frontend/src/`
- Franchise/sim documentation and audits under `docs/` and root `*_REPORT.md` / `*_AUDIT.md` files
- SimEngine franchise orchestration notes in `SimEngine/app/sim_engine/franchise/README.md`

Do not fabricate testimonials, press, customer logos, benchmarks, pricing, or licensing claims.

## Product Principles

1. **Season job first** — Every surface should reinforce living the NHL calendar as a GM, not browsing a disconnected dashboard.
2. **Systems must stay credible** — Cap, trades, draft, contracts, and scouting outcomes are product truth; UI must not paper over broken or simplified rules.
3. **Cinematic moments earn their place** — Lottery, draft, WJC, awards, deadline nights, and storylines are first-class rituals, not chrome around tables.
4. **Office hub is home** — Return players to a clear GM home base between tools and events.
5. **Don’t invent authority** — No fake NHL/EA endorsement, press, or social proof; keep the unofficial fan-project honesty.

## Accessibility & Inclusion

No product-specific accessibility standard was established. Default to solid web accessibility (keyboard, contrast, readable type) unless a stronger requirement is confirmed later.
