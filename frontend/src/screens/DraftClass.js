import React, { useMemo, useState } from "react";
import "../styles/game-ui.css";
import { useGameUI } from "../game/GameUIContext";

const POSITIONS = ["C", "LW", "RW", "D", "G"];
const REGIONS = ["ALL PLAYERS", "NORTH AMERICA", "EUROPE", "INTERNATIONAL"];
const PROFILE_TABS = ["OVERVIEW", "STATS", "ATTRIBUTES", "SCOUT REPORT", "CHARACTER"];
const LEAGUES = ["OHL", "WHL", "QMJHL", "NCAA", "USHL", "SHL", "LIIGA", "DEL", "CZECHIA"];
const COUNTRIES = ["Canada", "United States", "Sweden", "Finland", "Czechia", "Slovakia", "Germany", "Switzerland"];
const PLAYER_TYPES = [
  "Elite Playmaker",
  "Two-Way Forward",
  "Power Forward",
  "Sniper",
  "Offensive Defenseman",
  "Two-Way Defenseman",
  "Defensive Defenseman",
  "Hybrid Goalie",
  "Butterfly Goalie",
];

const FIRST_NAMES = [
  "Adam", "Noah", "Liam", "Ethan", "Owen", "Lucas", "Mason", "Logan", "Jack", "Caleb",
  "Ryan", "Nolan", "Cole", "Wyatt", "Carter", "Tyler", "Dylan", "Evan", "Miles", "Blake",
  "Leo", "Felix", "Axel", "Anton", "Emil", "Hugo", "Lukas", "Oscar", "Mikko", "Aleksi",
  "Tomas", "Jakub", "Matej", "David", "Samuel", "Ben", "Max", "Julian", "Jonas", "Rasmus",
];

const LAST_NAMES = [
  "Andersson", "Bennett", "Carlsson", "Dubois", "Eriksson", "Fischer", "Graves", "Harrison",
  "Ivanov", "Johansson", "Keller", "Larsson", "Miller", "Novak", "Olsen", "Pettersson",
  "Quinn", "Rossi", "Sandberg", "Thompson", "Ullman", "Varga", "Wilson", "Young", "Zetterberg",
  "MacLeod", "Lavoie", "Schneider", "Koskinen", "Horak", "Sullivan", "Bouchard", "Markovic",
];

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function seededNumber(seed) {
  let value = seed;
  value = (value ^ 61) ^ (value >>> 16);
  value += value << 3;
  value ^= value >>> 4;
  value *= 0x27d4eb2d;
  value ^= value >>> 15;
  return Math.abs(value);
}

function pick(list, seed) {
  return list[seededNumber(seed) % list.length];
}

function projectionForRank(rank) {
  if (rank <= 3) return "TOP 3";
  if (rank <= 5) return "TOP 5";
  if (rank <= 10) return "TOP 10";
  if (rank <= 20) return "TOP 20";
  if (rank <= 32) return "1ST RD";
  if (rank <= 64) return "2ND RD";
  if (rank <= 96) return "3RD RD";
  return "LATE RD";
}

function talentGrade(rank, seed) {
  const swing = seededNumber(seed) % 4;
  if (rank <= 2) return swing > 1 ? "A+" : "A";
  if (rank <= 5) return swing > 1 ? "A" : "A-";
  if (rank <= 12) return swing > 1 ? "A-" : "B+";
  if (rank <= 25) return swing > 1 ? "B+" : "B";
  if (rank <= 50) return swing > 1 ? "B" : "B-";
  if (rank <= 80) return swing > 1 ? "B-" : "C+";
  return swing > 1 ? "C+" : "C";
}

function countryFlag(country) {
  const map = {
    Canada: "🇨🇦",
    "United States": "🇺🇸",
    Sweden: "🇸🇪",
    Finland: "🇫🇮",
    Czechia: "🇨🇿",
    Slovakia: "🇸🇰",
    Germany: "🇩🇪",
    Switzerland: "🇨🇭",
  };
  return map[country] || "🌐";
}

function regionForCountry(country) {
  if (country === "Canada" || country === "United States") return "NORTH AMERICA";
  if (["Sweden", "Finland", "Czechia", "Slovakia", "Germany", "Switzerland"].includes(country)) return "EUROPE";
  return "INTERNATIONAL";
}

function generateProspects(count = 96) {
  return Array.from({ length: count }, (_, i) => {
    const rank = i + 1;
    const seed = 1000 + rank * 37;
    const country = pick(COUNTRIES, seed);
    const position = rank % 13 === 0 ? "G" : pick(POSITIONS.filter((p) => p !== "G"), seed + 9);
    const league = country === "Canada"
      ? pick(["OHL", "WHL", "QMJHL", "NCAA"], seed + 4)
      : country === "United States"
      ? pick(["NCAA", "USHL", "OHL"], seed + 5)
      : pick(["SHL", "LIIGA", "DEL", "CZECHIA", "NCAA"], seed + 6);

    const type = position === "G"
      ? pick(["Hybrid Goalie", "Butterfly Goalie"], seed + 7)
      : position === "D"
      ? pick(["Offensive Defenseman", "Two-Way Defenseman", "Defensive Defenseman"], seed + 8)
      : pick(["Elite Playmaker", "Two-Way Forward", "Power Forward", "Sniper"], seed + 10);

    const gp = clamp(34 + (seededNumber(seed + 11) % 26), 34, 62);
    const goals = position === "G" ? 0 : clamp(8 + Math.floor((100 - rank) / 3) + (seededNumber(seed + 12) % 18), 3, 52);
    const assists = position === "G" ? 0 : clamp(12 + Math.floor((100 - rank) / 2.5) + (seededNumber(seed + 13) % 22), 8, 72);
    const points = goals + assists;
    const wins = position === "G" ? clamp(12 + (seededNumber(seed + 14) % 22), 8, 36) : 0;
    const savePct = position === "G" ? (0.895 + (seededNumber(seed + 15) % 35) / 1000).toFixed(3) : null;
    const gaa = position === "G" ? (1.95 + (seededNumber(seed + 16) % 90) / 100).toFixed(2) : null;

    const stockRaw = (seededNumber(seed + 17) % 47) - 16;
    const stock = rank <= 8 ? Math.abs(stockRaw) + 5 : stockRaw;
    const completion = clamp(42 + Math.floor((100 - rank) / 4) + (seededNumber(seed + 18) % 17), 35, 84);
    const heightInches = position === "G"
      ? clamp(73 + (seededNumber(seed + 19) % 6), 72, 80)
      : clamp(69 + (seededNumber(seed + 20) % 8), 68, 78);
    const weight = position === "G"
      ? clamp(178 + (seededNumber(seed + 21) % 36), 170, 225)
      : clamp(165 + (seededNumber(seed + 22) % 42), 160, 220);

    return {
      id: `prospect-${rank}`,
      rank,
      firstName: pick(FIRST_NAMES, seed + 1),
      lastName: pick(LAST_NAMES, seed + 2),
      position,
      country,
      region: regionForCountry(country),
      league,
      team: `${league} Club ${rank}`,
      playerType: type,
      projection: projectionForRank(rank),
      talent: talentGrade(rank, seed),
      completion,
      stock,
      gp,
      goals,
      assists,
      points,
      wins,
      savePct,
      gaa,
      height: `${Math.floor(heightInches / 12)}'${heightInches % 12}"`,
      weight,
      age: rank % 9 === 0 ? 19 : 18,
      handedness: rank % 2 === 0 ? "Right" : "Left",
      birthday: `June ${clamp((rank * 3) % 28, 1, 28)}, ${rank % 9 === 0 ? 2005 : 2006}`,
      morale: clamp(64 + (seededNumber(seed + 23) % 34), 45, 99),
      character: clamp(62 + (seededNumber(seed + 24) % 35), 45, 99),
      fit: clamp(58 + (seededNumber(seed + 25) % 39), 40, 99),
      compete: clamp(60 + (seededNumber(seed + 26) % 39), 40, 99),
      leadership: clamp(55 + (seededNumber(seed + 27) % 40), 35, 99),
      workEthic: clamp(62 + (seededNumber(seed + 28) % 36), 40, 99),
      coachability: clamp(60 + (seededNumber(seed + 29) % 37), 40, 99),
      consistency: clamp(50 + (seededNumber(seed + 30) % 42), 35, 99),
      poise: clamp(55 + (seededNumber(seed + 31) % 40), 35, 99),
      skating: clamp(54 + Math.floor((100 - rank) / 4) + (seededNumber(seed + 32) % 20), 45, 96),
      shooting: clamp(54 + Math.floor((100 - rank) / 4) + (seededNumber(seed + 33) % 20), 45, 96),
      passing: clamp(54 + Math.floor((100 - rank) / 4) + (seededNumber(seed + 34) % 20), 45, 96),
      defense: clamp(50 + Math.floor((100 - rank) / 5) + (seededNumber(seed + 35) % 22), 40, 96),
      physical: clamp(48 + Math.floor((100 - rank) / 5) + (seededNumber(seed + 36) % 25), 38, 96),
      hockeyIQ: clamp(55 + Math.floor((100 - rank) / 4) + (seededNumber(seed + 37) % 20), 45, 98),
    };
  });
}

function splitName(fullName) {
  const parts = String(fullName || "").trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return { firstName: "Prospect", lastName: "Player" };
  if (parts.length === 1) return { firstName: parts[0], lastName: "Player" };
  return { firstName: parts.slice(0, -1).join(" "), lastName: parts[parts.length - 1] };
}

function mapBackendDraftBoard(entries) {
  const rows = Array.isArray(entries) ? entries : [];
  if (!rows.length) return [];
  const templates = generateProspects(Math.max(96, rows.length));
  return rows.map((row, i) => {
    const rank = Number(row?.rank) || i + 1;
    const base = templates[Math.min(Math.max(rank - 1, 0), templates.length - 1)] || templates[0];
    const nm = splitName(row?.name);
    const pos = String(row?.position || base.position || "C").toUpperCase();
    const stock = String(row?.trend || "").toUpperCase() === "UP" ? Number(row?.rank_delta || 1) : String(row?.trend || "").toUpperCase() === "DOWN" ? -Number(row?.rank_delta || 1) : 0;
    return {
      ...base,
      id: String(row?.key || base.id || `prospect-${rank}`),
      rank,
      firstName: nm.firstName,
      lastName: nm.lastName,
      position: pos,
      age: Number(row?.age) || base.age,
      league: row?.league_name || row?.league_code || base.league,
      team: row?.league_name || base.team,
      projection: base.projection,
      talent: String(row?.scout_tier || base.talent || "B"),
      completion: Math.max(35, Math.min(95, Math.round(Number(row?.scout_grade) || base.completion || 60))),
      stock,
      playerType: base.playerType,
      ovrHint: Number(row?.true_ovr || 0),
    };
  });
}

function ratingLabel(value) {
  if (value >= 90) return "Elite";
  if (value >= 82) return "Excellent";
  if (value >= 74) return "Good";
  if (value >= 64) return "Average";
  return "Concern";
}

function gradeFromValue(value) {
  if (value >= 94) return "A+";
  if (value >= 88) return "A";
  if (value >= 82) return "A-";
  if (value >= 76) return "B+";
  if (value >= 70) return "B";
  if (value >= 64) return "B-";
  if (value >= 58) return "C+";
  return "C";
}

function initials(player) {
  return `${player.firstName?.[0] || ""}${player.lastName?.[0] || ""}`.toUpperCase();
}

function fullName(player) {
  return `${player.firstName} ${player.lastName}`;
}

function stockClass(stock) {
  if (stock > 0) return "draft-trend-flag--up";
  if (stock < 0) return "draft-trend-flag--down";
  return "draft-trend-flag--same";
}

function stockText(stock) {
  if (stock > 0) return `↟ +${stock}`;
  if (stock < 0) return `↡ ${stock}`;
  return "—";
}

function strengthList(player) {
  const pool = [
    player.hockeyIQ >= 78 && "High-end hockey IQ and reads pressure early",
    player.passing >= 78 && "Creates offense through seams and controlled entries",
    player.shooting >= 78 && "Dangerous release from the slot and circles",
    player.skating >= 78 && "Strong acceleration and edge control",
    player.defense >= 78 && "Reliable defensive habits away from the puck",
    player.physical >= 78 && "Competes hard on walls and around the crease",
    player.workEthic >= 78 && "High work rate with clear development habits",
    player.poise >= 78 && "Composed under pressure in late-game situations",
  ].filter(Boolean);

  return pool.length ? pool.slice(0, 5) : [
    "Projectable frame with room to develop",
    "Shows flashes of high-end processing",
    "Useful habits in transition",
  ];
}

function weaknessList(player) {
  const pool = [
    player.skating < 70 && "Needs another gear in open ice",
    player.physical < 70 && "Could add strength before NHL minutes",
    player.defense < 70 && "Defensive reads are still inconsistent",
    player.shooting < 70 && "Shot selection can be predictable",
    player.passing < 70 && "Can force plays through traffic",
    player.consistency < 70 && "Game-to-game impact can fluctuate",
    player.coachability < 70 && "Scouts want quicker adjustments after feedback",
    player.leadership < 70 && "Still developing a louder presence in the room",
  ].filter(Boolean);

  return pool.length ? pool.slice(0, 4) : [
    "Needs pro pace adjustment",
    "Could become more consistent shift-to-shift",
    "Strength gains will decide ceiling",
  ];
}

function scoutSummary(player) {
  if (player.rank <= 5) {
    return `${fullName(player)} grades as a potential franchise-level piece with high-end tools, strong detail, and a profile that should translate quickly if development stays on track.`;
  }

  if (player.rank <= 16) {
    return `${fullName(player)} projects as a top-half first-round talent with enough translatable traits to become a major NHL contributor.`;
  }

  if (player.rank <= 32) {
    return `${fullName(player)} has first-round upside, but the final projection depends on whether the weaker parts of the profile catch up to the standout tools.`;
  }

  return `${fullName(player)} is a longer-view prospect with useful traits, development variance, and enough upside to justify serious scouting attention.`;
}

function PlayerHeadshot({ player, size = "md" }) {
  return (
    <div className={`dc-headshot dc-headshot--${size}`}>
      <div className="dc-headshot__halo" />
      <div className="dc-headshot__neck" />
      <div className="dc-headshot__face">
        <div className="dc-headshot__hair" />
        <div className="dc-headshot__eyes" />
        <div className="dc-headshot__smile" />
      </div>
      <div className="dc-headshot__jersey">
        <span>{initials(player)}</span>
      </div>
    </div>
  );
}

function TopHeader() {
  return (
    <header className="dc-topbar">
      <div className="dc-brand">
        <div className="dc-draft-logo">
          <span>ENTRY</span>
          <strong>DRAFT</strong>
        </div>
        <div>
          <h1>DRAFT CLASS</h1>
          <p>Central Scouting · Prospect Intelligence</p>
        </div>
      </div>

      <div className="dc-topbar__right">
        <span className="dc-alert">UPGRADE AVAILABLE</span>
        <span className="dc-currency">◆ 2398</span>
        <span className="dc-capacity">⚙ 53/53</span>
      </div>
    </header>
  );
}

function FilterBar({ region, setRegion, query, setQuery, sortMode, setSortMode }) {
  return (
    <div className="dc-filterbar">
      <button className="dc-bumper">L2</button>

      {REGIONS.map((r) => (
        <button
          key={r}
          className={`dc-tab ${region === r ? "is-active" : ""}`}
          onClick={() => setRegion(r)}
        >
          {r}
        </button>
      ))}

      <div className="dc-search">
        <button className="dc-bumper">R2</button>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search prospect, league, country, position..."
        />
      </div>

      <select value={sortMode} onChange={(e) => setSortMode(e.target.value)} className="dc-sort">
        <option value="rank">Rank</option>
        <option value="stock">Stock</option>
        <option value="points">Points</option>
        <option value="completion">Scouting Completion</option>
        <option value="fit">Team Fit</option>
      </select>
    </div>
  );
}

function ProspectRow({ player, selected, onSelect }) {
  return (
    <button className={`dc-prospect-row ${selected ? "is-selected" : ""}`} onClick={onSelect}>
      <div className="dc-rank">{player.rank}</div>

      <div className="dc-flag">{countryFlag(player.country)}</div>

      <PlayerHeadshot player={player} size="sm" />

      <div className="dc-player-main">
        <div className="dc-player-line">
          <span className="dc-position">{player.position}</span>
          <div>
            <strong>{player.firstName}</strong>
            <b>{player.lastName}</b>
          </div>
        </div>
        <p>{player.team} ({player.league}) | {player.height} {player.weight} lbs</p>
      </div>

      <div className="dc-archetype">{player.playerType}</div>

      <div className="dc-projection">
        <strong>{player.projection}</strong>
        <span>Projection</span>
      </div>

      <div className="dc-grade">
        <strong>{player.talent}</strong>
        <span>Talent</span>
      </div>

      <div className="dc-completion">
        <strong>{player.completion}%</strong>
        <span>Completion</span>
      </div>

      <div className="dc-stock">
        <span>STOCK</span>
        <strong className={stockClass(player.stock)}>{stockText(player.stock)}</strong>
      </div>
    </button>
  );
}

function ProspectBoard({ prospects, selectedId, setSelectedId }) {
  return (
    <section className="dc-board">
      {prospects.map((player) => (
        <ProspectRow
          key={player.id}
          player={player}
          selected={player.id === selectedId}
          onSelect={() => setSelectedId(player.id)}
        />
      ))}
    </section>
  );
}

function LeagueLeaders({ prospects }) {
  const groups = useMemo(() => {
    return LEAGUES.map((league) => {
      const players = prospects
        .filter((p) => p.league === league && p.position !== "G")
        .sort((a, b) => b.points - a.points)
        .slice(0, 3);

      return { league, players };
    }).filter((g) => g.players.length);
  }, [prospects]);

  return (
    <aside className="dc-leaders">
      <div className="dc-side-title">
        <button className="dc-bumper">L2</button>
        <h2>LEAGUE LEADERS - JUNIORS</h2>
        <button className="dc-bumper">R2</button>
      </div>

      <div className="dc-leader-scroll">
        {groups.map((group) => (
          <div key={group.league} className="dc-league-card">
            <div className="dc-league-card__head">
              <strong>{group.league}</strong>
              <span>GP</span>
              <span>G</span>
              <span>A</span>
              <span>PTS</span>
            </div>

            {group.players.map((p, index) => (
              <div className="dc-league-row" key={p.id}>
                <span>{index + 1}. {p.firstName[0]}. {p.lastName}</span>
                <span>{p.gp}</span>
                <span>{p.goals}</span>
                <span>{p.assists}</span>
                <strong>{p.points}</strong>
              </div>
            ))}
          </div>
        ))}
      </div>

      <button className="dc-view-full">△ VIEW FULL LEADERS</button>
    </aside>
  );
}

function OverviewTab({ player }) {
  return (
    <div className="dc-profile-body">
      <div className="dc-profile-left">
        <PlayerHeadshot player={player} size="lg" />

        <div className="dc-profile-name">
          <span>{player.firstName}</span>
          <strong>{player.lastName}</strong>
          <p>{player.position} | {player.playerType}</p>
          <small>{countryFlag(player.country)} {player.country}</small>
          <small>{player.team} ({player.league})</small>
          <small>{player.height} | {player.weight} lbs | Shoots {player.handedness} | Age: {player.age}</small>
        </div>
      </div>

      <div className="dc-profile-card dc-draft-projection">
        <span>DRAFT PROJECTION</span>
        <strong>{player.projection}</strong>
        <p>{player.rank <= 10 ? "Franchise Potential" : player.rank <= 32 ? "NHL Upside" : "Development Prospect"}</p>
      </div>

      <div className="dc-info-card">
        <h3>PLAYER INFO</h3>
        <div className="dc-info-grid">
          <span>Birthdate</span><b>{player.birthday}</b>
          <span>Hometown</span><b>{player.country}</b>
          <span>Draft Eligible</span><b>2024</b>
          <span>Ranking</span><b>{player.rank}</b>
          <span>Position Rank</span><b>{player.position}-{player.rank}</b>
          <span>Height</span><b>{player.height}</b>
          <span>Weight</span><b>{player.weight} lbs</b>
          <span>Shoots</span><b>{player.handedness}</b>
        </div>
      </div>

      <div className="dc-list-card dc-list-card--good">
        <h3>STRENGTHS</h3>
        <ul>
          {strengthList(player).map((s) => <li key={s}>{s}</li>)}
        </ul>
      </div>

      <div className="dc-list-card dc-list-card--bad">
        <h3>WEAKNESSES</h3>
        <ul>
          {weaknessList(player).map((s) => <li key={s}>{s}</li>)}
        </ul>
      </div>

      <div className="dc-summary-card">
        <h3>SCOUT SUMMARY</h3>
        <p>{scoutSummary(player)}</p>
      </div>
    </div>
  );
}

function StatsTab({ player }) {
  const isGoalie = player.position === "G";

  return (
    <div className="dc-stat-layout">
      <div className="dc-stat-card">
        <h3>SEASON STATS</h3>
        {!isGoalie ? (
          <div className="dc-big-stat-grid">
            <div><span>GP</span><strong>{player.gp}</strong></div>
            <div><span>G</span><strong>{player.goals}</strong></div>
            <div><span>A</span><strong>{player.assists}</strong></div>
            <div><span>PTS</span><strong>{player.points}</strong></div>
            <div><span>P/GP</span><strong>{(player.points / player.gp).toFixed(2)}</strong></div>
            <div><span>STOCK</span><strong>{stockText(player.stock)}</strong></div>
          </div>
        ) : (
          <div className="dc-big-stat-grid">
            <div><span>GP</span><strong>{player.gp}</strong></div>
            <div><span>W</span><strong>{player.wins}</strong></div>
            <div><span>SV%</span><strong>{player.savePct}</strong></div>
            <div><span>GAA</span><strong>{player.gaa}</strong></div>
            <div><span>STOCK</span><strong>{stockText(player.stock)}</strong></div>
            <div><span>FIT</span><strong>{player.fit}</strong></div>
          </div>
        )}
      </div>

      <div className="dc-stat-card">
        <h3>LEAGUE CONTEXT</h3>
        <p>
          Current production places this prospect among the stronger tracked players in {player.league}.
          Scouting confidence is at {player.completion}%, meaning the range can still move as more games are logged.
        </p>
      </div>

      <div className="dc-stat-card">
        <h3>DEVELOPMENT ETA</h3>
        <div className="dc-eta">
          <strong>{player.rank <= 8 ? "1-2 YEARS" : player.rank <= 32 ? "2-3 YEARS" : "3-5 YEARS"}</strong>
          <span>
            {player.rank <= 8
              ? "Could challenge for NHL minutes quickly."
              : player.rank <= 32
              ? "Likely needs one or two years of development."
              : "Longer runway with higher variance."}
          </span>
        </div>
      </div>
    </div>
  );
}

function AttributeBar({ label, value }) {
  return (
    <div className="dc-attribute">
      <div>
        <span>{label}</span>
        <b>{value}</b>
      </div>
      <div className="dc-attribute__track">
        <div style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}

function AttributesTab({ player }) {
  return (
    <div className="dc-attributes-layout">
      <div className="dc-attribute-card">
        <h3>PLAYER ATTRIBUTES</h3>
        <AttributeBar label="Skating" value={player.skating} />
        <AttributeBar label="Shooting" value={player.shooting} />
        <AttributeBar label="Passing" value={player.passing} />
        <AttributeBar label="Defense" value={player.defense} />
        <AttributeBar label="Physical" value={player.physical} />
        <AttributeBar label="Hockey IQ" value={player.hockeyIQ} />
      </div>

      <div className="dc-attribute-card">
        <h3>SCOUTING GRADES</h3>
        <div className="dc-grade-grid">
          <div><span>Skating</span><strong>{gradeFromValue(player.skating)}</strong></div>
          <div><span>Shooting</span><strong>{gradeFromValue(player.shooting)}</strong></div>
          <div><span>Passing</span><strong>{gradeFromValue(player.passing)}</strong></div>
          <div><span>Defense</span><strong>{gradeFromValue(player.defense)}</strong></div>
          <div><span>Physical</span><strong>{gradeFromValue(player.physical)}</strong></div>
          <div><span>IQ</span><strong>{gradeFromValue(player.hockeyIQ)}</strong></div>
        </div>
      </div>
    </div>
  );
}

function ScoutReportTab({ player }) {
  return (
    <div className="dc-scout-layout">
      <div className="dc-scout-card">
        <h3>SCOUT REPORT</h3>
        <p>{scoutSummary(player)}</p>
        <p>
          The current projection is <b>{player.projection}</b> with a talent grade of <b>{player.talent}</b>.
          The biggest deciding factor is whether the player can turn strong junior habits into repeatable pro pace.
        </p>
      </div>

      <div className="dc-scout-card">
        <h3>RISK PROFILE</h3>
        <ul>
          <li>Scouting completion: {player.completion}%</li>
          <li>Development volatility: {player.rank <= 10 ? "Low" : player.rank <= 32 ? "Medium" : "High"}</li>
          <li>Projection confidence: {ratingLabel(player.completion)}</li>
          <li>Draft movement: {stockText(player.stock)}</li>
        </ul>
      </div>

      <div className="dc-scout-card">
        <h3>TEAM FIT NOTES</h3>
        <p>
          Fit score is currently <b>{player.fit}</b>. This is based on team need, player archetype,
          willingness to join the organization, and projected development timeline.
        </p>
      </div>
    </div>
  );
}

function CharacterTab({ player }) {
  const rows = [
    ["Competitiveness", "High compete level. Wants to be the difference.", player.compete],
    ["Leadership", "Teammates respond well to his habits.", player.leadership],
    ["Work Ethic", "Consistently looks to improve.", player.workEthic],
    ["Coachability", "Takes feedback and applies it.", player.coachability],
    ["Consistency", "Performance stability over long sample.", player.consistency],
    ["Poise", "Calm under pressure. Rarely rattled.", player.poise],
  ];

  return (
    <div className="dc-character-layout">
      <div className="dc-character-card">
        <h3>PERSONALITY & CHARACTER</h3>
        {rows.map(([label, note, value]) => (
          <div className="dc-character-row" key={label}>
            <span>{label}</span>
            <p>{note}</p>
            <strong>{gradeFromValue(value)}</strong>
          </div>
        ))}
      </div>

      <div className="dc-character-card">
        <h3>MORALE & FIT</h3>
        <div className="dc-fit-row"><span>Morale</span><strong>{ratingLabel(player.morale)} ☻</strong></div>
        <div className="dc-fit-row"><span>Character</span><strong>{ratingLabel(player.character)} ☻</strong></div>
        <div className="dc-fit-row"><span>Willingness To Play For You</span><strong>{ratingLabel(player.fit)} ☻</strong></div>
        <div className="dc-fit-row"><span>Market Size Fit</span><strong>{player.fit >= 72 ? "Good" : "Questionable"} ★</strong></div>
        <div className="dc-fit-row"><span>Team Need Fit</span><strong>{ratingLabel(player.fit)} ★</strong></div>
        <div className="dc-fit-row"><span>Potential Impact</span><strong>{player.rank <= 5 ? "Franchise" : player.rank <= 32 ? "Core" : "Depth"} ★</strong></div>
      </div>

      <div className="dc-character-card dc-character-card--notes">
        <h3>SCOUT NOTES</h3>
        <p>
          {fullName(player)} shows a {ratingLabel(player.character).toLowerCase()} character profile and
          a {ratingLabel(player.fit).toLowerCase()} fit score for your organization.
        </p>
      </div>

      <div className="dc-summary-rings">
        <div><strong>{player.morale}</strong><span>MORALE</span></div>
        <div><strong>{player.character}</strong><span>CHARACTER</span></div>
        <div><strong>{player.fit}</strong><span>FIT</span></div>
      </div>
    </div>
  );
}

function PlayerProfile({ player }) {
  const [tab, setTab] = useState("OVERVIEW");

  return (
    <section className="dc-profile">
      <div className="dc-profile-header">
        <h2>PLAYER PROFILE</h2>
        <div className="dc-profile-tabs">
          <button className="dc-bumper">L2</button>
          {PROFILE_TABS.map((t) => (
            <button
              key={t}
              className={`dc-profile-tab ${tab === t ? "is-active" : ""}`}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          ))}
          <button className="dc-bumper">R2</button>
        </div>
      </div>

      {tab === "OVERVIEW" && <OverviewTab player={player} />}
      {tab === "STATS" && <StatsTab player={player} />}
      {tab === "ATTRIBUTES" && <AttributesTab player={player} />}
      {tab === "SCOUT REPORT" && <ScoutReportTab player={player} />}
      {tab === "CHARACTER" && <CharacterTab player={player} />}
    </section>
  );
}

function BottomLegend() {
  return (
    <footer className="dc-bottom-legend">
      <span>✕ SELECT</span>
      <span>○ BACK</span>
      <span>△ PLAYER INFO</span>
      <span>R3 SORT</span>
      <span>L3 PLAYER COMPARISON</span>
    </footer>
  );
}

export default function DraftClass() {
  const { franchiseState } = useGameUI();
  const [region, setRegion] = useState("ALL PLAYERS");
  const [query, setQuery] = useState("");
  const [sortMode, setSortMode] = useState("rank");

  const prospects = useMemo(() => {
    const live = mapBackendDraftBoard(franchiseState?.draft_class_rankings?.entries);
    return live.length ? live : generateProspects(96);
  }, [franchiseState?.draft_class_rankings?.entries]);
  const [selectedId, setSelectedId] = useState(prospects[0]?.id);

  const filteredProspects = useMemo(() => {
    const q = query.trim().toLowerCase();

    let list = prospects.filter((p) => {
      const regionPass = region === "ALL PLAYERS" || p.region === region;
      const queryPass =
        !q ||
        fullName(p).toLowerCase().includes(q) ||
        p.country.toLowerCase().includes(q) ||
        p.league.toLowerCase().includes(q) ||
        p.position.toLowerCase().includes(q) ||
        p.playerType.toLowerCase().includes(q);

      return regionPass && queryPass;
    });

    list = [...list].sort((a, b) => {
      if (sortMode === "stock") return b.stock - a.stock;
      if (sortMode === "points") return b.points - a.points;
      if (sortMode === "completion") return b.completion - a.completion;
      if (sortMode === "fit") return b.fit - a.fit;
      return a.rank - b.rank;
    });

    return list;
  }, [prospects, region, query, sortMode]);

  const selectedPlayer =
    filteredProspects.find((p) => p.id === selectedId) ||
    prospects.find((p) => p.id === selectedId) ||
    filteredProspects[0] ||
    prospects[0];

  return (
    <div className="game-root">
      <div className="game-canvas">
        <main className="dc-screen">
          <TopHeader />

          <FilterBar
            region={region}
            setRegion={setRegion}
            query={query}
            setQuery={setQuery}
            sortMode={sortMode}
            setSortMode={setSortMode}
          />

          <div className="dc-main-grid">
            <div className="dc-left">
              <ProspectBoard
                prospects={filteredProspects}
                selectedId={selectedPlayer?.id}
                setSelectedId={setSelectedId}
              />
              <BottomLegend />
            </div>

            <LeagueLeaders prospects={prospects} />
          </div>

          {selectedPlayer && <PlayerProfile player={selectedPlayer} />}
        </main>

        <style>{`
          .dc-screen {
            position: relative;
            z-index: 2;
            width: 100%;
            height: 100%;
            padding: 10px 12px;
            display: grid;
            grid-template-rows: 44px 44px minmax(280px, 1fr) minmax(300px, 0.95fr);
            gap: 8px;
            overflow: hidden;
            color: var(--g-text);
            font-family: var(--g-font-body);
          }

          .dc-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            min-height: 0;
          }

          .dc-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 0;
          }

          .dc-brand h1 {
            margin: 0;
            font-family: var(--g-font-head);
            letter-spacing: 0.08em;
            font-size: 1.4rem;
            line-height: 1;
          }

          .dc-brand p {
            margin: 2px 0 0;
            font-size: 0.58rem;
            letter-spacing: 0.12em;
            color: var(--g-silver-dim);
            text-transform: uppercase;
          }

          .dc-draft-logo {
            width: 50px;
            height: 36px;
            border-radius: 7px;
            border: 1px solid rgba(255,255,255,0.18);
            background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(4,8,16,0.95));
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: inset 0 0 0 1px rgba(56,189,248,0.16);
          }

          .dc-draft-logo span {
            font-size: 0.44rem;
            color: #9ae66e;
            letter-spacing: 0.1em;
          }

          .dc-draft-logo strong {
            font-size: 0.62rem;
            color: #fff;
            letter-spacing: 0.06em;
          }

          .dc-topbar__right {
            display: flex;
            align-items: center;
            gap: 8px;
            font-family: var(--g-font-head);
            font-size: 0.68rem;
            letter-spacing: 0.06em;
          }

          .dc-alert {
            background: linear-gradient(180deg, #ff5a5a, #d83636);
            color: #fff;
            padding: 7px 10px;
            clip-path: polygon(0 0, 92% 0, 100% 100%, 0 100%);
          }

          .dc-currency,
          .dc-capacity {
            border: 1px solid rgba(255,255,255,0.16);
            background: rgba(0,0,0,0.32);
            padding: 7px 10px;
            border-radius: 4px;
          }

          .dc-filterbar {
            display: grid;
            grid-template-columns: 42px repeat(4, minmax(110px, 1fr)) minmax(220px, 2.6fr) 140px;
            gap: 6px;
            min-height: 0;
          }

          .dc-bumper {
            border: 1px solid rgba(255,255,255,0.18);
            background: rgba(0,0,0,0.38);
            color: var(--g-silver);
            border-radius: 5px;
            font-family: var(--g-font-head);
            font-size: 0.62rem;
            letter-spacing: 0.08em;
            height: 100%;
            min-height: 24px;
          }

          .dc-tab,
          .dc-profile-tab,
          .dc-sort {
            border: 1px solid rgba(255,255,255,0.12);
            background: linear-gradient(180deg, rgba(31,41,55,0.9), rgba(10,15,25,0.92));
            color: var(--g-silver);
            border-radius: 4px;
            font-family: var(--g-font-head);
            font-size: 0.62rem;
            letter-spacing: 0.08em;
            cursor: pointer;
          }

          .dc-tab.is-active,
          .dc-profile-tab.is-active {
            color: #07110a;
            border-color: rgba(166,255,91,0.8);
            background: linear-gradient(180deg, #a5f765, #60c936);
            box-shadow: 0 0 18px rgba(132, 255, 86, 0.25);
          }

          .dc-search {
            display: flex;
            gap: 6px;
          }

          .dc-search input {
            flex: 1;
            min-width: 0;
            border-radius: 4px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(0,0,0,0.28);
            color: var(--g-text);
            padding: 0 10px;
            font-size: 0.74rem;
          }

          .dc-sort {
            padding: 0 8px;
          }

          .dc-main-grid {
            display: grid;
            grid-template-columns: minmax(720px, 1fr) 310px;
            gap: 10px;
            min-height: 0;
            overflow: hidden;
          }

          .dc-left {
            display: flex;
            flex-direction: column;
            min-height: 0;
            overflow: hidden;
          }

          .dc-board {
            flex: 1;
            min-height: 0;
            overflow-y: auto;
            border: 1px solid rgba(120, 255, 92, 0.35);
            background: rgba(0,0,0,0.22);
          }

          .dc-prospect-row {
            width: 100%;
            min-height: 72px;
            display: grid;
            grid-template-columns: 46px 34px 74px minmax(210px, 1.6fr) 120px 96px 80px 92px 92px;
            gap: 8px;
            align-items: center;
            border: 0;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            background: rgba(2,8,14,0.58);
            color: var(--g-silver);
            text-align: left;
            cursor: pointer;
            padding: 5px 10px;
          }

          .dc-prospect-row:hover {
            background: rgba(24,37,52,0.75);
          }

          .dc-prospect-row.is-selected {
            background: linear-gradient(90deg, rgba(99,150,49,0.26), rgba(12,18,28,0.78));
            box-shadow: inset 0 0 0 1px rgba(145,255,96,0.72);
          }

          .dc-rank {
            font-family: var(--g-font-head);
            font-size: 1.65rem;
            color: #dfe5ea;
            text-align: center;
          }

          .dc-flag {
            font-size: 1.05rem;
          }

          .dc-player-main {
            min-width: 0;
          }

          .dc-player-line {
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 0;
          }

          .dc-position {
            font-family: var(--g-font-head);
            color: #dce4ec;
            width: 24px;
            text-align: center;
          }

          .dc-player-main strong {
            display: block;
            font-size: 0.78rem;
            line-height: 1;
            color: #d6dde5;
            font-weight: 700;
          }

          .dc-player-main b {
            display: block;
            font-family: var(--g-font-head);
            font-size: 1.12rem;
            line-height: 1.05;
            color: #f3f6f8;
            letter-spacing: 0.035em;
          }

          .dc-player-main p {
            margin: 2px 0 0;
            font-size: 0.62rem;
            color: var(--g-silver-dim);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          }

          .dc-archetype {
            font-family: var(--g-font-head);
            font-size: 0.74rem;
            line-height: 1.1;
            text-align: center;
            color: #d8dce2;
          }

          .dc-projection,
          .dc-grade,
          .dc-completion,
          .dc-stock {
            text-align: center;
          }

          .dc-projection strong,
          .dc-grade strong,
          .dc-completion strong {
            display: block;
            font-family: var(--g-font-head);
            font-size: 1.02rem;
            color: #f4f7fa;
          }

          .dc-projection span,
          .dc-grade span,
          .dc-completion span,
          .dc-stock span {
            display: block;
            font-size: 0.58rem;
            color: var(--g-silver-dim);
          }

          .dc-stock strong {
            font-family: var(--g-font-head);
            font-size: 0.8rem;
          }

          .dc-headshot {
            position: relative;
            margin: 0 auto;
            overflow: hidden;
            background: radial-gradient(circle at 50% 10%, rgba(82,160,210,0.26), rgba(5,10,18,0.95));
            border-bottom: 2px solid rgba(56,189,248,0.45);
          }

          .dc-headshot--sm {
            width: 60px;
            height: 62px;
            border-radius: 18px 18px 0 0;
          }

          .dc-headshot--md {
            width: 90px;
            height: 110px;
          }

          .dc-headshot--lg {
            width: 155px;
            height: 170px;
            border-radius: 22px 22px 0 0;
          }

          .dc-headshot__face {
            position: absolute;
            left: 28%;
            right: 28%;
            top: 18%;
            height: 36%;
            background: #e3b891;
            border-radius: 45% 45% 48% 48%;
            z-index: 3;
          }

          .dc-headshot__hair {
            position: absolute;
            left: -12%;
            right: -12%;
            top: -18%;
            height: 42%;
            background: #5b3924;
            border-radius: 45% 45% 25% 25%;
          }

          .dc-headshot__eyes {
            position: absolute;
            left: 26%;
            right: 26%;
            top: 48%;
            height: 5%;
            border-left: 3px solid #17202b;
            border-right: 3px solid #17202b;
          }

          .dc-headshot__smile {
            position: absolute;
            left: 38%;
            right: 38%;
            bottom: 18%;
            height: 4px;
            border-bottom: 2px solid rgba(60,20,20,0.55);
            border-radius: 50%;
          }

          .dc-headshot__neck {
            position: absolute;
            left: 43%;
            right: 43%;
            top: 50%;
            height: 14%;
            background: #d2a27e;
            z-index: 2;
          }

          .dc-headshot__jersey {
            position: absolute;
            left: 15%;
            right: 15%;
            bottom: -6%;
            height: 42%;
            background: linear-gradient(160deg, #0f2b47, #07111d);
            border-radius: 40% 40% 8% 8%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: rgba(255,255,255,0.7);
            font-family: var(--g-font-head);
            font-size: 0.76rem;
          }

          .dc-headshot__halo {
            position: absolute;
            inset: -30%;
            background: radial-gradient(circle, rgba(96,165,250,0.22), transparent 60%);
          }

          .dc-leaders {
            min-height: 0;
            display: flex;
            flex-direction: column;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(0,0,0,0.3);
            padding: 8px;
            overflow: hidden;
          }

          .dc-side-title {
            display: grid;
            grid-template-columns: 34px 1fr 34px;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
          }

          .dc-side-title h2 {
            margin: 0;
            text-align: center;
            font-family: var(--g-font-head);
            font-size: 1rem;
            letter-spacing: 0.08em;
          }

          .dc-leader-scroll {
            flex: 1;
            min-height: 0;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 6px;
          }

          .dc-league-card {
            border: 1px solid rgba(255,255,255,0.14);
            background: rgba(6,12,18,0.82);
            padding: 6px;
          }

          .dc-league-card__head,
          .dc-league-row {
            display: grid;
            grid-template-columns: 1fr 34px 28px 28px 34px;
            gap: 4px;
            align-items: center;
          }

          .dc-league-card__head {
            font-family: var(--g-font-head);
            color: var(--g-silver);
            font-size: 0.65rem;
            border-bottom: 1px solid rgba(255,255,255,0.12);
            padding-bottom: 3px;
          }

          .dc-league-card__head strong {
            color: #38bdf8;
            font-size: 1rem;
          }

          .dc-league-row {
            font-size: 0.66rem;
            padding: 3px 0;
            color: #d5dbe2;
          }

          .dc-league-row span:not(:first-child),
          .dc-league-row strong {
            text-align: right;
            font-variant-numeric: tabular-nums;
          }

          .dc-league-row strong {
            color: #f6e58d;
          }

          .dc-view-full {
            margin-top: 7px;
            height: 28px;
            border: 1px solid rgba(255,255,255,0.14);
            background: rgba(8,14,22,0.9);
            color: var(--g-silver);
            font-family: var(--g-font-head);
            font-size: 0.62rem;
            letter-spacing: 0.08em;
          }

          .dc-bottom-legend {
            height: 26px;
            display: flex;
            align-items: center;
            gap: 18px;
            font-family: var(--g-font-head);
            font-size: 0.58rem;
            color: var(--g-silver);
            padding: 0 8px;
          }

          .dc-profile {
            min-height: 0;
            overflow: hidden;
            border-top: 4px solid rgba(180,190,220,0.95);
            background: linear-gradient(145deg, rgba(5,12,20,0.96), rgba(15,18,29,0.94));
            padding: 8px 12px;
            display: flex;
            flex-direction: column;
          }

          .dc-profile-header {
            display: grid;
            grid-template-columns: 220px 1fr;
            gap: 12px;
            align-items: center;
            margin-bottom: 8px;
          }

          .dc-profile-header h2 {
            margin: 0;
            font-family: var(--g-font-head);
            font-size: 1.1rem;
            letter-spacing: 0.08em;
          }

          .dc-profile-tabs {
            display: grid;
            grid-template-columns: 34px repeat(5, minmax(86px, 1fr)) 34px;
            gap: 4px;
          }

          .dc-profile-tab {
            height: 28px;
            font-size: 0.55rem;
          }

          .dc-profile-body {
            flex: 1;
            min-height: 0;
            display: grid;
            grid-template-columns: 220px minmax(170px, 0.7fr) minmax(220px, 1fr) minmax(220px, 1fr);
            grid-template-rows: 1fr 1fr 70px;
            gap: 8px;
            overflow: hidden;
          }

          .dc-profile-left {
            grid-row: 1 / 3;
            display: flex;
            gap: 12px;
            align-items: flex-start;
          }

          .dc-profile-name {
            min-width: 0;
          }

          .dc-profile-name span {
            display: block;
            font-family: var(--g-font-head);
            font-size: 0.82rem;
            color: #dbe2e9;
          }

          .dc-profile-name strong {
            display: block;
            font-family: var(--g-font-head);
            font-size: 1.35rem;
            line-height: 1;
          }

          .dc-profile-name p {
            margin: 4px 0;
            font-size: 0.82rem;
            font-weight: 700;
          }

          .dc-profile-name small {
            display: block;
            margin-top: 3px;
            color: var(--g-silver-dim);
            font-size: 0.64rem;
          }

          .dc-profile-card,
          .dc-info-card,
          .dc-list-card,
          .dc-summary-card,
          .dc-stat-card,
          .dc-attribute-card,
          .dc-scout-card,
          .dc-character-card {
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(12,20,28,0.72);
            padding: 10px;
            overflow: hidden;
          }

          .dc-draft-projection {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
          }

          .dc-draft-projection span {
            font-family: var(--g-font-head);
            font-size: 0.66rem;
            color: var(--g-silver-dim);
          }

          .dc-draft-projection strong {
            font-family: var(--g-font-head);
            font-size: 2.5rem;
            line-height: 1;
          }

          .dc-draft-projection p {
            margin: 4px 0 0;
            font-size: 0.7rem;
          }

          .dc-info-card {
            grid-row: 2 / 3;
          }

          .dc-info-card h3,
          .dc-list-card h3,
          .dc-summary-card h3,
          .dc-stat-card h3,
          .dc-attribute-card h3,
          .dc-scout-card h3,
          .dc-character-card h3 {
            margin: 0 0 8px;
            font-family: var(--g-font-head);
            font-size: 0.75rem;
            letter-spacing: 0.08em;
            color: #dce3ea;
          }

          .dc-info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 3px 8px;
            font-size: 0.62rem;
          }

          .dc-info-grid span {
            color: var(--g-silver-dim);
          }

          .dc-info-grid b {
            text-align: right;
          }

          .dc-list-card ul,
          .dc-scout-card ul {
            margin: 0;
            padding-left: 18px;
            font-size: 0.7rem;
            line-height: 1.6;
          }

          .dc-list-card--good h3 {
            color: #82f04f;
          }

          .dc-list-card--bad h3 {
            color: #ff5252;
          }

          .dc-summary-card {
            grid-column: 1 / -1;
          }

          .dc-summary-card p,
          .dc-scout-card p,
          .dc-stat-card p {
            margin: 0;
            font-size: 0.74rem;
            color: var(--g-silver);
            line-height: 1.45;
          }

          .dc-stat-layout,
          .dc-attributes-layout,
          .dc-scout-layout,
          .dc-character-layout {
            flex: 1;
            min-height: 0;
            display: grid;
            gap: 10px;
            overflow: hidden;
          }

          .dc-stat-layout {
            grid-template-columns: 1.2fr 1fr 1fr;
          }

          .dc-big-stat-grid,
          .dc-grade-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
          }

          .dc-big-stat-grid div,
          .dc-grade-grid div {
            background: rgba(0,0,0,0.25);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 10px;
            text-align: center;
          }

          .dc-big-stat-grid span,
          .dc-grade-grid span {
            display: block;
            font-size: 0.58rem;
            color: var(--g-silver-dim);
            letter-spacing: 0.08em;
          }

          .dc-big-stat-grid strong,
          .dc-grade-grid strong {
            display: block;
            font-family: var(--g-font-head);
            font-size: 1.5rem;
            color: #f4f7fb;
          }

          .dc-eta strong {
            display: block;
            font-family: var(--g-font-head);
            font-size: 1.7rem;
            color: #9af765;
          }

          .dc-eta span {
            display: block;
            margin-top: 8px;
            color: var(--g-silver-dim);
            font-size: 0.75rem;
            line-height: 1.45;
          }

          .dc-attributes-layout {
            grid-template-columns: 1.2fr 0.8fr;
          }

          .dc-attribute {
            margin-bottom: 10px;
          }

          .dc-attribute > div:first-child {
            display: flex;
            justify-content: space-between;
            font-size: 0.72rem;
            margin-bottom: 4px;
          }

          .dc-attribute__track {
            height: 10px;
            border-radius: 999px;
            background: rgba(0,0,0,0.5);
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
          }

          .dc-attribute__track div {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #38bdf8, #9af765);
          }

          .dc-scout-layout {
            grid-template-columns: 1.3fr 0.85fr 0.85fr;
          }

          .dc-character-layout {
            grid-template-columns: 1.2fr 1fr;
            grid-template-rows: 1fr 0.75fr;
          }

          .dc-character-row {
            display: grid;
            grid-template-columns: 120px 1fr 36px;
            gap: 10px;
            align-items: center;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            padding: 7px 0;
          }

          .dc-character-row span,
          .dc-fit-row span {
            font-weight: 700;
            font-size: 0.72rem;
          }

          .dc-character-row p {
            margin: 0;
            font-size: 0.68rem;
            color: var(--g-silver-dim);
          }

          .dc-character-row strong {
            font-family: var(--g-font-head);
            color: #eaf2fa;
          }

          .dc-fit-row {
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            padding: 8px 0;
            font-size: 0.72rem;
          }

          .dc-fit-row strong {
            color: #7df24c;
          }

          .dc-character-card--notes {
            min-height: 0;
          }

          .dc-character-card--notes p {
            font-size: 0.74rem;
            line-height: 1.45;
          }

          .dc-summary-rings {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            align-items: center;
            justify-items: center;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(12,20,28,0.72);
            padding: 12px;
          }

          .dc-summary-rings div {
            width: 82px;
            height: 82px;
            border-radius: 50%;
            border: 7px solid #69df4b;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: rgba(0,0,0,0.25);
          }

          .dc-summary-rings strong {
            font-family: var(--g-font-head);
            font-size: 1.5rem;
          }

          .dc-summary-rings span {
            font-size: 0.52rem;
            color: var(--g-silver-dim);
          }

          .draft-trend-flag--up {
            color: #6cf065 !important;
          }

          .draft-trend-flag--down {
            color: #ff6464 !important;
          }

          .draft-trend-flag--same {
            color: var(--g-silver-dim) !important;
          }

          @media (max-width: 1200px) {
            .dc-screen {
              grid-template-rows: 44px 84px minmax(300px, 1fr) minmax(320px, 1fr);
            }

            .dc-filterbar {
              grid-template-columns: 42px repeat(2, 1fr);
              grid-auto-rows: 38px;
            }

            .dc-search {
              grid-column: 1 / -2;
            }

            .dc-main-grid {
              grid-template-columns: 1fr;
            }

            .dc-leaders {
              display: none;
            }

            .dc-prospect-row {
              grid-template-columns: 42px 30px 58px minmax(200px,1fr) 100px 80px 60px 80px 80px;
            }
          }
        `}</style>
      </div>
    </div>
  );
}