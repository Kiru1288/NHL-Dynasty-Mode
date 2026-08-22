import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { getStroke } from "perfect-freehand";

import {
  useGameUI,
  HUB_WARMUP_STAGES,
  HUB_WARMUP_LABELS,
} from "../game/GameUIContext";
import {
  buildDefaultFranchiseTeamList,
  teamNameToNhlAbbr,
} from "../game/constants";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";
import { ClubBallBoard } from "./setupClubBalls";

const APP_STAGE = Object.freeze({
  CONFIGURE: "configure",
  STARTING: "starting",
});

const EASTERN_ORDER = [
  "BOS", "BUF", "DET", "FLA", "MTL", "OTT", "TBL", "TOR",
  "CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH",
];

const WESTERN_ORDER = [
  "UTA", "ANA", "CGY", "CHI", "COL", "DAL", "EDM", "LAK",
  "MIN", "NSH", "SEA", "SJS", "STL", "VAN", "VGK", "WPG",
];

const DEFAULT_TEAM_ORDER = [...EASTERN_ORDER, ...WESTERN_ORDER];

const DIVISION_BY_CODE = {
  BOS: ["Eastern", "Atlantic"],
  BUF: ["Eastern", "Atlantic"],
  DET: ["Eastern", "Atlantic"],
  FLA: ["Eastern", "Atlantic"],
  MTL: ["Eastern", "Atlantic"],
  OTT: ["Eastern", "Atlantic"],
  TBL: ["Eastern", "Atlantic"],
  TOR: ["Eastern", "Atlantic"],
  CAR: ["Eastern", "Metropolitan"],
  CBJ: ["Eastern", "Metropolitan"],
  NJD: ["Eastern", "Metropolitan"],
  NYI: ["Eastern", "Metropolitan"],
  NYR: ["Eastern", "Metropolitan"],
  PHI: ["Eastern", "Metropolitan"],
  PIT: ["Eastern", "Metropolitan"],
  WSH: ["Eastern", "Metropolitan"],
  CHI: ["Western", "Central"],
  COL: ["Western", "Central"],
  DAL: ["Western", "Central"],
  MIN: ["Western", "Central"],
  NSH: ["Western", "Central"],
  STL: ["Western", "Central"],
  UTA: ["Western", "Central"],
  WPG: ["Western", "Central"],
  ANA: ["Western", "Pacific"],
  CGY: ["Western", "Pacific"],
  EDM: ["Western", "Pacific"],
  LAK: ["Western", "Pacific"],
  SEA: ["Western", "Pacific"],
  SJS: ["Western", "Pacific"],
  VAN: ["Western", "Pacific"],
  VGK: ["Western", "Pacific"],
};

function clubConference(code) {
  return DIVISION_BY_CODE[code] || ["National", "Hockey League"];
}

function deedFileNo(code, dateText) {
  const year = String(dateText || "").match(/\d{4}/)?.[0] || "2026";
  const club = String(code || "NHL").slice(0, 3).toUpperCase();
  return `NHL-HO-${club}-${year}`;
}

function gmInitials(name) {
  const parts = String(name || "").trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "GM";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase();
}

const NHL_FUN_FACTS = [
  "Wayne Gretzky recorded four separate 200-point seasons.",
  "Glenn Hall started 502 consecutive regular-season games in goal.",
  "Mario Lemieux once scored five goals five different ways in one game.",
  "Nicklas Lidstrom played 20 NHL seasons and never missed the playoffs.",
  "Buffalo once drafted a fictional player named Taro Tsujimoto.",
  "The Stanley Cup predates the National Hockey League.",
  "Matt Murray won the Stanley Cup as a rookie twice.",
  "Gordie Howe played professional hockey in six different decades.",
  "Ron Hextall became the first NHL goalie to shoot and score himself.",
  "Anaheim was the first California franchise to win the Stanley Cup.",
  "Wayne Gretzky finished his NHL career with 1,963 assists.",
  "The Seattle Metropolitans were the first American Stanley Cup champions.",
];

function shuffleArray(array) {
  const result = [...array];
  for (let index = result.length - 1; index > 0; index -= 1) {
    const randomIndex = Math.floor(Math.random() * (index + 1));
    [result[index], result[randomIndex]] = [result[randomIndex], result[index]];
  }
  return result;
}

function normalizeCode(raw) {
  if (raw == null) return "";
  const text = String(raw).trim().toUpperCase();
  if (text.length <= 3) return text;
  return teamNameToNhlAbbr(text) || text.slice(0, 3);
}

function teamDisplayName(team) {
  return String(team?.name || team?.team_name || team?.display_name || "Team").trim();
}

function teamCodeFromRow(team) {
  const fromName = teamNameToNhlAbbr(team?.name || team?.team_name || "");
  if (fromName) return fromName;
  return normalizeCode(team?.abbreviation || team?.abbr || team?.team_id || team?.id || "");
}

function buildOrderedTeams(teams) {
  if (!Array.isArray(teams) || !teams.length) return [];

  const enriched = teams.map((raw, index) => ({
    raw,
    index,
    code: teamCodeFromRow(raw),
    name: teamDisplayName(raw),
    logo: resolveFranchiseTeamLogo(raw, teamDisplayName(raw)),
  }));

  const claimed = new Set();
  const ordered = [];

  DEFAULT_TEAM_ORDER.forEach((code) => {
    const match = enriched.find((item) => item.code === code && !claimed.has(item.index));
    if (match) {
      ordered.push(match);
      claimed.add(match.index);
    }
  });

  enriched.forEach((item) => {
    if (!claimed.has(item.index)) {
      ordered.push(item);
      claimed.add(item.index);
    }
  });

  return ordered;
}

function findOrderedIndexFromSetupIndex(orderedTeams, setupIndex) {
  if (setupIndex == null || setupIndex < 0) return -1;
  const found = orderedTeams.findIndex((item) => item.index === setupIndex);
  return found >= 0 ? found : -1;
}

function teamAccentForCode(code) {
  const colors = {
    ANA: ["#f47a20", "#b9975b"],
    BOS: ["#ffb81c", "#ffffff"],
    BUF: ["#003087", "#ffb81c"],
    CAR: ["#cc0000", "#a2aaad"],
    CBJ: ["#002654", "#ce1126"],
    CGY: ["#c8102e", "#f1be48"],
    CHI: ["#cf0a2c", "#ff671b"],
    COL: ["#6f263d", "#236192"],
    DAL: ["#006847", "#8f8f8c"],
    DET: ["#ce1126", "#ffffff"],
    EDM: ["#ff4c00", "#041e42"],
    FLA: ["#c8102e", "#b9975b"],
    LAK: ["#a2aaad", "#ffffff"],
    MIN: ["#154734", "#a6192e"],
    MTL: ["#af1e2d", "#192168"],
    NJD: ["#ce1126", "#000000"],
    NSH: ["#ffb81c", "#041e42"],
    NYI: ["#00539b", "#f47d30"],
    NYR: ["#0038a8", "#ce1126"],
    OTT: ["#c52032", "#c2912c"],
    PHI: ["#f74902", "#000000"],
    PIT: ["#fcb514", "#000000"],
    SEA: ["#99d9d9", "#e9072b"],
    SJS: ["#006d75", "#ea7200"],
    STL: ["#002f87", "#fcb514"],
    TBL: ["#002868", "#ffffff"],
    TOR: ["#00205b", "#ffffff"],
    UTA: ["#6cace4", "#010101"],
    VAN: ["#00843d", "#00205b"],
    VGK: ["#b4975a", "#333f42"],
    WPG: ["#041e42", "#7b303e"],
    WSH: ["#c8102e", "#041e42"],
  };
  return colors[code] || ["#c9a86a", "#9aa5b1"];
}

function formatContractDate(date = new Date()) {
  return date.toLocaleDateString("en-CA", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}

function getSvgPathFromStroke(points) {
  if (!points.length) return "";
  const max = points.length - 1;
  let path = `M ${points[0][0]} ${points[0][1]} Q`;
  for (let index = 0; index < max; index += 1) {
    const a = points[index];
    const b = points[index + 1];
    path += ` ${a[0]} ${a[1]} ${(a[0] + b[0]) / 2} ${(a[1] + b[1]) / 2}`;
  }
  return `${path} Z`;
}

function SignaturePad({ onInkChange }) {
  const svgRef = useRef(null);
  const liveRef = useRef([]);
  const strokesRef = useRef([]);
  const [paths, setPaths] = useState([]);

  const pointFromEvent = (event) => {
    const svg = svgRef.current;
    const rect = svg.getBoundingClientRect();
    return [
      event.clientX - rect.left,
      event.clientY - rect.top,
      event.pressure || 0.5,
    ];
  };

  const commitLive = (complete) => {
    const outline = getStroke(liveRef.current, {
      size: 7.5,
      thinning: 0.62,
      smoothing: 0.58,
      streamline: 0.42,
      simulatePressure: true,
      last: complete,
    });
    return getSvgPathFromStroke(outline);
  };

  const start = (event) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    liveRef.current = [pointFromEvent(event)];
    setPaths((current) => [...current, commitLive(false)]);
  };

  const move = (event) => {
    if (event.buttons !== 1 || !liveRef.current.length) return;
    event.preventDefault();
    liveRef.current.push(pointFromEvent(event));
    const next = commitLive(false);
    setPaths((current) => {
      const copy = current.slice();
      copy[copy.length - 1] = next;
      return copy;
    });
  };

  const end = () => {
    if (liveRef.current.length < 10) {
      liveRef.current = [];
      setPaths(strokesRef.current.slice());
      onInkChange(strokesRef.current.length > 0);
      return;
    }
    const next = commitLive(true);
    strokesRef.current.push(next);
    liveRef.current = [];
    setPaths(strokesRef.current.slice());
    onInkChange(true);
  };

  return (
    <div className="setup-signature-frame">
      <svg
        ref={svgRef}
        className="setup-signature-pad"
        aria-label="Sign the appointment deed"
        onPointerDown={start}
        onPointerMove={move}
        onPointerUp={end}
        onPointerCancel={end}
      >
        {paths.map((d, index) => (
          <path key={index} d={d} />
        ))}
      </svg>
      <button
        type="button"
        className="setup-signature-clear"
        onClick={() => {
          liveRef.current = [];
          strokesRef.current = [];
          setPaths([]);
          onInkChange(false);
        }}
      >
        Clear
      </button>
    </div>
  );
}

function AppointmentDeedSheet({
  selected,
  gmName,
  setGmName,
  playerUniverse,
  setPlayerUniverse,
  injuriesEnabled,
  setInjuriesEnabled,
  onAccept,
  loading,
  error,
  contractDate,
  signatureReady,
  setSignatureReady,
}) {
  const canContinue =
    Boolean(selected) && Boolean(gmName?.trim()) && signatureReady && !loading;
  const teamName = selected?.name || "National Hockey League Club";
  const teamCode = selected?.code || "NHL";
  const [conference, division] = clubConference(teamCode);
  const fileNo = deedFileNo(teamCode, contractDate);
  const logoSrc =
    selected?.logo ||
    resolveFranchiseTeamLogo(selected?.raw || selected, selected?.name || teamDisplayName(selected));

  return (
    <article className="setup-deed-sheet">
      <div className="setup-deed-paper">
        <header className="setup-deed-docket">
          <div>
            <span>Office of the Commissioner</span>
            <strong>Hockey Operations · Confidential</strong>
          </div>
          <div>
            <span>File no.</span>
            <strong>{fileNo}</strong>
          </div>
        </header>

        <p className="setup-deed-kicker">National Hockey League</p>
        <h2 className="setup-deed-title">
          General Manager
          <span>Appointment Deed</span>
        </h2>
        <p className="setup-deed-parties">
          This instrument is made between the League and the Club of Record named below,
          to take effect on the date executed.
        </p>

        <div className="setup-deed-club">
          {logoSrc ? <img src={logoSrc} alt="" /> : <em>{teamCode}</em>}
          <div>
            <strong>{teamName}</strong>
            <small>{teamCode} · {conference} · {division}</small>
          </div>
        </div>

        <label className="setup-deed-gm">
          <span>Appointed General Manager</span>
          <input
            type="text"
            value={gmName}
            onChange={(event) => setGmName(event.target.value)}
            placeholder="Your name"
            maxLength={80}
            autoComplete="off"
          />
        </label>

        <div className="setup-deed-options">
          <fieldset>
            <legend>Player names</legend>
            <div>
              <button
                type="button"
                className={playerUniverse !== "real_nhl" ? "setup-token is-on" : "setup-token"}
                aria-pressed={playerUniverse !== "real_nhl"}
                onClick={() => setPlayerUniverse("generated")}
              >
                <span className="setup-token-orb" aria-hidden="true" />
                <strong>Generated</strong>
              </button>
              <button
                type="button"
                className={playerUniverse === "real_nhl" ? "setup-token is-on" : "setup-token"}
                aria-pressed={playerUniverse === "real_nhl"}
                onClick={() => setPlayerUniverse("real_nhl")}
              >
                <span className="setup-token-orb" aria-hidden="true" />
                <strong>Real NHL</strong>
              </button>
            </div>
          </fieldset>
          <fieldset>
            <legend>Injuries</legend>
            <div>
              <button
                type="button"
                className={injuriesEnabled ? "setup-token is-on" : "setup-token"}
                aria-pressed={injuriesEnabled}
                onClick={() => setInjuriesEnabled(true)}
              >
                <span className="setup-token-orb" aria-hidden="true" />
                <strong>On</strong>
              </button>
              <button
                type="button"
                className={!injuriesEnabled ? "setup-token is-on" : "setup-token"}
                aria-pressed={!injuriesEnabled}
                onClick={() => setInjuriesEnabled(false)}
              >
                <span className="setup-token-orb" aria-hidden="true" />
                <strong>Off</strong>
              </button>
            </div>
          </fieldset>
        </div>

        <dl className="setup-deed-meta">
          <div>
            <dt>Appointment term</dt>
            <dd>Year one</dd>
          </div>
          <div>
            <dt>Effective date</dt>
            <dd>{contractDate || "—"}</dd>
          </div>
          <div>
            <dt>Player universe</dt>
            <dd>{playerUniverse === "real_nhl" ? "Real NHL players" : "Generated players"}</dd>
          </div>
          <div>
            <dt>League health system</dt>
            <dd>{injuriesEnabled ? "Enabled" : "Disabled"}</dd>
          </div>
        </dl>

        <ol className="setup-deed-articles">
          <li>
            <strong>Art. I — Appointment.</strong>
            {" "}The undersigned is appointed General Manager of the Club of Record
            for a term of one League year, renewable by continued operations.
          </li>
          <li>
            <strong>Art. II — Player universe.</strong>
            {" "}Rosters shall be drawn from{" "}
            {playerUniverse === "real_nhl" ? "the real NHL player universe" : "a generated player universe"}
            , as scheduled on this deed.
          </li>
          <li>
            <strong>Art. III — Health system.</strong>
            {" "}League injury administration is{" "}
            {injuriesEnabled ? "enabled" : "disabled"}
            {" "}for the term of this appointment.
          </li>
          <li>
            <strong>Art. IV — Powers.</strong>
            {" "}The General Manager accepts authority over hockey operations,
            transactions, contracts, staff, roster construction, scouting, and
            franchise strategy, subject to the Constitution and this deed.
          </li>
        </ol>

        <div className="setup-deed-sign">
          <span>General Manager signature · Executed in counterpart</span>
          <SignaturePad onInkChange={setSignatureReady} />
          <div className="setup-deed-sign-meta">
            <small>{signatureReady ? "Signature captured" : "Sign on the line above"}</small>
            <small>Witness: League Secretary</small>
          </div>
          <div className={`setup-deed-seal ${signatureReady ? "is-struck" : ""}`} aria-hidden="true">
            <em>NHL</em>
            <span>Hockey Ops</span>
          </div>
        </div>

        {error ? (
          <div className="setup-error" role="alert">
            {error}
          </div>
        ) : null}

        <button type="button" className="setup-accept-btn" disabled={!canContinue} onClick={onAccept}>
          <span>Begin franchise</span>
          <small>Execute deed · Open hockey operations</small>
        </button>
        <footer className="setup-deed-footer">
          <span>Page 1 of 1</span>
          <span>Schedule A attached</span>
          <span>{fileNo}</span>
        </footer>
      </div>
    </article>
  );
}

const TeamSelection = React.memo(function TeamSelection({
  teams,
  selectedIndex,
  onSelect,
  selected,
  gmName,
  playerUniverse,
  injuriesEnabled,
  contractDate,
  signatureReady,
}) {
  const clubList = teams.length ? teams : buildOrderedTeams(buildDefaultFranchiseTeamList());
  const teamCode = selected?.code || "NHL";
  const [conference, division] = clubConference(teamCode);
  const fileNo = deedFileNo(teamCode, contractDate);

  return (
    <section className="setup-team-selector">
      <header className="setup-panel-heading">
        <span>Schedule A · Club of record</span>
        <strong>Choose your club · {clubList.length} NHL clubs</strong>
      </header>
      <ClubBallBoard teams={clubList} selectedIndex={selectedIndex} onSelect={onSelect} />
      <aside className="setup-schedule">
        <div className="setup-schedule-head">
          <span>Exhibit A</span>
          <strong>Particulars of appointment</strong>
          <em>{fileNo}</em>
        </div>
        <dl className="setup-schedule-grid">
          <div>
            <dt>Club of record</dt>
            <dd>{selected?.name || "Select a club"}</dd>
          </div>
          <div>
            <dt>Abbreviation</dt>
            <dd>{teamCode}</dd>
          </div>
          <div>
            <dt>Conference</dt>
            <dd>{conference}</dd>
          </div>
          <div>
            <dt>Division</dt>
            <dd>{division}</dd>
          </div>
          <div>
            <dt>Appointed GM</dt>
            <dd>{gmName?.trim() || "To be named"}</dd>
          </div>
          <div>
            <dt>Effective</dt>
            <dd>{contractDate || "—"}</dd>
          </div>
          <div>
            <dt>Player names</dt>
            <dd>{playerUniverse === "real_nhl" ? "Real NHL" : "Generated"}</dd>
          </div>
          <div>
            <dt>Health system</dt>
            <dd>{injuriesEnabled ? "Enabled" : "Disabled"}</dd>
          </div>
        </dl>
        <p className="setup-schedule-clause">
          The Club of Record is the sole franchise named on this deed. Crest
          selection on this schedule constitutes identification of the Club
          for all hockey operations, transactions, and League filings during
          Year One. Execution of the signature block on page 1 binds the
          General Manager to the Articles herein.
        </p>
        <div className="setup-schedule-marks">
          <span>Initials {gmInitials(gmName)}</span>
          <span>{signatureReady ? "Deed executed" : "Awaiting signature"}</span>
          <span>Commissioner copy retained</span>
        </div>
      </aside>
    </section>
  );
});

function SetupLoadingScreen({
  selected,
  gmName,
  injuriesEnabled,
  playerUniverse,
  error,
  loading,
  warmup,
  onRetry,
  onBack,
}) {
  const facts = useMemo(() => shuffleArray(NHL_FUN_FACTS), []);
  const [factIndex, setFactIndex] = useState(0);

  useEffect(() => {
    const id = window.setInterval(() => {
      setFactIndex((current) => (current + 1) % facts.length);
    }, 8500);
    return () => window.clearInterval(id);
  }, [facts.length]);

  const failed = Boolean(error);
  const categories = useMemo(
    () =>
      Object.entries(HUB_WARMUP_LABELS).map(([key, label]) => ({
        key,
        label,
        status: warmup?.[key] || "waiting",
      })),
    [warmup]
  );
  const settled = categories.filter(({ status }) => status === "ready").length;

  return (
    <div className="setup-loading-screen" role="status" aria-live="polite">
      <div className="setup-loading-panel">
        <small>Franchise Mode</small>
        <h2>{selected?.name || "NHL"}</h2>
        <p>
          {failed
            ? error
            : playerUniverse === "real_nhl"
              ? "Opening the real NHL player universe."
              : `${gmName?.trim() || "General Manager"} is taking over hockey operations.`}
        </p>
        {!failed ? (
          <div className="setup-loading-meta">
            <span>{injuriesEnabled ? "Injuries on" : "Injuries off"}</span>
            <span>{playerUniverse === "real_nhl" ? "Real NHL" : "Generated"}</span>
          </div>
        ) : null}
        {!failed ? (
          <>
            <ul className="setup-loading-tasks">
              {categories.map(({ key, label, status }) => (
                <li key={key} className={`is-${status}`}>
                  <i aria-hidden="true" />
                  <span>{label}</span>
                  <em>{status === "ready" ? "Ready" : status === "loading" ? "Arriving" : "Queued"}</em>
                </li>
              ))}
            </ul>
            <div
              className={`setup-loading-bar ${settled === categories.length ? "is-complete" : ""}`}
              aria-hidden="true"
            >
              <span />
            </div>
          </>
        ) : null}
        {failed ? (
          <div className="setup-loading-actions">
            {typeof onRetry === "function" ? (
              <button type="button" className="setup-loading-retry" onClick={onRetry} disabled={loading && !failed}>
                Retry
              </button>
            ) : null}
            {typeof onBack === "function" ? (
              <button type="button" className="setup-loading-back" onClick={onBack}>
                Back to setup
              </button>
            ) : null}
          </div>
        ) : null}
        {!failed ? <blockquote>{facts[factIndex]}</blockquote> : null}
      </div>
    </div>
  );
}

export function SetupScreen() {
  const {
    teams,
    setupTeamIndex,
    setSetupTeamIndex,
    gmName,
    setGmName,
    playerUniverse,
    setPlayerUniverse,
    injuriesEnabled,
    setInjuriesEnabled,
    beginFranchise,
    loading,
    loadTeams,
    error,
    setError,
    hubWarmup,
    primeHubAssets,
  } = useGameUI();

  const [appStage, setAppStage] = useState(APP_STAGE.CONFIGURE);
  const [signatureReady, setSignatureReady] = useState(false);
  const [pickedTeamCode, setPickedTeamCode] = useState("");
  const [statusText, setStatusText] = useState("Select a club and configure the franchise.");

  useEffect(() => {
    loadTeams();
  }, [loadTeams]);

  useEffect(() => {
    primeHubAssets(HUB_WARMUP_STAGES.ENVIRONMENT);
    primeHubAssets(HUB_WARMUP_STAGES.CRESTS);
    primeHubAssets(HUB_WARMUP_STAGES.OPERATIONS);
  }, [primeHubAssets]);

  const orderedTeams = useMemo(
    () => buildOrderedTeams(teams.length ? teams : buildDefaultFranchiseTeamList()),
    [teams]
  );

  const orderedIndex = useMemo(() => {
    if (pickedTeamCode) {
      const byCode = orderedTeams.findIndex((item) => item.code === pickedTeamCode);
      if (byCode >= 0) return byCode;
    }
    return findOrderedIndexFromSetupIndex(orderedTeams, setupTeamIndex);
  }, [orderedTeams, pickedTeamCode, setupTeamIndex]);

  const selected = orderedTeams[orderedIndex] || null;
  const [accentPrimary, accentSecondary] = teamAccentForCode(selected?.code || "");
  const contractDate = useMemo(() => formatContractDate(), []);

  useEffect(() => {
    if (!pickedTeamCode) return;
    const match = orderedTeams.find((item) => item.code === pickedTeamCode);
    if (match && match.index !== setupTeamIndex) {
      setSetupTeamIndex(match.index);
    }
  }, [pickedTeamCode, orderedTeams, setupTeamIndex, setSetupTeamIndex]);

  const setTeamByOrderedIndex = useCallback(
    (nextIndex) => {
      if (!orderedTeams.length) return;
      const safeIndex = ((nextIndex % orderedTeams.length) + orderedTeams.length) % orderedTeams.length;
      const team = orderedTeams[safeIndex];
      if (!team) return;
      setPickedTeamCode(team.code || "");
      setSetupTeamIndex(team.index);
      setStatusText(`${team.name} selected.`);
    },
    [orderedTeams, setSetupTeamIndex]
  );

  const startFranchise = useCallback(async () => {
    setAppStage(APP_STAGE.STARTING);
    setStatusText(`Appointment executed. Opening ${selected?.name || "franchise"} hockey operations.`);
    try {
      const result = await Promise.resolve(beginFranchise());
      if (result && result.ok === false) {
        setStatusText(result.error || "Franchise start failed. Retry or return to setup.");
      }
    } catch (startError) {
      console.error("Unable to begin franchise", startError);
      setStatusText("Franchise start failed. Retry or return to setup.");
    }
  }, [beginFranchise, selected]);

  return (
    <div
      className="nhlcal-root setup-root"
      style={{
        "--team-accent": accentPrimary,
        "--team-accent-2": accentSecondary,
      }}
    >
      {appStage === APP_STAGE.CONFIGURE ? (
        <main className="setup-config-layout setup-config-layout--desk">
          <div className="setup-config-topline">
            <strong>Franchise Agreement</strong>
            <small>Complete the schedule, then sign and execute</small>
          </div>
          <div className="setup-config-grid">
            <AppointmentDeedSheet
              selected={selected}
              gmName={gmName}
              setGmName={setGmName}
              playerUniverse={playerUniverse}
              setPlayerUniverse={setPlayerUniverse}
              injuriesEnabled={injuriesEnabled}
              setInjuriesEnabled={setInjuriesEnabled}
              onAccept={startFranchise}
              loading={loading}
              error={error}
              contractDate={contractDate}
              signatureReady={signatureReady}
              setSignatureReady={setSignatureReady}
            />
            <TeamSelection
              teams={orderedTeams}
              selectedIndex={orderedIndex}
              onSelect={setTeamByOrderedIndex}
              selected={selected}
              gmName={gmName}
              playerUniverse={playerUniverse}
              injuriesEnabled={injuriesEnabled}
              contractDate={contractDate}
              signatureReady={signatureReady}
            />
          </div>
          <p className="setup-sr-status" aria-live="polite">
            {statusText}
          </p>
        </main>
      ) : null}

      {appStage === APP_STAGE.STARTING ? (
        <SetupLoadingScreen
          selected={selected}
          gmName={gmName}
          injuriesEnabled={injuriesEnabled}
          playerUniverse={playerUniverse}
          error={error}
          loading={loading}
          warmup={hubWarmup}
          onRetry={startFranchise}
          onBack={() => {
            setError(null);
            setAppStage(APP_STAGE.CONFIGURE);
            setStatusText("Select a club and configure the franchise.");
          }}
        />
      ) : null}

      <style>{SETUP_SCREEN_CSS}</style>
    </div>
  );
}

const SETUP_SCREEN_CSS = `
.nhlcal-root.setup-root {
  --setup-font: var(--font-ops-ui, Inter, ui-sans-serif, system-ui, sans-serif);
  --setup-text: #f0ede6;
  --setup-muted: rgba(229, 225, 216, 0.62);
  --setup-line: rgba(201, 168, 106, 0.28);
  --setup-gold: #c9a86a;
  position: fixed;
  z-index: 23000;
  inset: 0;
  display: flex;
  flex-direction: column;
  width: 100vw;
  height: 100dvh;
  overflow: hidden;
  font-family: var(--setup-font);
  color: var(--setup-text);
  background:
    radial-gradient(ellipse 80% 60% at 50% 0%, rgba(40, 32, 18, 0.45), transparent 58%),
    linear-gradient(180deg, #0c0e14, #08090d 55%, #050608);
}
.setup-root *,
.setup-root *::before,
.setup-root *::after { box-sizing: border-box; }
.setup-root button,
.setup-root input,
.setup-root fieldset,
.setup-root legend { font-family: var(--setup-font); }
.setup-root button { -webkit-tap-highlight-color: transparent; }
.setup-root button:focus-visible,
.setup-root input:focus-visible {
  outline: 2px solid color-mix(in srgb, var(--team-accent) 80%, #fff);
  outline-offset: 2px;
}

.setup-config-layout {
  position: relative;
  width: 100%;
  height: 100%;
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  gap: 2px;
  padding: 2px;
  overflow: hidden;
  animation: setupConfigArrive 480ms cubic-bezier(.2, .72, .2, 1) both;
}
.setup-config-layout--desk {
  position: fixed;
  inset: 0;
  height: auto;
  padding: clamp(14px, 3.2vh, 34px) clamp(14px, 4vw, 64px) clamp(14px, 3vh, 30px);
  gap: clamp(8px, 1.4vh, 16px);
}
.setup-config-topline {
  min-height: 28px;
  display: grid;
  grid-template-columns: auto 1fr;
  align-items: center;
  gap: 12px;
  padding: 0 10px 4px;
  text-transform: uppercase;
  letter-spacing: 0.13em;
  border-bottom: 1px solid rgba(201, 168, 106, 0.18);
}
.setup-config-topline small {
  justify-self: end;
  text-align: right;
  font-size: 10px;
  font-weight: 800;
  color: var(--setup-gold);
}
.setup-config-topline strong {
  font-size: clamp(13px, 1.3vw, 17px);
  font-weight: 900;
  color: var(--setup-gold);
}
.setup-config-grid {
  position: relative;
  z-index: 2;
  min-height: 0;
  height: 100%;
  display: grid;
  grid-template-columns: minmax(380px, 0.42fr) minmax(0, 0.58fr);
  gap: clamp(12px, 1.6vw, 22px);
  padding: clamp(12px, 1.8vh, 18px) clamp(14px, 1.8vw, 22px) clamp(12px, 1.8vh, 18px) 34px;
  border: 1px solid rgba(201, 168, 106, 0.32);
  border-radius: 3px;
  background:
    linear-gradient(90deg, rgba(18, 14, 10, 0.96) 0 22px, transparent 22px),
    linear-gradient(178deg, rgba(36, 30, 22, 0.96), rgba(12, 10, 9, 0.98));
  box-shadow:
    0 48px 120px rgba(0, 0, 0, 0.55),
    inset 0 0 0 1px rgba(255, 226, 168, 0.05),
    inset 10px 0 18px rgba(0, 0, 0, 0.28);
}
.setup-config-grid::before {
  content: "";
  position: absolute;
  z-index: 3;
  top: 18px;
  bottom: 18px;
  left: 9px;
  width: 8px;
  background:
    radial-gradient(circle at 50% 12%, #1a1612 4px, transparent 5px),
    radial-gradient(circle at 50% 32%, #1a1612 4px, transparent 5px),
    radial-gradient(circle at 50% 52%, #1a1612 4px, transparent 5px),
    radial-gradient(circle at 50% 72%, #1a1612 4px, transparent 5px),
    radial-gradient(circle at 50% 90%, #1a1612 4px, transparent 5px);
  border-right: 1px solid rgba(201, 168, 106, 0.22);
  pointer-events: none;
}
.setup-config-grid::after {
  content: "";
  position: absolute;
  inset: 7px;
  border: 1px solid rgba(201, 168, 106, 0.12);
  pointer-events: none;
}

.setup-team-selector {
  min-height: 0;
  display: grid;
  grid-template-rows: auto auto minmax(140px, 1fr);
  gap: 10px;
  overflow: hidden;
  padding: 10px 12px 12px;
  border: 1px solid rgba(201, 168, 106, 0.28);
  background:
    linear-gradient(180deg, rgba(255, 236, 190, 0.03), transparent 18%),
    linear-gradient(178deg, rgba(24, 20, 16, 0.92), rgba(11, 10, 9, 0.96));
  box-shadow: inset 0 0 0 6px rgba(8, 7, 6, 0.35), inset 0 0 0 7px rgba(201, 168, 106, 0.16);
}
.setup-panel-heading {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 2px 0 4px;
}
.setup-panel-heading span {
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--setup-muted);
}
.setup-panel-heading strong {
  font-size: 15px;
  font-weight: 900;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--setup-gold);
}

.setup-deed-sheet { min-height: 0; overflow: auto; }
.setup-deed-paper {
  position: relative;
  min-height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 14px 16px 10px 22px;
  border: 1px solid rgba(201, 168, 106, 0.28);
  background:
    linear-gradient(180deg, rgba(255, 236, 190, 0.035), transparent 16%),
    repeating-linear-gradient(
      180deg,
      transparent 0 27px,
      rgba(201, 168, 106, 0.035) 27px 28px
    ),
    linear-gradient(178deg, rgba(24, 20, 16, 0.94), rgba(11, 10, 9, 0.97));
  box-shadow: inset 0 0 0 6px rgba(8, 7, 6, 0.35), inset 0 0 0 7px rgba(201, 168, 106, 0.16);
}
.setup-deed-paper::before {
  content: "CONFIDENTIAL";
  position: absolute;
  z-index: 0;
  top: 46%;
  left: 50%;
  transform: translate(-50%, -50%) rotate(-28deg);
  font-size: clamp(42px, 5vw, 72px);
  font-weight: 900;
  letter-spacing: 0.22em;
  color: rgba(201, 168, 106, 0.055);
  pointer-events: none;
  white-space: nowrap;
}
.setup-deed-paper::after {
  content: "";
  position: absolute;
  z-index: 0;
  top: 16px;
  bottom: 16px;
  left: 11px;
  width: 2px;
  background: linear-gradient(180deg, transparent, rgba(168, 42, 42, 0.55) 8%, rgba(201, 168, 106, 0.55) 92%, transparent);
  pointer-events: none;
}
.setup-deed-paper > * { position: relative; z-index: 1; }
.setup-deed-docket {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 12px;
  align-items: end;
  margin: 0 0 12px;
  padding: 0 0 10px;
  border-bottom: 2px solid rgba(201, 168, 106, 0.42);
  box-shadow: 0 1px 0 rgba(201, 168, 106, 0.18);
}
.setup-deed-docket span,
.setup-deed-footer span {
  display: block;
  font-size: 8px;
  font-weight: 800;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--setup-muted);
}
.setup-deed-docket strong {
  display: block;
  margin-top: 3px;
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--setup-gold);
}
.setup-deed-docket > div:last-child { text-align: right; }
.setup-deed-kicker {
  margin: 0 0 4px;
  text-align: center;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--setup-gold);
}
.setup-deed-title {
  margin: 0 0 8px;
  text-align: center;
  font-size: clamp(20px, 2vw, 30px);
  font-weight: 900;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  line-height: 0.95;
  color: var(--setup-gold);
}
.setup-deed-title span { display: block; margin-top: 4px; font-size: 0.72em; }
.setup-deed-parties {
  margin: 0 0 12px;
  text-align: center;
  font-size: 11px;
  line-height: 1.45;
  font-style: italic;
  color: var(--setup-muted);
}
.setup-deed-club {
  display: grid;
  grid-template-columns: 56px minmax(0, 1fr);
  gap: 12px;
  align-items: center;
  margin: 0 0 12px;
  padding: 8px 0 12px;
  border-top: 1px solid var(--setup-line);
  border-bottom: 1px solid var(--setup-line);
}
.setup-deed-club img,
.setup-deed-club em {
  width: 56px;
  height: 56px;
  object-fit: contain;
  filter: drop-shadow(0 8px 10px rgba(0, 0, 0, 0.45));
}
.setup-deed-club em {
  display: grid;
  place-items: center;
  font-style: normal;
  font-weight: 900;
  color: var(--setup-gold);
}
.setup-deed-club strong {
  display: block;
  font-size: 15px;
  font-weight: 900;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.setup-deed-club small {
  display: block;
  margin-top: 3px;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--setup-muted);
}
.setup-deed-gm { display: grid; gap: 6px; margin: 0 0 12px; }
.setup-deed-gm span,
.setup-deed-options legend,
.setup-deed-meta dt,
.setup-deed-sign span {
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--setup-gold);
}
.setup-deed-gm input {
  width: 100%;
  min-height: 42px;
  border: 0;
  border-bottom: 2px solid var(--setup-line);
  background: transparent;
  color: var(--setup-text);
  font-size: 22px;
  font-weight: 900;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.setup-deed-gm input::placeholder { color: var(--setup-muted); text-transform: none; font-weight: 700; }
.setup-deed-options {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin: 0 0 10px;
}
.setup-deed-options fieldset { margin: 0; padding: 0; border: 0; min-width: 0; }
.setup-deed-options fieldset > div {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
  margin-top: 6px;
}
.setup-token {
  appearance: none;
  display: grid;
  grid-template-columns: 14px minmax(0, 1fr);
  align-items: center;
  gap: 8px;
  min-height: 34px;
  padding: 6px 8px;
  border: 1px solid rgba(201, 168, 106, 0.22);
  background: rgba(8, 7, 6, 0.25);
  color: var(--setup-text);
  cursor: pointer;
}
.setup-token-orb {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  border: 1px solid rgba(201, 168, 106, 0.4);
  background: #1c1710;
  box-shadow: none;
}
.setup-token.is-on {
  border-color: rgba(201, 168, 106, 0.7);
  background: rgba(201, 168, 106, 0.1);
}
.setup-token.is-on .setup-token-orb {
  border-color: #c9a86a;
  background: radial-gradient(circle at 32% 28%, #f3e2b0, #c9a86a 42%, #6a4e1c 100%);
}
.setup-token strong { font-size: 10px; font-weight: 900; letter-spacing: 0.08em; text-transform: uppercase; text-align: left; }
.setup-deed-meta { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 16px; margin: 0 0 12px; padding-bottom: 10px; border-bottom: 1px dashed rgba(201, 168, 106, 0.28); }
.setup-deed-meta dd { margin: 4px 0 0; font-size: 13px; font-weight: 900; text-transform: uppercase; }
.setup-deed-legal { margin: 0 0 12px; font-size: 12px; line-height: 1.45; color: var(--setup-muted); }
.setup-deed-articles {
  margin: 0 0 12px;
  padding: 0 0 0 18px;
  display: grid;
  gap: 8px;
  color: rgba(229, 225, 216, 0.78);
  font-size: 11px;
  line-height: 1.45;
}
.setup-deed-articles li::marker { color: var(--setup-gold); font-weight: 900; }
.setup-deed-articles strong { color: var(--setup-gold); letter-spacing: 0.04em; text-transform: uppercase; font-size: 10px; }
.setup-deed-sign {
  position: relative;
  display: grid;
  gap: 6px;
  margin: 0 0 10px;
}
.setup-deed-sign-meta {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 8px;
  align-items: center;
}
.setup-deed-seal {
  position: absolute;
  right: 6px;
  top: 22px;
  width: 76px;
  height: 76px;
  display: grid;
  place-content: center;
  justify-items: center;
  gap: 1px;
  border-radius: 50%;
  border: 2px solid rgba(201, 168, 106, 0.28);
  background: radial-gradient(circle at 35% 30%, rgba(80, 22, 22, 0.15), rgba(12, 10, 9, 0.2));
  transform: rotate(-14deg);
  opacity: 0.28;
  pointer-events: none;
}
.setup-deed-seal em {
  font-style: normal;
  font-size: 13px;
  font-weight: 900;
  letter-spacing: 0.12em;
  color: #c9a86a;
}
.setup-deed-seal span {
  font-size: 7px !important;
  letter-spacing: 0.14em;
  color: rgba(201, 168, 106, 0.8) !important;
}
.setup-deed-seal.is-struck {
  opacity: 0.92;
  border-color: #b4232a;
  box-shadow: 0 0 0 3px rgba(180, 35, 42, 0.18), inset 0 0 12px rgba(180, 35, 42, 0.25);
  background: radial-gradient(circle at 35% 30%, rgba(180, 35, 42, 0.35), rgba(40, 10, 10, 0.4));
}
.setup-deed-seal.is-struck em,
.setup-deed-seal.is-struck span { color: #e8b3b0 !important; }
.setup-signature-frame { position: relative; min-height: 88px; height: 12vh; max-height: 120px; padding-right: 88px; }
.setup-signature-pad {
  display: block;
  width: 100%;
  height: 100%;
  touch-action: none;
  cursor: crosshair;
  border: 0;
  border-bottom: 1px solid rgba(201, 168, 106, 0.45);
  background: transparent;
}
.setup-signature-pad path { fill: #e8d5a0; }
.setup-signature-clear {
  position: absolute;
  right: 8px;
  bottom: 8px;
  border: 0;
  background: transparent;
  color: var(--setup-gold);
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  cursor: pointer;
}
.setup-deed-sign small {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--setup-gold);
}
.setup-deed-paper .setup-accept-btn { width: 100%; margin-top: 4px; }
.setup-deed-footer {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid rgba(201, 168, 106, 0.28);
}
.setup-deed-footer span:nth-child(2) { text-align: center; color: var(--setup-gold); }
.setup-deed-footer span:last-child { text-align: right; }
.setup-error {
  padding: 10px 12px;
  border: 1px solid rgba(255, 96, 109, 0.42);
  background: rgba(100, 10, 18, 0.35);
  color: #ffd4d8;
  font-size: 11px;
}
.setup-accept-btn {
  min-height: 56px;
  display: grid;
  place-items: center;
  gap: 2px;
  padding: 8px 16px;
  border: 1px solid color-mix(in srgb, var(--team-accent) 64%, rgba(255,255,255,.18));
  background: linear-gradient(180deg, color-mix(in srgb, var(--team-accent) 15%, rgba(13,15,20,.98)), rgba(7,9,13,.98));
  color: #f0e8db;
  cursor: pointer;
}
.setup-accept-btn:hover:not(:disabled) { transform: translateY(-1px); }
.setup-accept-btn:disabled { opacity: 0.38; cursor: not-allowed; }
.setup-accept-btn span { font-size: 13px; font-weight: 900; letter-spacing: 0.08em; text-transform: uppercase; }
.setup-accept-btn small { font-size: 9px; font-weight: 750; letter-spacing: 0.12em; text-transform: uppercase; color: rgba(229, 221, 207, 0.53); }

.setup-club-ball-grid {
  display: grid;
  grid-template-columns: repeat(8, minmax(0, 1fr));
  align-content: start;
  gap: 8px 6px;
  min-height: 0;
  height: auto;
  overflow: visible;
  padding: 2px 2px 4px;
}
.setup-club-ball {
  appearance: none;
  display: grid;
  justify-items: center;
  gap: 6px;
  min-width: 0;
  padding: 4px 2px 2px;
  border: 0;
  background: transparent;
  color: var(--setup-text);
  cursor: pointer;
}
.setup-club-ball-orb {
  width: clamp(40px, 5.4vw, 56px);
  height: clamp(40px, 5.4vw, 56px);
  display: grid;
  place-items: center;
  border-radius: 50%;
  border: 1px solid rgba(201, 168, 106, 0.28);
  background: radial-gradient(circle at 32% 28%, #6a5a40, #1c1710 62%, #070605 100%);
  box-shadow: 0 10px 14px rgba(0, 0, 0, 0.5), inset -6px -8px 12px rgba(0, 0, 0, 0.45), inset 4px 5px 8px rgba(255, 236, 190, 0.12);
}
.setup-club-ball-orb img,
.setup-club-ball-orb em { width: 68%; height: 68%; object-fit: contain; pointer-events: none; }
.setup-club-ball-orb em { display: grid; place-items: center; font-style: normal; font-size: 11px; font-weight: 900; color: #f3e6c8; }
.setup-club-ball strong {
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: 10px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--setup-muted);
}
.setup-club-ball.is-selected .setup-club-ball-orb {
  border-color: rgba(201, 168, 106, 0.78);
  box-shadow: 0 0 0 2px rgba(201, 168, 106, 0.35), 0 12px 16px rgba(0, 0, 0, 0.5);
}
.setup-club-ball.is-selected strong { color: var(--setup-gold); }
.setup-club-ball:hover .setup-club-ball-orb,
.setup-club-ball:focus-visible .setup-club-ball-orb { border-color: rgba(201, 168, 106, 0.55); }

.setup-schedule {
  min-height: 0;
  overflow: auto;
  display: grid;
  align-content: start;
  gap: 10px;
  padding: 10px 12px 12px;
  border: 1px solid rgba(201, 168, 106, 0.28);
  background: linear-gradient(180deg, rgba(201, 168, 106, 0.05), rgba(8, 7, 6, 0.2));
}
.setup-schedule-head {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 10px;
  align-items: end;
  padding-bottom: 8px;
  border-bottom: 2px solid rgba(201, 168, 106, 0.38);
}
.setup-schedule-head span,
.setup-schedule-head em {
  font-size: 9px;
  font-weight: 800;
  font-style: normal;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--setup-muted);
}
.setup-schedule-head strong {
  font-size: 12px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--setup-gold);
}
.setup-schedule-grid {
  margin: 0;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px 16px;
}
.setup-schedule-grid dt {
  font-size: 8px;
  font-weight: 800;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--setup-muted);
}
.setup-schedule-grid dd {
  margin: 3px 0 0;
  font-size: 13px;
  font-weight: 900;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  color: var(--setup-text);
}
.setup-schedule-clause {
  margin: 0;
  padding-top: 4px;
  font-size: 11px;
  line-height: 1.5;
  color: rgba(229, 225, 216, 0.7);
}
.setup-schedule-marks {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  margin-top: 2px;
}
.setup-schedule-marks span {
  display: grid;
  place-items: center;
  min-height: 36px;
  padding: 6px 8px;
  border: 1px dashed rgba(201, 168, 106, 0.4);
  font-size: 8px;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  text-align: center;
  color: var(--setup-gold);
}

.setup-loading-screen {
  position: fixed;
  z-index: 25000;
  inset: 0;
  display: grid;
  place-items: center;
  padding: 22px;
  background:
    radial-gradient(circle at 50% 34%, color-mix(in srgb, var(--team-accent) 10%, transparent), transparent 42%),
    linear-gradient(180deg, rgba(4, 5, 8, 0.92), rgba(4, 5, 8, 0.96));
}
.setup-loading-panel {
  width: min(560px, 100%);
  display: grid;
  justify-items: center;
  gap: 7px;
  padding: 30px;
  border: 1px solid rgba(201, 168, 106, 0.22);
  background: linear-gradient(180deg, rgba(15, 14, 14, 0.92), rgba(8, 8, 10, 0.96));
  text-align: center;
  box-shadow: 0 36px 110px rgba(0, 0, 0, 0.5);
}
.setup-loading-panel > small {
  font-size: 9px;
  font-weight: 900;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--setup-muted);
}
.setup-loading-panel h2 {
  margin: 0;
  font-size: clamp(26px, 4vw, 46px);
  font-weight: 950;
  text-transform: uppercase;
}
.setup-loading-panel p { margin: 0; color: var(--setup-muted); font-size: 12px; }
.setup-loading-meta { display: flex; flex-wrap: wrap; justify-content: center; gap: 7px; margin: 10px 0; }
.setup-loading-meta span {
  padding: 6px 9px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: rgba(234, 229, 218, 0.62);
}
.setup-loading-panel blockquote {
  width: 100%;
  margin: 8px 0 0;
  padding: 12px 14px;
  border-left: 2px solid var(--team-accent);
  background: rgba(255, 255, 255, 0.02);
  color: rgba(235, 229, 217, 0.7);
  font-size: 11px;
  line-height: 1.45;
  text-align: left;
}
.setup-loading-tasks { width: 100%; margin: 6px 0 2px; padding: 0; list-style: none; display: grid; gap: 1px; }
.setup-loading-tasks li {
  display: grid;
  grid-template-columns: 14px 1fr auto;
  align-items: center;
  gap: 10px;
  padding: 8px 2px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.04);
  text-align: left;
}
.setup-loading-tasks li:last-child { border-bottom: none; }
.setup-loading-tasks i {
  width: 7px;
  height: 7px;
  margin-left: 3px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.14);
}
.setup-loading-tasks li.is-loading i { background: var(--team-accent); animation: setupTaskPulse 1.3s ease-in-out infinite; }
.setup-loading-tasks li.is-ready i { background: #6fbf8a; box-shadow: 0 0 10px rgba(111, 191, 138, 0.55); }
.setup-loading-tasks span { font-size: 11px; font-weight: 700; color: rgba(238, 232, 220, 0.78); }
.setup-loading-tasks em { font-style: normal; font-size: 9px; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase; color: rgba(236, 230, 218, 0.4); }
.setup-loading-tasks li.is-ready em { color: rgba(111, 191, 138, 0.8); }
.setup-loading-tasks li.is-loading em { color: var(--setup-gold); }
.setup-loading-bar { position: relative; width: 100%; height: 2px; overflow: hidden; background: rgba(255, 255, 255, 0.06); }
.setup-loading-bar span {
  position: absolute;
  inset: 0 auto 0 0;
  width: 38%;
  background: linear-gradient(90deg, transparent, var(--team-accent), var(--team-accent-2), transparent);
  animation: setupLoadingSweep 1.9s cubic-bezier(.5, 0, .5, 1) infinite;
}
.setup-loading-bar.is-complete span {
  width: 100%;
  animation: none;
  background: linear-gradient(90deg, var(--team-accent), var(--team-accent-2));
}
.setup-loading-actions { display: flex; gap: 8px; margin-top: 12px; }
.setup-loading-retry,
.setup-loading-back {
  border: 1px solid rgba(201, 168, 106, 0.35);
  background: rgba(201, 168, 106, 0.08);
  color: var(--setup-gold);
  padding: 8px 14px;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  cursor: pointer;
}

.setup-sr-status {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

@keyframes setupConfigArrive {
  from { opacity: 0; }
  to { opacity: 1; }
}
@keyframes setupLoadingSweep {
  from { transform: translateX(-100%); }
  to { transform: translateX(300%); }
}
@keyframes setupTaskPulse {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 1; }
}

@media (max-width: 1100px) {
  .setup-config-grid { grid-template-columns: minmax(320px, 0.44fr) minmax(0, 0.56fr); }
  .setup-club-ball-grid { grid-template-columns: repeat(6, minmax(0, 1fr)); }
  .setup-schedule-grid { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 900px) {
  .setup-config-layout--desk .setup-config-grid {
    grid-template-columns: minmax(0, 1fr);
    overflow-y: auto;
    padding-left: 34px;
  }
  .setup-team-selector { grid-template-rows: auto auto auto; }
  .setup-club-ball-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .setup-schedule-marks { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
  .setup-config-layout,
  .setup-accept-btn,
  .setup-loading-tasks li.is-loading i,
  .setup-loading-bar span { animation: none !important; transition: none !important; }
}
`;
