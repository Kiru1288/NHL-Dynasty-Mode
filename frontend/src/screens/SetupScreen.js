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
  const logoSrc =
    selected?.logo ||
    resolveFranchiseTeamLogo(selected?.raw || selected, selected?.name || teamDisplayName(selected));

  const steps = [
    { key: "team", label: "Team", done: Boolean(selected) },
    { key: "name", label: "Name", done: Boolean(gmName?.trim()) },
    { key: "sign", label: "Sign", done: signatureReady },
  ];

  return (
    <article className="setup-deed-sheet">
      <div className="setup-deed-paper">
        <h2 className="setup-deed-title">General Manager</h2>

        <div className="setup-deed-club">
          {logoSrc ? <img src={logoSrc} alt="" /> : <em>{teamCode}</em>}
          <div>
            <strong>{teamName}</strong>
            <small>{teamCode} · {conference} · {division}</small>
          </div>
        </div>

        <ol className="setup-steps" aria-label="Setup progress">
          {steps.map((step, i) => (
            <li key={step.key} className={step.done ? "is-done" : ""}>
              <b>{i + 1}</b> {step.label}{step.done ? " ✓" : ""}
            </li>
          ))}
        </ol>

        <label className="setup-deed-gm">
          <span className="setup-sr-only">General Manager name</span>
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

        <div className="setup-deed-sign">
          <SignaturePad onInkChange={setSignatureReady} />
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
          {!canContinue && !loading ? <small>Sign to unlock</small> : null}
        </button>
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

  return (
    <section className="setup-team-selector">
      <header className="setup-panel-heading">
        <strong>Select your team</strong>
      </header>
      <ClubBallBoard teams={clubList} selectedIndex={selectedIndex} onSelect={onSelect} />
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
  const crestSrc =
    selected?.logo ||
    resolveFranchiseTeamLogo(selected?.raw || selected, selected?.name || teamDisplayName(selected));

  return (
    <div className="setup-loading-screen" role="status" aria-live="polite">
      <div className="setup-loading-panel">
        {crestSrc ? (
          <div className="setup-loading-crest" aria-hidden="true">
            <span><img src={crestSrc} alt="" /></span>
          </div>
        ) : null}
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
            <div className="setup-loading-word" aria-hidden="true">
              {"LOADING".split("").map((ch, i) => (
                <b key={i} style={{ animationDelay: `${i * 0.08}s` }}>{ch}</b>
              ))}
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
    // OPERATIONS needs a franchise session — primed after /start in beginFranchise.
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

      {null}
    </div>
  );
}


