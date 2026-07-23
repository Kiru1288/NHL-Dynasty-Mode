import React, {
  forwardRef,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { useGameUI } from "../game/GameUIContext";
import { teamNameToNhlAbbr } from "../game/constants";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";
import cookieFontUrl from "../styles/Cookie/Cookie-Regular.ttf";

const EASTERN_ORDER = [
  "BOS",
  "BUF",
  "DET",
  "FLA",
  "MTL",
  "OTT",
  "TBL",
  "TOR",
  "CAR",
  "CBJ",
  "NJD",
  "NYI",
  "NYR",
  "PHI",
  "PIT",
  "WSH",
];

const WESTERN_ORDER = [
  "UTA",
  "ANA",
  "CGY",
  "CHI",
  "COL",
  "DAL",
  "EDM",
  "LAK",
  "MIN",
  "NSH",
  "SEA",
  "SJS",
  "STL",
  "VAN",
  "VGK",
  "WPG",
];

const DEFAULT_TEAM_ORDER = [...EASTERN_ORDER, ...WESTERN_ORDER];

const FIELD_KEYS = [
  "team",
  "gmName",
  "injuries",
  "signature",
  "start",
];

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

export default NHL_FUN_FACTS;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function shuffleArray(array) {
  const result = [...array];

  for (let i = result.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));

    [result[i], result[j]] = [result[j], result[i]];
  }

  return result;
}

function normalizeCode(raw) {
  if (raw == null) {
    return "";
  }

  const text = String(raw).trim().toUpperCase();

  if (text.length <= 3) {
    return text;
  }

  return teamNameToNhlAbbr(text) || text.slice(0, 3);
}

function teamDisplayName(team) {
  return String(
    team?.name ||
      team?.team_name ||
      team?.display_name ||
      "Team"
  ).trim();
}

function teamCodeFromRow(team) {
  const fromName = teamNameToNhlAbbr(
    team?.name || team?.team_name || ""
  );

  if (fromName) {
    return fromName;
  }

  return normalizeCode(
    team?.abbreviation ||
      team?.abbr ||
      team?.team_id ||
      team?.id ||
      ""
  );
}

function buildOrderedTeams(teams) {
  if (!Array.isArray(teams) || !teams.length) {
    return [];
  }

  const enriched = teams.map((raw, index) => ({
    raw,
    index,
    code: teamCodeFromRow(raw),
    name: teamDisplayName(raw),
    logo: resolveFranchiseTeamLogo(
      raw,
      teamDisplayName(raw)
    ),
  }));

  const claimed = new Set();
  const ordered = [];

  DEFAULT_TEAM_ORDER.forEach((code) => {
    const match = enriched.find(
      (item) =>
        item.code === code &&
        !claimed.has(item.index)
    );

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

function findOrderedIndexFromSetupIndex(
  orderedTeams,
  setupIndex
) {
  const found = orderedTeams.findIndex(
    (item) => item.index === setupIndex
  );

  return found >= 0 ? found : 0;
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

  return colors[code] || ["#e9a83c", "#13d8e7"];
}

function TeamLogoImg({
  team,
  size = 48,
  className = "",
}) {
  const name = teamDisplayName(team);

  const src =
    team?.logo ||
    resolveFranchiseTeamLogo(team, name);

  if (!src) {
    return (
      <div
        className={`setup-logo-empty ${className}`}
        style={{
          width: size,
          height: size,
        }}
        aria-hidden="true"
      />
    );
  }

  return (
    <img
      className={`setup-team-logo ${className}`}
      src={src}
      alt={name}
      draggable={false}
      style={{
        width: size,
        height: size,
      }}
    />
  );
}

function ExecutiveFigure({
  side = "center",
  primary = false,
}) {
  return (
    <div
      className={[
        "appointment-executive",
        `appointment-executive--${side}`,
        primary ? "is-primary" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      aria-hidden="true"
    >
      <div className="appointment-executive-halo" />

      <div className="appointment-executive-head">
        <span className="appointment-head-highlight" />
      </div>

      <div className="appointment-executive-neck" />

      <div className="appointment-executive-body">
        <span className="appointment-lapel appointment-lapel--left" />

        <span className="appointment-lapel appointment-lapel--right" />

        <span className="appointment-shirt" />

        <span className="appointment-tie" />

        <span className="appointment-club-pin" />
      </div>
    </div>
  );
}

function AppointmentDossier({
  team,
  hasGmSignature,
}) {
  if (!team) {
    return null;
  }

  return (
    <div
      className={`appointment-dossier ${
        hasGmSignature ? "is-signed" : ""
      }`}
    >
      <div
        className="appointment-dossier-spine"
        aria-hidden="true"
      />

      <div
        className="appointment-dossier-stitch appointment-dossier-stitch--top"
        aria-hidden="true"
      />

      <div
        className="appointment-dossier-stitch appointment-dossier-stitch--bottom"
        aria-hidden="true"
      />

      <div
        className="appointment-dossier-corner appointment-dossier-corner--tl"
        aria-hidden="true"
      />

      <div
        className="appointment-dossier-corner appointment-dossier-corner--tr"
        aria-hidden="true"
      />

      <div
        className="appointment-dossier-corner appointment-dossier-corner--bl"
        aria-hidden="true"
      />

      <div
        className="appointment-dossier-corner appointment-dossier-corner--br"
        aria-hidden="true"
      />

      <span className="appointment-dossier-kicker">
        Executive Appointment
      </span>

      <div className="appointment-dossier-crest">
        <TeamLogoImg
          team={{
            ...team.raw,
            logo: team.logo,
          }}
          size={68}
        />
      </div>

      <strong className="appointment-dossier-team">
        {team.name}
      </strong>

      <span className="appointment-dossier-office">
        Hockey Operations
      </span>

      <div
        className="appointment-dossier-rule"
        aria-hidden="true"
      />

      <div className="appointment-dossier-status">
        <span
          className="appointment-dossier-status-light"
          aria-hidden="true"
        />

        {hasGmSignature
          ? "Signature Captured"
          : "Access Pending Execution"}
      </div>

      <div
        className="appointment-dossier-clasp"
        aria-hidden="true"
      >
        <span />
      </div>
    </div>
  );
}

function AppointmentRoom({
  orderedTeams,
  selectedOrderedIndex,
  selectedTeam,
  onSelect,
  onStep,
  hasGmSignature,
}) {
  const visibleOffers = useMemo(() => {
    if (!orderedTeams.length) {
      return [];
    }

    return [-2, -1, 0, 1, 2].map((offset) => {
      const index =
        (selectedOrderedIndex +
          offset +
          orderedTeams.length) %
        orderedTeams.length;

      return {
        team: orderedTeams[index],
        index,
        offset,
      };
    });
  }, [orderedTeams, selectedOrderedIndex]);

  const sceneKey =
    selectedTeam?.code || selectedOrderedIndex;

  return (
    <div
      className="appointment-room"
      aria-label="NHL executive appointment room"
    >
      <header className="appointment-registry">
        <div className="appointment-registry-title">
          <span>National Hockey League</span>

          <strong>NHL Executive Registry</strong>
        </div>

        <div className="appointment-registry-file">
          <span>Appointment</span>

          <strong>
            // {selectedTeam?.code || "---"}
          </strong>
        </div>
      </header>

      <div
        className="appointment-scene"
        key={sceneKey}
      >
        <div
          className="appointment-scene-darkener"
          aria-hidden="true"
        />

        <div
          className="appointment-wall"
          aria-hidden="true"
        >
          <span className="appointment-wall-seam appointment-wall-seam--one" />

          <span className="appointment-wall-seam appointment-wall-seam--two" />

          <span className="appointment-wall-seam appointment-wall-seam--three" />

          <span className="appointment-wall-light appointment-wall-light--left" />

          <span className="appointment-wall-light appointment-wall-light--right" />
        </div>

        {selectedTeam ? (
          <div
            className="appointment-wall-crest"
            aria-hidden="true"
          >
            <TeamLogoImg
              team={{
                ...selectedTeam.raw,
                logo: selectedTeam.logo,
              }}
              size={320}
              className="appointment-wall-crest-image"
            />
          </div>
        ) : null}

        <div
          className="appointment-executive-lightbar"
          aria-hidden="true"
        />

        <div className="appointment-executives">
          <ExecutiveFigure side="left" />

          <ExecutiveFigure
            side="center"
            primary
          />

          <ExecutiveFigure side="right" />
        </div>

        <div
          className="appointment-table"
          aria-hidden="true"
        >
          <div className="appointment-table-surface">
            <div className="appointment-table-edge" />

            <div className="appointment-table-beam">
              <span className="appointment-table-beam-core" />

              {selectedTeam ? (
                <TeamLogoImg
                  team={{
                    ...selectedTeam.raw,
                    logo: selectedTeam.logo,
                  }}
                  size={142}
                  className="appointment-table-projection"
                />
              ) : null}
            </div>

            <span className="appointment-table-reflection appointment-table-reflection--left" />

            <span className="appointment-table-reflection appointment-table-reflection--right" />
          </div>
        </div>

        <AppointmentDossier
          team={selectedTeam}
          hasGmSignature={hasGmSignature}
        />

        <div
          className="appointment-seat-shadow"
          aria-hidden="true"
        />
      </div>

      <div className="appointment-rail-wrap">
        <span className="appointment-rail-label">
          Appointment Files
        </span>

        <div className="appointment-rail">
          <button
            type="button"
            className="appointment-rail-arrow"
            onClick={() => onStep(-1)}
            aria-label="Previous appointment"
          >
            ‹
          </button>

          <div className="appointment-rail-files">
            {visibleOffers.map(
              ({ team, index, offset }) => {
                const selected = offset === 0;

                return (
                  <button
                    key={`${team.code}-${team.index}-${offset}`}
                    type="button"
                    className={`appointment-offer-file ${
                      selected
                        ? "is-selected"
                        : ""
                    }`}
                    onClick={() => onSelect(index)}
                    aria-label={`Select ${team.name}`}
                    aria-current={
                      selected ? "true" : undefined
                    }
                  >
                    <span
                      className="appointment-offer-file-tab"
                      aria-hidden="true"
                    />

                    <TeamLogoImg
                      team={{
                        ...team.raw,
                        logo: team.logo,
                      }}
                      size={selected ? 38 : 30}
                      className="appointment-offer-logo"
                    />

                    <span className="appointment-offer-code">
                      {team.code || "NHL"}
                    </span>
                  </button>
                );
              }
            )}
          </div>

          <button
            type="button"
            className="appointment-rail-arrow"
            onClick={() => onStep(1)}
            aria-label="Next appointment"
          >
            ›
          </button>
        </div>

        <span className="appointment-rail-hint">
          Click file · Use ← →
        </span>
      </div>
    </div>
  );
}

function InjuriesToggle({
  enabled,
  onChange,
  isActive,
}) {
  return (
    <div
      className={`setup-injuries-toggle ${
        isActive ? "is-active" : ""
      }`}
      role="group"
      aria-label="Enable injuries in simulation"
    >
      <button
        type="button"
        className={`setup-injuries-btn ${
          enabled ? "is-selected" : ""
        }`}
        onClick={() => onChange(true)}
      >
        Yes
      </button>

      <button
        type="button"
        className={`setup-injuries-btn ${
          !enabled ? "is-selected" : ""
        }`}
        onClick={() => onChange(false)}
      >
        No
      </button>
    </div>
  );
}

function formatContractDate(date = new Date()) {
  return date.toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}

const SignaturePad = forwardRef(
  function SignaturePad(
    {
      isActive,
      onFocus,
      onSignatureChange,
    },
    ref
  ) {
    const wrapRef = useRef(null);
    const canvasRef = useRef(null);

    const drawingRef = useRef(false);
    const lastPointRef = useRef(null);

    const [hasInk, setHasInk] = useState(false);

    const syncInkState = useCallback(
      (nextHasInk) => {
        setHasInk(nextHasInk);

        onSignatureChange?.(nextHasInk);
      },
      [onSignatureChange]
    );

    const getCanvasPoint = useCallback(
      (event) => {
        const canvas = canvasRef.current;

        if (!canvas) {
          return null;
        }

        const rect =
          canvas.getBoundingClientRect();

        const clientX =
          event.clientX ??
          event.touches?.[0]?.clientX;

        const clientY =
          event.clientY ??
          event.touches?.[0]?.clientY;

        if (
          clientX == null ||
          clientY == null
        ) {
          return null;
        }

        return {
          x: clientX - rect.left,
          y: clientY - rect.top,
        };
      },
      []
    );

    const prepareContext = useCallback(
      (ctx) => {
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.strokeStyle = "#e8c894";
        ctx.lineWidth = 2.4;
      },
      []
    );

    const resizeCanvas = useCallback(() => {
      const canvas = canvasRef.current;
      const wrap = wrapRef.current;

      if (!canvas || !wrap) {
        return;
      }

      const rect = wrap.getBoundingClientRect();

      if (
        rect.width <= 0 ||
        rect.height <= 0
      ) {
        return;
      }

      const dpr =
        window.devicePixelRatio || 1;

      const snapshot = hasInk
        ? canvas.toDataURL("image/png")
        : "";

      canvas.width = Math.max(
        1,
        Math.floor(rect.width * dpr)
      );

      canvas.height = Math.max(
        1,
        Math.floor(rect.height * dpr)
      );

      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;

      const ctx = canvas.getContext("2d");

      if (!ctx) {
        return;
      }

      ctx.setTransform(
        dpr,
        0,
        0,
        dpr,
        0,
        0
      );

      prepareContext(ctx);

      if (snapshot) {
        const image = new Image();

        image.onload = () => {
          ctx.drawImage(
            image,
            0,
            0,
            rect.width,
            rect.height
          );
        };

        image.src = snapshot;
      }
    }, [
      hasInk,
      prepareContext,
    ]);

    useEffect(() => {
      resizeCanvas();

      const wrap = wrapRef.current;

      if (
        !wrap ||
        typeof ResizeObserver === "undefined"
      ) {
        return undefined;
      }

      const observer = new ResizeObserver(
        () => {
          resizeCanvas();
        }
      );

      observer.observe(wrap);

      return () => observer.disconnect();
    }, [resizeCanvas]);

    const beginStroke = useCallback(
      (event) => {
        if (
          event.pointerType === "mouse" &&
          event.button !== 0
        ) {
          return;
        }

        const point = getCanvasPoint(event);
        const canvas = canvasRef.current;

        if (!point || !canvas) {
          return;
        }

        event.preventDefault();

        drawingRef.current = true;
        lastPointRef.current = point;

        const ctx = canvas.getContext("2d");

        if (ctx) {
          prepareContext(ctx);

          ctx.beginPath();

          ctx.arc(
            point.x,
            point.y,
            1.1,
            0,
            Math.PI * 2
          );

          ctx.fillStyle = "#e8c894";

          ctx.fill();

          if (!hasInk) {
            syncInkState(true);
          }
        }

        if (
          typeof canvas.setPointerCapture ===
          "function"
        ) {
          canvas.setPointerCapture(
            event.pointerId
          );
        }
      },
      [
        getCanvasPoint,
        hasInk,
        prepareContext,
        syncInkState,
      ]
    );

    const continueStroke = useCallback(
      (event) => {
        if (!drawingRef.current) {
          return;
        }

        const canvas = canvasRef.current;

        const ctx =
          canvas?.getContext("2d");

        const point =
          getCanvasPoint(event);

        const lastPoint =
          lastPointRef.current;

        if (
          !canvas ||
          !ctx ||
          !point ||
          !lastPoint
        ) {
          return;
        }

        event.preventDefault();

        prepareContext(ctx);

        ctx.beginPath();

        ctx.moveTo(
          lastPoint.x,
          lastPoint.y
        );

        ctx.lineTo(
          point.x,
          point.y
        );

        ctx.stroke();

        lastPointRef.current = point;

        if (!hasInk) {
          syncInkState(true);
        }
      },
      [
        getCanvasPoint,
        hasInk,
        prepareContext,
        syncInkState,
      ]
    );

    const endStroke = useCallback(
      (event) => {
        if (!drawingRef.current) {
          return;
        }

        drawingRef.current = false;
        lastPointRef.current = null;

        const canvas = canvasRef.current;

        if (
          canvas &&
          typeof canvas.releasePointerCapture ===
            "function" &&
          canvas.hasPointerCapture?.(
            event.pointerId
          )
        ) {
          canvas.releasePointerCapture(
            event.pointerId
          );
        }
      },
      []
    );

    const clearSignature = useCallback(() => {
      const canvas = canvasRef.current;

      const ctx =
        canvas?.getContext("2d");

      if (!canvas || !ctx) {
        return;
      }

      ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      );

      syncInkState(false);
    }, [syncInkState]);

    return (
      <div
        ref={ref}
        className={[
          "setup-signature-pad",
          isActive ? "is-active" : "",
          hasInk ? "has-ink" : "",
        ]
          .filter(Boolean)
          .join(" ")}
        onFocus={onFocus}
        tabIndex={0}
      >
        <div
          ref={wrapRef}
          className="setup-signature-pad-surface"
        >
          <span
            className="setup-signature-placeholder"
            aria-hidden="true"
          >
            Sign Here
          </span>

          <canvas
            ref={canvasRef}
            className="setup-signature-canvas"
            aria-label="Draw your General Manager signature"
            onPointerDown={(event) => {
              onFocus?.();

              beginStroke(event);
            }}
            onPointerMove={continueStroke}
            onPointerUp={endStroke}
            onPointerCancel={endStroke}
            onPointerLeave={endStroke}
          />

          <div
            className="setup-signature-line"
            aria-hidden="true"
          />
        </div>

        <button
          type="button"
          className="setup-signature-clear"
          onClick={clearSignature}
        >
          Clear signature
        </button>
      </div>
    );
  }
);

function SetupLoadingScreen({
  selected,
  gmName,
  injuriesEnabled,
}) {
  const shuffledFacts = useMemo(
    () => shuffleArray(NHL_FUN_FACTS),
    []
  );

  const [factIndex, setFactIndex] =
    useState(0);

  useEffect(() => {
    const intervalId =
      window.setInterval(() => {
        setFactIndex(
          (current) =>
            (current + 1) %
            shuffledFacts.length
        );
      }, 10000);

    return () =>
      window.clearInterval(intervalId);
  }, [shuffledFacts.length]);

  return (
    <div
      className="setup-loading-screen"
      role="status"
      aria-live="polite"
    >
      <div className="setup-loading-noise" />

      <div className="setup-loading-card">
        <div
          className="setup-loading-spinner"
          aria-hidden="true"
        />

        <p className="setup-loading-kicker">
          Franchise Mode
        </p>

        <h2 className="setup-loading-title">
          Welcome to{" "}
          {selected?.name || "the NHL"}
        </h2>

        <p className="setup-loading-copy">
          GM{" "}
          {gmName?.trim() ||
            "General Manager"}{" "}
          has entered hockey operations.
        </p>

        <div className="setup-loading-steps">
          <span>Loading roster</span>

          <span>Building schedule</span>

          <span>
            {injuriesEnabled
              ? "Enabling injuries"
              : "Disabling injuries"}
          </span>
        </div>

        <div className="setup-fact-card">
          <span className="setup-fact-label">
            Hockey fact
          </span>

          <p>
            {shuffledFacts[factIndex]}
          </p>

          <div
            className="setup-fact-progress"
            key={factIndex}
          />
        </div>
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
    injuriesEnabled,
    setInjuriesEnabled,
    beginFranchise,
    loading,
    loadTeams,
    error,
  } = useGameUI();

  const [
    activeField,
    setActiveField,
  ] = useState("team");

  const [
    statusText,
    setStatusText,
  ] = useState(
    "Finalize your appointment."
  );

  const [
    hasGmSignature,
    setHasGmSignature,
  ] = useState(false);

  const startButtonRef = useRef(null);
  const gmInputRef = useRef(null);
  const signaturePadRef = useRef(null);

  useEffect(() => {
    loadTeams();
  }, [loadTeams]);

  const orderedTeams = useMemo(
    () => buildOrderedTeams(teams),
    [teams]
  );

  const orderedIndex = useMemo(
    () =>
      findOrderedIndexFromSetupIndex(
        orderedTeams,
        setupTeamIndex
      ),
    [
      orderedTeams,
      setupTeamIndex,
    ]
  );

  const selected =
    orderedTeams[orderedIndex] || null;

  const selectedCode =
    selected?.code || "";

  const [
    accentPrimary,
    accentSecondary,
  ] = teamAccentForCode(selectedCode);

  const contractDate = useMemo(
    () => formatContractDate(),
    []
  );

  const handleSignatureChange =
    useCallback((signed) => {
      setHasGmSignature(Boolean(signed));

      setStatusText(
        signed
          ? "Signature captured."
          : "Signature required."
      );
    }, []);

  useEffect(() => {
    if (
      selected?.index != null &&
      selected.index !== setupTeamIndex
    ) {
      setSetupTeamIndex(selected.index);
    }
  }, [
    selected,
    setupTeamIndex,
    setSetupTeamIndex,
  ]);

  const setTeamByOrderedIndex =
    useCallback(
      (nextOrderedIndex) => {
        if (!orderedTeams.length) {
          return;
        }

        const safeIndex =
          ((nextOrderedIndex %
            orderedTeams.length) +
            orderedTeams.length) %
          orderedTeams.length;

        const team =
          orderedTeams[safeIndex];

        if (!team) {
          return;
        }

        setSetupTeamIndex(team.index);

        setStatusText(
          `${team.name} appointment selected.`
        );
      },
      [
        orderedTeams,
        setSetupTeamIndex,
      ]
    );

  const handleInjuriesChange =
    useCallback(
      (enabled) => {
        setInjuriesEnabled(enabled);

        setStatusText(
          enabled
            ? "League Health System enabled."
            : "League Health System disabled."
        );
      },
      [setInjuriesEnabled]
    );

  const onStart = useCallback(() => {
    if (
      !selected ||
      !hasGmSignature
    ) {
      return;
    }

    setStatusText(
      `Executing appointment with ${selected.name}…`
    );

    beginFranchise();
  }, [
    beginFranchise,
    hasGmSignature,
    selected,
  ]);

  const cycleFieldValue = useCallback(
    (fieldKey, direction) => {
      const dir =
        direction >= 0 ? 1 : -1;

      if (fieldKey === "team") {
        setTeamByOrderedIndex(
          orderedIndex + dir
        );

        return;
      }

      if (fieldKey === "injuries") {
        handleInjuriesChange(
          !injuriesEnabled
        );
      }
    },
    [
      handleInjuriesChange,
      injuriesEnabled,
      orderedIndex,
      setTeamByOrderedIndex,
    ]
  );

  useEffect(() => {
    function onKeyDown(event) {
      const targetTag =
        event.target?.tagName;

      const isInputFocused =
        targetTag === "INPUT" ||
        targetTag === "TEXTAREA";

      if (
        isInputFocused &&
        activeField === "gmName"
      ) {
        if (event.key === "Enter") {
          event.preventDefault();

          setActiveField("signature");

          signaturePadRef.current?.focus?.();
        }

        return;
      }

      if (event.key === "ArrowUp") {
        event.preventDefault();

        const current =
          FIELD_KEYS.indexOf(activeField);

        setActiveField(
          FIELD_KEYS[
            clamp(
              current - 1,
              0,
              FIELD_KEYS.length - 1
            )
          ]
        );

        return;
      }

      if (event.key === "ArrowDown") {
        event.preventDefault();

        const current =
          FIELD_KEYS.indexOf(activeField);

        setActiveField(
          FIELD_KEYS[
            clamp(
              current + 1,
              0,
              FIELD_KEYS.length - 1
            )
          ]
        );

        return;
      }

      if (event.key === "ArrowLeft") {
        event.preventDefault();

        cycleFieldValue(
          activeField,
          -1
        );

        return;
      }

      if (event.key === "ArrowRight") {
        event.preventDefault();

        cycleFieldValue(
          activeField,
          1
        );

        return;
      }

      if (event.key === "Enter") {
        event.preventDefault();

        if (
          activeField === "start" &&
          hasGmSignature
        ) {
          onStart();
        } else if (
          activeField === "gmName"
        ) {
          gmInputRef.current?.focus?.();
        } else if (
          activeField === "signature"
        ) {
          signaturePadRef.current?.focus?.();

          if (hasGmSignature) {
            setActiveField("start");

            startButtonRef.current?.focus?.();
          }
        }
      }
    }

    window.addEventListener(
      "keydown",
      onKeyDown
    );

    return () =>
      window.removeEventListener(
        "keydown",
        onKeyDown
      );
  }, [
    activeField,
    cycleFieldValue,
    hasGmSignature,
    onStart,
  ]);

  return (
    <div
      className="nhlcal-root setup-root"
      style={{
        "--team-accent": accentPrimary,
        "--team-accent-2":
          accentSecondary,
      }}
    >
      {loading ? (
        <SetupLoadingScreen
          selected={selected}
          gmName={gmName}
          injuriesEnabled={
            injuriesEnabled
          }
        />
      ) : null}

      <main className="setup-main">
        <section
          className={`setup-appointment-panel ${
            activeField === "team"
              ? "is-active-panel"
              : ""
          }`}
        >
          <AppointmentRoom
            orderedTeams={orderedTeams}
            selectedOrderedIndex={
              orderedIndex
            }
            selectedTeam={selected}
            onSelect={(index) => {
              setActiveField("team");

              setTeamByOrderedIndex(index);
            }}
            onStep={(direction) => {
              setActiveField("team");

              setTeamByOrderedIndex(
                orderedIndex + direction
              );
            }}
            hasGmSignature={
              hasGmSignature
            }
          />
        </section>

        <section
          className="setup-config-panel"
          aria-label="General Manager Agreement"
        >
          <div className="setup-contract-scroll">
            <div
              className="setup-contract-ornament"
              aria-hidden="true"
            >
              ✦ ✦ ✦
            </div>

            <h2 className="setup-contract-title">
              GENERAL MANAGER AGREEMENT
            </h2>

            <p className="setup-contract-intro">
              This Agreement is made between
            </p>

            <div className="setup-contract-parties">
              <div className="setup-contract-team-badge">
                {selected?.logo ? (
                  <img
                    className="setup-contract-logo"
                    src={selected.logo}
                    alt=""
                    draggable={false}
                  />
                ) : (
                  <div
                    className="setup-contract-logo setup-logo-empty"
                    aria-hidden="true"
                  />
                )}
              </div>

              <div className="setup-contract-party-copy">
                <h3 className="setup-contract-team-name">
                  {selected?.name ||
                    "Select a team"}
                </h3>

                <div
                  className={`setup-contract-gm-field ${
                    activeField === "gmName"
                      ? "is-active"
                      : ""
                  }`}
                >
                  <label
                    className="setup-contract-gm-label"
                    htmlFor="setup-gm-name"
                  >
                    General Manager
                  </label>

                  <input
                    id="setup-gm-name"
                    ref={gmInputRef}
                    className="setup-contract-gm-input"
                    value={gmName}
                    onChange={(event) =>
                      setGmName(
                        event.target.value
                      )
                    }
                    onFocus={() =>
                      setActiveField("gmName")
                    }
                    placeholder="Enter GM name"
                    maxLength={80}
                    autoComplete="off"
                  />
                </div>
              </div>

              <div className="setup-contract-year">
                <span className="setup-contract-year-label">
                  Year One
                </span>
              </div>
            </div>

            <div
              className="setup-contract-divider"
              aria-hidden="true"
            />

            <p className="setup-contract-body">
              This General Manager Agreement is
              made between the selected Club and
              the undersigned General Manager.
              By signing below, the General
              Manager accepts authority over
              hockey operations.
            </p>

            <div
              className="setup-contract-divider setup-contract-divider--thin"
              aria-hidden="true"
            />

            <article
              className={`setup-clause ${
                activeField === "injuries"
                  ? "is-active"
                  : ""
              }`}
            >
              <h4 className="setup-clause-heading">
                Clause 01 — League Health System
              </h4>

              <p className="setup-clause-text">
                Confirm whether injuries will
                operate during this franchise.
              </p>

              <p className="setup-clause-question">
                Enable injuries?
              </p>

              <InjuriesToggle
                enabled={injuriesEnabled}
                onChange={
                  handleInjuriesChange
                }
                isActive={
                  activeField === "injuries"
                }
              />
            </article>

            <div
              className="setup-contract-divider setup-contract-divider--thin"
              aria-hidden="true"
            />

            <article className="setup-clause">
              <h4 className="setup-clause-heading">
                Clause 02 — Role and Authority
              </h4>

              <p className="setup-clause-text">
                The General Manager receives
                authority over transactions,
                contracts, staff, and franchise
                strategy.
              </p>
            </article>

            <div
              className="setup-contract-divider setup-contract-divider--thin"
              aria-hidden="true"
            />

            <article className="setup-clause">
              <h4 className="setup-clause-heading">
                Clause 03 — Term
              </h4>

              <p className="setup-clause-text">
                Appointment begins immediately
                upon execution of this
                agreement.
              </p>
            </article>

            <div
              className="setup-contract-divider setup-contract-divider--thin"
              aria-hidden="true"
            />

            <article className="setup-clause">
              <h4 className="setup-clause-heading">
                Clause 04 — Acceptance
              </h4>

              <p className="setup-clause-text">
                Signature confirms acceptance
                of the General Manager
                appointment.
              </p>
            </article>

            <div
              className="setup-contract-divider"
              aria-hidden="true"
            />

            <div className="setup-signature-area">
              <div className="setup-signature-fields">
                <div
                  className={`setup-signature-block ${
                    activeField === "signature"
                      ? "is-active"
                      : ""
                  }`}
                >
                  <span className="setup-signature-label">
                    General Manager Signature
                  </span>

                  <SignaturePad
                    ref={signaturePadRef}
                    isActive={
                      activeField === "signature"
                    }
                    onFocus={() =>
                      setActiveField(
                        "signature"
                      )
                    }
                    onSignatureChange={
                      handleSignatureChange
                    }
                  />
                </div>

                <div className="setup-date-block">
                  <span className="setup-signature-label">
                    Date
                  </span>

                  <time
                    className="setup-date-value"
                    dateTime={contractDate}
                  >
                    {contractDate}
                  </time>
                </div>
              </div>

              <div
                className="setup-wax-seal"
                aria-hidden="true"
              >
                <div className="setup-wax-seal-inner">
                  {selected?.logo ? (
                    <img
                      className="setup-wax-seal-logo"
                      src={selected.logo}
                      alt=""
                      draggable={false}
                    />
                  ) : (
                    <span className="setup-wax-seal-mark">
                      NHL
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>

          <p
            className="setup-status setup-sr-status"
            aria-live="polite"
          >
            {statusText}
          </p>

          {error ? (
            <div className="setup-error">
              {error}
            </div>
          ) : null}

          {hasGmSignature ? (
            <button
              ref={startButtonRef}
              type="button"
              className={`setup-start-btn ${
                activeField === "start"
                  ? "is-active"
                  : ""
              }`}
              disabled={
                loading ||
                !teams.length ||
                !orderedTeams.length
              }
              onClick={onStart}
              onFocus={() =>
                setActiveField("start")
              }
            >
              <span>ACCEPT APPOINTMENT</span>

              <small>
                EXECUTE GENERAL MANAGER AGREEMENT
              </small>
            </button>
          ) : (
            <p className="setup-signature-hint">
              Draw your signature to execute the
              appointment.
            </p>
          )}
        </section>
      </main>

      <style>{SETUP_SCREEN_CSS}</style>
    </div>
  );
}

const SETUP_SCREEN_CSS = `
@font-face {
  font-family: "CookieAgreement";
  src: url("${cookieFontUrl}") format("truetype");
  font-weight: 400;
  font-style: normal;
  font-display: swap;
}

.nhlcal-root.setup-root {
  --text: #f0e0c0;
  --muted: rgba(200, 175, 130, 0.62);
  --gold: #e8c894;
  --red: #ff606d;

  position: relative;
  display: flex;
  flex-direction: column;

  width: 100%;
  height: 100dvh;
  max-height: 100dvh;

  overflow: hidden;

  color: var(--text);

  background:
    radial-gradient(
      circle at 18% 0%,
      color-mix(
        in srgb,
        var(--team-accent) 10%,
        rgba(191, 109, 54, 0.12)
      ),
      transparent 32%
    ),
    radial-gradient(
      circle at 86% 12%,
      rgba(120, 34, 23, 0.1),
      transparent 30%
    ),
    linear-gradient(
      180deg,
      #0c0e12,
      #06080a
    );

  font-family:
    Inter,
    ui-sans-serif,
    system-ui,
    -apple-system,
    BlinkMacSystemFont,
    "Segoe UI",
    sans-serif;
}

.setup-root *,
.setup-root *::before,
.setup-root *::after {
  box-sizing: border-box;
}

.setup-root button,
.setup-root input {
  font-family: inherit;
}

.setup-root::before {
  content: "";

  position: fixed;
  inset: 0;

  pointer-events: none;

  opacity: 0.14;

  background:
    repeating-linear-gradient(
      155deg,
      rgba(255, 255, 255, 0.03) 0 1px,
      transparent 1px 17px
    );
}

.setup-main {
  position: relative;
  z-index: 2;

  flex: 1;
  min-height: 0;

  display: grid;

  grid-template-columns:
    minmax(0, 1.15fr)
    minmax(320px, 0.85fr);

  gap: 16px;

  padding: 18px 20px 16px;
}

.setup-appointment-panel {
  position: relative;

  min-height: 0;

  overflow: hidden;

  display: flex;
  flex-direction: column;

  border:
    1px solid
    rgba(255, 255, 255, 0.08);

  border-radius:
    7px 14px 9px 18px;

  background:
    radial-gradient(
      circle at 50% 18%,
      color-mix(
        in srgb,
        var(--team-accent) 9%,
        transparent
      ),
      transparent 34%
    ),
    linear-gradient(
      180deg,
      rgba(10, 11, 13, 0.99),
      rgba(3, 4, 6, 1)
    );

  box-shadow:
    0 28px 68px rgba(0, 0, 0, 0.62),
    inset 0 0 90px rgba(0, 0, 0, 0.5);
}

.setup-appointment-panel::after {
  content: "";

  position: absolute;
  inset: 0;

  z-index: 20;

  pointer-events: none;

  box-shadow:
    inset 60px 0 85px rgba(0, 0, 0, 0.54),
    inset -60px 0 85px rgba(0, 0, 0, 0.54),
    inset 0 -55px 70px rgba(0, 0, 0, 0.42);
}

.setup-appointment-panel.is-active-panel {
  border-color:
    color-mix(
      in srgb,
      var(--team-accent) 35%,
      rgba(255, 255, 255, 0.12)
    );

  box-shadow:
    0 28px 68px rgba(0, 0, 0, 0.62),
    0 0 34px
      color-mix(
        in srgb,
        var(--team-accent) 12%,
        transparent
      ),
    inset 0 0 90px rgba(0, 0, 0, 0.5);
}

.appointment-room {
  position: relative;
  z-index: 1;

  flex: 1;
  min-height: 0;

  display: grid;

  grid-template-rows:
    auto
    minmax(0, 1fr)
    auto;

  overflow: hidden;
}

.appointment-registry {
  position: relative;
  z-index: 30;

  min-height: 62px;

  display: flex;
  align-items: flex-start;
  justify-content: space-between;

  gap: 16px;

  padding: 15px 18px 10px;

  border-bottom:
    1px solid
    rgba(255, 255, 255, 0.055);

  background:
    linear-gradient(
      180deg,
      rgba(0, 0, 0, 0.46),
      rgba(0, 0, 0, 0)
    );
}

.appointment-registry-title,
.appointment-registry-file {
  display: grid;
  gap: 2px;
}

.appointment-registry-title span,
.appointment-registry-file span {
  font-size: 8px;
  font-weight: 800;

  letter-spacing: 0.2em;
  text-transform: uppercase;

  color:
    rgba(228, 219, 200, 0.38);
}

.appointment-registry-title strong,
.appointment-registry-file strong {
  font-size: 11px;
  font-weight: 950;

  letter-spacing: 0.16em;
  text-transform: uppercase;

  color:
    rgba(241, 234, 219, 0.82);
}

.appointment-registry-file {
  text-align: right;
}

.appointment-registry-file strong {
  color:
    color-mix(
      in srgb,
      var(--team-accent) 64%,
      #f1eadb
    );
}

.appointment-scene {
  position: relative;

  min-height: 0;

  overflow: hidden;
  isolation: isolate;

  perspective: 1100px;

  background:
    radial-gradient(
      ellipse at 50% 23%,
      color-mix(
        in srgb,
        var(--team-accent) 10%,
        transparent
      ),
      transparent 31%
    ),
    linear-gradient(
      180deg,
      #0a0b0d 0%,
      #050607 48%,
      #020304 100%
    );
}

.appointment-scene::before {
  content: "";

  position: absolute;
  inset: 0;

  z-index: 50;

  pointer-events: none;

  background: #020304;

  animation:
    appointmentSceneReveal
    620ms
    cubic-bezier(.2, .75, .2, 1)
    forwards;
}

.appointment-scene::after {
  content: "";

  position: absolute;
  inset: 0;

  z-index: 25;

  pointer-events: none;

  background:
    radial-gradient(
      ellipse at 50% 34%,
      transparent 0 38%,
      rgba(0, 0, 0, 0.18) 56%,
      rgba(0, 0, 0, 0.72) 100%
    ),
    linear-gradient(
      90deg,
      rgba(0, 0, 0, 0.42),
      transparent 18%,
      transparent 82%,
      rgba(0, 0, 0, 0.42)
    );
}

.appointment-scene-darkener {
  position: absolute;
  inset: 0;

  z-index: 24;

  pointer-events: none;

  background:
    linear-gradient(
      180deg,
      transparent 52%,
      rgba(0, 0, 0, 0.64)
    );
}

.appointment-wall {
  position: absolute;

  z-index: 0;

  inset: 0 0 38%;

  overflow: hidden;

  border-bottom:
    1px solid
    rgba(255, 255, 255, 0.055);

  background:
    linear-gradient(
      180deg,
      rgba(255, 255, 255, 0.018),
      transparent 22%
    ),
    radial-gradient(
      circle at 50% 46%,
      color-mix(
        in srgb,
        var(--team-accent) 10%,
        rgba(255, 255, 255, 0.02)
      ),
      transparent 40%
    ),
    #090a0c;
}

.appointment-wall::before {
  content: "";

  position: absolute;
  inset: 0;

  opacity: 0.42;

  background:
    repeating-linear-gradient(
      90deg,
      transparent 0 152px,
      rgba(255, 255, 255, 0.045) 153px,
      transparent 154px
    );
}

.appointment-wall::after {
  content: "";

  position: absolute;

  top: 0;
  left: 12%;
  right: 12%;

  height: 1px;

  background:
    linear-gradient(
      90deg,
      transparent,
      color-mix(
        in srgb,
        var(--team-accent) 55%,
        white
      ),
      transparent
    );

  box-shadow:
    0 0 34px 9px
      color-mix(
        in srgb,
        var(--team-accent) 10%,
        transparent
      );
}

.appointment-wall-seam {
  position: absolute;

  top: 0;
  bottom: 0;

  width: 1px;

  background:
    linear-gradient(
      180deg,
      transparent,
      rgba(255, 255, 255, 0.07),
      transparent
    );
}

.appointment-wall-seam--one {
  left: 25%;
}

.appointment-wall-seam--two {
  left: 50%;
}

.appointment-wall-seam--three {
  left: 75%;
}

.appointment-wall-light {
  position: absolute;

  top: -14%;

  width: 16%;
  height: 116%;

  opacity: 0.16;

  filter: blur(10px);

  background:
    linear-gradient(
      180deg,
      rgba(255, 255, 255, 0.12),
      transparent 62%
    );
}

.appointment-wall-light--left {
  left: 8%;
  transform: skewX(-8deg);
}

.appointment-wall-light--right {
  right: 8%;
  transform: skewX(8deg);
}

.appointment-wall-crest {
  position: absolute;

  z-index: 1;

  top: 1.5%;
  left: 50%;

  width: min(42%, 340px);

  aspect-ratio: 1;

  transform: translateX(-50%);

  display: grid;
  place-items: center;

  opacity: 0;

  animation:
    appointmentCrestReveal
    700ms
    140ms
    ease-out
    forwards;
}

.appointment-wall-crest::before {
  content: "";

  position: absolute;
  inset: 2%;

  border-radius: 50%;

  background:
    radial-gradient(
      circle,
      color-mix(
        in srgb,
        var(--team-accent) 17%,
        transparent
      ),
      transparent 66%
    );

  filter: blur(18px);
}

.appointment-wall-crest-image {
  position: relative;

  width: 100% !important;
  height: 100% !important;

  opacity: 0.14;

  filter:
    saturate(0.72)
    contrast(1.06)
    drop-shadow(
      0 0 45px
      color-mix(
        in srgb,
        var(--team-accent) 36%,
        transparent
      )
    );
}

.appointment-executive-lightbar {
  position: absolute;

  z-index: 2;

  top: 30.5%;
  left: 18%;
  right: 18%;

  height: 2px;

  opacity: 0;

  background:
    linear-gradient(
      90deg,
      transparent,
      rgba(255, 255, 255, 0.26),
      color-mix(
        in srgb,
        var(--team-accent) 36%,
        white
      ),
      rgba(255, 255, 255, 0.26),
      transparent
    );

  box-shadow:
    0 14px 55px
      color-mix(
        in srgb,
        var(--team-accent) 11%,
        transparent
      );

  animation:
    appointmentLightbarReveal
    650ms
    170ms
    ease-out
    forwards;
}

.appointment-executives {
  position: absolute;

  z-index: 3;

  top: 17.5%;
  left: 16%;
  right: 16%;

  height: 34%;

  display: flex;
  align-items: flex-end;
  justify-content: space-between;

  gap: 7%;

  opacity: 0;

  transform: translateY(10px);

  animation:
    appointmentExecutivesReveal
    600ms
    200ms
    cubic-bezier(.18, .72, .24, 1)
    forwards;
}

.appointment-executive {
  position: relative;

  width: 25%;
  max-width: 132px;
  height: 100%;

  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-end;

  transform-origin: center bottom;

  opacity: 0.72;

  animation:
    appointmentExecutiveIdle
    5.8s
    ease-in-out
    infinite;
}

.appointment-executive--left {
  transform:
    scale(0.86)
    rotate(1.2deg);

  animation-delay: -1.8s;
}

.appointment-executive--right {
  transform:
    scale(0.86)
    rotate(-1.2deg);

  animation-delay: -3.4s;
}

.appointment-executive.is-primary {
  z-index: 2;

  opacity: 0.95;

  transform: scale(1.08);
}

.appointment-executive-halo {
  position: absolute;

  top: 0;
  left: 50%;

  width: 120%;
  height: 84%;

  transform: translateX(-50%);

  opacity: 0.46;

  filter: blur(13px);

  background:
    radial-gradient(
      ellipse at 50% 20%,
      rgba(255, 255, 255, 0.11),
      transparent 58%
    );
}

.appointment-executive-head {
  position: relative;

  z-index: 2;

  width: 46px;
  height: 50px;

  flex: 0 0 auto;

  border:
    1px solid
    rgba(255, 255, 255, 0.07);

  border-radius:
    48% 48% 44% 44% /
    29% 29% 58% 58%;

  background:
    radial-gradient(
      ellipse at 38% 20%,
      rgba(255, 255, 255, 0.11),
      transparent 28%
    ),
    linear-gradient(
      90deg,
      #090a0b,
      #1a1c1f 44%,
      #090a0b
    );

  box-shadow:
    inset 0 5px 6px
      rgba(255, 255, 255, 0.03),
    inset 0 -11px 14px
      rgba(0, 0, 0, 0.68),
    0 13px 25px
      rgba(0, 0, 0, 0.72);
}

.appointment-head-highlight {
  position: absolute;

  top: 5px;
  left: 14%;
  right: 14%;

  height: 2px;

  border-radius: 50%;

  background:
    linear-gradient(
      90deg,
      transparent,
      rgba(255, 255, 255, 0.22),
      transparent
    );
}

.appointment-executive-neck {
  position: relative;

  z-index: 1;

  width: 20px;
  height: 9px;

  margin-top: -2px;

  background: #0a0b0c;
}

.appointment-executive-body {
  position: relative;

  z-index: 1;

  width: 100%;
  height: 83px;

  margin-top: -1px;

  overflow: hidden;

  clip-path:
    polygon(
      20% 4%,
      39% 0,
      61% 0,
      80% 4%,
      100% 25%,
      92% 100%,
      8% 100%,
      0 25%
    );

  border-top:
    1px solid
    rgba(255, 255, 255, 0.07);

  background:
    radial-gradient(
      ellipse at 50% 0%,
      rgba(255, 255, 255, 0.055),
      transparent 42%
    ),
    linear-gradient(
      90deg,
      #050607,
      #15171a 42%,
      #111316 58%,
      #050607
    );

  box-shadow:
    inset 0 -26px 36px
      rgba(0, 0, 0, 0.76),
    0 25px 34px
      rgba(0, 0, 0, 0.72);
}

.appointment-shirt {
  position: absolute;

  top: 0;
  left: 42%;

  width: 16%;
  height: 46%;

  clip-path:
    polygon(
      0 0,
      100% 0,
      72% 100%,
      28% 100%
    );

  opacity: 0.62;

  background:
    linear-gradient(
      180deg,
      rgba(224, 226, 226, 0.72),
      rgba(120, 126, 130, 0.16)
    );
}

.appointment-lapel {
  position: absolute;

  top: 1px;

  width: 38%;
  height: 55%;

  opacity: 0.72;
}

.appointment-lapel--left {
  left: 10%;

  clip-path:
    polygon(
      0 0,
      100% 0,
      72% 100%,
      44% 46%
    );

  background:
    linear-gradient(
      135deg,
      rgba(255, 255, 255, 0.035),
      transparent 48%
    );
}

.appointment-lapel--right {
  right: 10%;

  clip-path:
    polygon(
      0 0,
      100% 0,
      56% 46%,
      28% 100%
    );

  background:
    linear-gradient(
      225deg,
      rgba(255, 255, 255, 0.035),
      transparent 48%
    );
}

.appointment-tie {
  position: absolute;

  z-index: 3;

  top: 7px;
  left: 47.5%;

  width: 5%;
  min-width: 5px;
  height: 39px;

  clip-path:
    polygon(
      50% 0,
      100% 18%,
      69% 100%,
      31% 100%,
      0 18%
    );

  background:
    linear-gradient(
      180deg,
      color-mix(
        in srgb,
        var(--team-accent) 74%,
        #181818
      ),
      #070707
    );

  box-shadow:
    0 0 8px
      color-mix(
        in srgb,
        var(--team-accent) 18%,
        transparent
      );
}

.appointment-club-pin {
  position: absolute;

  z-index: 3;

  top: 26px;
  right: 25%;

  width: 6px;
  height: 6px;

  border-radius: 50%;

  background:
    var(--team-accent);

  box-shadow:
    0 0 10px
      color-mix(
        in srgb,
        var(--team-accent) 58%,
        transparent
      );
}

.appointment-table {
  position: absolute;

  z-index: 4;

  left: 2%;
  right: 2%;
  bottom: -15%;

  height: 67%;

  pointer-events: none;
}

.appointment-table-surface {
  position: absolute;
  inset: 0;

  overflow: hidden;

  clip-path:
    polygon(
      34% 0,
      66% 0,
      100% 100%,
      0 100%
    );

  background:
    linear-gradient(
      90deg,
      rgba(0, 0, 0, 0.7),
      transparent 27%,
      transparent 73%,
      rgba(0, 0, 0, 0.7)
    ),
    repeating-linear-gradient(
      94deg,
      rgba(255, 255, 255, 0.017) 0 1px,
      transparent 1px 18px
    ),
    linear-gradient(
      180deg,
      #181310,
      #110d0a 34%,
      #080706
    );

  box-shadow:
    inset 0 3px 0
      rgba(255, 255, 255, 0.07),
    inset 0 -90px 110px
      rgba(0, 0, 0, 0.72),
    0 -34px 85px
      rgba(0, 0, 0, 0.8);
}

.appointment-table-edge {
  position: absolute;

  z-index: 6;

  top: 0;
  left: 0;
  right: 0;

  height: 2px;

  background:
    linear-gradient(
      90deg,
      transparent 31%,
      rgba(255, 255, 255, 0.22),
      color-mix(
        in srgb,
        var(--team-accent) 32%,
        white
      ),
      rgba(255, 255, 255, 0.22),
      transparent 69%
    );
}

.appointment-table-beam {
  position: absolute;

  z-index: 3;

  top: 0;
  left: 44.5%;

  width: 11%;
  height: 100%;

  transform-origin: top center;

  overflow: hidden;

  clip-path:
    polygon(
      39% 0,
      61% 0,
      100% 100%,
      0 100%
    );

  opacity: 0;

  background:
    linear-gradient(
      90deg,
      transparent,
      color-mix(
        in srgb,
        var(--team-accent) 18%,
        transparent
      ) 18%,
      color-mix(
        in srgb,
        var(--team-accent) 40%,
        rgba(255, 255, 255, 0.05)
      ) 50%,
      color-mix(
        in srgb,
        var(--team-accent) 18%,
        transparent
      ) 82%,
      transparent
    );

  box-shadow:
    inset 0 0 22px
      color-mix(
        in srgb,
        var(--team-accent) 22%,
        transparent
      ),
    0 0 34px
      color-mix(
        in srgb,
        var(--team-accent) 14%,
        transparent
      );

  animation:
    appointmentBeamEnter
    720ms
    90ms
    cubic-bezier(.18, .78, .22, 1)
    forwards;
}

.appointment-table-beam-core {
  position: absolute;

  top: 0;
  bottom: 0;
  left: 49%;

  width: 2%;
  min-width: 1px;

  opacity: 0.72;

  background:
    linear-gradient(
      180deg,
      color-mix(
        in srgb,
        var(--team-accent) 72%,
        white
      ),
      var(--team-accent) 48%,
      transparent
    );

  box-shadow:
    0 0 16px 3px
      color-mix(
        in srgb,
        var(--team-accent) 30%,
        transparent
      );
}

.appointment-table-projection {
  position: absolute;

  top: 40%;
  left: 50%;

  width: 130% !important;
  height: 130% !important;

  transform:
    translate(-50%, -50%);

  opacity: 0.075;

  filter:
    grayscale(0.4)
    saturate(0.8)
    drop-shadow(
      0 0 24px
      color-mix(
        in srgb,
        var(--team-accent) 40%,
        transparent
      )
    );
}

.appointment-table-reflection {
  position: absolute;

  top: 6%;

  height: 76%;
  width: 1px;

  opacity: 0.12;

  transform-origin: top;

  background:
    linear-gradient(
      180deg,
      rgba(255, 255, 255, 0.7),
      transparent
    );
}

.appointment-table-reflection--left {
  left: 29%;

  transform: rotate(15deg);
}

.appointment-table-reflection--right {
  right: 29%;

  transform: rotate(-15deg);
}

.appointment-dossier {
  position: absolute;

  z-index: 10;

  left: 50%;
  bottom: 4.5%;

  width:
    clamp(
      210px,
      35%,
      292px
    );

  aspect-ratio: 1.34 / 1;

  transform:
    translateX(-50%)
    perspective(850px)
    rotateX(54deg)
    rotateZ(-0.8deg);

  transform-origin: center bottom;

  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;

  gap: 5px;

  padding:
    19px 30px 22px;

  overflow: hidden;

  color:
    rgba(225, 216, 196, 0.9);

  border:
    1px solid
    color-mix(
      in srgb,
      var(--team-accent) 32%,
      rgba(185, 160, 120, 0.38)
    );

  border-radius:
    5px 9px 4px 7px;

  background:
    radial-gradient(
      circle at 18% 12%,
      rgba(255, 255, 255, 0.055),
      transparent 24%
    ),
    repeating-linear-gradient(
      92deg,
      rgba(255, 255, 255, 0.018) 0 1px,
      transparent 1px 5px
    ),
    repeating-linear-gradient(
      8deg,
      rgba(0, 0, 0, 0.2) 0 1px,
      transparent 1px 7px
    ),
    linear-gradient(
      145deg,
      #171719,
      #0c0c0e 48%,
      #050506
    );

  box-shadow:
    0 32px 48px
      rgba(0, 0, 0, 0.75),
    0 8px 12px
      rgba(0, 0, 0, 0.62),
    inset 0 0 34px
      rgba(0, 0, 0, 0.64),
    inset 0 1px 0
      rgba(255, 255, 255, 0.06);

  opacity: 0;

  animation:
    appointmentDossierOffer
    720ms
    230ms
    cubic-bezier(.16, .82, .22, 1)
    forwards;
}

.appointment-dossier::before {
  content: "";

  position: absolute;
  inset: 8px;

  pointer-events: none;

  border:
    1px solid
    rgba(255, 255, 255, 0.04);

  border-radius:
    3px 7px 3px 5px;
}

.appointment-dossier::after {
  content: "";

  position: absolute;
  inset: 0;

  pointer-events: none;

  background:
    linear-gradient(
      112deg,
      transparent 0 42%,
      rgba(255, 255, 255, 0.035) 48%,
      transparent 54%
    );

  transform: translateX(-120%);

  animation:
    appointmentDossierSheen
    900ms
    620ms
    ease-out
    forwards;
}

.appointment-dossier.is-signed {
  border-color:
    color-mix(
      in srgb,
      var(--team-accent) 62%,
      rgba(232, 200, 148, 0.58)
    );

  box-shadow:
    0 32px 48px
      rgba(0, 0, 0, 0.75),
    0 8px 12px
      rgba(0, 0, 0, 0.62),
    0 0 28px
      color-mix(
        in srgb,
        var(--team-accent) 16%,
        transparent
      ),
    inset 0 0 34px
      rgba(0, 0, 0, 0.64);
}

.appointment-dossier-spine {
  position: absolute;

  top: 0;
  bottom: 0;
  left: 0;

  width: 16px;

  border-right:
    1px solid
    rgba(255, 255, 255, 0.045);

  background:
    linear-gradient(
      90deg,
      rgba(0, 0, 0, 0.72),
      rgba(255, 255, 255, 0.025),
      rgba(0, 0, 0, 0.38)
    );

  box-shadow:
    5px 0 9px
      rgba(0, 0, 0, 0.32);
}

.appointment-dossier-stitch {
  position: absolute;

  left: 20px;
  right: 10px;

  height: 1px;

  opacity: 0.5;

  background:
    repeating-linear-gradient(
      90deg,
      var(--team-accent) 0 5px,
      transparent 5px 11px
    );
}

.appointment-dossier-stitch--top {
  top: 10px;
}

.appointment-dossier-stitch--bottom {
  bottom: 10px;
}

.appointment-dossier-corner {
  position: absolute;

  width: 23px;
  height: 23px;

  border-color:
    rgba(190, 175, 145, 0.2);

  border-style: solid;
}

.appointment-dossier-corner--tl {
  top: 14px;
  left: 25px;

  border-width: 1px 0 0 1px;
}

.appointment-dossier-corner--tr {
  top: 14px;
  right: 14px;

  border-width: 1px 1px 0 0;
}

.appointment-dossier-corner--bl {
  bottom: 14px;
  left: 25px;

  border-width: 0 0 1px 1px;
}

.appointment-dossier-corner--br {
  bottom: 14px;
  right: 14px;

  border-width: 0 1px 1px 0;
}

.appointment-dossier-kicker {
  position: relative;

  z-index: 2;

  font-size: 7px;
  font-weight: 900;

  letter-spacing: 0.19em;
  text-transform: uppercase;

  color:
    rgba(215, 202, 178, 0.43);
}

.appointment-dossier-crest {
  position: relative;

  z-index: 2;

  width: 70px;
  height: 62px;

  display: grid;
  place-items: center;

  margin-top: 1px;
}

.appointment-dossier-crest::before {
  content: "";

  position: absolute;
  inset: 4px;

  border-radius: 50%;

  filter: blur(9px);

  background:
    radial-gradient(
      circle,
      color-mix(
        in srgb,
        var(--team-accent) 15%,
        transparent
      ),
      transparent 70%
    );
}

.appointment-dossier-crest .setup-team-logo {
  position: relative;

  z-index: 1;

  opacity: 0.82;

  filter:
    grayscale(0.22)
    drop-shadow(
      0 4px 8px
      rgba(0, 0, 0, 0.6)
    );
}

.appointment-dossier-team {
  position: relative;

  z-index: 2;

  max-width: 100%;

  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;

  font-size:
    clamp(
      11px,
      1.2vw,
      15px
    );

  font-weight: 950;

  letter-spacing: 0.08em;
  text-transform: uppercase;

  color:
    rgba(239, 230, 211, 0.9);
}

.appointment-dossier-office {
  position: relative;

  z-index: 2;

  font-size: 7px;
  font-weight: 850;

  letter-spacing: 0.18em;
  text-transform: uppercase;

  color:
    color-mix(
      in srgb,
      var(--team-accent-2) 42%,
      rgba(220, 206, 178, 0.6)
    );
}

.appointment-dossier-rule {
  position: relative;

  z-index: 2;

  width: 54%;
  height: 1px;

  margin: 3px 0 1px;

  background:
    linear-gradient(
      90deg,
      transparent,
      color-mix(
        in srgb,
        var(--team-accent) 44%,
        rgba(220, 200, 160, 0.32)
      ),
      transparent
    );
}

.appointment-dossier-status {
  position: relative;

  z-index: 2;

  display: flex;
  align-items: center;

  gap: 6px;

  font-size: 7px;
  font-weight: 950;

  letter-spacing: 0.12em;
  text-transform: uppercase;

  color:
    rgba(214, 196, 163, 0.56);
}

.appointment-dossier.is-signed
.appointment-dossier-status {
  color:
    color-mix(
      in srgb,
      var(--team-accent-2) 50%,
      #e8c894
    );
}

.appointment-dossier-status-light {
  width: 5px;
  height: 5px;

  border-radius: 50%;

  background:
    rgba(185, 150, 100, 0.52);

  box-shadow:
    0 0 7px
      rgba(185, 150, 100, 0.25);
}

.appointment-dossier.is-signed
.appointment-dossier-status-light {
  background: var(--team-accent);

  box-shadow:
    0 0 9px
      color-mix(
        in srgb,
        var(--team-accent) 72%,
        transparent
      );
}

.appointment-dossier-clasp {
  position: absolute;

  z-index: 5;

  top: 50%;
  right: -2px;

  width: 20px;
  height: 34px;

  transform: translateY(-50%);

  border:
    1px solid
    rgba(215, 196, 150, 0.32);

  border-radius:
    3px 0 0 3px;

  background:
    linear-gradient(
      90deg,
      #463b2d,
      #827056 45%,
      #2e261d
    );

  box-shadow:
    -4px 4px 9px
      rgba(0, 0, 0, 0.48);
}

.appointment-dossier-clasp span {
  position: absolute;

  top: 50%;
  left: 50%;

  width: 7px;
  height: 7px;

  transform:
    translate(-50%, -50%);

  border-radius: 50%;

  border:
    1px solid
    rgba(0, 0, 0, 0.5);

  background:
    color-mix(
      in srgb,
      var(--team-accent) 52%,
      #3e3427
    );

  box-shadow:
    0 0 7px
      color-mix(
        in srgb,
        var(--team-accent) 24%,
        transparent
      );
}

.appointment-seat-shadow {
  position: absolute;

  z-index: 12;

  left: 50%;
  bottom: -16%;

  width: 38%;
  height: 28%;

  transform: translateX(-50%);

  border-radius: 50%;

  filter: blur(18px);

  background:
    radial-gradient(
      ellipse,
      rgba(0, 0, 0, 0.82),
      transparent 70%
    );
}

.appointment-rail-wrap {
  position: relative;

  z-index: 30;

  padding: 8px 14px 12px;

  border-top:
    1px solid
    rgba(255, 255, 255, 0.055);

  background:
    linear-gradient(
      180deg,
      rgba(8, 9, 11, 0.96),
      rgba(3, 4, 5, 0.99)
    );
}

.appointment-rail-label,
.appointment-rail-hint {
  display: block;

  text-align: center;

  font-size: 7px;
  font-weight: 900;

  letter-spacing: 0.18em;
  text-transform: uppercase;

  color:
    rgba(226, 217, 199, 0.34);
}

.appointment-rail {
  display: grid;

  grid-template-columns:
    34px
    minmax(0, 1fr)
    34px;

  align-items: center;

  gap: 8px;

  margin: 5px 0 4px;
}

.appointment-rail-files {
  min-width: 0;

  display: grid;

  grid-template-columns:
    repeat(
      5,
      minmax(0, 1fr)
    );

  align-items: end;

  gap: 6px;
}

.appointment-rail-arrow {
  width: 34px;
  height: 42px;

  border: none;

  background: transparent;

  color:
    rgba(224, 214, 195, 0.56);

  font-size: 28px;

  cursor: pointer;

  transition:
    color 150ms ease,
    transform 150ms ease;
}

.appointment-rail-arrow:hover,
.appointment-rail-arrow:focus-visible {
  color:
    color-mix(
      in srgb,
      var(--team-accent-2) 52%,
      #eee4d2
    );

  transform: scale(1.08);

  outline: none;
}

.appointment-offer-file {
  position: relative;

  min-width: 0;
  height: 55px;

  padding: 9px 5px 5px;

  overflow: visible;

  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;

  gap: 2px;

  border:
    1px solid
    rgba(255, 255, 255, 0.06);

  border-radius:
    3px 5px 3px 4px;

  background:
    repeating-linear-gradient(
      95deg,
      rgba(255, 255, 255, 0.015) 0 1px,
      transparent 1px 7px
    ),
    linear-gradient(
      160deg,
      rgba(29, 29, 31, 0.94),
      rgba(8, 8, 10, 0.98)
    );

  color:
    rgba(226, 217, 199, 0.48);

  cursor: pointer;

  opacity: 0.58;

  transform:
    translateY(0)
    scale(0.9);

  box-shadow:
    0 8px 14px
      rgba(0, 0, 0, 0.28);

  transition:
    transform 170ms ease,
    opacity 170ms ease,
    border-color 170ms ease,
    box-shadow 170ms ease;
}

.appointment-offer-file:hover {
  opacity: 0.88;

  transform:
    translateY(-3px)
    scale(0.94);

  border-color:
    rgba(255, 255, 255, 0.15);
}

.appointment-offer-file.is-selected {
  height: 63px;

  opacity: 1;

  transform:
    translateY(-5px)
    scale(1);

  border-color:
    color-mix(
      in srgb,
      var(--team-accent) 52%,
      rgba(255, 255, 255, 0.15)
    );

  background:
    radial-gradient(
      circle at 50% 16%,
      color-mix(
        in srgb,
        var(--team-accent) 12%,
        transparent
      ),
      transparent 48%
    ),
    linear-gradient(
      160deg,
      rgba(34, 34, 37, 0.98),
      rgba(7, 7, 9, 1)
    );

  box-shadow:
    0 13px 22px
      rgba(0, 0, 0, 0.48),
    0 0 18px
      color-mix(
        in srgb,
        var(--team-accent) 12%,
        transparent
      );
}

.appointment-offer-file-tab {
  position: absolute;

  top: -5px;
  left: 8px;

  width: 38%;
  height: 7px;

  border:
    1px solid
    rgba(255, 255, 255, 0.055);

  border-bottom: none;

  border-radius:
    3px 5px 0 0;

  background: #171719;
}

.appointment-offer-file.is-selected
.appointment-offer-file-tab {
  border-color:
    color-mix(
      in srgb,
      var(--team-accent) 36%,
      rgba(255, 255, 255, 0.08)
    );

  background:
    color-mix(
      in srgb,
      var(--team-accent) 12%,
      #171719
    );
}

.appointment-offer-logo {
  max-width: 38px;
  max-height: 38px;

  filter:
    saturate(0.72)
    drop-shadow(
      0 4px 7px
      rgba(0, 0, 0, 0.55)
    );
}

.appointment-offer-file.is-selected
.appointment-offer-logo {
  filter:
    saturate(0.9)
    drop-shadow(
      0 4px 8px
      rgba(0, 0, 0, 0.6)
    )
    drop-shadow(
      0 0 8px
      color-mix(
        in srgb,
        var(--team-accent) 18%,
        transparent
      )
    );
}

.appointment-offer-code {
  font-size: 7px;
  font-weight: 950;

  letter-spacing: 0.14em;

  color:
    rgba(230, 221, 203, 0.46);
}

.appointment-offer-file.is-selected
.appointment-offer-code {
  color:
    color-mix(
      in srgb,
      var(--team-accent-2) 44%,
      #e7dcc8
    );
}

.setup-team-logo {
  object-fit: contain;

  filter:
    drop-shadow(
      0 6px 14px
      rgba(0, 0, 0, 0.45)
    );
}

.setup-logo-empty {
  border-radius: 12px;

  border:
    1px dashed
    rgba(191, 109, 54, 0.28);

  background:
    rgba(0, 0, 0, 0.2);
}

.setup-config-panel {
  position: relative;

  min-height: 0;

  overflow: hidden;

  display: flex;
  flex-direction: column;

  color: #d8c9a8;

  transform: rotate(0.35deg);

  clip-path:
    polygon(
      1.5% 0.8%,
      6% 0%,
      11% 1.8%,
      17% 0.4%,
      24% 2.2%,
      31% 0.2%,
      39% 1.6%,
      47% 0%,
      55% 2%,
      63% 0.5%,
      71% 1.4%,
      79% 0.2%,
      87% 2.5%,
      94% 0.8%,
      99% 2.5%,
      100% 7%,
      98.5% 14%,
      100% 22%,
      99.2% 31%,
      100% 41%,
      98.8% 52%,
      100% 63%,
      99% 74%,
      100% 84%,
      98.5% 93%,
      96% 99%,
      89% 97.5%,
      81% 100%,
      73% 98.2%,
      64% 99.6%,
      55% 97.8%,
      46% 100%,
      37% 98.5%,
      28% 99.8%,
      19% 97.2%,
      10% 99.5%,
      3% 96.5%,
      0.5% 91%,
      0% 82%,
      1.8% 71%,
      0% 60%,
      2.2% 49%,
      0% 38%,
      1.5% 27%,
      0% 16%,
      1.2% 7%
    );

  background:
    radial-gradient(
      ellipse at 12% 8%,
      rgba(219, 129, 52, 0.32),
      transparent 22%
    ),
    radial-gradient(
      ellipse at 94% 6%,
      rgba(184, 87, 39, 0.38),
      transparent 20%
    ),
    radial-gradient(
      ellipse at 100% 88%,
      rgba(96, 38, 25, 0.55),
      transparent 24%
    ),
    radial-gradient(
      ellipse at 48% 52%,
      rgba(0, 0, 0, 0.28),
      transparent 48%
    ),
    linear-gradient(
      168deg,
      rgba(38, 32, 26, 0.98),
      rgba(18, 16, 14, 0.99) 42%,
      rgba(10, 12, 14, 1)
    );

  box-shadow:
    0 28px 70px
      rgba(0, 0, 0, 0.72),
    0 0 0 1px
      rgba(120, 68, 32, 0.45),
    inset 0 0 90px
      rgba(0, 0, 0, 0.65);

  filter:
    drop-shadow(
      0 18px 42px
      rgba(0, 0, 0, 0.55)
    );
}

.setup-config-panel::before {
  content: "";

  position: absolute;
  inset: 0;

  z-index: 4;

  pointer-events: none;

  opacity: 0.36;

  background:
    repeating-linear-gradient(
      8deg,
      rgba(255, 235, 190, 0.06) 0 1px,
      transparent 1px 5px
    ),
    repeating-linear-gradient(
      104deg,
      rgba(0, 0, 0, 0.42) 0 1px,
      transparent 1px 9px
    );
}

.setup-contract-scroll {
  position: relative;

  z-index: 5;

  flex: 1;
  min-height: 0;

  overflow-x: hidden;
  overflow-y: auto;

  display: flex;
  flex-direction: column;

  gap: 8px;

  padding: 18px 24px 10px;

  scrollbar-width: thin;

  scrollbar-color:
    rgba(191, 109, 54, 0.45)
    rgba(0, 0, 0, 0.2);
}

.setup-contract-scroll::-webkit-scrollbar {
  width: 6px;
}

.setup-contract-scroll::-webkit-scrollbar-thumb {
  border-radius: 4px;

  background:
    rgba(191, 109, 54, 0.5);
}

.setup-contract-ornament {
  margin-bottom: -2px;

  text-align: center;

  font-family:
    "CookieAgreement",
    Georgia,
    serif;

  font-size: 14px;

  letter-spacing: 0.45em;

  color:
    rgba(220, 162, 92, 0.55);
}

.setup-contract-title {
  margin: 0;

  text-align: center;

  font-family:
    "CookieAgreement",
    Georgia,
    serif;

  font-size:
    clamp(
      32px,
      2.8vw,
      48px
    );

  font-weight: 400;

  line-height: 0.92;

  color: #e8c894;

  text-shadow:
    0 2px 0
      rgba(0, 0, 0, 0.85),
    0 0 18px
      rgba(199, 139, 68, 0.22);
}

.setup-contract-intro {
  margin: 2px 0 0;

  text-align: center;

  font-family:
    "CookieAgreement",
    Georgia,
    serif;

  font-size:
    clamp(
      16px,
      1.6vw,
      22px
    );

  color:
    rgba(220, 190, 140, 0.82);
}

.setup-contract-parties {
  display: grid;

  grid-template-columns:
    auto
    minmax(0, 1fr)
    auto;

  align-items: center;

  gap: 10px;

  margin-top: 2px;
}

.setup-contract-team-badge {
  width: 56px;
  height: 56px;

  display: grid;
  place-items: center;

  border-radius: 50%;

  border:
    2px solid
    rgba(191, 109, 54, 0.45);

  background:
    radial-gradient(
      circle,
      rgba(0, 0, 0, 0.35),
      rgba(0, 0, 0, 0.65)
    );

  box-shadow:
    inset 0 0 18px
      rgba(0, 0, 0, 0.55);
}

.setup-contract-logo {
  width: 42px;
  height: 42px;

  object-fit: contain;
}

.setup-contract-party-copy {
  min-width: 0;
}

.setup-contract-team-name {
  margin: 0;

  font-size:
    clamp(
      14px,
      1.4vw,
      18px
    );

  font-weight: 800;

  letter-spacing: 0.04em;
  text-transform: uppercase;

  line-height: 1.15;

  color: #f0e0c0;
}

.setup-contract-gm-field {
  margin-top: 4px;
}

.setup-contract-gm-label,
.setup-signature-label {
  display: block;

  margin-bottom: 3px;

  font-family:
    "CookieAgreement",
    Georgia,
    serif;

  font-size: 16px;

  color:
    rgba(220, 190, 140, 0.78);
}

.setup-contract-gm-input {
  width: 100%;

  padding: 4px 2px;

  border: none;

  border-bottom:
    1px solid
    rgba(191, 109, 54, 0.55);

  outline: none;

  background: transparent;

  color: #f0e0c0;

  font-size: 13px;
  font-weight: 700;
}

.setup-contract-gm-field.is-active
.setup-contract-gm-input {
  border-color:
    rgba(220, 162, 92, 0.75);

  box-shadow:
    0 8px 12px -10px
      rgba(220, 162, 92, 0.7);
}

.setup-contract-gm-input::placeholder {
  color:
    rgba(200, 175, 130, 0.45);
}

.setup-contract-year {
  text-align: right;
}

.setup-contract-year-label {
  font-family:
    "CookieAgreement",
    Georgia,
    serif;

  font-size:
    clamp(
      17px,
      1.8vw,
      22px
    );

  color: #e8c894;
}

.setup-contract-divider {
  height: 1px;

  background:
    linear-gradient(
      90deg,
      transparent,
      rgba(191, 109, 54, 0.65) 12%,
      rgba(220, 162, 92, 0.85) 50%,
      rgba(191, 109, 54, 0.65) 88%,
      transparent
    );
}

.setup-contract-divider--thin {
  opacity: 0.55;
}

.setup-contract-body,
.setup-clause-text {
  margin: 0;

  font-size: 11.5px;

  line-height: 1.48;

  color:
    rgba(210, 190, 155, 0.88);
}

.setup-contract-body {
  text-align: justify;
}

.setup-clause {
  display: grid;

  gap: 4px;
}

.setup-clause.is-active {
  margin: -4px -8px;

  padding: 6px 8px;

  border-radius: 4px;

  background:
    rgba(0, 0, 0, 0.22);

  box-shadow:
    inset 0 0 0 1px
      rgba(220, 162, 92, 0.22);
}

.setup-clause-heading {
  margin: 0;

  font-family:
    "CookieAgreement",
    Georgia,
    serif;

  font-size:
    clamp(
      17px,
      1.7vw,
      22px
    );

  font-weight: 400;

  line-height: 1.05;

  color: #e0b878;
}

.setup-clause-question {
  margin: 0;

  font-family:
    "CookieAgreement",
    Georgia,
    serif;

  font-size: 16px;

  color:
    rgba(232, 200, 148, 0.9);
}

.setup-injuries-toggle {
  display: grid;

  grid-template-columns:
    1fr 1fr;

  gap: 8px;

  max-width: 200px;
}

.setup-injuries-btn {
  min-height: 36px;

  padding: 0 10px;

  border:
    1px solid
    rgba(191, 109, 54, 0.45);

  border-radius:
    4px 10px 5px 8px;

  background:
    linear-gradient(
      180deg,
      rgba(0, 0, 0, 0.32),
      rgba(0, 0, 0, 0.18)
    ),
    rgba(40, 32, 24, 0.55);

  color:
    rgba(200, 175, 130, 0.72);

  font-family:
    "CookieAgreement",
    Georgia,
    serif;

  font-size: 22px;

  cursor: pointer;
}

.setup-injuries-btn.is-selected {
  border:
    2px solid
    rgba(220, 162, 92, 0.85);

  background:
    linear-gradient(
      180deg,
      rgba(120, 34, 23, 0.35),
      rgba(0, 0, 0, 0.25)
    ),
    rgba(60, 38, 22, 0.65);

  color: #f0d8a8;

  box-shadow:
    0 0 18px
      rgba(199, 139, 68, 0.28);
}

.setup-signature-area {
  position: relative;

  display: grid;

  grid-template-columns:
    minmax(0, 1fr)
    auto;

  align-items: end;

  gap: 10px;

  min-height: 96px;
}

.setup-signature-fields {
  display: grid;

  grid-template-columns:
    minmax(0, 1.55fr)
    minmax(0, 0.85fr);

  gap: 14px;

  align-items: end;
}

.setup-signature-pad {
  display: grid;

  gap: 4px;

  outline: none;
}

.setup-signature-pad-surface {
  position: relative;

  min-height: 72px;

  overflow: hidden;

  border:
    1px solid
    rgba(191, 109, 54, 0.42);

  border-radius:
    4px 8px 3px 7px;

  background:
    linear-gradient(
      rgba(255, 235, 190, 0.04) 1px,
      transparent 1px
    ),
    linear-gradient(
      90deg,
      rgba(255, 235, 190, 0.04) 1px,
      transparent 1px
    ),
    rgba(0, 0, 0, 0.2);

  background-size:
    14px 14px,
    14px 14px,
    auto;
}

.setup-signature-block.is-active
.setup-signature-pad-surface {
  box-shadow:
    0 0 0 1px
      rgba(220, 162, 92, 0.35),
    0 0 18px
      rgba(199, 139, 68, 0.16);
}

.setup-signature-canvas {
  position: absolute;
  inset: 0;

  z-index: 2;

  width: 100%;
  height: 100%;

  touch-action: none;

  cursor: crosshair;
}

.setup-signature-line {
  position: absolute;

  z-index: 1;

  left: 8px;
  right: 8px;
  bottom: 8px;

  height: 1px;

  pointer-events: none;

  background:
    linear-gradient(
      90deg,
      rgba(120, 68, 32, 0.35),
      rgba(220, 162, 92, 0.8) 48%,
      rgba(120, 68, 32, 0.35)
    );
}

.setup-signature-placeholder {
  position: absolute;

  z-index: 1;

  left: 12px;
  bottom: 14px;

  pointer-events: none;

  font-family:
    "CookieAgreement",
    Georgia,
    serif;

  font-size: 22px;

  color:
    rgba(191, 109, 54, 0.48);

  transition:
    opacity 150ms ease;
}

.setup-signature-pad.has-ink
.setup-signature-placeholder {
  opacity: 0;
}

.setup-signature-clear {
  justify-self: start;

  padding: 0;

  border: none;

  background: transparent;

  color:
    rgba(220, 190, 140, 0.62);

  font-size: 10px;
  font-weight: 800;

  letter-spacing: 0.08em;
  text-transform: uppercase;

  cursor: pointer;
}

.setup-signature-clear:hover {
  color: #e8c894;
}

.setup-date-value {
  display: block;

  padding-bottom: 4px;

  border-bottom:
    1px solid
    rgba(191, 109, 54, 0.45);

  color: #f0e0c0;

  font-size: 12px;
  font-weight: 700;
}

.setup-wax-seal {
  width: 64px;
  height: 64px;

  display: grid;
  place-items: center;

  border:
    3px solid
    rgba(120, 30, 20, 0.85);

  border-radius: 50%;

  transform: rotate(-8deg);

  background:
    radial-gradient(
      circle at 35% 30%,
      #c44a2a,
      #7a1e14 58%,
      #4a0f0a
    );

  box-shadow:
    0 8px 22px
      rgba(0, 0, 0, 0.55),
    inset 0 0 18px
      rgba(0, 0, 0, 0.45);
}

.setup-wax-seal-inner {
  width: 46px;
  height: 46px;

  display: grid;
  place-items: center;

  border:
    2px dashed
    rgba(255, 200, 160, 0.35);

  border-radius: 50%;

  background:
    rgba(0, 0, 0, 0.18);
}

.setup-wax-seal-logo {
  width: 32px;
  height: 32px;

  object-fit: contain;
}

.setup-wax-seal-mark {
  font-family:
    "CookieAgreement",
    Georgia,
    serif;

  font-size: 22px;

  color:
    rgba(255, 220, 180, 0.75);
}

.setup-config-panel >
.setup-start-btn,
.setup-config-panel >
.setup-signature-hint {
  position: relative;

  z-index: 10;

  flex-shrink: 0;

  margin: 0 16px 14px;
}

.setup-error {
  position: relative;

  z-index: 10;

  margin: 0 16px 8px;

  padding: 10px 12px;

  border:
    1px solid
    rgba(255, 96, 109, 0.45);

  border-radius: 10px;

  background:
    rgba(120, 0, 20, 0.55);

  color: #ffd6da;

  font-size: 13px;
  font-weight: 800;
}

.setup-signature-hint {
  padding: 12px 14px;

  border:
    1px dashed
    rgba(191, 109, 54, 0.35);

  border-radius:
    5px 10px 4px 8px;

  background:
    rgba(0, 0, 0, 0.22);

  text-align: center;

  font-family:
    "CookieAgreement",
    Georgia,
    serif;

  font-size: 18px;

  color:
    rgba(220, 190, 140, 0.72);
}

.setup-start-btn {
  min-height: 52px;

  display: grid;
  place-items: center;

  gap: 2px;

  border:
    2px solid
    rgba(191, 109, 54, 0.7);

  border-radius:
    5px 12px 6px 10px;

  background:
    linear-gradient(
      180deg,
      rgba(80, 48, 22, 0.85),
      rgba(30, 20, 12, 0.92)
    );

  color: #e8c894;

  cursor: pointer;

  box-shadow:
    0 14px 32px
      rgba(0, 0, 0, 0.45);

  transition:
    transform 150ms ease,
    border-color 150ms ease,
    box-shadow 150ms ease;

  animation:
    setupStartReveal
    320ms
    ease-out;
}

.setup-start-btn span {
  font-family:
    "CookieAgreement",
    Georgia,
    serif;

  font-size:
    clamp(
      22px,
      2.2vw,
      28px
    );

  font-weight: 400;
}

.setup-start-btn small {
  font-size: 9px;
  font-weight: 800;

  letter-spacing: 0.14em;
  text-transform: uppercase;

  color:
    rgba(220, 190, 140, 0.72);
}

.setup-start-btn:hover:not(:disabled),
.setup-start-btn.is-active:not(:disabled) {
  transform: translateY(-1px);

  border-color:
    rgba(232, 200, 148, 0.9);

  box-shadow:
    0 18px 40px
      rgba(0, 0, 0, 0.5),
    0 0 28px
      rgba(199, 139, 68, 0.32);
}

.setup-start-btn:disabled {
  opacity: 0.45;

  cursor: not-allowed;
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

.setup-loading-screen {
  position: fixed;

  z-index: 9999;

  inset: 0;

  display: grid;
  place-items: center;

  padding: 24px;

  overflow: hidden;

  background:
    radial-gradient(
      circle at 50% 22%,
      color-mix(
        in srgb,
        var(--team-accent) 16%,
        transparent
      ),
      transparent 32%
    ),
    linear-gradient(
      180deg,
      #06131f,
      #020a11
    );
}

.setup-loading-noise {
  position: absolute;
  inset: 0;

  opacity: 0.2;

  pointer-events: none;

  background:
    repeating-linear-gradient(
      155deg,
      rgba(255, 255, 255, 0.04) 0 1px,
      transparent 1px 16px
    );
}

.setup-loading-card {
  position: relative;

  z-index: 2;

  width:
    min(
      760px,
      100%
    );

  padding: 34px;

  border:
    1px solid
    color-mix(
      in srgb,
      var(--team-accent) 32%,
      rgba(255, 255, 255, 0.12)
    );

  border-radius: 26px;

  background:
    rgba(5, 17, 27, 0.88);

  box-shadow:
    0 34px 90px
      rgba(0, 0, 0, 0.48);

  text-align: center;

  backdrop-filter: blur(14px);
}

.setup-loading-spinner {
  width: 72px;
  height: 72px;

  margin:
    0 auto 18px;

  border:
    4px solid
    rgba(255, 255, 255, 0.12);

  border-top-color:
    var(--team-accent);

  border-right-color:
    var(--team-accent-2);

  border-radius: 50%;

  animation:
    setupSpin
    0.92s
    linear
    infinite;

  box-shadow:
    0 0 28px
      color-mix(
        in srgb,
        var(--team-accent) 22%,
        transparent
      );
}

.setup-loading-kicker {
  margin: 0;

  font-size: 11px;
  font-weight: 950;

  letter-spacing: 0.2em;
  text-transform: uppercase;

  color:
    var(--team-accent-2);
}

.setup-loading-title {
  margin: 8px 0;

  font-size:
    clamp(
      28px,
      4vw,
      46px
    );

  line-height: 1.03;

  font-weight: 950;
}

.setup-loading-copy {
  margin: 0 auto;

  max-width: 580px;

  color: var(--muted);

  font-size: 15px;

  line-height: 1.5;
}

.setup-loading-steps {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;

  gap: 8px;

  margin: 22px 0;
}

.setup-loading-steps span {
  padding: 8px 11px;

  border:
    1px solid
    rgba(255, 255, 255, 0.1);

  border-radius: 999px;

  background:
    rgba(255, 255, 255, 0.035);

  color: var(--text);

  font-size: 11px;
  font-weight: 850;

  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.setup-fact-card {
  position: relative;

  overflow: hidden;

  margin-top: 20px;

  padding:
    18px 18px 20px;

  border:
    1px solid
    rgba(255, 255, 255, 0.09);

  border-radius: 18px;

  background:
    rgba(4, 10, 18, 0.58);

  text-align: left;
}

.setup-fact-label {
  display: block;

  margin-bottom: 8px;

  font-size: 10px;
  font-weight: 950;

  letter-spacing: 0.16em;
  text-transform: uppercase;

  color:
    var(--team-accent-2);
}

.setup-fact-card p {
  margin: 0;

  color: #e9f7fb;

  font-size:
    clamp(
      15px,
      2vw,
      19px
    );

  line-height: 1.55;

  font-weight: 750;
}

.setup-fact-progress {
  position: absolute;

  left: 0;
  bottom: 0;

  height: 3px;

  background:
    linear-gradient(
      90deg,
      var(--team-accent),
      var(--team-accent-2)
    );

  animation:
    setupFactProgress
    10s
    linear
    forwards;
}

@keyframes appointmentSceneReveal {
  0% {
    opacity: 1;
  }

  55% {
    opacity: 0.74;
  }

  100% {
    opacity: 0;
  }
}

@keyframes appointmentCrestReveal {
  from {
    opacity: 0;

    transform:
      translateX(-50%)
      scale(0.9);
  }

  to {
    opacity: 1;

    transform:
      translateX(-50%)
      scale(1);
  }
}

@keyframes appointmentLightbarReveal {
  from {
    opacity: 0;

    transform: scaleX(0.25);
  }

  to {
    opacity: 0.72;

    transform: scaleX(1);
  }
}

@keyframes appointmentExecutivesReveal {
  from {
    opacity: 0;

    transform: translateY(12px);
  }

  to {
    opacity: 1;

    transform: translateY(0);
  }
}

@keyframes appointmentExecutiveIdle {
  0%,
  100% {
    margin-bottom: 0;
  }

  50% {
    margin-bottom: 2px;
  }
}

@keyframes appointmentBeamEnter {
  from {
    opacity: 0;

    transform: scaleY(0);
  }

  to {
    opacity: 1;

    transform: scaleY(1);
  }
}

@keyframes appointmentDossierOffer {
  0% {
    opacity: 0;

    transform:
      translateX(-50%)
      perspective(850px)
      rotateX(54deg)
      translateY(-110px)
      scale(0.78);
  }

  48% {
    opacity: 1;
  }

  100% {
    opacity: 1;

    transform:
      translateX(-50%)
      perspective(850px)
      rotateX(54deg)
      rotateZ(-0.8deg)
      translateY(0)
      scale(1);
  }
}

@keyframes appointmentDossierSheen {
  from {
    transform: translateX(-120%);
  }

  to {
    transform: translateX(150%);
  }
}

@keyframes setupStartReveal {
  from {
    opacity: 0;

    transform: translateY(8px);
  }

  to {
    opacity: 1;

    transform: none;
  }
}

@keyframes setupSpin {
  to {
    transform: rotate(360deg);
  }
}

@keyframes setupFactProgress {
  from {
    width: 0%;
  }

  to {
    width: 100%;
  }
}

@media (max-width: 980px) {
  .nhlcal-root.setup-root {
    height: auto;
    max-height: none;

    overflow: auto;
  }

  .setup-main {
    grid-template-columns: 1fr;
  }

  .setup-config-panel {
    transform: none;
  }

  .setup-contract-scroll {
    max-height: 62vh;
  }

  .appointment-scene {
    min-height: 540px;
  }
}

@media (max-width: 620px) {
  .setup-main {
    padding: 12px;
  }

  .setup-contract-scroll {
    padding:
      16px 14px 8px;
  }

  .setup-config-panel >
  .setup-start-btn,
  .setup-config-panel >
  .setup-signature-hint {
    margin:
      0 12px 12px;
  }

  .setup-contract-parties {
    grid-template-columns:
      auto
      minmax(0, 1fr);

    grid-template-rows:
      auto auto;
  }

  .setup-contract-year {
    grid-column: 1 / -1;

    text-align: left;
  }

  .setup-signature-area,
  .setup-signature-fields {
    grid-template-columns: 1fr;
  }

  .setup-wax-seal {
    justify-self: end;
  }

  .appointment-registry {
    padding:
      12px 13px 8px;
  }

  .appointment-executives {
    left: 10%;
    right: 10%;
  }

  .appointment-dossier {
    width: 48%;
  }

  .appointment-rail-wrap {
    padding-inline: 8px;
  }

  .appointment-rail {
    grid-template-columns:
      28px
      minmax(0, 1fr)
      28px;

    gap: 4px;
  }

  .appointment-rail-files {
    gap: 3px;
  }

  .appointment-offer-file {
    height: 48px;

    padding-inline: 2px;
  }

  .appointment-offer-file.is-selected {
    height: 56px;
  }

  .appointment-offer-logo {
    max-width: 30px;
    max-height: 30px;
  }

  .setup-loading-card {
    padding:
      24px 18px;
  }
}
`;