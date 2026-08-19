import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

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
import setupTheme from "../soundtrack/JJ's Energy - Felix Weber (FIFA 2014 World Cup Brazil OST).mp3";
import { getStroke } from "perfect-freehand";
import {
  ArcRotateCamera,
  Color3,
  Color4,
  DirectionalLight,
  DynamicTexture,
  Engine,
  HemisphericLight,
  Mesh,
  MeshBuilder,
  ParticleSystem,
  PBRMaterial,
  PointLight,
  PointerEventTypes,
  Ray,
  Scene,
  SceneLoader,
  SpotLight,
  StandardMaterial,
  Texture,
  TransformNode,
  UniversalCamera,
  Vector3,
  VertexBuffer,
  VertexData,
} from "@babylonjs/core";
import "@babylonjs/loaders/glTF";
import darkOfficeGlb
  from "../pictures/modern_office.glb?url";
import executiveDeskGlb
  from "../pictures/office_pics/ambani_executive_office_desk_with_walnut_finish.glb?url";
import leatherChairGlb
  from "../pictures/office_pics/worn_leather_office_chair.glb?url";
import manSittingGlb
  from "../pictures/office_pics/man_sitting.glb?url";
import manDressedInSuitGlb
  from "../pictures/office_pics/man_dressed_in_suit.glb?url";
import hockeyStickGlb
  from "../pictures/office_pics/hockey_stick (1).glb?url";
import trophyCupGlb
  from "../pictures/office_pics/trophy_cup.glb?url";
import officePropsGlb
  from "../pictures/office_pics/office_props_pack.glb?url";
import contractGlb
  from "../pictures/contract.glb?url";
import clipboardGlb
  from "../pictures/ps1_style_patient_sheet_with_clipboard.glb?url";

/*
  ============================================================================
  GLB ASSETS
  ============================================================================
  These are the exact files supplied for the GLB-first franchise setup.

  IMPORTANT:
  - Vite resolves these imports at build time.
  - Keep the ?url suffix.
  - This component never touches the existing music/audio system.
  - The backend/state integration remains useGameUI() + beginFranchise().
*/

/* ============================================================================
   TEAM / SETUP DATA
   ========================================================================== */

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

const DEFAULT_TEAM_ORDER = [
  ...EASTERN_ORDER,
  ...WESTERN_ORDER,
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


/* ============================================================================
   APP FLOW
   ==========================================================================

   THIS IS THE LOCKED ORDER.

   1. Intro cinematic launches FIRST.
      - no team selected visually
      - generic NHL executive floor
      - office hallway GLB
      - dark office GLB
      - seated executive

   2. Intro fades into configuration.
      - select team
      - GM name
      - real/generated
      - injuries
      - NO signature here

   3. Signed deed starts the franchise and shows the loading screen.
*/

const APP_STAGE = Object.freeze({
  INTRO: "intro",
  CONFIGURE: "configure",
  APPOINTMENT: "appointment",
  STARTING: "starting",
});

const CINEMATIC_MODE = Object.freeze({
  INTRO: "intro",
  APPOINTMENT: "appointment",
});

const CINEMATIC_STAGE = Object.freeze({
  LOADING: "loading",
  HALLWAY: "hallway",
  OFFICE_ENTRY: "office_entry",
  MEETING: "meeting",
  CONTRACT: "contract",
  SIGNING: "signing",
  SIGNED: "signed",
  HANDSHAKE: "handshake",
  WELCOME: "welcome",
});

const CINEMATIC_STAGE_COPY = {
  [CINEMATIC_STAGE.LOADING]: "Preparing executive floor",
  [CINEMATIC_STAGE.HALLWAY]: "Executive floor",
  [CINEMATIC_STAGE.OFFICE_ENTRY]: "Executive office",
  [CINEMATIC_STAGE.MEETING]: "Hockey operations",
  [CINEMATIC_STAGE.CONTRACT]: "Appointment agreement",
  [CINEMATIC_STAGE.SIGNING]: "Signature required",
  [CINEMATIC_STAGE.SIGNED]: "Appointment executed",
  [CINEMATIC_STAGE.HANDSHAKE]: "Appointment confirmed",
  [CINEMATIC_STAGE.WELCOME]: "Welcome",
};


/* ============================================================================
   PUCKCEPTION OPENING HALLWAY
   ==========================================================================

   The corridor the player physically walks before the office. It is built
   procedurally so the first frame costs almost nothing: geometry, collision,
   camera and the two hero jerseys are ready immediately while the heavier
   office GLBs stream in behind the player.

   Corridor runs along -Z (entrance) to +Z (office door).
*/

const HALL = Object.freeze({
  width: 3.5,
  height: 3.15,
  startZ: -22,
  doorZ: 1.15,
  eyeHeight: 1.68,
  runnerWidth: 1.9,
  tkachukZ: -16.9,
});

const HALL_PHASE = Object.freeze({
  BOOTING: "booting",
  EXPLORING: "exploring",
  DOOR: "door",
  OFFICE: "office",
  SETTLED: "settled",
});

/*
  Memorabilia copy. Small elegant cards only — never a dashboard.
*/
const EXHIBIT_CARDS = Object.freeze({
  karlsson: {
    kicker: "Game-worn / authenticated",
    title: "Erik Karlsson",
    subtitle: "Ottawa Senators · No. 65 · Defence",
    lines: [
      "White Reebok away sweater, hung back-out. KARLSSON over a red 65, shoulder O crests, and the NHL 100 patch still on the right sleeve.",
      "Game 1, Eastern Conference Second Round, 2017 Stanley Cup Playoffs. Ottawa hosts the New York Rangers to open the series.",
      "Signed in silver across the numbers. The fight-strap stitching from that 2016-17 run is still on the hem.",
    ],
    footer: "Round 2 · Game 1 · 2017",
  },
  ovechkin: {
    kicker: "Rookie era / Koho",
    title: "Alex Ovechkin",
    subtitle: "Washington Capitals · No. 8 · Left Wing",
    lines: [
      "The black screaming-eagle Capitals sweater from Ovechkin's first NHL seasons, cut and stitched in the old Koho pattern before the league changed suppliers.",
      "Fifty-two goals and one hundred and six points as a rookie, and the Calder Trophy in a class nobody expected to be that good.",
      "Hung back-out on purpose. In this building the nameplate and the number are the whole point.",
    ],
    footer: "Rookie season · Washington",
  },
  cup: {
    kicker: "Championship hardware",
    title: "The Cup",
    subtitle: "On loan · travel case open",
    lines: [
      "Silver and nickel over a barrel base, engraved band after engraved band. It goes back in the case tonight.",
      "The polishing cloth beside it is not decorative. Fingerprints show on this thing within seconds.",
    ],
    footer: "Do not lift by the bowl",
  },
  masks: {
    kicker: "Goaltending",
    title: "Three Eras Of Nerve",
    subtitle: "Fibreglass · cage · modern shell",
    lines: [
      "A moulded fibreglass face piece from the era when a goaltender's whole protection was four millimetres of resin.",
      "Beside it, a bare wire cage, and a modern painted shell with a certified cat-eye.",
      "They are hung by the office door on purpose. Everyone who walks in has to look at them first.",
    ],
    footer: "Hung, not stored",
  },
  photos: {
    kicker: "Archive wall",
    title: "Forty Years Of Rooms Like This",
    subtitle: "Photographs · credentials · stubs",
    lines: [
      "Playoff credentials, ticket stubs and press photographs, most of them from buildings that no longer exist.",
      "Nothing here is arranged chronologically. It was hung whenever it arrived.",
    ],
    footer: "Unsorted, deliberately",
  },
  whiteboard: {
    kicker: "Coaching staff",
    title: "The Breakout",
    subtitle: "Do not erase",
    lines: [
      "Nine arrows, four hash marks, two question marks and a circled area labelled only \"NO\".",
      "Somebody added a fifth option in red marker. Nobody has admitted to it.",
    ],
    footer: "It has never been run in a game",
  },
  equipment: {
    kicker: "Equipment room",
    title: "Sticks, Pads And One Bad Night",
    subtitle: "Wood · composite · taped",
    lines: [
      "An old one-piece wooden stick, two modern composites, and one snapped shaft with the tape still wrapped around the blade.",
      "The pads have been leaned in that corner long enough to leave a mark on the panelling.",
    ],
    footer: "Nothing here is for sale",
  },
  tkachuk: {
    kicker: "Editorial position",
    title: "Fuck Brady Tkachuk",
    subtitle: "Ottawa Senators · No. 7 · Right wall",
    lines: [
      "The only object in the corridor that was not professionally mounted.",
      "Brady Tkachuk, spelled correctly, because management has standards even when it is being petty.",
      "There is a dart tray underneath it. Management considers that a feature.",
    ],
    footer: "Press the interact key to throw",
  },
  door: {
    kicker: "Hockey operations",
    title: "The Office",
    subtitle: "Executive suite",
    lines: [
      "Walnut, brass, and warm light leaking under the sill.",
      "Whoever is behind that door is already expecting you.",
    ],
    footer: "Interact to enter",
  },
});

const HALL_FUN_LABELS = Object.freeze({
  doNotTouch: "DO NOT TOUCH",
  tradeRequest: "0 DAYS WITHOUT A TRADE REQUEST",
  officeSign: "HOCKEY OPERATIONS",
});


/* ============================================================================
   PHYSICAL SCALE
   ==========================================================================

   One Babylon unit == approximately one real-world meter.

   These values intentionally replace the old "make it visible" scaling.
   Every model is normalized to a role-specific real-world dimension.

   If a particular downloaded GLB was authored facing the opposite direction,
   adjust ONLY that asset's yaw here. Do not add random 180-degree rotations
   elsewhere in the scene.
*/

const ASSET_CALIBRATION = Object.freeze({
  office: {
    measure: "width",
    target: 8.4,
    position: new Vector3(0, 0, 3.35),
    yaw: 0,
  },

  desk: {
    measure: "height",
    target: 0.76,
    position: new Vector3(0, 0, 3.05),
    yaw: 0,
    alignWideToX: true,
  },

  chair: {
    measure: "height",
    target: 1.22,
    position: new Vector3(0, 0, 4.28),
    yaw: Math.PI,
  },

  seatedExecutive: {
    measure: "height",
    target: 1.34,
    position: new Vector3(0, 0.02, 4.18),
    yaw: Math.PI,
  },

  standingExecutive: {
    measure: "height",
    target: 1.78,
    position: new Vector3(0, 0, 4.05),
    yaw: Math.PI,
  },

  hockeyStick: {
    measure: "longest",
    target: 1.65,
    position: new Vector3(-2.65, 0.05, 5.25),
    yaw: -0.16,
    orientLongestToY: true,
  },

  trophy: {
    measure: "height",
    target: 0.46,
    position: new Vector3(2.65, 0.84, 5.18),
    yaw: -0.1,
  },

  props: {
    measure: "longest",
    target: 1.35,
    position: new Vector3(2.35, 0.82, 4.78),
    yaw: Math.PI,
  },

  clipboard: {
    measure: "longest",
    target: 0.34,
    position: new Vector3(-0.70, 0.775, 2.86),
    yaw: -0.08,
    layFlat: true,
  },

  contract: {
    measure: "longest",
    target: 0.34,
    position: new Vector3(0.22, 0.777, 2.92),
    yaw: 0.04,
    layFlat: true,
  },
});


/* ============================================================================
   DYNAMIC CAMERA SHOTS
   ========================================================================== */

const CAMERA_SHOTS = Object.freeze({
  hallwayStart: {
    kind: "FIXED",
    position: new Vector3(0, HALL.eyeHeight, HALL.startZ + 0.9),
    target: new Vector3(0, HALL.eyeHeight - 0.16, HALL.startZ + 6.4),
    fov: 0.9,
    skipSafety: true,
  },
  hallwayMid: {
    kind: "FIXED",
    position: new Vector3(-0.18, HALL.eyeHeight, -8.4),
    target: new Vector3(0.1, HALL.eyeHeight - 0.14, 1.2),
    fov: 0.84,
    skipSafety: true,
  },
  doorApproach: {
    kind: "FIXED",
    position: new Vector3(0, HALL.eyeHeight - 0.02, HALL.doorZ - 2.05),
    target: new Vector3(0, HALL.eyeHeight - 0.2, HALL.doorZ + 0.6),
    fov: 0.8,
    skipSafety: true,
  },
  doorThreshold: {
    kind: "FIXED",
    position: new Vector3(0, HALL.eyeHeight - 0.05, HALL.doorZ + 0.35),
    target: new Vector3(0, HALL.eyeHeight - 0.26, HALL.doorZ + 5.2),
    fov: 0.86,
    skipSafety: true,
  },

  /*
    The office beats are authored rather than derived.

    The room GLB is normalized around its own centre at z = 3.35, which puts its
    near wall behind the corridor doorway — an automatically fitted wide shot
    would reverse the camera back through that wall. These coordinates keep the
    camera inside the room and on the player's side of the desk, which is where
    the document is.
  */
  officeReveal: {
    kind: "FIXED",
    position: new Vector3(0.72, 1.72, 1.34),
    target: new Vector3(0.05, 1.14, 3.5),
    fov: 0.92,
    skipSafety: true,
  },
  officeAddress: {
    kind: "FIXED",
    position: new Vector3(-0.62, 1.68, 1.86),
    target: new Vector3(0.38, 1.22, 3.78),
    fov: 0.82,
    skipSafety: true,
  },
  deskApproach: {
    kind: "FIXED",
    position: new Vector3(0.3, 1.5, 2.02),
    target: new Vector3(0.2, 0.82, 3.02),
    fov: 0.74,
    skipSafety: true,
  },
  contractReveal: {
    kind: "FIXED",
    position: new Vector3(0.23, 1.08, 2.44),
    target: new Vector3(0.22, 0.78, 2.95),
    fov: 0.62,
    skipSafety: true,
  },
  officeWide: {
    kind: "ROOM",
    asset: "office",
    coverage: "wide",
    cameraDepth: 0.02,
    lookDepth: 0.58,
    eyeHeight: 1.72,
    lookHeight: 1.22,
    side: 0.55,
    fov: 0.88,
  },
  deskThreeQuarter: {
    kind: "DESK",
    asset: "desk",
    coverage: "medium",
    fov: 0.72,
  },
  chairThreeQuarter: {
    kind: "CHAIR",
    asset: "chair",
    coverage: "medium",
    fov: 0.7,
  },
  gmHero: {
    kind: "HUMAN",
    asset: "seatedExecutive",
    coverage: "medium",
    fov: 0.68,
  },
  gmClose: {
    kind: "HUMAN",
    asset: "seatedExecutive",
    coverage: "close",
    fov: 0.58,
  },
  stickProp: {
    kind: "PROP",
    asset: "hockeyStick",
    coverage: "close",
    fov: 0.62,
  },
  trophyProp: {
    kind: "PROP",
    asset: "trophy",
    coverage: "close",
    fov: 0.6,
  },
  officeHero: {
    kind: "HERO",
    assets: ["desk", "chair", "seatedExecutive"],
    coverage: "wide",
    fov: 0.78,
  },
  officeEntry: {
    kind: "HERO",
    assets: ["desk", "chair", "seatedExecutive", "office"],
    coverage: "wide",
    fov: 0.84,
  },
  seatedExecutiveWide: {
    kind: "HUMAN",
    asset: "seatedExecutive",
    coverage: "wide",
    fov: 0.74,
  },
  seatedExecutiveClose: {
    kind: "HUMAN",
    asset: "seatedExecutive",
    coverage: "close",
    fov: 0.58,
  },
  contract: {
    kind: "PROP",
    asset: "contract",
    coverage: "close",
    fov: 0.56,
  },
  standingExecutive: {
    kind: "HUMAN",
    asset: "standingExecutive",
    coverage: "medium",
    fov: 0.66,
  },
  branding: {
    kind: "PROP",
    asset: "trophy",
    coverage: "close",
    fov: 0.64,
  },
});


/* ============================================================================
   CONTRACT
   ========================================================================== */

const CONTRACT_TEXTURE = Object.freeze({
  width: 1200,
  height: 1697,
});

const SIGNATURE_ZONE = Object.freeze({
  x: 112,
  y: 1254,
  width: 710,
  height: 220,
});


/* ============================================================================
   BASIC HELPERS
   ========================================================================== */

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function easeInOutCubic(t) {
  return t < 0.5
    ? 4 * t * t * t
    : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function easeInOutQuint(t) {
  return t < 0.5
    ? 16 * t * t * t * t * t
    : 1 - Math.pow(-2 * t + 2, 5) / 2;
}

function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

function easeOutQuint(t) {
  return 1 - Math.pow(1 - t, 5);
}

function smoothStep(t) {
  return t * t * (3 - 2 * t);
}

function shuffleArray(array) {
  const result = [...array];

  for (let index = result.length - 1; index > 0; index -= 1) {
    const randomIndex = Math.floor(Math.random() * (index + 1));
    [result[index], result[randomIndex]] = [
      result[randomIndex],
      result[index],
    ];
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
    team?.name ||
      team?.team_name ||
      ""
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
  if (
    setupIndex == null ||
    setupIndex < 0
  ) {
    return -1;
  }

  const found = orderedTeams.findIndex(
    (item) => item.index === setupIndex
  );

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

function colorFromHex(hex, fallback = "#ffffff") {
  try {
    return Color3.FromHexString(hex || fallback);
  } catch (_error) {
    return Color3.FromHexString(fallback);
  }
}

function sleep(ms, cancelledRef) {
  return new Promise((resolve) => {
    const timeout = window.setTimeout(() => {
      resolve(!cancelledRef.current);
    }, ms);

    if (cancelledRef.current) {
      window.clearTimeout(timeout);
      resolve(false);
    }
  });
}

function tween({
  duration,
  cancelledRef,
  easing = easeInOutCubic,
  onUpdate,
}) {
  return new Promise((resolve) => {
    const started = performance.now();

    const frame = (now) => {
      if (cancelledRef.current) {
        resolve(false);
        return;
      }

      const raw = clamp(
        (now - started) / Math.max(duration, 1),
        0,
        1
      );

      onUpdate(
        easing(raw),
        raw
      );

      if (raw >= 1) {
        resolve(true);
        return;
      }

      window.requestAnimationFrame(frame);
    };

    window.requestAnimationFrame(frame);
  });
}

function vectorLerp(from, to, t) {
  return new Vector3(
    lerp(from.x, to.x, t),
    lerp(from.y, to.y, t),
    lerp(from.z, to.z, t)
  );
}

/* ============================================================================
   SPRING MOTION
   ========================================================================== */

class VectorSpring {
  constructor(value) {
    this.value = value.clone();
    this.velocity = Vector3.Zero();
    this.target = value.clone();
  }

  setTarget(value) {
    this.target.copyFrom(value);
  }

  snap(value) {
    this.value.copyFrom(value);
    this.target.copyFrom(value);
    this.velocity.set(0, 0, 0);
  }

  update(
    dt,
    {
      stiffness = 38,
      damping = 11,
    } = {}
  ) {
    const delta = this.target.subtract(
      this.value
    );

    this.velocity.addInPlace(
      delta.scale(
        stiffness * dt
      )
    );

    this.velocity.scaleInPlace(
      Math.exp(
        -damping * dt
      )
    );

    this.value.addInPlace(
      this.velocity.scale(dt)
    );

    return this.value;
  }
}


/* ============================================================================
   GLB BOUNDS / NORMALIZATION
   ========================================================================== */

function meshListForAsset(asset) {
  return (asset?.meshes || [])
    .filter(
      (mesh) =>
        mesh &&
        typeof mesh.getBoundingInfo === "function" &&
        (mesh.getTotalVertices?.() || 0) > 0
    );
}

function computeAssetBounds(asset) {
  const meshes = meshListForAsset(asset);

  if (!meshes.length) {
    return {
      min: Vector3.Zero(),
      max: Vector3.Zero(),
      center: Vector3.Zero(),
      size: Vector3.Zero(),
    };
  }

  let min = new Vector3(
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY
  );

  let max = new Vector3(
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY
  );

  meshes.forEach((mesh) => {
    mesh.computeWorldMatrix(true);

    const box =
      mesh.getBoundingInfo()
        .boundingBox;

    min = Vector3.Minimize(
      min,
      box.minimumWorld
    );

    max = Vector3.Maximize(
      max,
      box.maximumWorld
    );
  });

  return {
    min,
    max,
    center: min.add(max).scale(0.5),
    size: max.subtract(min),
  };
}
function buildCameraProfile(asset) {
  if (!asset) {
    return null;
  }

  const bounds = computeAssetBounds(asset);
  const width = Math.abs(bounds.size.x);
  const height = Math.abs(bounds.size.y);
  const depth = Math.abs(bounds.size.z);

  return {
    bounds,
    center: bounds.center.clone(),
    width,
    height,
    depth,
    radius:
      Math.sqrt(
        width * width +
        height * height +
        depth * depth
      ) / 2,
  };
}

function buildMeshCameraProfile(mesh) {
  if (!mesh?.getBoundingInfo) {
    return null;
  }

  mesh.computeWorldMatrix(true);

  const box = mesh.getBoundingInfo().boundingBox;
  const min = box.minimumWorld.clone();
  const max = box.maximumWorld.clone();
  const size = max.subtract(min);
  const center = min.add(max).scale(0.5);
  const width = Math.abs(size.x);
  const height = Math.abs(size.y);
  const depth = Math.abs(size.z);

  return {
    bounds: {
      min,
      max,
      center,
      size,
    },
    center,
    width,
    height,
    depth,
    radius:
      Math.sqrt(
        width * width +
        height * height +
        depth * depth
      ) / 2,
  };
}

function cameraTargetFromProfile(
  profile,
  {
    targetRatio = 0.5,
    verticalOffset = 0,
  } = {}
) {
  const { bounds } = profile;

  return new Vector3(
    bounds.center.x,
    bounds.min.y +
      bounds.size.y * targetRatio +
      verticalOffset,
    bounds.center.z
  );
}

function distanceToFitObject({
  profile,
  fov,
  aspect = 16 / 9,
  padding = 1.25,
}) {
  const vertical = Math.max(profile.height, 0.05);
  const horizontal = Math.max(profile.width, 0.05);
  const safeAspect = Math.max(aspect, 0.25);

  const horizontalFov =
    2 *
    Math.atan(
      Math.tan(fov / 2) *
        safeAspect
    );

  const verticalDistance =
    (vertical / 2) /
    Math.tan(fov / 2);

  const horizontalDistance =
    (horizontal / 2) /
    Math.tan(horizontalFov / 2);

  return Math.max(
    verticalDistance,
    horizontalDistance,
    0.5
  ) * padding;
}

function getPreferredViewDirection(asset) {
  if (!asset?.root) {
    return new Vector3(0, 0, -1);
  }

  const yaw = asset.root.rotation?.y || 0;

  return new Vector3(
    Math.sin(yaw),
    0,
    Math.cos(yaw)
  ).normalize();
}

function unionAssetBounds(assetList) {
  const valid = assetList.map((asset) => computeAssetBounds(asset)).filter(
    (bounds) => bounds && bounds.size.length() > 0.01
  );

  if (!valid.length) {
    return null;
  }

  let min = valid[0].min.clone();
  let max = valid[0].max.clone();
  valid.forEach((bounds) => {
    min = Vector3.Minimize(min, bounds.min);
    max = Vector3.Maximize(max, bounds.max);
  });

  return {
    min,
    max,
    center: min.add(max).scale(0.5),
    size: max.subtract(min),
  };
}

function pullBackIfTooClose(position, target, minDistance) {
  const offset = position.subtract(target);
  const length = offset.length();
  if (length >= minDistance) {
    return position;
  }
  if (length < 0.001) {
    return target.add(new Vector3(0.4, 1.2, -minDistance));
  }
  return target.add(offset.normalize().scale(minDistance));
}

function composeEntityShot({
  scene,
  camera,
  assets,
  shot,
  logoPlane,
}) {
  const fov = shot.fov ?? 0.74;
  const aspect =
    scene.getEngine?.().getAspectRatio?.(camera) || 16 / 9;
  const asset = shot.asset
    ? shot.asset === "logoPlane"
      ? { root: logoPlane, meshes: logoPlane ? [logoPlane] : [] }
      : assets?.[shot.asset]
    : null;
  const profile = asset ? buildCameraProfile(asset) : null;
  const ignoreMeshes = asset?.meshes || [];

  let position;
  let target;

  if (shot.kind === "FIXED") {
    /*
      Authored corridor framing. The hallway is procedural and its own walls
      would otherwise trip the occlusion search below.
    */
    return {
      shot,
      profile: null,
      asset: null,
      target: shot.target.clone(),
      position: shot.position.clone(),
      fov,
    };
  }

  if ((shot.kind === "ROOM" || shot.kind === "WALK") && profile) {
    const bounds = profile.bounds;
    const camZ = bounds.min.z + profile.depth * (shot.cameraDepth ?? 0.06);
    const lookZ = bounds.min.z + profile.depth * (shot.lookDepth ?? 0.62);
    position = new Vector3(
      bounds.center.x + (shot.side ?? 0.3),
      bounds.min.y + (shot.eyeHeight ?? 1.62),
      camZ
    );
    target = new Vector3(
      bounds.center.x,
      bounds.min.y + (shot.lookHeight ?? 1.36),
      lookZ
    );
  } else if (shot.kind === "HUMAN" && profile) {
    const bounds = profile.bounds;
    const heightRatio = shot.coverage === "close" ? 0.78 : 0.72;
    target = new Vector3(
      bounds.center.x,
      bounds.min.y + profile.height * heightRatio,
      bounds.center.z
    );
    const fitProfile = {
      ...profile,
      height: profile.height * (shot.coverage === "close" ? 0.48 : 0.78),
      width: profile.width * 0.85,
    };
    const distance = distanceToFitObject({
      profile: fitProfile,
      fov,
      aspect,
      padding: shot.coverage === "close" ? 1.2 : 1.48,
    });
    const yaw = Math.PI * 0.28;
    const dir = new Vector3(Math.sin(yaw), 0.06, Math.cos(yaw)).normalize();
    position = target.add(dir.scale(Math.max(distance, 1.15)));
    position.y = Math.max(position.y, target.y - 0.12);
  } else if (shot.kind === "DESK" && profile) {
    const bounds = profile.bounds;
    target = new Vector3(
      bounds.center.x,
      bounds.max.y + 0.05,
      bounds.center.z
    );
    const distance =
      Math.max(profile.width, profile.depth) *
        (shot.coverage === "wide" ? 1.7 : 1.35) +
      1.55;
    const yaw = Math.PI * 0.22;
    position = new Vector3(
      target.x + Math.sin(yaw) * distance,
      target.y + Math.max(1.05, profile.height * 0.95),
      target.z + Math.cos(yaw) * distance * 0.82
    );
  } else if (shot.kind === "CHAIR" && profile) {
    const bounds = profile.bounds;
    target = new Vector3(
      bounds.center.x,
      bounds.min.y + profile.height * 0.58,
      bounds.center.z
    );
    const distance = distanceToFitObject({
      profile,
      fov,
      aspect,
      padding: 1.4,
    });
    const yaw = -Math.PI * 0.3;
    position = target.add(
      new Vector3(Math.sin(yaw), 0.12, Math.cos(yaw)).scale(distance)
    );
  } else if (shot.kind === "PROP" && profile) {
    target = profile.center.clone();
    const longest = Math.max(profile.width, profile.height, profile.depth, 0.12);
    const distance = Math.max(0.95, longest * 2.2);
    position = target.add(new Vector3(0.42, 0.28, distance));
  } else if (shot.kind === "HERO") {
    const names = shot.assets || ["desk", "chair", "seatedExecutive"];
    const union = unionAssetBounds(
      names.map((name) => assets?.[name]).filter(Boolean)
    );
    if (union) {
      target = new Vector3(
        union.center.x,
        union.min.y + Math.max(1.05, union.size.y * 0.48),
        union.center.z
      );
      const distance = Math.max(
        Math.max(union.size.x, union.size.z) * 1.05 + 2.8,
        3.2
      );
      const yaw = 0.38;
      position = new Vector3(
        target.x + Math.sin(yaw) * distance,
        1.55,
        target.z - Math.cos(yaw) * distance
      );
    }
  } else if (profile) {
    target = cameraTargetFromProfile(profile, { targetRatio: 0.55 });
    const distance = distanceToFitObject({
      profile,
      fov,
      aspect,
      padding: 1.4,
    });
    position = target.add(new Vector3(0.4, 0.35, distance));
  }

  if (!position || !target) {
    return null;
  }

  position = pullBackIfTooClose(position, target, 1.35);
  if (position.y < 0.45) {
    position.y = 0.45;
  }

  position = findSafeCameraPosition({
    scene,
    desiredPosition: position,
    target,
    ignoreMeshes,
  });
  position = pullBackIfTooClose(position, target, 0.7);

  return {
    shot,
    profile,
    asset,
    target,
    position,
    fov,
  };
}

function findSafeCameraPosition({
  scene,
  desiredPosition,
  target,
  ignoreMeshes = [],
}) {
  const ignored = new Set(
    ignoreMeshes.filter(Boolean)
  );

  const candidates = [
    desiredPosition.clone(),
    desiredPosition.add(new Vector3(0.45, 0.08, 0)),
    desiredPosition.add(new Vector3(-0.45, 0.08, 0)),
    desiredPosition.add(new Vector3(0.85, 0.12, 0.20)),
    desiredPosition.add(new Vector3(-0.85, 0.12, 0.20)),
    desiredPosition.add(new Vector3(0, 0.20, -0.45)),
    desiredPosition.add(new Vector3(0, 0.20, 0.45)),
  ];

  for (const candidate of candidates) {
    const direction = target.subtract(candidate);
    const length = direction.length();

    if (length <= 0.001) {
      continue;
    }

    const ray = new Ray(
      candidate,
      direction.normalize(),
      Math.max(length - 0.08, 0.001)
    );

    const hit = scene.pickWithRay(
      ray,
      (mesh) => {
        if (!mesh || ignored.has(mesh)) {
          return false;
        }

        return (
          mesh.isEnabled?.() !== false &&
          mesh.isVisible !== false &&
          (mesh.getTotalVertices?.() || 0) > 8
        );
      }
    );

    if (!hit?.hit) {
      return candidate;
    }
  }

  return desiredPosition;
}

function createCameraDirector({
  scene,
  camera,
  assets,
  logoPlane,
  cancelledRef,
}) {
  function getProfile(name) {
    if (name === "logoPlane") {
      return buildMeshCameraProfile(logoPlane);
    }

    return buildCameraProfile(assets?.[name]);
  }

  function getAsset(name) {
    if (name === "logoPlane") {
      return null;
    }

    return assets?.[name];
  }

  function resolveShot(shotName) {
    const shot = CAMERA_SHOTS[shotName];

    if (!shot) {
      console.warn(`Unknown camera shot: ${shotName}`);
      return null;
    }

    return composeEntityShot({
      scene,
      camera,
      assets,
      shot,
      logoPlane,
    });
  }

  function snap(shotName) {
    const resolved = resolveShot(shotName);

    if (!resolved) {
      return false;
    }

    camera.position.copyFrom(resolved.position);
    camera.fov = resolved.fov;
    camera.rotation.z = 0;
    camera.setTarget(resolved.target);
    return true;
  }

  async function focus(
    shotName,
    {
      duration = 1200,
      walking = false,
      breathing = false,
      easing = easeInOutQuint,
    } = {}
  ) {
    const resolved = resolveShot(shotName);

    if (!resolved) {
      return false;
    }

    const desiredPosition = resolved.position;
    const target = resolved.target;
    const desiredFov = resolved.fov;
    const startingFov = camera.fov;
    const fromPosition = camera.position.clone();
    const currentForward = camera.getForwardRay(1).direction;
    const fromTarget = camera.position.add(
      currentForward.scale(5)
    );

    return tween({
      duration,
      cancelledRef,
      easing,
      onUpdate: (t, raw) => {
        const position = vectorLerp(
          fromPosition,
          desiredPosition,
          t
        );

        const currentTarget = vectorLerp(
          fromTarget,
          target,
          t
        );

        const envelope = Math.sin(raw * Math.PI);

        if (walking) {
          const stride = Math.sin(raw * Math.PI * 10);
          const lateral = Math.sin(raw * Math.PI * 5);

          position.y += stride * 0.014 * envelope;
          position.x += lateral * 0.01 * envelope;
          currentTarget.x += lateral * 0.008 * envelope;
          currentTarget.y += stride * 0.003 * envelope;
        }

        if (breathing) {
          position.y +=
            Math.sin(raw * Math.PI * 3) *
            0.005 *
            envelope;
        }

        camera.position.copyFrom(position);
        camera.fov = lerp(
          startingFov,
          desiredFov,
          t
        );
        camera.rotation.z = 0;
        camera.setTarget(currentTarget);
      },
    });
  }

  function targetFor(shotName) {
    return resolveShot(shotName)?.target?.clone() || null;
  }

  return {
    focus,
    snap,
    targetFor,
  };
}

function measurementFromBounds(
  bounds,
  measure
) {
  if (measure === "width") {
    return Math.abs(bounds.size.x);
  }

  if (measure === "height") {
    return Math.abs(bounds.size.y);
  }

  if (measure === "depth") {
    return Math.abs(bounds.size.z);
  }

  return Math.max(
    Math.abs(bounds.size.x),
    Math.abs(bounds.size.y),
    Math.abs(bounds.size.z)
  );
}

function prepareImportedMaterials(asset) {
  (asset?.meshes || []).forEach((mesh) => {
    const material = mesh.material;
    if (!material) {
      return;
    }

    if (material instanceof PBRMaterial) {
      material.roughness = clamp(
        Number.isFinite(material.roughness) ? material.roughness : 0.65,
        0.28,
        0.92
      );
      material.metallic = clamp(
        Number.isFinite(material.metallic) ? material.metallic : 0,
        0,
        1
      );
      material.environmentIntensity = 0;
      material.directIntensity = 1.6;
      material.specularIntensity = 1.05;
      material.maxSimultaneousLights = 16;
      material.ambientColor = new Color3(1, 1, 1);

      if (!material.albedoTexture) {
        material.albedoColor = new Color3(0.62, 0.54, 0.44);
      }

      material.emissiveColor = new Color3(0.22, 0.2, 0.16);
      material.emissiveIntensity = 0.55;
    }
  });
}

function collectDoorMeshes(asset) {
  return (asset?.meshes || []).filter((mesh) => {
    const name = String(mesh?.name || "").toLowerCase();
    return /door|porte|gate/.test(name);
  });
}

function createEntranceDoors(scene, hallwayAsset) {
  if (!hallwayAsset) {
    return null;
  }

  const bounds = computeAssetBounds(hallwayAsset);
  const height = Math.max(
    2.15,
    Math.min(2.7, Math.abs(bounds.size.y) * 0.82 || 2.35)
  );
  const opening = Math.min(1.7, Math.max(1.35, Math.abs(bounds.size.x) * 0.22));
  const z = bounds.max.z - 0.06;
  const y = bounds.min.y;

  const wood = new StandardMaterial("setup-door-wood", scene);
  wood.diffuseColor = new Color3(0.32, 0.2, 0.12);
  wood.specularColor = new Color3(0.08, 0.06, 0.04);
  wood.emissiveColor = new Color3(0.06, 0.04, 0.025);

  const makeLeaf = (name, side) => {
    const hinge = new TransformNode(`${name}-hinge`, scene);
    hinge.position.set(
      bounds.center.x + side * (opening / 2),
      y,
      z
    );

    const leaf = MeshBuilder.CreateBox(
      name,
      {
        width: opening / 2 - 0.02,
        height,
        depth: 0.07,
      },
      scene
    );

    leaf.parent = hinge;
    leaf.position.set(side * ((opening / 4) - 0.01), height / 2, 0);
    leaf.material = wood;
    leaf.receiveShadows = true;
    leaf.isPickable = false;

    return hinge;
  };

  return {
    left: makeLeaf("setup-door-left", -1),
    right: makeLeaf("setup-door-right", 1),
    found: [],
  };
}

function openEntranceDoors(doors, cancelledRef, duration) {
  if (!doors) {
    return Promise.resolve(true);
  }

  const found = doors.found || [];

  found.forEach((mesh) => {
    mesh.rotationQuaternion = null;
  });

  return tween({
    duration,
    cancelledRef,
    easing: easeOutCubic,
    onUpdate: (t) => {
      if (doors.left) {
        doors.left.rotation.y = lerp(0, -1.42, t);
      }

      if (doors.right) {
        doors.right.rotation.y = lerp(0, 1.42, t);
      }

      found.forEach((mesh, index) => {
        const swing = index % 2 === 0 ? -1.35 : 1.35;
        mesh.rotation.y = lerp(0.4 * Math.sign(swing), swing, t);
      });
    },
  });
}

function hideNamedMeshes(
  asset,
  keywords
) {
  const safeKeywords = keywords.map(
    (keyword) =>
      String(keyword).toLowerCase()
  );

  (asset?.meshes || []).forEach((mesh) => {
    const name = String(
      mesh?.name || ""
    ).toLowerCase();

    if (
      safeKeywords.some(
        (keyword) =>
          name.includes(keyword)
      )
    ) {
      mesh.setEnabled(false);
    }
  });
}

function createAssetRoot(
  scene,
  name,
  imported
) {
  const root = new TransformNode(
    `${name}-root`,
    scene
  );

  imported.meshes.forEach((mesh) => {
    if (
      mesh &&
      !mesh.parent
    ) {
      mesh.parent = root;
    }
  });

  return root;
}

function orientLongestAxisToY(
  root,
  asset
) {
  root.rotationQuaternion = null;
  root.rotation.set(0, 0, 0);

  let bounds =
    computeAssetBounds(asset);

  const dimensions = [
    ["x", Math.abs(bounds.size.x)],
    ["y", Math.abs(bounds.size.y)],
    ["z", Math.abs(bounds.size.z)],
  ].sort(
    (a, b) => b[1] - a[1]
  );

  const longestAxis =
    dimensions[0]?.[0];

  if (longestAxis === "x") {
    root.rotation.z =
      Math.PI / 2;
  } else if (longestAxis === "z") {
    root.rotation.x =
      Math.PI / 2;
  }

  bounds = computeAssetBounds(asset);

  return bounds;
}

function layThinnestAxisOnY(
  root,
  asset
) {
  root.rotationQuaternion = null;

  let bounds =
    computeAssetBounds(asset);

  const dimensions = [
    ["x", Math.abs(bounds.size.x)],
    ["y", Math.abs(bounds.size.y)],
    ["z", Math.abs(bounds.size.z)],
  ].sort(
    (a, b) => a[1] - b[1]
  );

  const thinnestAxis =
    dimensions[0]?.[0];

  if (thinnestAxis === "x") {
    root.rotation.z +=
      Math.PI / 2;
  } else if (thinnestAxis === "z") {
    root.rotation.x +=
      Math.PI / 2;
  }

  bounds = computeAssetBounds(asset);

  return bounds;
}

function alignWideAxisToX(
  root,
  asset
) {
  const bounds =
    computeAssetBounds(asset);

  if (
    Math.abs(bounds.size.z) >
    Math.abs(bounds.size.x)
  ) {
    root.rotation.y +=
      Math.PI / 2;
  }
}

function normalizeLoadedAsset(
  asset,
  calibration
) {
  const {
    root,
  } = asset;

  root.position.set(0, 0, 0);
  root.scaling.set(1, 1, 1);
  root.rotationQuaternion = null;
  root.rotation.set(0, 0, 0);

  if (
    calibration.orientLongestToY
  ) {
    orientLongestAxisToY(
      root,
      asset
    );
  }

  if (
    calibration.layFlat
  ) {
    layThinnestAxisOnY(
      root,
      asset
    );
  }

  root.rotation.y +=
    calibration.yaw || 0;

  if (
    calibration.alignWideToX
  ) {
    alignWideAxisToX(
      root,
      asset
    );
  }

  let bounds =
    computeAssetBounds(asset);

  const currentMeasurement =
    measurementFromBounds(
      bounds,
      calibration.measure
    );

  const safeMeasurement =
    Math.max(
      currentMeasurement,
      0.000001
    );

  const scale =
    calibration.target /
    safeMeasurement;

  root.scaling.set(
    scale,
    scale,
    scale
  );

  bounds =
    computeAssetBounds(asset);

  const desired =
    calibration.position;

  root.position.addInPlace(
    new Vector3(
      desired.x -
        bounds.center.x,
      desired.y -
        bounds.min.y,
      desired.z -
        bounds.center.z
    )
  );

  computeAssetBounds(asset);

  return asset;
}

async function loadGlbAsset({
  scene,
  name,
  url,
  calibration,
  enabled = true,
  materialOptions,
}) {
  const imported =
    await SceneLoader.ImportMeshAsync(
      "",
      "",
      url,
      scene
    );

  const root =
    createAssetRoot(
      scene,
      name,
      imported
    );

  const asset = {
    name,
    root,
    meshes: imported.meshes,
    animationGroups:
      imported.animationGroups || [],
    skeletons:
      imported.skeletons || [],
    transformNodes:
      imported.transformNodes || [],
  };

  prepareImportedMaterials(
    asset,
    materialOptions
  );

  normalizeLoadedAsset(
    asset,
    calibration
  );

  const debugBounds = computeAssetBounds(asset);

  console.table({
    asset: name,
    width: Number(debugBounds.size.x.toFixed(3)),
    height: Number(debugBounds.size.y.toFixed(3)),
    depth: Number(debugBounds.size.z.toFixed(3)),
    centerX: Number(debugBounds.center.x.toFixed(3)),
    centerY: Number(debugBounds.center.y.toFixed(3)),
    centerZ: Number(debugBounds.center.z.toFixed(3)),
  });

  root.setEnabled(enabled);

  return asset;
}

function setAssetEnabled(
  asset,
  enabled
) {
  asset?.root?.setEnabled(
    Boolean(enabled)
  );
}

function playBestAnimation(
  asset,
  keywords,
  {
    loop = true,
    speedRatio = 1,
  } = {}
) {
  if (!asset?.animationGroups?.length) {
    return null;
  }

  const lowerKeywords =
    keywords.map(
      (keyword) =>
        String(keyword).toLowerCase()
    );

  const group =
    asset.animationGroups.find(
      (animationGroup) => {
        const name = String(
          animationGroup?.name || ""
        ).toLowerCase();

        return lowerKeywords.some(
          (keyword) =>
            name.includes(keyword)
        );
      }
    ) ||
    asset.animationGroups[0];

  try {
    group.stop();
    group.start(
      loop,
      speedRatio
    );
  } catch (_error) {
    try {
      group.play(loop);
    } catch (_secondError) {
      return null;
    }
  }

  return group;
}


/* ============================================================================
   2D TEAM LOGO
   ========================================================================== */

function TeamLogo({
  team,
  size = 54,
  className = "",
  decorative = false,
}) {
  const label =
    team?.name ||
    team?.code ||
    "NHL club";

  const src =
    team?.logo ||
    resolveFranchiseTeamLogo(
      team?.raw || team,
      team?.name ||
        teamDisplayName(team)
    );

  if (!src) {
    return (
      <span
        className={`setup-logo-fallback setup-logo-coin ${className}`}
        style={{
          width: size,
          height: size,
        }}
        aria-hidden={
          decorative
            ? "true"
            : undefined
        }
      >
        {team?.code || "NHL"}
      </span>
    );
  }

  return (
    <span
      className={`setup-logo-coin ${className}`}
      style={{
        width: size,
        height: size,
      }}
    >
      <img
        src={src}
        className="setup-team-logo"
        alt={
          decorative ? "" : label
        }
        draggable={false}
        style={{
          width: size * 0.72,
          height: size * 0.72,
        }}
      />
    </span>
  );
}


/* ============================================================================
   APPOINTMENT DEED AND CLUB PICKER
   ========================================================================== */

/*
  The opening hallway synthesises its own lounge bed, HVAC and room tone, so
  the pre-rendered menu theme stays out of the way for the whole of this
  screen. It remains available as a fallback bed if the 3D floor cannot run.
*/
function useSetupStageMusic(enabled) {
  useEffect(() => {
    if (!enabled) {
      return undefined;
    }

    const audio = new Audio(setupTheme);
    audio.loop = true;
    audio.volume = 0.28;
    audio.preload = "auto";

    const tryPlay = () => {
      audio.play().catch(() => {});
    };

    tryPlay();
    window.addEventListener("pointerdown", tryPlay);
    window.addEventListener("keydown", tryPlay);

    return () => {
      window.removeEventListener("pointerdown", tryPlay);
      window.removeEventListener("keydown", tryPlay);
      audio.pause();
      audio.src = "";
    };
  }, [enabled]);
}

function getSvgPathFromStroke(points) {
  if (!points.length) {
    return "";
  }

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
    if (event.buttons !== 1 || !liveRef.current.length) {
      return;
    }
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
    Boolean(selected) &&
    Boolean(gmName?.trim()) &&
    signatureReady &&
    !loading;

  const teamName =
    selected?.name ||
    "National Hockey League Club";
  const teamCode = selected?.code || "NHL";
  const logoSrc =
    selected?.logo ||
    resolveFranchiseTeamLogo(
      selected?.raw || selected,
      selected?.name || teamDisplayName(selected)
    );

  return (
    <article className="setup-deed-sheet">
      <div className="setup-deed-paper">
        <p className="setup-deed-kicker">National Hockey League</p>
        <h2 className="setup-deed-title">
          General Manager
          <span>Appointment Deed</span>
        </h2>

        <div className="setup-deed-club">
          {logoSrc ? (
            <img src={logoSrc} alt="" />
          ) : (
            <em>{teamCode}</em>
          )}
          <div>
            <strong>{teamName}</strong>
            <small>{teamCode} / Hockey Operations</small>
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
                className={
                  playerUniverse !== "real_nhl"
                    ? "setup-token is-on"
                    : "setup-token"
                }
                aria-pressed={playerUniverse !== "real_nhl"}
                onClick={() => setPlayerUniverse("generated")}
              >
                <span className="setup-token-orb" aria-hidden="true" />
                <strong>Generated</strong>
              </button>
              <button
                type="button"
                className={
                  playerUniverse === "real_nhl"
                    ? "setup-token is-on"
                    : "setup-token"
                }
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
                className={
                  injuriesEnabled ? "setup-token is-on" : "setup-token"
                }
                aria-pressed={injuriesEnabled}
                onClick={() => setInjuriesEnabled(true)}
              >
                <span className="setup-token-orb" aria-hidden="true" />
                <strong>On</strong>
              </button>
              <button
                type="button"
                className={
                  !injuriesEnabled ? "setup-token is-on" : "setup-token"
                }
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
            <dd>
              {playerUniverse === "real_nhl"
                ? "Real NHL players"
                : "Generated players"}
            </dd>
          </div>
          <div>
            <dt>League health system</dt>
            <dd>{injuriesEnabled ? "Enabled" : "Disabled"}</dd>
          </div>
        </dl>

        <p className="setup-deed-legal">
          By executing this agreement, the General Manager accepts authority
          over hockey operations, transactions, contracts, staff, roster
          construction, scouting, and franchise strategy subject to the rules
          of Franchise Mode.
        </p>

        <div className="setup-deed-sign">
          <span>General Manager signature</span>
          <SignaturePad onInkChange={setSignatureReady} />
          <small>
            {signatureReady
              ? "Signature captured"
              : "Sign on the line above"}
          </small>
        </div>

        {error ? (
          <div className="setup-error" role="alert">
            {error}
          </div>
        ) : null}

        <button
          type="button"
          className="setup-accept-btn"
          disabled={!canContinue}
          onClick={onAccept}
        >
          <span>Begin franchise</span>
          <small>Open hockey operations</small>
        </button>
      </div>
    </article>
  );
}

const TeamSelection = React.memo(function TeamSelection({
  teams,
  selectedIndex,
  onSelect,
}) {
  const clubList = teams.length
    ? teams
    : buildOrderedTeams(buildDefaultFranchiseTeamList());

  return (
    <section className="setup-team-selector">
      <header className="setup-panel-heading">
        <span>Choose your club</span>
        <strong>{clubList.length} NHL clubs</strong>
      </header>
      <ClubBallBoard
        teams={clubList}
        selectedIndex={selectedIndex}
        onSelect={onSelect}
      />
    </section>
  );
});

function ConfigurationPanel({
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
  return (
    <AppointmentDeedSheet
      selected={selected}
      gmName={gmName}
      setGmName={setGmName}
      playerUniverse={playerUniverse}
      setPlayerUniverse={setPlayerUniverse}
      injuriesEnabled={injuriesEnabled}
      setInjuriesEnabled={setInjuriesEnabled}
      onAccept={onAccept}
      loading={loading}
      error={error}
      contractDate={contractDate}
      signatureReady={signatureReady}
      setSignatureReady={setSignatureReady}
    />
  );
}

function drawWrappedText(
  ctx,
  text,
  x,
  y,
  maxWidth,
  lineHeight
) {
  const words =
    String(text || "")
      .split(/\s+/);

  let line = "";
  let cursorY = y;

  words.forEach((word) => {
    const next =
      line
        ? `${line} ${word}`
        : word;

    if (
      ctx.measureText(next).width >
        maxWidth &&
      line
    ) {
      ctx.fillText(
        line,
        x,
        cursorY
      );

      line = word;
      cursorY += lineHeight;
    } else {
      line = next;
    }
  });

  if (line) {
    ctx.fillText(
      line,
      x,
      cursorY
    );
  }

  return cursorY;
}

function drawContractTexture({
  texture,
  team,
  gmName,
  playerUniverse,
  injuriesEnabled,
  contractDate,
  executed = false,
}) {
  const ctx =
    texture.getContext();

  const width =
    CONTRACT_TEXTURE.width;

  const height =
    CONTRACT_TEXTURE.height;

  const teamCode =
    team?.code || "NHL";

  const teamName =
    team?.name ||
    "National Hockey League Club";

  ctx.clearRect(
    0,
    0,
    width,
    height
  );

  ctx.fillStyle =
    "#f1ebdc";

  ctx.fillRect(
    0,
    0,
    width,
    height
  );

  const paperGradient =
    ctx.createLinearGradient(
      0,
      0,
      width,
      height
    );

  paperGradient.addColorStop(
    0,
    "rgba(255,255,255,.20)"
  );

  paperGradient.addColorStop(
    0.55,
    "rgba(160,130,90,.025)"
  );

  paperGradient.addColorStop(
    1,
    "rgba(60,40,25,.055)"
  );

  ctx.fillStyle =
    paperGradient;

  ctx.fillRect(
    0,
    0,
    width,
    height
  );

  ctx.strokeStyle =
    "#48443b";

  ctx.lineWidth = 3;

  ctx.strokeRect(
    44,
    44,
    width - 88,
    height - 88
  );

  ctx.strokeStyle =
    "#a8936d";

  ctx.lineWidth = 1;

  ctx.strokeRect(
    58,
    58,
    width - 116,
    height - 116
  );

  ctx.fillStyle =
    "#222426";

  ctx.textAlign =
    "center";

  ctx.font =
    "800 25px Inter, Arial";

  ctx.fillText(
    "NATIONAL HOCKEY LEAGUE",
    width / 2,
    122
  );

  ctx.font =
    "900 50px Inter, Arial";

  ctx.fillText(
    "GENERAL MANAGER",
    width / 2,
    198
  );

  ctx.fillText(
    "APPOINTMENT",
    width / 2,
    255
  );

  ctx.strokeStyle =
    "#8e713e";

  ctx.lineWidth = 4;

  ctx.beginPath();
  ctx.moveTo(
    145,
    292
  );
  ctx.lineTo(
    width - 145,
    292
  );
  ctx.stroke();

  ctx.font =
    "900 38px Inter, Arial";

  ctx.fillStyle =
    "#25272a";

  ctx.fillText(
    teamName.toUpperCase(),
    width / 2,
    370,
    930
  );

  ctx.font =
    "700 19px Inter, Arial";

  ctx.fillStyle =
    "#6b5f49";

  ctx.fillText(
    `${teamCode} / HOCKEY OPERATIONS`,
    width / 2,
    410,
    930
  );

  ctx.textAlign =
    "left";

  ctx.fillStyle =
    "#6b5f49";

  ctx.font =
    "800 18px Inter, Arial";

  ctx.fillText(
    "APPOINTED GENERAL MANAGER",
    112,
    490
  );

  ctx.fillStyle =
    "#25272a";

  ctx.font =
    "900 34px Inter, Arial";

  ctx.fillText(
    (
      gmName?.trim() ||
      "GENERAL MANAGER"
    ).toUpperCase(),
    112,
    535,
    930
  );

  ctx.strokeStyle =
    "#b5a687";

  ctx.lineWidth = 2;

  ctx.beginPath();
  ctx.moveTo(
    112,
    565
  );
  ctx.lineTo(
    1088,
    565
  );
  ctx.stroke();

  ctx.fillStyle =
    "#6b5f49";

  ctx.font =
    "800 18px Inter, Arial";

  ctx.fillText(
    "PLAYER UNIVERSE",
    112,
    650
  );

  ctx.fillText(
    "LEAGUE HEALTH SYSTEM",
    650,
    650
  );

  ctx.fillText(
    "APPOINTMENT TERM",
    112,
    765
  );

  ctx.fillText(
    "EFFECTIVE DATE",
    650,
    765
  );

  ctx.fillStyle =
    "#25272a";

  ctx.font =
    "900 27px Inter, Arial";

  ctx.fillText(
    playerUniverse === "real_nhl"
      ? "REAL NHL PLAYERS"
      : "GENERATED PLAYERS",
    112,
    691,
    460
  );

  ctx.fillText(
    injuriesEnabled
      ? "ENABLED"
      : "DISABLED",
    650,
    691,
    420
  );

  ctx.fillText(
    "YEAR ONE",
    112,
    806,
    460
  );

  ctx.fillText(
    contractDate,
    650,
    806,
    420
  );

  ctx.strokeStyle =
    "#c1b49a";

  ctx.lineWidth = 1;

  ctx.beginPath();
  ctx.moveTo(
    112,
    860
  );
  ctx.lineTo(
    1088,
    860
  );
  ctx.stroke();

  ctx.fillStyle =
    "#343638";

  ctx.font =
    "500 20px Inter, Arial";

  drawWrappedText(
    ctx,
    "By executing this agreement, the General Manager accepts authority over hockey operations, transactions, contracts, staff, roster construction, scouting, and franchise strategy subject to the rules of Franchise Mode.",
    112,
    920,
    976,
    36
  );

  ctx.fillStyle =
    "#6b5f49";

  ctx.font =
    "800 18px Inter, Arial";

  ctx.fillText(
    "GENERAL MANAGER SIGNATURE",
    SIGNATURE_ZONE.x,
    1215
  );

  ctx.strokeStyle =
    "#4e473b";

  ctx.lineWidth = 2;

  ctx.beginPath();
  ctx.moveTo(
    SIGNATURE_ZONE.x,
    SIGNATURE_ZONE.y +
      SIGNATURE_ZONE.height
  );

  ctx.lineTo(
    SIGNATURE_ZONE.x +
      SIGNATURE_ZONE.width,
    SIGNATURE_ZONE.y +
      SIGNATURE_ZONE.height
  );

  ctx.stroke();

  ctx.fillStyle =
    "#8b7b60";

  ctx.font =
    "700 15px Inter, Arial";

  ctx.fillText(
    "SIGN ABOVE THIS LINE",
    SIGNATURE_ZONE.x,
    SIGNATURE_ZONE.y +
      SIGNATURE_ZONE.height +
      33
  );

  if (executed) {
    const stampX = 980;
    const stampY = 1370;

    ctx.save();

    ctx.translate(
      stampX,
      stampY
    );

    ctx.rotate(-0.13);

    ctx.strokeStyle =
      "rgba(118,25,19,.82)";

    ctx.fillStyle =
      "rgba(118,25,19,.82)";

    ctx.lineWidth = 9;

    ctx.beginPath();
    ctx.arc(
      0,
      0,
      82,
      0,
      Math.PI * 2
    );
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(
      0,
      0,
      63,
      0,
      Math.PI * 2
    );
    ctx.stroke();

    ctx.textAlign =
      "center";

    ctx.font =
      "900 24px Inter, Arial";

    ctx.fillText(
      teamCode,
      0,
      -6
    );

    ctx.font =
      "900 16px Inter, Arial";

    ctx.fillText(
      "EXECUTED",
      0,
      27
    );

    ctx.restore();
  }

  texture.update(true);
}


/* ============================================================================
   TEAM BRANDING PLANES
   ========================================================================== */

function createTeamLogoPlane({
  scene,
  team,
  position,
  width = 1.55,
  height = 1.55,
}) {
  const plane =
    MeshBuilder.CreatePlane(
      "team-branding-logo",
      {
        width,
        height,
        sideOrientation:
          Mesh.DOUBLESIDE,
      },
      scene
    );

  plane.position.copyFrom(
    position
  );

  plane.rotation.set(
    0,
    0,
    0
  );

  plane.isPickable = false;

  const material =
    new StandardMaterial(
      "team-branding-logo-material",
      scene
    );

  material.backFaceCulling = false;
  material.diffuseColor =
    Color3.White();

  material.specularColor =
    new Color3(
      0.02,
      0.02,
      0.02
    );

  if (team?.logo) {
    /*
      The prior version used invertY=false and then compensated by rotating
      the whole plane. That is exactly the sort of orientation stacking that
      caused upside-down logos.

      Here the image is loaded with invertY=true and the plane itself is never
      given a 180-degree Z rotation.
    */
    const logo =
      new Texture(
        team.logo,
        scene,
        false,
        true
      );

    logo.hasAlpha = true;

    material.diffuseTexture =
      logo;

    material.opacityTexture =
      logo;

    material.useAlphaFromDiffuseTexture =
      true;

    material.emissiveTexture =
      logo;

    material.emissiveColor =
      new Color3(
        0.10,
        0.10,
        0.10
      );
  }

  plane.material =
    material;

  return plane;
}


/* ============================================================================
   PARTICLES
   ========================================================================== */

function makeRadialParticleTexture(
  scene,
  name
) {
  const texture =
    new DynamicTexture(
      name,
      {
        width: 64,
        height: 64,
      },
      scene,
      false
    );

  const ctx =
    texture.getContext();

  const gradient =
    ctx.createRadialGradient(
      32,
      32,
      0,
      32,
      32,
      32
    );

  gradient.addColorStop(
    0,
    "rgba(255,255,255,1)"
  );

  gradient.addColorStop(
    0.18,
    "rgba(255,255,255,.72)"
  );

  gradient.addColorStop(
    0.52,
    "rgba(255,255,255,.20)"
  );

  gradient.addColorStop(
    1,
    "rgba(255,255,255,0)"
  );

  ctx.fillStyle =
    gradient;

  ctx.fillRect(
    0,
    0,
    64,
    64
  );

  texture.hasAlpha = true;
  texture.update(false);

  return texture;
}

function createDustParticles(
  scene,
  accentPrimary
) {
  const texture =
    makeRadialParticleTexture(
      scene,
      "office-dust-particle"
    );

  const dust =
    new ParticleSystem(
      "office-dust",
      420,
      scene
    );

  dust.particleTexture =
    texture;

  dust.emitter =
    new Vector3(
      0,
      1.65,
      2.7
    );

  dust.minEmitBox =
    new Vector3(
      -4.0,
      -1.5,
      -3.5
    );

  dust.maxEmitBox =
    new Vector3(
      4.0,
      1.6,
      3.8
    );

  dust.color1 =
    new Color4(
      0.82,
      0.78,
      0.70,
      0.18
    );

  const accent =
    colorFromHex(
      accentPrimary,
      "#c9a86a"
    );

  dust.color2 =
    new Color4(
      accent.r,
      accent.g,
      accent.b,
      0.10
    );

  dust.colorDead =
    new Color4(
      0.1,
      0.1,
      0.1,
      0
    );

  dust.minSize = 0.006;
  dust.maxSize = 0.022;

  dust.minLifeTime = 5;
  dust.maxLifeTime = 11;

  dust.emitRate = 34;

  dust.minEmitPower = 0.004;
  dust.maxEmitPower = 0.025;

  dust.direction1 =
    new Vector3(
      -0.03,
      0.025,
      -0.015
    );

  dust.direction2 =
    new Vector3(
      0.03,
      0.075,
      0.025
    );

  dust.gravity =
    new Vector3(
      0,
      0.001,
      0
    );

  dust.updateSpeed = 0.012;
  dust.blendMode =
    ParticleSystem.BLENDMODE_STANDARD;

  dust.start();

  const deskMotes =
    new ParticleSystem(
      "desk-light-motes",
      120,
      scene
    );

  deskMotes.particleTexture =
    texture;

  deskMotes.emitter =
    new Vector3(
      -0.7,
      1.2,
      3.0
    );

  deskMotes.minEmitBox =
    new Vector3(
      -1.0,
      -0.5,
      -0.7
    );

  deskMotes.maxEmitBox =
    new Vector3(
      1.0,
      0.85,
      0.7
    );

  deskMotes.color1 =
    new Color4(
      1.0,
      0.74,
      0.43,
      0.18
    );

  deskMotes.color2 =
    new Color4(
      accent.r,
      accent.g,
      accent.b,
      0.09
    );

  deskMotes.colorDead =
    new Color4(
      0,
      0,
      0,
      0
    );

  deskMotes.minSize = 0.004;
  deskMotes.maxSize = 0.012;

  deskMotes.minLifeTime = 2.5;
  deskMotes.maxLifeTime = 5.5;

  deskMotes.emitRate = 10;

  deskMotes.direction1 =
    new Vector3(
      -0.01,
      0.04,
      -0.01
    );

  deskMotes.direction2 =
    new Vector3(
      0.01,
      0.09,
      0.01
    );

  deskMotes.minEmitPower = 0.005;
  deskMotes.maxEmitPower = 0.018;

  deskMotes.start();

  return {
    dust,
    deskMotes,
  };
}


/* ============================================================================
   LIGHTING / RENDERING
   ========================================================================== */

/* ============================================================================
   HALLWAY MATERIAL LIBRARY
   ==========================================================================

   Every corridor surface is generated at runtime from a 2D canvas. That keeps
   the first frame almost free, and because the same painter feeds both the
   albedo and the bump map the surfaces actually respond to the practical
   lights instead of reading as flat primitives.

   Materials are created once and shared through the cache below, so a hundred
   props still only cost a handful of GPU materials.
*/

const HALL_GOLD = "#c9a86a";

function hashRandom(seed) {
  const value = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
  return value - Math.floor(value);
}

function paintGrain(ctx, width, height, alpha, count = 5200, size = 2) {
  ctx.save();
  ctx.globalAlpha = alpha;
  for (let i = 0; i < count; i += 1) {
    const tone = Math.floor(hashRandom(i * 1.7) * 255);
    ctx.fillStyle = `rgb(${tone},${tone},${tone})`;
    ctx.fillRect(
      hashRandom(i * 3.1) * width,
      hashRandom(i * 5.9) * height,
      size,
      size
    );
  }
  ctx.restore();
}

function paintVignette(ctx, width, height, strength = 0.4) {
  const gradient = ctx.createRadialGradient(
    width / 2,
    height / 2,
    Math.min(width, height) * 0.18,
    width / 2,
    height / 2,
    Math.max(width, height) * 0.72
  );
  gradient.addColorStop(0, "rgba(0,0,0,0)");
  gradient.addColorStop(1, `rgba(0,0,0,${strength})`);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
}

function paintPlaster(ctx, width, height) {
  ctx.fillStyle = "#22242a";
  ctx.fillRect(0, 0, width, height);

  for (let i = 0; i < 220; i += 1) {
    const x = hashRandom(i * 2.3) * width;
    const y = hashRandom(i * 4.7) * height;
    const r = 24 + hashRandom(i * 7.1) * 130;
    const light = hashRandom(i * 9.3) > 0.5;
    const blob = ctx.createRadialGradient(x, y, 0, x, y, r);
    blob.addColorStop(
      0,
      light ? "rgba(255,252,244,0.045)" : "rgba(0,0,0,0.07)"
    );
    blob.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = blob;
    ctx.fillRect(x - r, y - r, r * 2, r * 2);
  }

  // trowel direction
  ctx.globalAlpha = 0.05;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1;
  for (let i = 0; i < 90; i += 1) {
    const y = hashRandom(i * 11.1) * height;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.bezierCurveTo(
      width * 0.3,
      y + (hashRandom(i * 2.9) - 0.5) * 26,
      width * 0.7,
      y + (hashRandom(i * 6.3) - 0.5) * 26,
      width,
      y
    );
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  paintGrain(ctx, width, height, 0.05);
}

function paintWalnut(ctx, width, height) {
  ctx.fillStyle = "#2b1d15";
  ctx.fillRect(0, 0, width, height);

  const bands = 5;
  for (let b = 0; b < bands; b += 1) {
    const y0 = (b / bands) * height;
    const bandHeight = height / bands;
    const shade = 0.86 + hashRandom(b * 3.7) * 0.3;
    const gradient = ctx.createLinearGradient(0, y0, 0, y0 + bandHeight);
    gradient.addColorStop(0, `rgba(64,40,25,${0.55 * shade})`);
    gradient.addColorStop(0.5, `rgba(46,29,19,${0.72 * shade})`);
    gradient.addColorStop(1, `rgba(28,17,11,${0.68 * shade})`);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, y0, width, bandHeight);

    // grain lines follow the band
    for (let g = 0; g < 46; g += 1) {
      const y = y0 + hashRandom(b * 31 + g * 1.9) * bandHeight;
      const dark = hashRandom(b * 17 + g * 5.3) > 0.35;
      ctx.strokeStyle = dark
        ? "rgba(22,12,7,0.42)"
        : "rgba(139,98,58,0.20)";
      ctx.lineWidth = 0.6 + hashRandom(g * 7.7) * 1.9;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.bezierCurveTo(
        width * 0.28,
        y + (hashRandom(g * 2.1) - 0.5) * 20,
        width * 0.66,
        y + (hashRandom(g * 4.3) - 0.5) * 20,
        width,
        y + (hashRandom(g * 8.9) - 0.5) * 8
      );
      ctx.stroke();
    }

    // seam between boards
    ctx.strokeStyle = "rgba(8,4,2,0.75)";
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(0, y0);
    ctx.lineTo(width, y0);
    ctx.stroke();
  }

  // ray flecks
  ctx.globalAlpha = 0.16;
  for (let i = 0; i < 320; i += 1) {
    ctx.fillStyle = hashRandom(i * 1.3) > 0.5 ? "#8a5f37" : "#160b05";
    ctx.fillRect(
      hashRandom(i * 3.3) * width,
      hashRandom(i * 6.7) * height,
      2 + hashRandom(i * 9.1) * 12,
      1
    );
  }
  ctx.globalAlpha = 1;

  paintGrain(ctx, width, height, 0.04);
}

function paintExecutiveRunner(ctx, width, height) {
  ctx.fillStyle = "#4d1522";
  ctx.fillRect(0, 0, width, height);

  // deep pile weave
  for (let y = 0; y < height; y += 3) {
    const tone = 0.5 + hashRandom(y * 2.7) * 0.5;
    ctx.fillStyle = `rgba(${Math.floor(24 * tone)},${Math.floor(
      6 * tone
    )},${Math.floor(12 * tone)},0.34)`;
    ctx.fillRect(0, y, width, 1.6);
  }
  for (let x = 0; x < width; x += 3) {
    ctx.fillStyle =
      hashRandom(x * 5.1) > 0.5
        ? "rgba(122,32,49,0.20)"
        : "rgba(38,8,15,0.24)";
    ctx.fillRect(x, 0, 1.4, height);
  }

  // woven border bands
  const border = width * 0.055;
  ctx.fillStyle = "rgba(24,6,11,0.6)";
  ctx.fillRect(0, 0, border, height);
  ctx.fillRect(width - border, 0, border, height);
  ctx.fillStyle = "rgba(168,132,74,0.24)";
  ctx.fillRect(border * 0.86, 0, border * 0.16, height);
  ctx.fillRect(width - border * 1.02, 0, border * 0.16, height);

  // worn centre track from decades of the same walk
  const wear = ctx.createLinearGradient(width * 0.3, 0, width * 0.7, 0);
  wear.addColorStop(0, "rgba(0,0,0,0)");
  wear.addColorStop(0.5, "rgba(196,168,140,0.11)");
  wear.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = wear;
  ctx.fillRect(0, 0, width, height);

  // scuffed edges
  ctx.globalAlpha = 0.3;
  for (let i = 0; i < 260; i += 1) {
    const edge = hashRandom(i * 1.9) > 0.5;
    ctx.fillStyle = "rgba(20,5,9,0.8)";
    ctx.fillRect(
      edge ? hashRandom(i * 3.7) * border : width - hashRandom(i * 4.1) * border,
      hashRandom(i * 8.3) * height,
      3,
      2 + hashRandom(i * 2.2) * 9
    );
  }
  ctx.globalAlpha = 1;

  paintGrain(ctx, width, height, 0.06);
}

function paintDarkStoneFloor(ctx, width, height) {
  ctx.fillStyle = "#15161a";
  ctx.fillRect(0, 0, width, height);

  for (let i = 0; i < 60; i += 1) {
    ctx.strokeStyle = `rgba(150,150,158,${0.03 + hashRandom(i) * 0.05})`;
    ctx.lineWidth = 0.7 + hashRandom(i * 3.3) * 1.6;
    ctx.beginPath();
    const y = hashRandom(i * 5.1) * height;
    ctx.moveTo(0, y);
    ctx.bezierCurveTo(
      width * 0.25,
      y + (hashRandom(i * 2.1) - 0.5) * 180,
      width * 0.7,
      y + (hashRandom(i * 7.7) - 0.5) * 180,
      width,
      y + (hashRandom(i * 9.9) - 0.5) * 90
    );
    ctx.stroke();
  }

  // stone joints
  ctx.strokeStyle = "rgba(0,0,0,0.5)";
  ctx.lineWidth = 3;
  for (let i = 1; i < 4; i += 1) {
    ctx.beginPath();
    ctx.moveTo((i / 4) * width, 0);
    ctx.lineTo((i / 4) * width, height);
    ctx.stroke();
  }

  paintGrain(ctx, width, height, 0.05);
}

function paintBrushedBrass(ctx, width, height) {
  const base = ctx.createLinearGradient(0, 0, 0, height);
  base.addColorStop(0, "#8a6d3a");
  base.addColorStop(0.36, "#d5b276");
  base.addColorStop(0.58, "#a98a51");
  base.addColorStop(1, "#6d5228");
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, width, height);

  ctx.globalAlpha = 0.22;
  for (let i = 0; i < 900; i += 1) {
    ctx.fillStyle = hashRandom(i * 1.1) > 0.5 ? "#ffe6b8" : "#4a3818";
    ctx.fillRect(
      hashRandom(i * 3.9) * width,
      hashRandom(i * 6.1) * height,
      10 + hashRandom(i * 2.7) * 70,
      1
    );
  }
  ctx.globalAlpha = 1;
}

function paintPolishedSilver(ctx, width, height) {
  const base = ctx.createLinearGradient(0, 0, 0, height);
  base.addColorStop(0, "#3c4049");
  base.addColorStop(0.22, "#e6e9ef");
  base.addColorStop(0.4, "#9aa0ab");
  base.addColorStop(0.62, "#f2f4f8");
  base.addColorStop(0.8, "#6d737e");
  base.addColorStop(1, "#c8ccd4");
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, width, height);

  // faint hand marks — this thing gets touched constantly
  ctx.globalAlpha = 0.09;
  for (let i = 0; i < 40; i += 1) {
    const x = hashRandom(i * 2.9) * width;
    const y = hashRandom(i * 4.3) * height;
    const r = 12 + hashRandom(i * 6.7) * 34;
    const smudge = ctx.createRadialGradient(x, y, 0, x, y, r);
    smudge.addColorStop(0, "rgba(120,120,130,0.9)");
    smudge.addColorStop(1, "rgba(120,120,130,0)");
    ctx.fillStyle = smudge;
    ctx.fillRect(x - r, y - r, r * 2, r * 2);
  }
  ctx.globalAlpha = 1;
}

function paintCorridorEnvironment(ctx, width, height) {
  /*
    Cheap spherical environment used only as a reflection source for brass and
    the championship trophy. Dark room, warm ceiling pools, one cool spill.
  */
  const base = ctx.createLinearGradient(0, 0, 0, height);
  base.addColorStop(0, "#1b1a19");
  base.addColorStop(0.44, "#3a332a");
  base.addColorStop(0.52, "#171719");
  base.addColorStop(1, "#07070a");
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, width, height);

  const pools = [
    [0.12, 0.2, "rgba(255,214,150,0.85)"],
    [0.34, 0.16, "rgba(255,204,138,0.7)"],
    [0.58, 0.22, "rgba(255,226,176,0.8)"],
    [0.79, 0.17, "rgba(255,198,128,0.65)"],
    [0.46, 0.62, "rgba(150,176,214,0.32)"],
  ];

  pools.forEach(([u, v, colour], index) => {
    const x = u * width;
    const y = v * height;
    const r = (0.05 + hashRandom(index * 3.1) * 0.07) * width;
    const glow = ctx.createRadialGradient(x, y, 0, x, y, r);
    glow.addColorStop(0, colour);
    glow.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = glow;
    ctx.fillRect(x - r, y - r, r * 2, r * 2);
  });

  // horizon band so metal picks up a believable break
  ctx.fillStyle = "rgba(0,0,0,0.42)";
  ctx.fillRect(0, height * 0.5, width, 3);
}

function makeHallTexture(scene, name, width, height, paint, options = {}) {
  const texture = new DynamicTexture(
    name,
    { width, height },
    scene,
    options.mip !== false
  );
  const ctx = texture.getContext();
  ctx.clearRect(0, 0, width, height);
  paint(ctx, width, height);
  texture.update(false);

  const wrap =
    options.clamp === true ? Texture.CLAMP_ADDRESSMODE : Texture.WRAP_ADDRESSMODE;
  texture.wrapU = wrap;
  texture.wrapV = wrap;
  texture.uScale = options.uScale ?? 1;
  texture.vScale = options.vScale ?? 1;
  texture.hasAlpha = Boolean(options.hasAlpha);
  texture.anisotropicFilteringLevel = options.anisotropy ?? 4;
  return texture;
}

/*
  Convert an albedo canvas into a matching height field so a single painter can
  drive both colour and relief.
*/
function makeBumpFromPaint(scene, name, width, height, paint, options = {}) {
  return makeHallTexture(
    scene,
    name,
    width,
    height,
    (ctx, w, h) => {
      paint(ctx, w, h);
      ctx.globalCompositeOperation = "saturation";
      ctx.fillStyle = "#808080";
      ctx.fillRect(0, 0, w, h);
      ctx.globalCompositeOperation = "source-over";
    },
    options
  );
}

function createHallMaterials(scene) {
  const materials = new Map();
  const textures = new Map();

  function texture(key, factory) {
    if (!textures.has(key)) {
      textures.set(key, factory());
    }
    return textures.get(key);
  }

  const environment = texture("environment", () =>
    makeHallTexture(scene, "hall-env", 1024, 512, paintCorridorEnvironment, {
      clamp: true,
      anisotropy: 2,
    })
  );
  environment.coordinatesMode = Texture.SPHERICAL_MODE;

  function pbr(key, configure) {
    if (materials.has(key)) {
      return materials.get(key);
    }
    const material = new PBRMaterial(`hall-${key}`, scene);
    material.environmentTexture = environment;
    material.environmentIntensity = 0.32;
    material.directIntensity = 1.35;
    material.specularIntensity = 0.9;
    material.maxSimultaneousLights = 8;
    material.metallic = 0;
    material.roughness = 0.72;
    configure(material);
    materials.set(key, material);
    return material;
  }

  function flat(key, hex, options = {}) {
    return pbr(key, (material) => {
      material.albedoColor = colorFromHex(hex, "#888888");
      material.metallic = options.metallic ?? 0;
      material.roughness = options.roughness ?? 0.7;
      if (options.emissive) {
        material.emissiveColor = colorFromHex(options.emissive, "#000000");
        material.emissiveIntensity = options.emissiveIntensity ?? 1;
      }
      if (options.alpha != null) {
        material.alpha = options.alpha;
      }
      if (options.environmentIntensity != null) {
        material.environmentIntensity = options.environmentIntensity;
      }
    });
  }

  const api = {
    environment,

    plaster: () =>
      pbr("plaster", (material) => {
        material.albedoTexture = texture("plaster", () =>
          makeHallTexture(scene, "tex-plaster", 1024, 1024, paintPlaster, {
            uScale: 5,
            vScale: 1.6,
          })
        );
        material.bumpTexture = texture("plaster-bump", () =>
          makeBumpFromPaint(
            scene,
            "tex-plaster-bump",
            512,
            512,
            paintPlaster,
            { uScale: 5, vScale: 1.6 }
          )
        );
        material.bumpTexture.level = 0.35;
        material.roughness = 0.88;
        material.environmentIntensity = 0.12;
      }),

    walnut: () =>
      pbr("walnut", (material) => {
        material.albedoTexture = texture("walnut", () =>
          makeHallTexture(scene, "tex-walnut", 1024, 1024, paintWalnut, {
            uScale: 3.2,
            vScale: 1,
          })
        );
        material.bumpTexture = texture("walnut-bump", () =>
          makeBumpFromPaint(scene, "tex-walnut-bump", 512, 512, paintWalnut, {
            uScale: 3.2,
            vScale: 1,
          })
        );
        material.bumpTexture.level = 0.42;
        material.roughness = 0.44;
        material.metallic = 0.04;
        material.environmentIntensity = 0.4;
      }),

    walnutDark: () =>
      pbr("walnut-dark", (material) => {
        material.albedoTexture = texture("walnut", () =>
          makeHallTexture(scene, "tex-walnut", 1024, 1024, paintWalnut, {
            uScale: 3.2,
            vScale: 1,
          })
        );
        material.albedoColor = new Color3(0.62, 0.6, 0.58);
        material.roughness = 0.34;
        material.metallic = 0.06;
        material.environmentIntensity = 0.46;
      }),

    runner: () =>
      pbr("runner", (material) => {
        material.albedoTexture = texture("runner", () =>
          makeHallTexture(
            scene,
            "tex-runner",
            512,
            1024,
            paintExecutiveRunner,
            { uScale: 1, vScale: 9 }
          )
        );
        material.bumpTexture = texture("runner-bump", () =>
          makeBumpFromPaint(
            scene,
            "tex-runner-bump",
            256,
            512,
            paintExecutiveRunner,
            { uScale: 1, vScale: 9 }
          )
        );
        material.bumpTexture.level = 0.9;
        material.roughness = 0.97;
        material.environmentIntensity = 0.04;
      }),

    stone: () =>
      pbr("stone", (material) => {
        material.albedoTexture = texture("stone", () =>
          makeHallTexture(
            scene,
            "tex-stone",
            1024,
            1024,
            paintDarkStoneFloor,
            { uScale: 2, vScale: 8 }
          )
        );
        material.roughness = 0.3;
        material.metallic = 0.1;
        material.environmentIntensity = 0.34;
      }),

    brass: () =>
      pbr("brass", (material) => {
        material.albedoTexture = texture("brass", () =>
          makeHallTexture(scene, "tex-brass", 512, 512, paintBrushedBrass)
        );
        material.metallic = 0.92;
        material.roughness = 0.28;
        material.environmentIntensity = 0.95;
      }),

    silver: () =>
      pbr("silver", (material) => {
        material.albedoTexture = texture("silver", () =>
          makeHallTexture(scene, "tex-silver", 512, 512, paintPolishedSilver)
        );
        material.metallic = 1;
        material.roughness = 0.14;
        material.environmentIntensity = 1.5;
      }),

    steel: () => flat("steel", "#8f939a", { metallic: 0.85, roughness: 0.36 }),
    blackMetal: () =>
      flat("black-metal", "#191b1f", { metallic: 0.6, roughness: 0.45 }),
    charcoal: () => flat("charcoal", "#20222a", { roughness: 0.82 }),
    rubber: () => flat("rubber", "#0d0e10", { roughness: 0.94 }),
    leather: () => flat("leather", "#31241c", { roughness: 0.6 }),
    tape: () => flat("tape", "#1c1c1e", { roughness: 0.86 }),
    whiteTape: () => flat("white-tape", "#d8d3c6", { roughness: 0.88 }),
    paper: () => flat("paper", "#cfc7b4", { roughness: 0.9 }),
    matBoard: () => flat("mat-board", "#0e0f12", { roughness: 0.9 }),
    canvasCream: () => flat("canvas-cream", "#b9b2a0", { roughness: 0.85 }),

    glass: () =>
      pbr("glass", (material) => {
        material.albedoColor = new Color3(0.02, 0.024, 0.03);
        material.alpha = 0.16;
        material.metallic = 0.24;
        material.roughness = 0.05;
        material.environmentIntensity = 1.35;
        material.backFaceCulling = false;
      }),

    acrylic: () =>
      pbr("acrylic", (material) => {
        material.albedoColor = new Color3(0.05, 0.055, 0.06);
        material.alpha = 0.2;
        material.metallic = 0.1;
        material.roughness = 0.08;
        material.environmentIntensity = 1.1;
        material.backFaceCulling = false;
      }),

    warmLeak: () =>
      flat("warm-leak", "#2a1d0e", {
        emissive: "#ffbe6a",
        emissiveIntensity: 2.6,
        roughness: 0.6,
      }),

    lampLens: () =>
      flat("lamp-lens", "#3a3327", {
        emissive: "#ffd9a0",
        emissiveIntensity: 2.1,
        roughness: 0.4,
      }),

    coldLens: () =>
      flat("cold-lens", "#26292f", {
        emissive: "#cfe0ff",
        emissiveIntensity: 1.2,
        roughness: 0.4,
      }),

    /*
      Painted surfaces (photographs, posters, plaques, jerseys) each need their
      own texture, but they all share this factory so the PBR setup and the
      texture cache stay in one place.
    */
    art(key, width, height, paint, options = {}) {
      const cacheKey = `art:${key}`;
      if (materials.has(cacheKey)) {
        return materials.get(cacheKey);
      }
      const material = new PBRMaterial(`hall-art-${key}`, scene);
      material.environmentTexture = environment;
      material.environmentIntensity = options.environmentIntensity ?? 0.16;
      material.directIntensity = 1.5;
      material.metallic = options.metallic ?? 0;
      material.roughness = options.roughness ?? 0.78;
      material.maxSimultaneousLights = 8;
      material.albedoTexture = makeHallTexture(
        scene,
        `tex-art-${key}`,
        width,
        height,
        paint,
        { clamp: true, ...options.texture }
      );
      if (options.bump) {
        material.bumpTexture = makeBumpFromPaint(
          scene,
          `tex-art-${key}-bump`,
          Math.min(width, 512),
          Math.min(height, 512),
          paint,
          { clamp: true }
        );
        material.bumpTexture.level = options.bump;
      }
      if (options.emissive) {
        material.emissiveTexture = material.albedoTexture;
        material.emissiveIntensity = options.emissive;
      }
      materials.set(cacheKey, material);
      return material;
    },

    flat,
  };

  return api;
}

/* ============================================================================
   HALLWAY ART PAINTERS
   ==========================================================================

   Each memorabilia surface gets a real illustration rather than a coloured
   rectangle: woven cloth, layered twill numbers, stitching, newsprint columns,
   etched brass. Everything is drawn once at load and cached.
*/

function paintClothWeave(ctx, width, height, alpha = 0.14) {
  ctx.save();
  ctx.globalAlpha = alpha;
  for (let x = 0; x < width; x += 3) {
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.fillRect(x, 0, 1, height);
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(x + 1.5, 0, 1, height);
  }
  for (let y = 0; y < height; y += 3) {
    ctx.fillStyle = "rgba(0,0,0,0.4)";
    ctx.fillRect(0, y, width, 1);
  }
  ctx.restore();
  paintGrain(ctx, width, height, 0.035, 4200, 3);
}

function drawStitchRun(ctx, points, colour, dash = [7, 6], lineWidth = 2.4) {
  if (points.length < 2) return;
  ctx.save();
  ctx.strokeStyle = colour;
  ctx.lineWidth = lineWidth;
  ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i][0], points[i][1]);
  }
  ctx.stroke();
  ctx.restore();
}

/*
  Layered twill: an applique number is several stacked fabric layers, so it is
  drawn as progressively tighter outlines with stitching on the boundary.
*/
function drawTwill(ctx, text, x, y, size, layers, options = {}) {
  ctx.save();
  ctx.font = `900 ${size}px "Arial Black", "Haettenschweiler", Impact, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineJoin = "round";
  ctx.miterLimit = 2;

  layers.forEach(({ colour, width }) => {
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.strokeText(text, x, y);
  });

  ctx.fillStyle = options.fill || "#ffffff";
  ctx.fillText(text, x, y);

  if (options.stitch !== false) {
    ctx.strokeStyle = "rgba(0,0,0,0.32)";
    ctx.lineWidth = 1.6;
    ctx.setLineDash([5, 5]);
    ctx.strokeText(text, x, y);
    ctx.setLineDash([]);
  }

  // twill nap
  ctx.globalAlpha = 0.16;
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = 1;
  for (let i = -size; i < size; i += 4) {
    ctx.beginPath();
    ctx.moveTo(x + i, y - size * 0.62);
    ctx.lineTo(x + i + size * 0.4, y + size * 0.62);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  ctx.restore();
}

function drawJerseyLaces(ctx, x, y, width, height) {
  ctx.save();
  ctx.fillStyle = "rgba(18,14,10,0.5)";
  ctx.fillRect(x - width / 2, y, width, height);
  ctx.strokeStyle = "#d8cdb2";
  ctx.lineWidth = 6;
  ctx.lineCap = "round";
  for (let i = 0; i < 4; i += 1) {
    const t = y + (i / 4) * height + height * 0.08;
    ctx.beginPath();
    ctx.moveTo(x - width / 2, t);
    ctx.lineTo(x + width / 2, t + height * 0.14);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x + width / 2, t);
    ctx.lineTo(x - width / 2, t + height * 0.14);
    ctx.stroke();
  }
  ctx.restore();
}

function clipHangingJersey(ctx, width, height) {
  const cx = width / 2;
  const collarY = height * 0.1;
  const sleeveY = height * 0.265;
  const armpitY = height * 0.55;
  const hemY = height * 0.9;
  ctx.beginPath();
  ctx.moveTo(cx - width * 0.09, collarY);
  ctx.quadraticCurveTo(cx, height * 0.16, cx + width * 0.09, collarY);
  ctx.lineTo(cx + width * 0.185, collarY + height * 0.018);
  ctx.lineTo(cx + width * 0.445, sleeveY);
  ctx.lineTo(cx + width * 0.418, height * 0.6);
  ctx.lineTo(cx + width * 0.262, armpitY);
  ctx.lineTo(cx + width * 0.275, hemY);
  ctx.quadraticCurveTo(cx, hemY + height * 0.022, cx - width * 0.275, hemY);
  ctx.lineTo(cx - width * 0.262, armpitY);
  ctx.lineTo(cx - width * 0.418, height * 0.6);
  ctx.lineTo(cx - width * 0.445, sleeveY);
  ctx.lineTo(cx - width * 0.185, collarY + height * 0.018);
  ctx.closePath();
}

/*
  Ottawa Senators Reebok-era white away sweater, reverse, as photographed:
  KARLSSON nameplate, large 65 in red with a black outline, shoulder O crests,
  Reebok mark under the collar, NHL 100 patch on the right sleeve.
*/
function paintKarlssonJersey(ctx, width, height) {
  const red = "#c8102e";
  const black = "#111111";
  const white = "#f4f1ea";

  ctx.clearRect(0, 0, width, height);

  const cx = width / 2;
  const hemY = height * 0.9;

  clipHangingJersey(ctx, width, height);
  ctx.save();
  ctx.clip();

  ctx.fillStyle = white;
  ctx.fillRect(0, 0, width, height);

  // sleeve: black cuff, then a thick red band
  ctx.fillStyle = black;
  ctx.fillRect(0, height * 0.5, width * 0.2, height * 0.14);
  ctx.fillRect(width * 0.8, height * 0.5, width * 0.2, height * 0.14);
  ctx.fillStyle = red;
  ctx.fillRect(0, height * 0.4, width * 0.2, height * 0.11);
  ctx.fillRect(width * 0.8, height * 0.4, width * 0.2, height * 0.11);

  // hem: black / red / black
  [
    [height * 0.78, 22, black],
    [height * 0.808, 28, red],
    [height * 0.848, 22, black],
  ].forEach(([y, thickness, colour]) => {
    ctx.fillStyle = colour;
    ctx.fillRect(0, y, width, thickness);
  });

  // Reebok mark, centered just below the collar
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(cx - width * 0.055, height * 0.168, width * 0.11, height * 0.028);
  ctx.strokeStyle = "rgba(0,0,0,0.28)";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(cx - width * 0.055, height * 0.168, width * 0.11, height * 0.028);
  ctx.fillStyle = black;
  ctx.font = `800 ${Math.round(height * 0.016)}px "Arial Black", sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("Reebok", cx, height * 0.183);

  // nameplate
  ctx.fillStyle = white;
  ctx.fillRect(cx - width * 0.24, height * 0.215, width * 0.48, height * 0.08);
  drawTwill(
    ctx,
    "KARLSSON",
    cx,
    height * 0.258,
    Math.round(height * 0.052),
    [{ colour: black, width: 6 }],
    { fill: black }
  );

  // back number
  drawTwill(
    ctx,
    "65",
    cx,
    height * 0.52,
    Math.round(height * 0.3),
    [{ colour: black, width: 28 }],
    { fill: red }
  );

  // shoulder O crests — red disc, black O
  [-1, 1].forEach((side) => {
    const sx = cx + side * width * 0.3;
    const sy = height * 0.195;
    ctx.fillStyle = red;
    ctx.beginPath();
    ctx.arc(sx, sy, width * 0.05, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = black;
    ctx.lineWidth = 4;
    ctx.stroke();
    ctx.fillStyle = black;
    ctx.font = `900 ${Math.round(width * 0.048)}px "Arial Black", sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("O", sx, sy + 1);
  });

  // sleeve numbers
  [-1, 1].forEach((side) => {
    drawTwill(
      ctx,
      "65",
      cx + side * width * 0.345,
      height * 0.355,
      Math.round(width * 0.072),
      [{ colour: black, width: 10 }],
      { fill: red }
    );
  });

  // NHL centennial patch on the right sleeve
  const patchX = cx + width * 0.345;
  const patchY = height * 0.455;
  ctx.fillStyle = "#c5cdd6";
  ctx.beginPath();
  ctx.arc(patchX, patchY, width * 0.032, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#1d4e89";
  ctx.lineWidth = 4;
  ctx.stroke();
  ctx.fillStyle = "#1d4e89";
  ctx.font = `900 ${Math.round(width * 0.028)}px "Arial Black", sans-serif`;
  ctx.fillText("100", patchX, patchY + 1);

  paintClothWeave(ctx, width, height, 0.12);

  drawStitchRun(
    ctx,
    [
      [cx - width * 0.185, height * 0.13],
      [cx - width * 0.26, height * 0.55],
      [cx - width * 0.275, hemY],
    ],
    "rgba(0,0,0,0.28)"
  );
  drawStitchRun(
    ctx,
    [
      [cx + width * 0.185, height * 0.13],
      [cx + width * 0.26, height * 0.55],
      [cx + width * 0.275, hemY],
    ],
    "rgba(0,0,0,0.28)"
  );

  const fold = ctx.createLinearGradient(0, 0, width, 0);
  fold.addColorStop(0, "rgba(0,0,0,0.22)");
  fold.addColorStop(0.22, "rgba(0,0,0,0.04)");
  fold.addColorStop(0.5, "rgba(255,255,255,0.08)");
  fold.addColorStop(0.78, "rgba(0,0,0,0.05)");
  fold.addColorStop(1, "rgba(0,0,0,0.24)");
  ctx.fillStyle = fold;
  ctx.fillRect(0, 0, width, height);

  ctx.restore();

  // silver autograph across the 65
  ctx.save();
  ctx.translate(cx + width * 0.02, height * 0.58);
  ctx.rotate(-0.08);
  ctx.strokeStyle = "rgba(210, 216, 224, 0.92)";
  ctx.lineWidth = 6;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(-width * 0.16, 0);
  ctx.bezierCurveTo(
    -width * 0.1,
    -width * 0.05,
    -width * 0.04,
    width * 0.04,
    width * 0.02,
    -width * 0.01
  );
  ctx.bezierCurveTo(
    width * 0.08,
    -width * 0.05,
    width * 0.12,
    width * 0.03,
    width * 0.18,
    -width * 0.005
  );
  ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.fillStyle = "#16161a";
  ctx.fillRect(cx + width * 0.11, hemY - height * 0.012, width * 0.09, height * 0.04);
  ctx.fillStyle = "#0b0b0d";
  ctx.fillRect(cx + width * 0.125, hemY + height * 0.012, width * 0.06, height * 0.016);
  ctx.restore();
}

/*
  Washington Capitals rookie-era sweater, reverse. Black screaming-eagle Koho
  cut: copper name and number, KOHO under the collar, V-stripes on the sleeves.
*/
function paintOvechkinJerseyBack(ctx, width, height) {
  const body = "#0c0c0e";
  const copper = "#c17a28";
  const blue = "#1a3f8f";
  const cream = "#efeae0";

  ctx.clearRect(0, 0, width, height);

  const cx = width / 2;
  const hemY = height * 0.9;

  clipHangingJersey(ctx, width, height);
  ctx.save();
  ctx.clip();

  const bodyFill = ctx.createLinearGradient(0, 0, 0, height);
  bodyFill.addColorStop(0, "#16161a");
  bodyFill.addColorStop(0.5, body);
  bodyFill.addColorStop(1, "#08080a");
  ctx.fillStyle = bodyFill;
  ctx.fillRect(0, 0, width, height);

  // V-shaped sleeve banding (blue / white / copper / white / blue)
  const sleeveBands = [
    [0.0, blue],
    [0.22, cream],
    [0.36, copper],
    [0.62, cream],
    [0.74, blue],
  ];
  [-1, 1].forEach((side) => {
    const innerX = cx + side * width * 0.28;
    const outerX = side < 0 ? 0 : width;
    sleeveBands.forEach(([t, colour], index) => {
      const next = sleeveBands[index + 1] ? sleeveBands[index + 1][0] : 1;
      const y0 = height * 0.42 + t * height * 0.16;
      const y1 = height * 0.42 + next * height * 0.16;
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.moveTo(innerX, y0);
      ctx.lineTo(outerX, y0 + height * 0.04);
      ctx.lineTo(outerX, y1 + height * 0.04);
      ctx.lineTo(innerX, y1);
      ctx.closePath();
      ctx.fill();
    });
  });

  // hem banding to match the sleeves
  [
    [height * 0.785, 16, blue],
    [height * 0.805, 8, cream],
    [height * 0.816, 18, copper],
    [height * 0.838, 8, cream],
    [height * 0.85, 18, blue],
  ].forEach(([y, thickness, colour]) => {
    ctx.fillStyle = colour;
    ctx.fillRect(0, y, width, thickness);
  });

  // KOHO sits under the collar on this cut, not on the hem
  ctx.save();
  ctx.font = `800 ${Math.round(height * 0.022)}px "Arial Black", sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = cream;
  ctx.letterSpacing = "3px";
  ctx.fillText("KOHO", cx, height * 0.175);
  ctx.restore();

  // arched nameplate
  const name = "OVECHKIN";
  const nameSize = Math.round(height * 0.046);
  const nameY = height * 0.245;
  const spread = 0.22;
  [...name].forEach((letter, i) => {
    const t = name.length <= 1 ? 0.5 : i / (name.length - 1);
    const angle = -spread / 2 + t * spread;
    ctx.save();
    ctx.translate(cx + Math.sin(angle) * width * 0.42, nameY + (1 - Math.cos(angle)) * height * 0.08);
    ctx.rotate(angle);
    drawTwill(
      ctx,
      letter,
      0,
      0,
      nameSize,
      [{ colour: cream, width: 8 }],
      { fill: copper }
    );
    ctx.restore();
  });

  // back number — copper fill, white then black outline
  drawTwill(
    ctx,
    "8",
    cx,
    height * 0.5,
    Math.round(height * 0.32),
    [
      { colour: "#0a0a0c", width: 32 },
      { colour: cream, width: 16 },
    ],
    { fill: copper }
  );

  // sleeve numbers sit above the V
  [-1, 1].forEach((side) => {
    drawTwill(
      ctx,
      "8",
      cx + side * width * 0.345,
      height * 0.355,
      Math.round(width * 0.08),
      [
        { colour: "#0a0a0c", width: 10 },
        { colour: cream, width: 5 },
      ],
      { fill: copper }
    );
  });

  // NHL shield on the lower right hem
  ctx.save();
  ctx.fillStyle = "#d4c4a0";
  ctx.beginPath();
  ctx.moveTo(cx + width * 0.16, height * 0.868);
  ctx.lineTo(cx + width * 0.2, height * 0.868);
  ctx.lineTo(cx + width * 0.195, height * 0.895);
  ctx.quadraticCurveTo(cx + width * 0.18, height * 0.905, cx + width * 0.165, height * 0.895);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  paintClothWeave(ctx, width, height, 0.14);

  drawStitchRun(
    ctx,
    [
      [cx - width * 0.185, height * 0.13],
      [cx - width * 0.26, height * 0.55],
      [cx - width * 0.275, hemY],
    ],
    "rgba(180,150,100,0.28)"
  );
  drawStitchRun(
    ctx,
    [
      [cx + width * 0.185, height * 0.13],
      [cx + width * 0.26, height * 0.55],
      [cx + width * 0.275, hemY],
    ],
    "rgba(180,150,100,0.28)"
  );

  const fold = ctx.createLinearGradient(0, 0, width, 0);
  fold.addColorStop(0, "rgba(0,0,0,0.38)");
  fold.addColorStop(0.18, "rgba(255,255,255,0.04)");
  fold.addColorStop(0.5, "rgba(255,255,255,0.07)");
  fold.addColorStop(0.82, "rgba(255,255,255,0.03)");
  fold.addColorStop(1, "rgba(0,0,0,0.4)");
  ctx.fillStyle = fold;
  ctx.fillRect(0, 0, width, height);

  ctx.restore();
}

function paintEtchedBrassPlate(ctx, width, height, lines) {
  paintBrushedBrass(ctx, width, height);

  ctx.fillStyle = "rgba(0,0,0,0.16)";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(60,44,18,0.6)";
  ctx.lineWidth = 3;
  ctx.strokeRect(9, 9, width - 18, height - 18);

  const total = lines.length;
  lines.forEach((line, index) => {
    const text = typeof line === "string" ? line : line.text;
    const scale = typeof line === "string" ? 1 : line.scale || 1;
    const size = Math.round((height / (total + 1.1)) * 0.62 * scale);
    const y = height * ((index + 0.85) / (total + 0.7));
    ctx.save();
    ctx.font = `${scale > 1 ? 900 : 700} ${size}px "Georgia", "Times New Roman", serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.letterSpacing = `${Math.max(1, size * 0.08)}px`;
    // engraved: dark cut with a light lip below it
    ctx.fillStyle = "rgba(255,240,205,0.34)";
    ctx.fillText(text, width / 2, y + 2);
    ctx.fillStyle = "rgba(38,26,8,0.92)";
    ctx.fillText(text, width / 2, y);
    ctx.restore();
  });
}

function paintVintageHockeyPhoto(ctx, width, height, seed, caption) {
  const warm = hashRandom(seed) > 0.45;

  // paper border
  ctx.fillStyle = warm ? "#cabfa4" : "#c3bdb0";
  ctx.fillRect(0, 0, width, height);

  const bx = width * 0.055;
  const by = height * 0.05;
  const bw = width - bx * 2;
  const bh = height * (caption ? 0.79 : 0.9);

  // ice
  const ice = ctx.createLinearGradient(0, by, 0, by + bh);
  ice.addColorStop(0, warm ? "#7d6a4e" : "#6f6f72");
  ice.addColorStop(0.42, warm ? "#c6b48d" : "#b9bcc0");
  ctx.fillStyle = ice;
  ctx.fillRect(bx, by, bw, bh);

  // boards + crowd
  ctx.fillStyle = warm ? "#4a3d28" : "#3d3f45";
  ctx.fillRect(bx, by, bw, bh * 0.3);
  for (let i = 0; i < 420; i += 1) {
    const tone = 40 + hashRandom(seed * 13 + i) * 90;
    ctx.fillStyle = `rgba(${tone},${tone * 0.93},${tone * 0.82},0.6)`;
    ctx.fillRect(
      bx + hashRandom(seed * 7 + i * 1.7) * bw,
      by + hashRandom(seed * 3 + i * 2.3) * bh * 0.27,
      3,
      3
    );
  }
  ctx.fillStyle = warm ? "#ddd2b4" : "#d6d8dc";
  ctx.fillRect(bx, by + bh * 0.3, bw, bh * 0.02);

  // rink markings
  ctx.strokeStyle = "rgba(90,70,70,0.36)";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(bx, by + bh * 0.66);
  ctx.lineTo(bx + bw, by + bh * 0.62);
  ctx.stroke();

  // skaters
  const count = 3 + Math.floor(hashRandom(seed * 5) * 3);
  for (let i = 0; i < count; i += 1) {
    const px = bx + bw * (0.14 + hashRandom(seed * 11 + i * 3.1) * 0.7);
    const py = by + bh * (0.44 + hashRandom(seed * 17 + i * 5.3) * 0.34);
    const scale = bh * (0.2 + hashRandom(seed * 19 + i) * 0.13);
    const dark = hashRandom(seed * 23 + i) > 0.5;
    ctx.fillStyle = dark ? "rgba(28,22,16,0.9)" : "rgba(236,230,214,0.9)";
    // torso
    ctx.beginPath();
    ctx.ellipse(px, py, scale * 0.2, scale * 0.3, hashRandom(i) * 0.4 - 0.2, 0, Math.PI * 2);
    ctx.fill();
    // head
    ctx.beginPath();
    ctx.arc(px + scale * 0.05, py - scale * 0.38, scale * 0.11, 0, Math.PI * 2);
    ctx.fill();
    // legs
    ctx.strokeStyle = ctx.fillStyle;
    ctx.lineWidth = scale * 0.09;
    ctx.beginPath();
    ctx.moveTo(px, py + scale * 0.24);
    ctx.lineTo(px - scale * 0.16, py + scale * 0.58);
    ctx.moveTo(px, py + scale * 0.24);
    ctx.lineTo(px + scale * 0.2, py + scale * 0.52);
    ctx.stroke();
    // stick
    ctx.lineWidth = scale * 0.05;
    ctx.beginPath();
    ctx.moveTo(px + scale * 0.16, py - scale * 0.02);
    ctx.lineTo(px + scale * 0.6, py + scale * 0.5);
    ctx.stroke();
  }

  // period grain, sepia wash and vignette
  paintGrain(ctx, width, height, 0.09, 3400, 2);
  ctx.fillStyle = warm ? "rgba(120,86,40,0.22)" : "rgba(60,66,80,0.18)";
  ctx.fillRect(bx, by, bw, bh);
  paintVignette(ctx, width, height, 0.34);

  if (caption) {
    ctx.fillStyle = "rgba(38,30,20,0.78)";
    ctx.font = `italic 600 ${Math.round(height * 0.048)}px "Georgia", serif`;
    ctx.textAlign = "center";
    ctx.fillText(caption, width / 2, height * 0.93);
  }
}

function paintChampionshipFrontPage(ctx, width, height) {
  ctx.fillStyle = "#d9d2bd";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "rgba(150,128,88,0.16)";
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = "#22201b";
  ctx.font = `900 ${Math.round(height * 0.045)}px "Georgia", serif`;
  ctx.textAlign = "center";
  ctx.fillText("THE MORNING LEDGER", width / 2, height * 0.07);
  ctx.strokeStyle = "#22201b";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(width * 0.05, height * 0.09);
  ctx.lineTo(width * 0.95, height * 0.09);
  ctx.stroke();

  ctx.font = `900 ${Math.round(height * 0.135)}px "Arial Black", Impact, sans-serif`;
  ctx.fillText("CHAMPIONS", width / 2, height * 0.235);
  ctx.font = `700 ${Math.round(height * 0.036)}px "Georgia", serif`;
  ctx.fillText(
    "City spills into the streets as the Cup finally comes home",
    width / 2,
    height * 0.29
  );

  // hero photo block
  const px = width * 0.08;
  const py = height * 0.33;
  const pw = width * 0.84;
  const ph = height * 0.34;
  const photo = ctx.createLinearGradient(0, py, 0, py + ph);
  photo.addColorStop(0, "#5c5346");
  photo.addColorStop(1, "#221f1b");
  ctx.fillStyle = photo;
  ctx.fillRect(px, py, pw, ph);
  // raised cup silhouette
  ctx.fillStyle = "rgba(240,236,226,0.9)";
  ctx.beginPath();
  ctx.moveTo(width / 2 - pw * 0.06, py + ph * 0.18);
  ctx.lineTo(width / 2 + pw * 0.06, py + ph * 0.18);
  ctx.lineTo(width / 2 + pw * 0.035, py + ph * 0.42);
  ctx.lineTo(width / 2 + pw * 0.018, py + ph * 0.78);
  ctx.lineTo(width / 2 - pw * 0.018, py + ph * 0.78);
  ctx.lineTo(width / 2 - pw * 0.035, py + ph * 0.42);
  ctx.closePath();
  ctx.fill();
  for (let i = 0; i < 300; i += 1) {
    ctx.fillStyle = `rgba(255,255,255,${0.05 + hashRandom(i) * 0.14})`;
    ctx.fillRect(
      px + hashRandom(i * 3.1) * pw,
      py + ph * 0.5 + hashRandom(i * 5.7) * ph * 0.5,
      2,
      2
    );
  }
  paintGrain(ctx, width, height, 0.06, 2600, 2);

  // body columns
  ctx.fillStyle = "rgba(34,32,27,0.62)";
  const cols = 3;
  for (let c = 0; c < cols; c += 1) {
    const colX = width * 0.08 + c * (width * 0.84 / cols) + 6;
    const colW = width * 0.84 / cols - 16;
    for (let line = 0; line < 22; line += 1) {
      const y = height * 0.71 + line * (height * 0.011);
      const w = colW * (0.72 + hashRandom(c * 31 + line) * 0.28);
      ctx.fillRect(colX, y, w, 2.4);
    }
  }
  paintVignette(ctx, width, height, 0.2);
}

function paintCredentialBoard(ctx, width, height) {
  // dark mat with stubs and laminates pinned to it
  ctx.fillStyle = "#0d0e11";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "rgba(255,255,255,0.03)";
  ctx.fillRect(0, 0, width, height);

  const stubs = [
    [0.05, 0.06, 0.4, 0.15, "#cdbfa0", "STANLEY CUP FINAL"],
    [0.53, 0.05, 0.42, 0.14, "#c2ccd6", "SEMI-FINAL · GAME 7"],
    [0.06, 0.26, 0.42, 0.15, "#d3c3a2", "ROUND 2 · GAME 1"],
    [0.55, 0.24, 0.39, 0.16, "#bcae92", "OPENING NIGHT"],
  ];

  stubs.forEach(([x, y, w, h, tone, label], index) => {
    const px = x * width;
    const py = y * height;
    const pw = w * width;
    const ph = h * height;
    ctx.save();
    ctx.translate(px + pw / 2, py + ph / 2);
    ctx.rotate((hashRandom(index * 4.7) - 0.5) * 0.12);
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(-pw / 2 + 4, -ph / 2 + 5, pw, ph);
    ctx.fillStyle = tone;
    ctx.fillRect(-pw / 2, -ph / 2, pw, ph);
    // perforation
    ctx.fillStyle = "rgba(0,0,0,0.28)";
    for (let d = 0; d < 24; d += 1) {
      ctx.fillRect(-pw / 2 + pw * 0.72, -ph / 2 + (d / 24) * ph, 2, ph / 40);
    }
    ctx.fillStyle = "rgba(30,24,16,0.86)";
    ctx.font = `800 ${Math.round(ph * 0.19)}px "Georgia", serif`;
    ctx.textAlign = "left";
    ctx.fillText(label, -pw / 2 + pw * 0.06, -ph / 2 + ph * 0.32);
    ctx.font = `600 ${Math.round(ph * 0.14)}px "Courier New", monospace`;
    ctx.fillText("SEC 12 · ROW C · SEAT 4", -pw / 2 + pw * 0.06, -ph / 2 + ph * 0.58);
    ctx.fillText("ADMIT ONE", -pw / 2 + pw * 0.06, -ph / 2 + ph * 0.82);
    ctx.restore();
  });

  // hanging laminate credentials
  const creds = [
    [0.12, 0.52, "#8f2230", "MEDIA"],
    [0.4, 0.55, "#1d3a6b", "ALL ACCESS"],
    [0.68, 0.52, "#2c4a2a", "ICE LEVEL"],
  ];
  creds.forEach(([x, y, tone, label], index) => {
    const px = x * width;
    const py = y * height;
    const pw = width * 0.2;
    const ph = height * 0.33;
    ctx.save();
    ctx.translate(px + pw / 2, py);
    ctx.rotate((hashRandom(index * 9.1) - 0.5) * 0.14);
    // lanyard
    ctx.strokeStyle = "rgba(210,200,180,0.5)";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(-pw * 0.2, -height * 0.1);
    ctx.lineTo(0, 0);
    ctx.lineTo(pw * 0.2, -height * 0.1);
    ctx.stroke();
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fillRect(-pw / 2 + 4, 5, pw, ph);
    ctx.fillStyle = "#e6e0d0";
    ctx.fillRect(-pw / 2, 0, pw, ph);
    ctx.fillStyle = tone;
    ctx.fillRect(-pw / 2, 0, pw, ph * 0.3);
    ctx.fillStyle = "#f2ede0";
    ctx.font = `900 ${Math.round(ph * 0.13)}px "Arial Black", sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText(label, 0, ph * 0.2);
    ctx.fillStyle = "rgba(30,26,20,0.6)";
    ctx.fillRect(-pw * 0.34, ph * 0.4, pw * 0.68, ph * 0.34);
    ctx.font = `700 ${Math.round(ph * 0.09)}px "Courier New", monospace`;
    ctx.fillStyle = "rgba(40,34,26,0.8)";
    ctx.fillText("PLAYOFFS", 0, ph * 0.88);
    ctx.restore();
  });

  paintGrain(ctx, width, height, 0.05);
  paintVignette(ctx, width, height, 0.36);
}

function paintRinkDiagram(ctx, width, height) {
  ctx.fillStyle = "#e8e4d8";
  ctx.fillRect(0, 0, width, height);

  const m = width * 0.06;
  const w = width - m * 2;
  const h = height - m * 2;
  const r = h * 0.24;

  ctx.strokeStyle = "#2a2a2e";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(m + r, m);
  ctx.lineTo(m + w - r, m);
  ctx.quadraticCurveTo(m + w, m, m + w, m + r);
  ctx.lineTo(m + w, m + h - r);
  ctx.quadraticCurveTo(m + w, m + h, m + w - r, m + h);
  ctx.lineTo(m + r, m + h);
  ctx.quadraticCurveTo(m, m + h, m, m + h - r);
  ctx.lineTo(m, m + r);
  ctx.quadraticCurveTo(m, m, m + r, m);
  ctx.stroke();

  // blue lines and centre
  ctx.strokeStyle = "#2a49a8";
  ctx.lineWidth = 7;
  [0.34, 0.66].forEach((t) => {
    ctx.beginPath();
    ctx.moveTo(m + w * t, m);
    ctx.lineTo(m + w * t, m + h);
    ctx.stroke();
  });
  ctx.strokeStyle = "#b32036";
  ctx.lineWidth = 7;
  ctx.beginPath();
  ctx.moveTo(m + w * 0.5, m);
  ctx.lineTo(m + w * 0.5, m + h);
  ctx.stroke();
  ctx.lineWidth = 4;
  [0.09, 0.91].forEach((t) => {
    ctx.beginPath();
    ctx.moveTo(m + w * t, m);
    ctx.lineTo(m + w * t, m + h);
    ctx.stroke();
  });

  // circles
  ctx.strokeStyle = "#b32036";
  ctx.lineWidth = 3;
  [
    [0.2, 0.25],
    [0.2, 0.75],
    [0.8, 0.25],
    [0.8, 0.75],
    [0.5, 0.5],
  ].forEach(([u, v]) => {
    ctx.beginPath();
    ctx.arc(m + w * u, m + h * v, h * 0.11, 0, Math.PI * 2);
    ctx.stroke();
  });

  // handwritten coaching markings in grease pencil
  ctx.strokeStyle = "rgba(24,26,32,0.82)";
  ctx.lineWidth = 5;
  ctx.lineCap = "round";
  const arrows = [
    [0.16, 0.78, 0.32, 0.56],
    [0.32, 0.56, 0.5, 0.68],
    [0.5, 0.68, 0.72, 0.36],
    [0.2, 0.28, 0.42, 0.34],
  ];
  arrows.forEach(([x0, y0, x1, y1]) => {
    const ax = m + w * x0;
    const ay = m + h * y0;
    const bx = m + w * x1;
    const by = m + h * y1;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.quadraticCurveTo((ax + bx) / 2, ay - h * 0.1, bx, by);
    ctx.stroke();
    const angle = Math.atan2(by - ay + h * 0.05, bx - ax);
    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.lineTo(bx - Math.cos(angle - 0.5) * 22, by - Math.sin(angle - 0.5) * 22);
    ctx.moveTo(bx, by);
    ctx.lineTo(bx - Math.cos(angle + 0.5) * 22, by - Math.sin(angle + 0.5) * 22);
    ctx.stroke();
  });

  ctx.font = `700 ${Math.round(height * 0.07)}px "Bradley Hand", "Segoe Script", cursive`;
  ctx.fillStyle = "rgba(150,26,38,0.85)";
  ctx.fillText("D TO D — QUICK", m + w * 0.12, m + h * 0.16);
  ctx.fillText("NO", m + w * 0.62, m + h * 0.84);
  ctx.beginPath();
  ctx.arc(m + w * 0.645, m + h * 0.81, h * 0.07, 0, Math.PI * 2);
  ctx.stroke();

  paintGrain(ctx, width, height, 0.05);
}

function paintWhiteboardBreakout(ctx, width, height) {
  ctx.fillStyle = "#f2f2ef";
  ctx.fillRect(0, 0, width, height);
  // ghosting from years of erasing
  ctx.globalAlpha = 0.08;
  for (let i = 0; i < 90; i += 1) {
    ctx.fillStyle = hashRandom(i) > 0.5 ? "#2b3a6b" : "#8c1e2a";
    ctx.fillRect(
      hashRandom(i * 2.3) * width,
      hashRandom(i * 4.9) * height,
      40 + hashRandom(i * 7.1) * 200,
      6 + hashRandom(i * 3.3) * 22
    );
  }
  ctx.globalAlpha = 1;

  ctx.strokeStyle = "#1f2b52";
  ctx.lineWidth = 5;
  ctx.strokeRect(width * 0.05, height * 0.12, width * 0.9, height * 0.74);

  ctx.font = `900 ${Math.round(height * 0.085)}px "Arial Black", sans-serif`;
  ctx.fillStyle = "#1f2b52";
  ctx.textAlign = "left";
  ctx.fillText("BREAKOUT — OPTION 7B", width * 0.06, height * 0.09);

  const markers = ["#1f2b52", "#8c1e2a", "#1b6b3a", "#6b3a8c"];
  ctx.lineCap = "round";
  for (let i = 0; i < 16; i += 1) {
    ctx.strokeStyle = markers[i % markers.length];
    ctx.lineWidth = 4 + hashRandom(i * 5.1) * 3;
    const x0 = width * (0.1 + hashRandom(i * 2.1) * 0.75);
    const y0 = height * (0.2 + hashRandom(i * 3.7) * 0.6);
    const x1 = width * (0.1 + hashRandom(i * 6.3) * 0.78);
    const y1 = height * (0.2 + hashRandom(i * 8.9) * 0.6);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.bezierCurveTo(
      x0 + (hashRandom(i * 11.3) - 0.5) * width * 0.3,
      y0 + (hashRandom(i * 13.7) - 0.5) * height * 0.4,
      x1 + (hashRandom(i * 17.1) - 0.5) * width * 0.3,
      y1 + (hashRandom(i * 19.3) - 0.5) * height * 0.4,
      x1,
      y1
    );
    ctx.stroke();
    const angle = Math.atan2(y1 - y0, x1 - x0);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - Math.cos(angle - 0.6) * 18, y1 - Math.sin(angle - 0.6) * 18);
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - Math.cos(angle + 0.6) * 18, y1 - Math.sin(angle + 0.6) * 18);
    ctx.stroke();
  }

  ["X", "O", "X", "O", "X", "O", "?", "?"].forEach((glyph, i) => {
    ctx.fillStyle = glyph === "?" ? "#8c1e2a" : markers[i % markers.length];
    ctx.font = `900 ${Math.round(height * 0.11)}px "Arial Black", sans-serif`;
    ctx.fillText(
      glyph,
      width * (0.12 + hashRandom(i * 23.1) * 0.74),
      height * (0.24 + hashRandom(i * 29.7) * 0.58)
    );
  });

  ctx.strokeStyle = "#8c1e2a";
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.arc(width * 0.72, height * 0.66, height * 0.11, 0, Math.PI * 2);
  ctx.stroke();
  ctx.font = `900 ${Math.round(height * 0.075)}px "Arial Black", sans-serif`;
  ctx.fillStyle = "#8c1e2a";
  ctx.textAlign = "center";
  ctx.fillText("NO", width * 0.72, height * 0.69);

  ctx.textAlign = "left";
  ctx.font = `800 ${Math.round(height * 0.05)}px "Segoe Script", cursive`;
  ctx.fillStyle = "#1b6b3a";
  ctx.fillText("IF THIS FAILS SEE OPTION 7C", width * 0.08, height * 0.94);
}

function paintTkachukProtestPoster(ctx, width, height) {
  ctx.fillStyle = "#0b0b0d";
  ctx.fillRect(0, 0, width, height);

  // cheap screen-printed stock
  const wash = ctx.createLinearGradient(0, 0, width, height);
  wash.addColorStop(0, "rgba(178,25,57,0.42)");
  wash.addColorStop(1, "rgba(0,0,0,0.2)");
  ctx.fillStyle = wash;
  ctx.fillRect(0, 0, width, height);

  ctx.save();
  ctx.translate(width / 2, height / 2);
  ctx.rotate(-0.02);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  ctx.font = `900 ${Math.round(height * 0.22)}px "Arial Black", Impact, sans-serif`;
  ctx.lineWidth = 12;
  ctx.strokeStyle = "rgba(0,0,0,0.75)";
  ctx.strokeText("F*CK", 0, -height * 0.24);
  ctx.fillStyle = "#f4f1e8";
  ctx.fillText("F*CK", 0, -height * 0.24);

  ctx.font = `900 ${Math.round(height * 0.17)}px "Arial Black", Impact, sans-serif`;
  ctx.strokeText("BRADY", 0, -height * 0.02);
  ctx.fillStyle = "#f4f1e8";
  ctx.fillText("BRADY", 0, -height * 0.02);

  ctx.font = `900 ${Math.round(height * 0.19)}px "Arial Black", Impact, sans-serif`;
  ctx.lineWidth = 14;
  ctx.strokeText("TKACHUK", 0, height * 0.18);
  ctx.fillStyle = "#ff2d4b";
  ctx.fillText("TKACHUK", 0, height * 0.18);

  ctx.font = `800 ${Math.round(height * 0.055)}px "Arial Black", sans-serif`;
  ctx.fillStyle = "rgba(244,241,232,0.88)";
  ctx.letterSpacing = "4px";
  ctx.fillText("NO. 7", 0, height * 0.3);

  ctx.font = `800 ${Math.round(height * 0.038)}px "Arial", sans-serif`;
  ctx.fillStyle = "rgba(244,241,232,0.6)";
  ctx.letterSpacing = "5px";
  ctx.fillText("MANAGEMENT REGRETS NOTHING", 0, height * 0.42);
  ctx.restore();

  // ink flaws and tape
  paintGrain(ctx, width, height, 0.1, 4200, 3);
  ctx.fillStyle = "rgba(226,220,198,0.34)";
  [
    [0.02, 0.02, 0.16, 0.05, -0.5],
    [0.82, 0.01, 0.16, 0.05, 0.5],
    [0.03, 0.93, 0.15, 0.05, 0.42],
    [0.83, 0.94, 0.15, 0.05, -0.4],
  ].forEach(([x, y, w, h, rot]) => {
    ctx.save();
    ctx.translate(width * (x + w / 2), height * (y + h / 2));
    ctx.rotate(rot);
    ctx.fillRect(-width * w * 0.5, -height * h * 0.5, width * w, height * h);
    ctx.restore();
  });
  paintVignette(ctx, width, height, 0.28);
}

function paintDartTargetPoster(ctx, width, height) {
  ctx.fillStyle = "#141414";
  ctx.fillRect(0, 0, width, height);

  // washed-out photo of a rival in full flight
  const shot = ctx.createLinearGradient(0, 0, 0, height);
  shot.addColorStop(0, "#3a4756");
  shot.addColorStop(1, "#12161c");
  ctx.fillStyle = shot;
  ctx.fillRect(width * 0.05, height * 0.05, width * 0.9, height * 0.9);

  ctx.fillStyle = "rgba(226,222,210,0.82)";
  const cx = width * 0.5;
  const cy = height * 0.48;
  // body
  ctx.beginPath();
  ctx.ellipse(cx, cy + height * 0.02, width * 0.13, height * 0.19, 0.08, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(cx + width * 0.02, cy - height * 0.24, width * 0.075, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(226,222,210,0.82)";
  ctx.lineWidth = width * 0.045;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(cx - width * 0.05, cy + height * 0.16);
  ctx.lineTo(cx - width * 0.16, cy + height * 0.36);
  ctx.moveTo(cx + width * 0.05, cy + height * 0.16);
  ctx.lineTo(cx + width * 0.15, cy + height * 0.34);
  ctx.stroke();

  // concentric rings
  const rings = [0.42, 0.32, 0.22, 0.12, 0.05];
  rings.forEach((r, index) => {
    ctx.strokeStyle =
      index % 2 === 0 ? "rgba(255,45,75,0.9)" : "rgba(244,241,232,0.85)";
    ctx.lineWidth = width * 0.014;
    ctx.beginPath();
    ctx.arc(cx, cy, width * r, 0, Math.PI * 2);
    ctx.stroke();
  });
  ctx.fillStyle = "rgba(255,45,75,0.95)";
  ctx.beginPath();
  ctx.arc(cx, cy, width * 0.025, 0, Math.PI * 2);
  ctx.fill();

  // crosshair
  ctx.strokeStyle = "rgba(244,241,232,0.5)";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(cx - width * 0.46, cy);
  ctx.lineTo(cx + width * 0.46, cy);
  ctx.moveTo(cx, cy - width * 0.46);
  ctx.lineTo(cx, cy + width * 0.46);
  ctx.stroke();

  ctx.font = `900 ${Math.round(height * 0.055)}px "Arial Black", sans-serif`;
  ctx.fillStyle = "#f4f1e8";
  ctx.textAlign = "center";
  ctx.fillText("PRACTICE TARGET", cx, height * 0.115);
  ctx.font = `700 ${Math.round(height * 0.035)}px "Arial", sans-serif`;
  ctx.fillStyle = "rgba(244,241,232,0.6)";
  ctx.fillText("PROVIDED BY HOCKEY OPERATIONS", cx, height * 0.94);

  // existing dart scars
  ctx.fillStyle = "rgba(0,0,0,0.7)";
  for (let i = 0; i < 26; i += 1) {
    ctx.beginPath();
    ctx.arc(
      cx + (hashRandom(i * 3.7) - 0.5) * width * 0.7,
      cy + (hashRandom(i * 5.1) - 0.5) * height * 0.6,
      1.6 + hashRandom(i * 7.3) * 2.2,
      0,
      Math.PI * 2
    );
    ctx.fill();
  }
  paintGrain(ctx, width, height, 0.07);
}

function paintPennant(ctx, width, height, label, tone) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = tone;
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "rgba(0,0,0,0.28)";
  ctx.fillRect(0, 0, width * 0.06, height);
  ctx.fillStyle = "rgba(240,234,218,0.9)";
  ctx.fillRect(width * 0.06, height * 0.1, width * 0.02, height * 0.8);

  ctx.save();
  ctx.translate(width * 0.55, height * 0.5);
  ctx.font = `900 ${Math.round(height * 0.3)}px "Arial Black", sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "rgba(240,234,218,0.95)";
  ctx.letterSpacing = "4px";
  ctx.fillText(label, 0, 0);
  ctx.restore();

  // felt fuzz
  paintGrain(ctx, width, height, 0.1, 3000, 3);
  paintVignette(ctx, width, height, 0.3);
}

function paintProgramCover(ctx, width, height) {
  ctx.fillStyle = "#a8261f";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "rgba(0,0,0,0.22)";
  ctx.fillRect(0, 0, width, height * 0.18);

  ctx.fillStyle = "#f0e8d4";
  ctx.font = `900 ${Math.round(height * 0.075)}px "Georgia", serif`;
  ctx.textAlign = "center";
  ctx.fillText("OFFICIAL PROGRAM", width / 2, height * 0.12);

  ctx.fillStyle = "rgba(20,16,12,0.55)";
  ctx.fillRect(width * 0.1, height * 0.24, width * 0.8, height * 0.44);
  ctx.fillStyle = "rgba(240,232,212,0.85)";
  ctx.beginPath();
  ctx.arc(width * 0.5, height * 0.44, width * 0.14, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#a8261f";
  ctx.font = `900 ${Math.round(height * 0.09)}px "Arial Black", sans-serif`;
  ctx.fillText("NHL", width * 0.5, height * 0.47);

  ctx.fillStyle = "#f0e8d4";
  ctx.font = `800 ${Math.round(height * 0.05)}px "Georgia", serif`;
  ctx.fillText("TONIGHT'S LINEUPS", width / 2, height * 0.76);
  ctx.font = `700 ${Math.round(height * 0.04)}px "Georgia", serif`;
  ctx.fillText("FIFTY CENTS", width / 2, height * 0.86);
  paintGrain(ctx, width, height, 0.09);
  paintVignette(ctx, width, height, 0.34);
}

function paintPinBoard(ctx, width, height) {
  ctx.fillStyle = "#111216";
  ctx.fillRect(0, 0, width, height);
  const tones = ["#b32036", "#1f3f8f", "#c9a86a", "#1b6b3a", "#e8e3d6", "#6b3a8c"];
  for (let row = 0; row < 4; row += 1) {
    for (let col = 0; col < 7; col += 1) {
      const cx = width * ((col + 0.7) / 7.4);
      const cy = height * ((row + 0.7) / 4.4);
      const r = Math.min(width / 7.4, height / 4.4) * 0.32;
      const tone = tones[(row * 7 + col) % tones.length];
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.beginPath();
      ctx.arc(cx + 2, cy + 3, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = tone;
      ctx.beginPath();
      if ((row + col) % 3 === 0) {
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
      } else if ((row + col) % 3 === 1) {
        ctx.rect(cx - r * 0.8, cy - r * 0.9, r * 1.6, r * 1.8);
      } else {
        ctx.moveTo(cx, cy - r);
        ctx.lineTo(cx + r, cy);
        ctx.lineTo(cx, cy + r);
        ctx.lineTo(cx - r, cy);
        ctx.closePath();
      }
      ctx.fill();
      ctx.strokeStyle = "rgba(255,240,210,0.5)";
      ctx.lineWidth = 1.6;
      ctx.stroke();
      // enamel highlight
      const gloss = ctx.createRadialGradient(
        cx - r * 0.3,
        cy - r * 0.4,
        0,
        cx - r * 0.3,
        cy - r * 0.4,
        r
      );
      gloss.addColorStop(0, "rgba(255,255,255,0.5)");
      gloss.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = gloss;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  paintVignette(ctx, width, height, 0.4);
}

function paintScratchedNote(ctx, width, height) {
  ctx.fillStyle = "#d8d0ba";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "rgba(150,130,90,0.12)";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(60,80,140,0.18)";
  ctx.lineWidth = 1.6;
  for (let y = height * 0.16; y < height * 0.95; y += height * 0.09) {
    ctx.beginPath();
    ctx.moveTo(width * 0.06, y);
    ctx.lineTo(width * 0.94, y);
    ctx.stroke();
  }

  ctx.fillStyle = "rgba(30,26,20,0.82)";
  ctx.font = `700 ${Math.round(height * 0.085)}px "Segoe Script", cursive`;
  ctx.textAlign = "left";
  ctx.fillText("CALL BACK LIST", width * 0.08, height * 0.14);
  ctx.font = `600 ${Math.round(height * 0.07)}px "Segoe Script", cursive`;
  ctx.fillText("1. winger, left side", width * 0.08, height * 0.25);
  ctx.fillText("2. 3rd pair D", width * 0.08, height * 0.34);
  ctx.fillText("3. goalie coach", width * 0.08, height * 0.43);

  // a rival crest scribbled into oblivion
  ctx.save();
  ctx.translate(width * 0.62, height * 0.68);
  ctx.fillStyle = "rgba(40,60,120,0.6)";
  ctx.beginPath();
  ctx.arc(0, 0, height * 0.16, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(20,18,14,0.92)";
  ctx.lineWidth = 5;
  ctx.lineCap = "round";
  for (let i = 0; i < 22; i += 1) {
    ctx.beginPath();
    ctx.moveTo(
      (hashRandom(i * 2.7) - 0.5) * height * 0.46,
      (hashRandom(i * 5.3) - 0.5) * height * 0.46
    );
    ctx.lineTo(
      (hashRandom(i * 7.9) - 0.5) * height * 0.46,
      (hashRandom(i * 9.1) - 0.5) * height * 0.46
    );
    ctx.stroke();
  }
  ctx.restore();

  ctx.fillStyle = "rgba(150,26,38,0.8)";
  ctx.font = `800 ${Math.round(height * 0.075)}px "Segoe Script", cursive`;
  ctx.fillText("NOT THEM", width * 0.1, height * 0.86);

  // coffee ring
  ctx.strokeStyle = "rgba(110,74,40,0.28)";
  ctx.lineWidth = 7;
  ctx.beginPath();
  ctx.arc(width * 0.22, height * 0.63, height * 0.14, 0, Math.PI * 2);
  ctx.stroke();
  paintGrain(ctx, width, height, 0.05);
}

function paintMiniBanner(ctx, width, height, top, bottom, tone) {
  ctx.fillStyle = tone;
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(201,168,106,0.85)";
  ctx.lineWidth = width * 0.045;
  ctx.strokeRect(width * 0.05, height * 0.04, width * 0.9, height * 0.92);

  ctx.textAlign = "center";
  ctx.fillStyle = "rgba(240,234,218,0.94)";
  ctx.font = `900 ${Math.round(height * 0.11)}px "Arial Black", sans-serif`;
  ctx.fillText(top, width / 2, height * 0.24);
  ctx.font = `900 ${Math.round(height * 0.2)}px "Arial Black", sans-serif`;
  ctx.fillText(bottom, width / 2, height * 0.52);
  ctx.font = `800 ${Math.round(height * 0.075)}px "Georgia", serif`;
  ctx.fillStyle = "rgba(201,168,106,0.9)";
  ctx.fillText("CHAMPIONS", width / 2, height * 0.72);

  // felt + hanging fringe
  paintGrain(ctx, width, height, 0.1, 2600, 3);
  ctx.fillStyle = "rgba(201,168,106,0.8)";
  for (let i = 0; i < 14; i += 1) {
    ctx.fillRect(
      width * (0.06 + (i / 14) * 0.88),
      height * 0.95,
      width * 0.02,
      height * 0.05
    );
  }
  paintVignette(ctx, width, height, 0.34);
}

/* ============================================================================
   HALLWAY GEOMETRY
   ==========================================================================

   Helpers first. Every art surface uses createFacePanel so the texture
   orientation, UVs and facing direction are explicit rather than inherited
   from a primitive's default winding.
*/

function createFacePanel(
  scene,
  name,
  { width, height, columns = 1, rows = 1, relief = 0, reliefShape }
) {
  const positions = [];
  const uvs = [];
  const indices = [];

  for (let r = 0; r <= rows; r += 1) {
    const v = r / rows;
    for (let c = 0; c <= columns; c += 1) {
      const u = c / columns;
      const z = relief ? relief * (reliefShape ? reliefShape(u, v) : 1) : 0;
      positions.push((u - 0.5) * width, (0.5 - v) * height, z);
      // DynamicTexture.update(false) keeps canvas Y unflipped, so v=0 is
      // the top of the painting. Mapping 1-v here hung every sweater upside down.
      uvs.push(u, v);
    }
  }

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < columns; c += 1) {
      const a = r * (columns + 1) + c;
      const b = a + 1;
      const d = a + columns + 1;
      const e = d + 1;
      indices.push(a, d, b, b, d, e);
    }
  }

  const normals = [];
  VertexData.ComputeNormals(positions, indices, normals);

  // The panel is authored to face +Z; keep the shading consistent with that.
  let facing = 0;
  for (let i = 2; i < normals.length; i += 3) {
    facing += normals[i];
  }
  if (facing < 0) {
    for (let i = 0; i < normals.length; i += 1) {
      normals[i] = -normals[i];
    }
  }

  const mesh = new Mesh(name, scene);
  const data = new VertexData();
  data.positions = positions;
  data.indices = indices;
  data.uvs = uvs;
  data.normals = normals;
  data.applyToMesh(mesh);
  mesh.isPickable = false;
  mesh.receiveShadows = true;
  return mesh;
}

function hangingJerseyRelief(u, v) {
  const torso = Math.pow(Math.sin(Math.PI * Math.max(0.04, Math.min(v, 0.96))), 0.7);
  const across = Math.max(0.18, 1 - Math.pow((u - 0.5) * 2.05, 2));
  return 0.28 + 0.72 * torso * across;
}

function place(mesh, parent, options = {}) {
  if (parent) {
    mesh.parent = parent;
  }
  const [x = 0, y = 0, z = 0] = options.at || [];
  mesh.position.set(x, y, z);
  if (options.turn) {
    const [rx = 0, ry = 0, rz = 0] = options.turn;
    mesh.rotation.set(rx, ry, rz);
  }
  if (options.material) {
    mesh.material = options.material;
  }
  mesh.checkCollisions = Boolean(options.collide);
  mesh.isPickable = Boolean(options.pickable);
  mesh.receiveShadows = options.receiveShadows !== false;
  return mesh;
}

function solid(scene, name, parent, options) {
  const mesh = MeshBuilder.CreateBox(
    name,
    {
      width: options.width ?? 0.1,
      height: options.height ?? 0.1,
      depth: options.depth ?? 0.1,
    },
    scene
  );
  return place(mesh, parent, options);
}

function rod(scene, name, parent, options) {
  const mesh = MeshBuilder.CreateCylinder(
    name,
    {
      height: options.height ?? 0.1,
      diameterTop: options.top ?? options.diameter ?? 0.02,
      diameterBottom: options.bottom ?? options.diameter ?? 0.02,
      tessellation: options.sides ?? 12,
    },
    scene
  );
  return place(mesh, parent, options);
}

function orb(scene, name, parent, options) {
  const mesh = MeshBuilder.CreateSphere(
    name,
    {
      diameter: options.diameter ?? 0.1,
      segments: options.segments ?? 12,
      slice: options.slice ?? 1,
    },
    scene
  );
  if (options.squash) {
    mesh.scaling.set(...options.squash);
  }
  return place(mesh, parent, options);
}

function artPanel(scene, name, parent, options) {
  const mesh = createFacePanel(scene, name, {
    width: options.width,
    height: options.height,
    columns: options.columns ?? 1,
    rows: options.rows ?? 1,
    relief: options.relief ?? 0,
    reliefShape: options.reliefShape,
  });
  return place(mesh, parent, options);
}

/*
  Anchor whose local +Z points away from the wall and into the corridor.
*/
function wallMount(scene, parent, side, z, y = 0) {
  const node = new TransformNode(`hall-mount${side}-${z.toFixed(2)}`, scene);
  node.parent = parent;
  node.position.set(side * (HALL.width / 2 - 0.004), y, z);
  node.rotation.y = side < 0 ? Math.PI / 2 : -Math.PI / 2;
  return node;
}

function contactShadowTexture(scene, cache) {
  if (!cache.contactShadow) {
    cache.contactShadow = makeHallTexture(
      scene,
      "hall-contact-shadow",
      128,
      128,
      (ctx, w, h) => {
        ctx.clearRect(0, 0, w, h);
        const gradient = ctx.createRadialGradient(
          w / 2,
          h / 2,
          0,
          w / 2,
          h / 2,
          w / 2
        );
        gradient.addColorStop(0, "rgba(0,0,0,0.82)");
        gradient.addColorStop(0.45, "rgba(0,0,0,0.5)");
        gradient.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, w, h);
      },
      { clamp: true, hasAlpha: true }
    );
  }
  return cache.contactShadow;
}

/*
  Soft grounding pool under floor props. Cheap, and it does more for the sense
  of physical weight than a shadow map at this corridor length would.
*/
function contactShadow(scene, cache, parent, { x = 0, z = 0, radius = 0.3, strength = 1 }) {
  if (!cache.contactShadowMaterial) {
    const material = new StandardMaterial("hall-contact-shadow-mat", scene);
    material.diffuseTexture = contactShadowTexture(scene, cache);
    material.diffuseTexture.hasAlpha = true;
    material.opacityTexture = material.diffuseTexture;
    material.diffuseColor = Color3.Black();
    material.specularColor = Color3.Black();
    material.emissiveColor = Color3.Black();
    material.disableLighting = true;
    material.backFaceCulling = false;
    material.alpha = 0.85;
    cache.contactShadowMaterial = material;
  }

  const mesh = MeshBuilder.CreateGround(
    `hall-shadow-${x.toFixed(2)}-${z.toFixed(2)}`,
    { width: radius * 2, height: radius * 2 },
    scene
  );
  mesh.parent = parent;
  mesh.position.set(x, 0.004, z);
  mesh.material = cache.contactShadowMaterial;
  mesh.isPickable = false;
  mesh.receiveShadows = false;
  mesh.visibility = clamp(strength, 0.1, 1);
  return mesh;
}

/*
  Museum-grade frame: mitred outer moulding, gilt fillet, recessed mat, art,
  and a glass sheet that catches the practical lights.
*/
function buildFramedPiece(
  scene,
  materials,
  parent,
  {
    name,
    width,
    height,
    art,
    depth = 0.055,
    moulding = 0.038,
    mat = 0.03,
    tilt = 0,
    glass = true,
    frameMaterial,
  }
) {
  const group = new TransformNode(`frame-${name}`, scene);
  group.parent = parent;
  group.rotation.z = tilt;

  const wood = frameMaterial || materials.walnutDark();
  const outerW = width + moulding * 2;
  const outerH = height + moulding * 2;

  // backing box gives the piece real thickness against the wall
  solid(scene, `frame-${name}-back`, group, {
    width: outerW,
    height: outerH,
    depth,
    at: [0, 0, depth / 2],
    material: materials.matBoard(),
  });

  // moulding rails
  const rails = [
    [outerW, moulding, 0, (outerH - moulding) / 2],
    [outerW, moulding, 0, -(outerH - moulding) / 2],
    [moulding, outerH - moulding * 2, (outerW - moulding) / 2, 0],
    [moulding, outerH - moulding * 2, -(outerW - moulding) / 2, 0],
  ];
  rails.forEach(([w, h, x, y], index) => {
    solid(scene, `frame-${name}-rail${index}`, group, {
      width: w,
      height: h,
      depth: depth + 0.012,
      at: [x, y, depth / 2 + 0.006],
      material: wood,
    });
  });

  // gilt fillet just inside the moulding
  const fillet = 0.006;
  [
    [width + fillet * 2, fillet, 0, (height + fillet) / 2],
    [width + fillet * 2, fillet, 0, -(height + fillet) / 2],
    [fillet, height, (width + fillet) / 2, 0],
    [fillet, height, -(width + fillet) / 2, 0],
  ].forEach(([w, h, x, y], index) => {
    solid(scene, `frame-${name}-fillet${index}`, group, {
      width: w,
      height: h,
      depth: 0.014,
      at: [x, y, depth + 0.004],
      material: materials.brass(),
    });
  });

  if (mat > 0) {
    artPanel(scene, `frame-${name}-mat`, group, {
      width,
      height,
      at: [0, 0, depth - 0.004],
      material: materials.matBoard(),
    });
  }

  const artMesh = artPanel(scene, `frame-${name}-art`, group, {
    width: width - mat * 2,
    height: height - mat * 2,
    at: [0, 0, depth - 0.002],
    material: art,
  });

  if (glass) {
    const pane = artPanel(scene, `frame-${name}-glass`, group, {
      width,
      height,
      at: [0, 0, depth + 0.002],
      material: materials.glass(),
    });
    pane.isPickable = false;
  }

  return { group, artMesh, outerW, outerH };
}

function buildBrassPlaque(scene, materials, parent, { name, width, height, lines, at }) {
  const plate = materials.art(
    `plaque-${name}`,
    Math.round(width * 900),
    Math.round(height * 900),
    (ctx, w, h) => paintEtchedBrassPlate(ctx, w, h, lines),
    { metallic: 0.7, roughness: 0.3, environmentIntensity: 0.7, bump: 0.3 }
  );

  const group = new TransformNode(`plaque-${name}`, scene);
  group.parent = parent;
  group.position.set(at[0], at[1], at[2]);

  solid(scene, `plaque-${name}-body`, group, {
    width: width + 0.012,
    height: height + 0.012,
    depth: 0.014,
    at: [0, 0, 0.007],
    material: materials.walnutDark(),
  });
  artPanel(scene, `plaque-${name}-face`, group, {
    width,
    height,
    at: [0, 0, 0.0155],
    material: plate,
  });
  // standoff screws
  [-1, 1].forEach((side) => {
    rod(scene, `plaque-${name}-screw${side}`, group, {
      diameter: 0.008,
      height: 0.006,
      sides: 8,
      at: [side * (width / 2 - 0.014), -height / 2 + 0.012, 0.018],
      turn: [Math.PI / 2, 0, 0],
      material: materials.brass(),
    });
  });

  return group;
}

/* --------------------------------------------------------------------------
   HERO EXHIBIT ONE — framed, authenticated, museum lit
   -------------------------------------------------------------------------- */

function buildKarlssonExhibit(scene, materials, parent, cache) {
  const mount = wallMount(scene, parent, -1, -19.6, 0);
  const group = new TransformNode("karlsson-exhibit", scene);
  group.parent = mount;
  group.position.set(0, 1.62, 0);

  const jerseyMaterial = materials.art(
    "karlsson-jersey",
    900,
    1200,
    paintKarlssonJersey,
    { roughness: 0.86, bump: 0.35, environmentIntensity: 0.14, emissive: 0.16 }
  );
  jerseyMaterial.albedoTexture.hasAlpha = true;
  jerseyMaterial.useAlphaFromAlbedoTexture = true;
  jerseyMaterial.transparencyMode = 1;
  jerseyMaterial.backFaceCulling = false;

  const boxWidth = 0.98;
  const boxHeight = 1.28;
  const depth = 0.14;

  // deep shadow box
  solid(scene, "karlsson-case-back", group, {
    width: boxWidth,
    height: boxHeight,
    depth,
    at: [0, 0, depth / 2],
    material: materials.charcoal(),
  });

  const rails = [
    [boxWidth + 0.09, 0.045, 0, (boxHeight + 0.045) / 2],
    [boxWidth + 0.09, 0.045, 0, -(boxHeight + 0.045) / 2],
    [0.045, boxHeight + 0.09, (boxWidth + 0.045) / 2, 0],
    [0.045, boxHeight + 0.09, -(boxWidth + 0.045) / 2, 0],
  ];
  rails.forEach(([w, h, x, y], index) => {
    solid(scene, `karlsson-rail${index}`, group, {
      width: w,
      height: h,
      depth: depth + 0.03,
      at: [x, y, depth / 2 + 0.015],
      material: materials.walnutDark(),
    });
  });

  [
    [boxWidth + 0.008, 0.008, 0, (boxHeight + 0.008) / 2],
    [boxWidth + 0.008, 0.008, 0, -(boxHeight + 0.008) / 2],
    [0.008, boxHeight, (boxWidth + 0.008) / 2, 0],
    [0.008, boxHeight, -(boxWidth + 0.008) / 2, 0],
  ].forEach(([w, h, x, y], index) => {
    solid(scene, `karlsson-fillet${index}`, group, {
      width: w,
      height: h,
      depth: 0.018,
      at: [x, y, depth + 0.012],
      material: materials.brass(),
    });
  });

  // the sweater itself, mounted over a form so it keeps cloth depth
  const jersey = createFacePanel(scene, "karlsson-jersey", {
    width: 0.86,
    height: 1.14,
    columns: 18,
    rows: 22,
    relief: 0.022,
    reliefShape: hangingJerseyRelief,
  });
  jersey.parent = group;
  jersey.position.set(0, 0.02, depth - 0.028);
  jersey.material = jerseyMaterial;

  // authentication hologram in the corner
  solid(scene, "karlsson-hologram", group, {
    width: 0.05,
    height: 0.05,
    depth: 0.003,
    at: [boxWidth / 2 - 0.07, -boxHeight / 2 + 0.07, depth - 0.02],
    material: materials.silver(),
  });

  const pane = artPanel(scene, "karlsson-glass", group, {
    width: boxWidth,
    height: boxHeight,
    at: [0, 0, depth + 0.014],
    material: materials.glass(),
  });
  pane.isPickable = false;

  buildBrassPlaque(scene, materials, group, {
    name: "karlsson",
    width: 0.56,
    height: 0.15,
    at: [0, -boxHeight / 2 - 0.16, 0.03],
    lines: [
      { text: "ERIK KARLSSON — No. 65", scale: 1.18 },
      "Ottawa Senators · Game-Worn",
      "Round 2, Game 1 · vs. New York Rangers · 2017",
    ],
  });

  // museum picture light above the case
  const lightArm = new TransformNode("karlsson-light-arm", scene);
  lightArm.parent = group;
  lightArm.position.set(0, boxHeight / 2 + 0.12, 0.06);
  rod(scene, "karlsson-light-stem", lightArm, {
    diameter: 0.016,
    height: 0.14,
    sides: 10,
    at: [0, 0, 0.07],
    turn: [Math.PI / 2.4, 0, 0],
    material: materials.brass(),
  });
  const hood = rod(scene, "karlsson-light-hood", lightArm, {
    top: 0.062,
    bottom: 0.062,
    height: 0.44,
    sides: 16,
    at: [0, 0.05, 0.15],
    turn: [0, 0, Math.PI / 2],
    material: materials.brass(),
  });
  hood.scaling.set(1, 1, 0.55);
  rod(scene, "karlsson-light-lens", lightArm, {
    diameter: 0.05,
    height: 0.4,
    sides: 14,
    at: [0, 0.015, 0.16],
    turn: [0, 0, Math.PI / 2],
    material: materials.lampLens(),
  });

  const groupMeshes = group.getChildMeshes();
  const spot = new SpotLight(
    "karlsson-museum-spot",
    new Vector3(0, 0, 0),
    new Vector3(0, -1, 0),
    Math.PI / 2.6,
    6,
    scene
  );
  group.computeWorldMatrix(true);
  const worldAnchor = group.getAbsolutePosition().clone();
  spot.position = new Vector3(
    worldAnchor.x + 0.42,
    worldAnchor.y + boxHeight / 2 + 0.2,
    worldAnchor.z
  );
  spot.setDirectionToTarget(worldAnchor);
  spot.diffuse = new Color3(1, 0.9, 0.72);
  spot.intensity = 16;
  spot.range = 4.2;
  spot.includedOnlyMeshes = groupMeshes;

  cache.lights.push(spot);

  return {
    group,
    mount,
    meshes: groupMeshes,
    focus: new Vector3(worldAnchor.x, worldAnchor.y, worldAnchor.z),
  };
}

/* --------------------------------------------------------------------------
   HERO EXHIBIT TWO — hanging, back out, man-cave rather than museum
   -------------------------------------------------------------------------- */

function buildOvechkinExhibit(scene, materials, parent, cache) {
  const mount = wallMount(scene, parent, 1, -18.7, 0);
  const group = new TransformNode("ovechkin-exhibit", scene);
  group.parent = mount;
  group.position.set(0, 1.74, 0);

  const jerseyMaterial = materials.art(
    "ovechkin-jersey",
    900,
    1200,
    paintOvechkinJerseyBack,
    { roughness: 0.9, bump: 0.38, environmentIntensity: 0.12, emissive: 0.14 }
  );
  jerseyMaterial.albedoTexture.hasAlpha = true;
  jerseyMaterial.useAlphaFromAlbedoTexture = true;
  jerseyMaterial.transparencyMode = 1;
  jerseyMaterial.backFaceCulling = false;

  // walnut backer board, so the sweater is not floating on plaster
  solid(scene, "ovechkin-backer", group, {
    width: 1.06,
    height: 1.42,
    depth: 0.028,
    at: [0, -0.02, 0.014],
    material: materials.walnut(),
  });
  [
    [1.1, 0.024, 0, 0.71],
    [1.1, 0.024, 0, -0.75],
  ].forEach(([w, h, x, y], index) => {
    solid(scene, `ovechkin-backer-trim${index}`, group, {
      width: w,
      height: h,
      depth: 0.04,
      at: [x, y, 0.02],
      material: materials.walnutDark(),
    });
  });

  // brass hanging rail on standoffs
  [-1, 1].forEach((side) => {
    rod(scene, `ovechkin-standoff${side}`, group, {
      diameter: 0.022,
      height: 0.1,
      sides: 10,
      at: [side * 0.45, 0.62, 0.06],
      turn: [Math.PI / 2, 0, 0],
      material: materials.brass(),
    });
  });
  const rail = rod(scene, "ovechkin-rail", group, {
    diameter: 0.026,
    height: 1.02,
    sides: 12,
    at: [0, 0.62, 0.11],
    turn: [0, 0, Math.PI / 2],
    material: materials.brass(),
  });
  void rail;

  // wooden hanger under the rail
  solid(scene, "ovechkin-hanger-bar", group, {
    width: 0.5,
    height: 0.016,
    depth: 0.026,
    at: [0, 0.56, 0.11],
    material: materials.walnutDark(),
  });
  [-1, 1].forEach((side) => {
    solid(scene, `ovechkin-hanger-arm${side}`, group, {
      width: 0.26,
      height: 0.014,
      depth: 0.024,
      at: [side * 0.13, 0.585, 0.11],
      turn: [0, 0, side * -0.2],
      material: materials.walnutDark(),
    });
  });
  rod(scene, "ovechkin-hanger-hook", group, {
    diameter: 0.008,
    height: 0.08,
    sides: 8,
    at: [0, 0.615, 0.11],
    material: materials.steel(),
  });

  const jersey = createFacePanel(scene, "ovechkin-jersey", {
    width: 0.94,
    height: 1.24,
    columns: 18,
    rows: 22,
    relief: 0.028,
    reliefShape: hangingJerseyRelief,
  });
  jersey.parent = group;
  jersey.position.set(0, -0.06, 0.14);
  jersey.material = jerseyMaterial;

  buildBrassPlaque(scene, materials, group, {
    name: "ovechkin",
    width: 0.48,
    height: 0.13,
    at: [0, -0.88, 0.03],
    lines: [
      { text: "ALEX OVECHKIN — No. 8", scale: 1.16 },
      "Washington Capitals · Koho · Rookie Era",
    ],
  });

  const groupMeshes = group.getChildMeshes();
  group.computeWorldMatrix(true);
  const worldAnchor = group.getAbsolutePosition().clone();
  const spot = new SpotLight(
    "ovechkin-spot",
    new Vector3(worldAnchor.x - 0.5, worldAnchor.y + 0.86, worldAnchor.z - 0.1),
    new Vector3(0, -1, 0),
    Math.PI / 2.4,
    5,
    scene
  );
  spot.setDirectionToTarget(worldAnchor);
  spot.diffuse = new Color3(1, 0.86, 0.66);
  spot.intensity = 15;
  spot.range = 4.4;
  spot.includedOnlyMeshes = groupMeshes;
  cache.lights.push(spot);

  return {
    group,
    mount,
    meshes: groupMeshes,
    focus: new Vector3(worldAnchor.x, worldAnchor.y, worldAnchor.z),
  };
}

/* --------------------------------------------------------------------------
   EQUIPMENT AND CLUTTER
   -------------------------------------------------------------------------- */

function buildPuck(scene, materials, parent, { at, turn, name }) {
  const puck = rod(scene, `puck-${name}`, parent, {
    diameter: 0.076,
    height: 0.026,
    sides: 20,
    at,
    turn,
    material: materials.rubber(),
  });
  // faint sidewall lettering ring
  rod(scene, `puck-${name}-band`, puck, {
    diameter: 0.0765,
    height: 0.008,
    sides: 20,
    at: [0, 0, 0],
    material: materials.blackMetal(),
  });
  return puck;
}

function buildStick(
  scene,
  materials,
  parent,
  { name, at, turn, length = 1.62, wooden = false, broken = false, taped = true }
) {
  const group = new TransformNode(`stick-${name}`, scene);
  group.parent = parent;
  group.position.set(at[0], at[1], at[2]);
  if (turn) {
    group.rotation.set(turn[0], turn[1], turn[2]);
  }

  const shaftMaterial = wooden ? materials.leather() : materials.blackMetal();
  const shaftLength = broken ? length * 0.52 : length * 0.78;

  const shaft = solid(scene, `stick-${name}-shaft`, group, {
    width: wooden ? 0.03 : 0.026,
    height: shaftLength,
    depth: wooden ? 0.022 : 0.019,
    at: [0, shaftLength / 2 + length * 0.2, 0],
    material: shaftMaterial,
  });
  void shaft;

  if (broken) {
    // splintered break, angled so the fracture reads from a distance
    solid(scene, `stick-${name}-splinter`, group, {
      width: 0.024,
      height: 0.07,
      depth: 0.015,
      at: [0.008, shaftLength + length * 0.22, 0],
      turn: [0.12, 0, 0.26],
      material: materials.canvasCream(),
    });
  }

  // grip tape at the butt end
  if (taped) {
    solid(scene, `stick-${name}-grip`, group, {
      width: wooden ? 0.034 : 0.03,
      height: 0.2,
      depth: wooden ? 0.026 : 0.023,
      at: [0, shaftLength + length * 0.11, 0],
      material: materials.tape(),
    });
  }

  // heel and blade
  const heel = solid(scene, `stick-${name}-heel`, group, {
    width: 0.03,
    height: 0.13,
    depth: 0.021,
    at: [0, length * 0.14, 0.02],
    turn: [0.5, 0, 0],
    material: shaftMaterial,
  });
  void heel;

  const blade = solid(scene, `stick-${name}-blade`, group, {
    width: 0.028,
    height: 0.075,
    depth: 0.31,
    at: [0, length * 0.045, 0.15],
    turn: [0, 0, 0],
    material: broken ? materials.whiteTape() : shaftMaterial,
  });
  blade.rotation.y = 0.12;

  // blade tape, still wrapped
  solid(scene, `stick-${name}-blade-tape`, group, {
    width: 0.031,
    height: 0.078,
    depth: 0.2,
    at: [0, length * 0.045, 0.19],
    turn: [0, 0.12, 0],
    material: broken ? materials.tape() : materials.whiteTape(),
  });

  return group;
}

function buildGoaliePads(scene, materials, parent, { at, turn }) {
  const group = new TransformNode("goalie-pads", scene);
  group.parent = parent;
  group.position.set(at[0], at[1], at[2]);
  if (turn) group.rotation.set(turn[0], turn[1], turn[2]);

  [-1, 1].forEach((side) => {
    const pad = new TransformNode(`goalie-pad${side}`, scene);
    pad.parent = group;
    pad.position.set(side * 0.17, 0, side * 0.02);
    pad.rotation.z = side * 0.05;

    // three stacked rolls, thigh rise, knee break — the classic silhouette
    solid(scene, `goalie-pad${side}-face`, pad, {
      width: 0.28,
      height: 0.9,
      depth: 0.1,
      at: [0, 0.45, 0],
      material: materials.canvasCream(),
    });
    [0.14, 0.45, 0.72].forEach((h, index) => {
      rod(scene, `goalie-pad${side}-roll${index}`, pad, {
        diameter: 0.085,
        height: 0.28,
        sides: 12,
        at: [0, h, -0.055],
        turn: [0, 0, Math.PI / 2],
        material: index === 1 ? materials.leather() : materials.canvasCream(),
      });
    });
    solid(scene, `goalie-pad${side}-thigh`, pad, {
      width: 0.26,
      height: 0.24,
      depth: 0.09,
      at: [0, 1.0, 0.02],
      turn: [-0.18, 0, 0],
      material: materials.canvasCream(),
    });
    solid(scene, `goalie-pad${side}-knee`, pad, {
      width: 0.2,
      height: 0.14,
      depth: 0.11,
      at: [side * 0.04, 0.58, 0.07],
      material: materials.leather(),
    });
    // straps
    [0.2, 0.5, 0.8].forEach((h, index) => {
      solid(scene, `goalie-pad${side}-strap${index}`, pad, {
        width: 0.3,
        height: 0.024,
        depth: 0.13,
        at: [0, h, -0.01],
        material: materials.tape(),
      });
    });
  });

  return group;
}

function buildHockeyGlove(scene, materials, parent, { name, at, turn, tone }) {
  const group = new TransformNode(`glove-${name}`, scene);
  group.parent = parent;
  group.position.set(at[0], at[1], at[2]);
  if (turn) group.rotation.set(turn[0], turn[1], turn[2]);

  const shell = tone || materials.leather();

  solid(scene, `glove-${name}-back`, group, {
    width: 0.14,
    height: 0.19,
    depth: 0.1,
    at: [0, 0.095, 0],
    material: shell,
  });
  // cuff roll
  rod(scene, `glove-${name}-cuff`, group, {
    diameter: 0.115,
    height: 0.13,
    sides: 14,
    at: [0, 0.235, 0],
    turn: [Math.PI / 2, 0, 0],
    material: shell,
  });
  // fingers
  for (let i = 0; i < 4; i += 1) {
    solid(scene, `glove-${name}-finger${i}`, group, {
      width: 0.032,
      height: 0.09,
      depth: 0.038,
      at: [-0.05 + i * 0.034, 0.012, 0.026],
      turn: [0.2, 0, 0],
      material: shell,
    });
  }
  // thumb
  solid(scene, `glove-${name}-thumb`, group, {
    width: 0.045,
    height: 0.1,
    depth: 0.048,
    at: [0.075, 0.075, 0.02],
    turn: [0.1, 0, -0.5],
    material: shell,
  });
  // palm
  solid(scene, `glove-${name}-palm`, group, {
    width: 0.13,
    height: 0.16,
    depth: 0.02,
    at: [0, 0.09, -0.052],
    material: materials.tape(),
  });

  return group;
}

function buildDisplayCase(
  scene,
  materials,
  parent,
  { name, at, width = 0.34, height = 0.24, depth = 0.22, pucks = 2 }
) {
  const group = new TransformNode(`case-${name}`, scene);
  group.parent = parent;
  group.position.set(at[0], at[1], at[2]);

  // walnut plinth
  solid(scene, `case-${name}-base`, group, {
    width,
    height: 0.032,
    depth,
    at: [0, 0.016, 0],
    material: materials.walnutDark(),
  });
  solid(scene, `case-${name}-reveal`, group, {
    width: width - 0.03,
    height: 0.008,
    depth: depth - 0.03,
    at: [0, 0.036, 0],
    material: materials.brass(),
  });

  // pucks on small posts
  for (let i = 0; i < pucks; i += 1) {
    const x = pucks === 1 ? 0 : -width * 0.26 + i * ((width * 0.52) / Math.max(1, pucks - 1));
    rod(scene, `case-${name}-post${i}`, group, {
      diameter: 0.018,
      height: 0.05,
      sides: 10,
      at: [x, 0.065, 0],
      material: materials.brass(),
    });
    const puck = buildPuck(scene, materials, group, {
      name: `${name}-${i}`,
      at: [x, 0.1, 0],
      turn: [Math.PI / 2, 0.3 + i * 0.5, 0],
    });
    // silver signature stroke across the face
    artPanel(scene, `case-${name}-sig${i}`, puck, {
      width: 0.055,
      height: 0.03,
      at: [0, -0.014, 0],
      turn: [Math.PI / 2, 0, 0],
      material: materials.silver(),
    });
  }

  // acrylic hood
  const hood = solid(scene, `case-${name}-hood`, group, {
    width,
    height,
    depth,
    at: [0, height / 2 + 0.04, 0],
    material: materials.acrylic(),
  });
  hood.isPickable = false;

  return group;
}

function buildPuckBucket(scene, materials, parent, { at }) {
  const group = new TransformNode("puck-bucket", scene);
  group.parent = parent;
  group.position.set(at[0], at[1], at[2]);

  const pail = rod(scene, "puck-bucket-body", group, {
    top: 0.3,
    bottom: 0.24,
    height: 0.32,
    sides: 20,
    at: [0, 0.16, 0],
    material: materials.blackMetal(),
  });
  void pail;
  rod(scene, "puck-bucket-lip", group, {
    diameter: 0.31,
    height: 0.02,
    sides: 20,
    at: [0, 0.32, 0],
    material: materials.steel(),
  });
  // handle
  const handle = MeshBuilder.CreateTorus(
    "puck-bucket-handle",
    { diameter: 0.3, thickness: 0.012, tessellation: 18 },
    scene
  );
  place(handle, group, {
    at: [0, 0.36, 0],
    turn: [0, 0, Math.PI / 2],
    material: materials.steel(),
  });

  // pucks stacked and spilling
  for (let i = 0; i < 9; i += 1) {
    buildPuck(scene, materials, group, {
      name: `bucket-${i}`,
      at: [
        (hashRandom(i * 2.3) - 0.5) * 0.16,
        0.05 + i * 0.027,
        (hashRandom(i * 5.7) - 0.5) * 0.16,
      ],
      turn: [Math.PI / 2, hashRandom(i * 3.1) * 3, hashRandom(i * 7.7) * 0.2],
    });
  }

  return group;
}

function buildEquipmentBag(scene, materials, parent, { at, turn }) {
  const group = new TransformNode("equipment-bag", scene);
  group.parent = parent;
  group.position.set(at[0], at[1], at[2]);
  if (turn) group.rotation.set(turn[0], turn[1], turn[2]);

  const body = orb(scene, "equipment-bag-body", group, {
    diameter: 0.5,
    segments: 14,
    at: [0, 0.22, 0],
    material: materials.tape(),
    squash: [1.55, 0.82, 1],
  });
  void body;
  // end caps read as a duffel rather than a boulder
  [-1, 1].forEach((side) => {
    rod(scene, `equipment-bag-end${side}`, group, {
      diameter: 0.38,
      height: 0.03,
      sides: 16,
      at: [side * 0.38, 0.22, 0],
      turn: [0, 0, Math.PI / 2],
      material: materials.blackMetal(),
    });
  });
  // zip and strap
  solid(scene, "equipment-bag-zip", group, {
    width: 0.72,
    height: 0.012,
    depth: 0.02,
    at: [0, 0.4, 0],
    material: materials.steel(),
  });
  solid(scene, "equipment-bag-strap", group, {
    width: 0.5,
    height: 0.035,
    depth: 0.02,
    at: [0, 0.42, 0.1],
    turn: [0.4, 0, 0],
    material: materials.leather(),
  });

  return group;
}

function buildTrashCan(scene, materials, parent, { at }) {
  const group = new TransformNode("dented-bin", scene);
  group.parent = parent;
  group.position.set(at[0], at[1], at[2]);

  const body = rod(scene, "dented-bin-body", group, {
    top: 0.31,
    bottom: 0.26,
    height: 0.56,
    sides: 22,
    at: [0, 0.28, 0],
    material: materials.steel(),
  });
  // the dent, which is the joke
  body.scaling.x = 0.9;
  solid(scene, "dented-bin-dent", group, {
    width: 0.2,
    height: 0.18,
    depth: 0.1,
    at: [0.11, 0.33, 0.1],
    turn: [0.2, 0.6, 0.3],
    material: materials.blackMetal(),
  });
  rod(scene, "dented-bin-lip", group, {
    diameter: 0.32,
    height: 0.022,
    sides: 22,
    at: [0, 0.57, 0],
    material: materials.blackMetal(),
  });
  for (let i = 0; i < 3; i += 1) {
    rod(scene, `dented-bin-band${i}`, group, {
      diameter: 0.3 - i * 0.012,
      height: 0.014,
      sides: 22,
      at: [0, 0.1 + i * 0.18, 0],
      material: materials.blackMetal(),
    });
  }
  // crumpled paper spilling over the rim
  for (let i = 0; i < 3; i += 1) {
    orb(scene, `dented-bin-paper${i}`, group, {
      diameter: 0.1,
      segments: 6,
      at: [
        (hashRandom(i * 3.7) - 0.5) * 0.18,
        0.58 + hashRandom(i * 5.1) * 0.04,
        (hashRandom(i * 9.1) - 0.5) * 0.18,
      ],
      material: materials.paper(),
    });
  }

  return group;
}

function buildTapeRolls(scene, materials, parent, { at }) {
  const group = new TransformNode("tape-rolls", scene);
  group.parent = parent;
  group.position.set(at[0], at[1], at[2]);

  const stack = [
    [0, 0.021, 0, materials.whiteTape()],
    [0, 0.063, 0, materials.tape()],
    [0.085, 0.021, 0.03, materials.whiteTape()],
  ];
  stack.forEach(([x, y, z, mat], index) => {
    const roll = rod(scene, `tape-roll${index}`, group, {
      diameter: 0.1,
      height: 0.038,
      sides: 18,
      at: [x, y, z],
      material: mat,
    });
    rod(scene, `tape-core${index}`, roll, {
      diameter: 0.042,
      height: 0.04,
      sides: 14,
      at: [0, 0, 0],
      material: materials.leather(),
    });
  });

  return group;
}

function buildSkateGuards(scene, materials, parent, { at, turn }) {
  const group = new TransformNode("skate-guards", scene);
  group.parent = parent;
  group.position.set(at[0], at[1], at[2]);
  if (turn) group.rotation.set(turn[0], turn[1], turn[2]);

  [-1, 1].forEach((side) => {
    const guard = new TransformNode(`skate-guard${side}`, scene);
    guard.parent = group;
    guard.position.set(side * 0.055, 0, 0);
    guard.rotation.y = side * 0.1;

    solid(scene, `skate-guard${side}-body`, guard, {
      width: 0.026,
      height: 0.05,
      depth: 0.3,
      at: [0, 0.025, 0],
      material: materials.blackMetal(),
    });
    [-1, 1].forEach((end) => {
      rod(scene, `skate-guard${side}-cap${end}`, guard, {
        diameter: 0.05,
        height: 0.026,
        sides: 10,
        at: [0, 0.026, end * 0.15],
        turn: [0, 0, Math.PI / 2],
        material: materials.blackMetal(),
      });
    });
    // coiled spring between the halves
    for (let i = 0; i < 5; i += 1) {
      rod(scene, `skate-guard${side}-coil${i}`, guard, {
        diameter: 0.03,
        height: 0.006,
        sides: 8,
        at: [0, 0.05 + i * 0.007, 0],
        material: materials.steel(),
      });
    }
  });

  return group;
}

function buildCoffeeCup(scene, materials, parent, { at }) {
  const group = new TransformNode("forgotten-coffee", scene);
  group.parent = parent;
  group.position.set(at[0], at[1], at[2]);

  rod(scene, "forgotten-coffee-cup", group, {
    top: 0.085,
    bottom: 0.062,
    height: 0.115,
    sides: 18,
    at: [0, 0.057, 0],
    material: materials.paper(),
  });
  rod(scene, "forgotten-coffee-lid", group, {
    top: 0.09,
    bottom: 0.088,
    height: 0.016,
    sides: 18,
    at: [0, 0.121, 0],
    material: materials.blackMetal(),
  });
  rod(scene, "forgotten-coffee-tab", group, {
    diameter: 0.022,
    height: 0.008,
    sides: 8,
    at: [0.03, 0.13, 0],
    material: materials.blackMetal(),
  });
  rod(scene, "forgotten-coffee-sleeve", group, {
    top: 0.079,
    bottom: 0.07,
    height: 0.045,
    sides: 18,
    at: [0, 0.055, 0],
    material: materials.leather(),
  });

  return group;
}

function buildClipboard(scene, materials, parent, { at, turn }) {
  const group = new TransformNode("equipment-clipboard", scene);
  group.parent = parent;
  group.position.set(at[0], at[1], at[2]);
  if (turn) group.rotation.set(turn[0], turn[1], turn[2]);

  solid(scene, "equipment-clipboard-board", group, {
    width: 0.23,
    height: 0.006,
    depth: 0.31,
    at: [0, 0.003, 0],
    material: materials.leather(),
  });
  solid(scene, "equipment-clipboard-paper", group, {
    width: 0.2,
    height: 0.004,
    depth: 0.27,
    at: [0, 0.008, -0.008],
    material: materials.paper(),
  });
  solid(scene, "equipment-clipboard-clip", group, {
    width: 0.09,
    height: 0.014,
    depth: 0.045,
    at: [0, 0.014, 0.13],
    material: materials.steel(),
  });
  rod(scene, "equipment-clipboard-pencil", group, {
    diameter: 0.008,
    height: 0.17,
    sides: 6,
    at: [0.06, 0.014, -0.02],
    turn: [0, 0.4, Math.PI / 2],
    material: materials.brass(),
  });

  return group;
}

function buildStopwatchAndWhistle(scene, materials, parent, { at }) {
  const group = new TransformNode("timing-kit", scene);
  group.parent = parent;
  group.position.set(at[0], at[1], at[2]);

  // stopwatch
  const watch = rod(scene, "stopwatch-body", group, {
    diameter: 0.075,
    height: 0.022,
    sides: 22,
    at: [0, 0.011, 0],
    material: materials.steel(),
  });
  rod(scene, "stopwatch-face", watch, {
    diameter: 0.062,
    height: 0.024,
    sides: 22,
    at: [0, 0, 0],
    material: materials.canvasCream(),
  });
  rod(scene, "stopwatch-crown", group, {
    diameter: 0.016,
    height: 0.016,
    sides: 10,
    at: [0, 0.011, 0.044],
    turn: [Math.PI / 2, 0, 0],
    material: materials.brass(),
  });
  const ring = MeshBuilder.CreateTorus(
    "stopwatch-ring",
    { diameter: 0.03, thickness: 0.005, tessellation: 14 },
    scene
  );
  place(ring, group, {
    at: [0, 0.011, 0.062],
    turn: [Math.PI / 2, 0, 0],
    material: materials.brass(),
  });

  // whistle
  const whistle = new TransformNode("whistle", scene);
  whistle.parent = group;
  whistle.position.set(0.11, 0.012, -0.03);
  whistle.rotation.y = 0.7;
  solid(scene, "whistle-body", whistle, {
    width: 0.055,
    height: 0.022,
    depth: 0.024,
    at: [0, 0.011, 0],
    material: materials.steel(),
  });
  rod(scene, "whistle-chamber", whistle, {
    diameter: 0.03,
    height: 0.024,
    sides: 14,
    at: [0.022, 0.013, 0],
    turn: [Math.PI / 2, 0, 0],
    material: materials.steel(),
  });
  rod(scene, "whistle-lanyard-loop", whistle, {
    diameter: 0.01,
    height: 0.012,
    sides: 8,
    at: [-0.03, 0.012, 0],
    turn: [0, 0, Math.PI / 2],
    material: materials.brass(),
  });
  // cord coiled beside it
  for (let i = 0; i < 14; i += 1) {
    const angle = (i / 14) * Math.PI * 2;
    rod(scene, `whistle-cord${i}`, group, {
      diameter: 0.006,
      height: 0.03,
      sides: 6,
      at: [0.11 + Math.cos(angle) * 0.05, 0.004, -0.03 + Math.sin(angle) * 0.05],
      turn: [Math.PI / 2, angle, 0],
      material: materials.tape(),
    });
  }

  return group;
}

/* --------------------------------------------------------------------------
   GOALTENDING MASKS — three unmistakable silhouettes
   -------------------------------------------------------------------------- */

function buildFibreglassMask(scene, materials, parent, { at, turn }) {
  const group = new TransformNode("mask-fibreglass", scene);
  group.parent = parent;
  group.position.set(at[0], at[1], at[2]);
  if (turn) group.rotation.set(turn[0], turn[1], turn[2]);

  const shellMaterial = materials.flat("mask-resin", "#cfc6ae", {
    roughness: 0.42,
    environmentIntensity: 0.5,
  });

  const shell = orb(scene, "mask-fibreglass-shell", group, {
    diameter: 0.24,
    segments: 20,
    at: [0, 0, 0],
    material: shellMaterial,
    squash: [0.92, 1.16, 0.72],
  });

  // brow ridge and cheekbones so it is a face, not an egg
  solid(scene, "mask-fibreglass-brow", group, {
    width: 0.18,
    height: 0.022,
    depth: 0.03,
    at: [0, 0.05, 0.082],
    turn: [0.2, 0, 0],
    material: shellMaterial,
  });
  [-1, 1].forEach((side) => {
    orb(scene, `mask-fibreglass-cheek${side}`, group, {
      diameter: 0.07,
      segments: 10,
      at: [side * 0.06, -0.03, 0.07],
      material: shellMaterial,
      squash: [1, 1.1, 0.6],
    });
  });

  // eye slots and mouth cut
  const voidMaterial = materials.flat("mask-void", "#08080a", {
    roughness: 0.95,
  });
  [-1, 1].forEach((side) => {
    solid(scene, `mask-fibreglass-eye${side}`, group, {
      width: 0.05,
      height: 0.026,
      depth: 0.016,
      at: [side * 0.045, 0.028, 0.086],
      turn: [0, side * -0.2, side * 0.08],
      material: voidMaterial,
    });
  });
  solid(scene, "mask-fibreglass-mouth", group, {
    width: 0.085,
    height: 0.018,
    depth: 0.016,
    at: [0, -0.058, 0.075],
    material: voidMaterial,
  });
  // breathing holes
  for (let i = 0; i < 8; i += 1) {
    rod(scene, `mask-fibreglass-hole${i}`, group, {
      diameter: 0.011,
      height: 0.014,
      sides: 8,
      at: [-0.035 + (i % 4) * 0.023, -0.026 - Math.floor(i / 4) * 0.016, 0.083],
      turn: [Math.PI / 2, 0, 0],
      material: voidMaterial,
    });
  }
  // nose bridge
  solid(scene, "mask-fibreglass-nose", group, {
    width: 0.02,
    height: 0.05,
    depth: 0.026,
    at: [0, -0.012, 0.09],
    material: shellMaterial,
  });

  // strap and hanging hook
  solid(scene, "mask-fibreglass-strap", group, {
    width: 0.24,
    height: 0.016,
    depth: 0.014,
    at: [0, -0.03, -0.075],
    material: materials.leather(),
  });

  shell.isPickable = false;
  return group;
}

function buildCageMask(scene, materials, parent, { at, turn }) {
  const group = new TransformNode("mask-cage", scene);
  group.parent = parent;
  group.position.set(at[0], at[1], at[2]);
  if (turn) group.rotation.set(turn[0], turn[1], turn[2]);

  const helmetMaterial = materials.flat("mask-helmet", "#16181c", {
    roughness: 0.48,
    environmentIntensity: 0.4,
  });

  const helmet = orb(scene, "mask-cage-helmet", group, {
    diameter: 0.24,
    segments: 18,
    slice: 0.56,
    at: [0, 0.02, 0],
    material: helmetMaterial,
    squash: [0.94, 1, 0.9],
  });
  helmet.rotation.x = 0.2;

  solid(scene, "mask-cage-brim", group, {
    width: 0.21,
    height: 0.014,
    depth: 0.05,
    at: [0, 0.035, 0.075],
    turn: [0.3, 0, 0],
    material: helmetMaterial,
  });

  // welded wire cage — vertical bars plus horizontal runs on an arc
  const wire = materials.steel();
  for (let i = 0; i < 7; i += 1) {
    const t = (i / 6 - 0.5) * 2;
    rod(scene, `mask-cage-vbar${i}`, group, {
      diameter: 0.007,
      height: 0.15,
      sides: 6,
      at: [t * 0.085, -0.045, 0.09 - Math.abs(t) * 0.028],
      turn: [0.12, 0, 0],
      material: wire,
    });
  }
  for (let i = 0; i < 4; i += 1) {
    rod(scene, `mask-cage-hbar${i}`, group, {
      diameter: 0.007,
      height: 0.185,
      sides: 6,
      at: [0, 0.012 - i * 0.037, 0.088 - i * 0.006],
      turn: [0, 0, Math.PI / 2],
      material: wire,
    });
  }
  // chin cup
  orb(scene, "mask-cage-chin", group, {
    diameter: 0.1,
    segments: 10,
    at: [0, -0.1, 0.05],
    material: materials.leather(),
    squash: [1, 0.55, 0.8],
  });
  solid(scene, "mask-cage-strap", group, {
    width: 0.22,
    height: 0.014,
    depth: 0.012,
    at: [0, -0.03, -0.07],
    material: materials.leather(),
  });

  return group;
}

function buildModernMask(scene, materials, parent, { at, turn }) {
  const group = new TransformNode("mask-modern", scene);
  group.parent = parent;
  group.position.set(at[0], at[1], at[2]);
  if (turn) group.rotation.set(turn[0], turn[1], turn[2]);

  const paintedShell = materials.art(
    "mask-paint",
    512,
    512,
    (ctx, w, h) => {
      ctx.fillStyle = "#0e1016";
      ctx.fillRect(0, 0, w, h);
      // airbrushed flames and a crest, kept abstract
      const flame = ctx.createLinearGradient(0, h, w, 0);
      flame.addColorStop(0, "rgba(178,25,57,0.9)");
      flame.addColorStop(0.5, "rgba(230,120,40,0.8)");
      flame.addColorStop(1, "rgba(20,22,30,0)");
      ctx.fillStyle = flame;
      for (let i = 0; i < 9; i += 1) {
        ctx.beginPath();
        const x = (i / 9) * w;
        ctx.moveTo(x, h);
        ctx.bezierCurveTo(
          x + w * 0.06,
          h * 0.6,
          x - w * 0.04,
          h * 0.45,
          x + w * 0.09,
          h * 0.18
        );
        ctx.bezierCurveTo(
          x + w * 0.02,
          h * 0.5,
          x + w * 0.12,
          h * 0.62,
          x + w * 0.1,
          h
        );
        ctx.closePath();
        ctx.fill();
      }
      ctx.strokeStyle = "rgba(201,168,106,0.9)";
      ctx.lineWidth = 8;
      ctx.beginPath();
      ctx.arc(w * 0.5, h * 0.4, w * 0.12, 0, Math.PI * 2);
      ctx.stroke();
      // clearcoat sheen
      const gloss = ctx.createLinearGradient(0, 0, w, h);
      gloss.addColorStop(0, "rgba(255,255,255,0.18)");
      gloss.addColorStop(0.4, "rgba(255,255,255,0)");
      ctx.fillStyle = gloss;
      ctx.fillRect(0, 0, w, h);
    },
    { roughness: 0.2, metallic: 0.15, environmentIntensity: 0.85 }
  );

  const shell = orb(scene, "mask-modern-shell", group, {
    diameter: 0.25,
    segments: 22,
    at: [0, 0, 0],
    material: paintedShell,
    squash: [0.96, 1.08, 0.94],
  });
  void shell;

  // dome vents
  for (let i = 0; i < 3; i += 1) {
    solid(scene, `mask-modern-vent${i}`, group, {
      width: 0.05,
      height: 0.008,
      depth: 0.02,
      at: [(i - 1) * 0.045, 0.1, 0.05],
      turn: [0.5, 0, 0],
      material: materials.blackMetal(),
    });
  }

  // cat-eye cage
  const wire = materials.flat("mask-modern-cage", "#c9ccd2", {
    metallic: 0.9,
    roughness: 0.24,
  });
  const eyeBars = [
    [0.024, 0.03],
    [0.052, -0.006],
    [0.052, -0.042],
    [0.024, -0.072],
  ];
  eyeBars.forEach(([halfWidth, y], index) => {
    rod(scene, `mask-modern-hbar${index}`, group, {
      diameter: 0.008,
      height: halfWidth * 2 + 0.14,
      sides: 6,
      at: [0, y, 0.113],
      turn: [0, 0, Math.PI / 2],
      material: wire,
    });
  });
  for (let i = 0; i < 5; i += 1) {
    const x = (i / 4 - 0.5) * 0.16;
    rod(scene, `mask-modern-vbar${i}`, group, {
      diameter: 0.008,
      height: 0.12,
      sides: 6,
      at: [x, -0.02, 0.112 - Math.abs(x) * 0.12],
      material: wire,
    });
  }
  // cage surround
  const surround = MeshBuilder.CreateTorus(
    "mask-modern-surround",
    { diameter: 0.2, thickness: 0.011, tessellation: 22 },
    scene
  );
  place(surround, group, {
    at: [0, -0.02, 0.105],
    turn: [Math.PI / 2, 0, 0],
    material: wire,
  });
  surround.scaling.set(1, 1, 1.2);

  // backplate
  orb(scene, "mask-modern-backplate", group, {
    diameter: 0.2,
    segments: 14,
    slice: 0.5,
    at: [0, -0.05, -0.07],
    turn: [Math.PI, 0, 0],
    material: materials.blackMetal(),
    squash: [1, 0.8, 0.6],
  });
  solid(scene, "mask-modern-strap", group, {
    width: 0.2,
    height: 0.016,
    depth: 0.012,
    at: [0, 0.02, -0.1],
    material: materials.leather(),
  });

  return group;
}

/* --------------------------------------------------------------------------
   CHAMPIONSHIP TROPHY DISPLAY
   -------------------------------------------------------------------------- */

function buildChampionshipCup(scene, materials, parent, cache, { at }) {
  const group = new TransformNode("championship-display", scene);
  group.parent = parent;
  group.position.set(at[0], at[1], at[2]);

  // low walnut pedestal, deliberately not centred in the corridor
  solid(scene, "cup-pedestal", group, {
    width: 0.62,
    height: 0.28,
    depth: 0.62,
    at: [0, 0.14, 0],
    material: materials.walnut(),
    collide: true,
  });
  solid(scene, "cup-pedestal-cap", group, {
    width: 0.68,
    height: 0.03,
    depth: 0.68,
    at: [0, 0.295, 0],
    material: materials.walnutDark(),
  });
  solid(scene, "cup-pedestal-reveal", group, {
    width: 0.64,
    height: 0.008,
    depth: 0.64,
    at: [0, 0.315, 0],
    material: materials.brass(),
  });

  const cup = new TransformNode("championship-cup", scene);
  cup.parent = group;
  cup.position.set(0, 0.32, 0);

  const silver = materials.silver();

  // barrel of engraved bands
  for (let i = 0; i < 5; i += 1) {
    const band = rod(scene, `cup-band${i}`, cup, {
      top: 0.147 - i * 0.001,
      bottom: 0.149 - i * 0.001,
      height: 0.096,
      sides: 34,
      at: [0, 0.05 + i * 0.098, 0],
      material: silver,
    });
    // engraved seam between bands
    rod(scene, `cup-band-seam${i}`, cup, {
      diameter: 0.152,
      height: 0.006,
      sides: 34,
      at: [0, 0.098 + i * 0.098, 0],
      material: materials.blackMetal(),
    });
    band.receiveShadows = true;
  }

  // plinth under the barrel
  rod(scene, "cup-plinth", cup, {
    top: 0.152,
    bottom: 0.168,
    height: 0.03,
    sides: 34,
    at: [0, 0.015, 0],
    material: silver,
  });

  // collar taper into the bowl
  rod(scene, "cup-collar", cup, {
    top: 0.084,
    bottom: 0.146,
    height: 0.075,
    sides: 30,
    at: [0, 0.578, 0],
    material: silver,
  });

  // the bowl
  const bowlProfile = [
    new Vector3(0.0, 0.0, 0),
    new Vector3(0.076, 0.006, 0),
    new Vector3(0.104, 0.05, 0),
    new Vector3(0.128, 0.12, 0),
    new Vector3(0.142, 0.185, 0),
    new Vector3(0.147, 0.212, 0),
    new Vector3(0.138, 0.214, 0),
    new Vector3(0.128, 0.17, 0),
    new Vector3(0.1, 0.09, 0),
    new Vector3(0.062, 0.03, 0),
    new Vector3(0.0, 0.02, 0),
  ];
  const bowl = MeshBuilder.CreateLathe(
    "cup-bowl",
    {
      shape: bowlProfile,
      tessellation: 40,
      sideOrientation: Mesh.DOUBLESIDE,
    },
    scene
  );
  bowl.parent = cup;
  bowl.position.set(0, 0.62, 0);
  bowl.material = silver;
  bowl.isPickable = false;
  bowl.receiveShadows = true;

  // lip ring, the sharpest highlight in an otherwise restrained corridor
  const lip = MeshBuilder.CreateTorus(
    "cup-lip",
    { diameter: 0.292, thickness: 0.012, tessellation: 40 },
    scene
  );
  place(lip, cup, { at: [0, 0.832, 0], material: silver });

  const cupMeshes = [];
  cup.getChildMeshes().forEach((mesh) => cupMeshes.push(mesh));

  // travel case, hinged open behind the pedestal
  const travelCase = new TransformNode("cup-travel-case", scene);
  travelCase.parent = group;
  travelCase.position.set(-0.62, 0, -0.16);
  travelCase.rotation.y = 0.32;

  solid(scene, "cup-case-body", travelCase, {
    width: 0.52,
    height: 0.34,
    depth: 0.46,
    at: [0, 0.17, 0],
    material: materials.blackMetal(),
    collide: true,
  });
  solid(scene, "cup-case-lid", travelCase, {
    width: 0.52,
    height: 0.06,
    depth: 0.46,
    at: [0, 0.5, -0.28],
    turn: [-1.05, 0, 0],
    material: materials.blackMetal(),
  });
  solid(scene, "cup-case-liner", travelCase, {
    width: 0.46,
    height: 0.02,
    depth: 0.4,
    at: [0, 0.35, 0],
    material: materials.leather(),
  });
  [-1, 1].forEach((side) => {
    solid(scene, `cup-case-latch${side}`, travelCase, {
      width: 0.06,
      height: 0.05,
      depth: 0.02,
      at: [side * 0.17, 0.3, 0.235],
      material: materials.steel(),
    });
    solid(scene, `cup-case-corner${side}`, travelCase, {
      width: 0.05,
      height: 0.05,
      depth: 0.05,
      at: [side * 0.235, 0.03, 0.21],
      material: materials.steel(),
    });
  });
  // stencilled routing label
  artPanel(scene, "cup-case-label", travelCase, {
    width: 0.24,
    height: 0.1,
    at: [0, 0.2, 0.232],
    material: materials.art(
      "cup-case-label",
      512,
      220,
      (ctx, w, h) => {
        ctx.fillStyle = "#151517";
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = "rgba(226,220,198,0.8)";
        ctx.lineWidth = 6;
        ctx.strokeRect(12, 12, w - 24, h - 24);
        ctx.fillStyle = "rgba(226,220,198,0.9)";
        ctx.font = `900 ${Math.round(h * 0.22)}px "Arial Black", sans-serif`;
        ctx.textAlign = "center";
        ctx.fillText("FRAGILE", w / 2, h * 0.4);
        ctx.font = `700 ${Math.round(h * 0.15)}px "Courier New", monospace`;
        ctx.fillText("HOCKEY OPERATIONS", w / 2, h * 0.66);
        ctx.fillText("HAND CARRY ONLY", w / 2, h * 0.85);
        paintGrain(ctx, w, h, 0.08, 1200, 2);
      },
      { roughness: 0.85 }
    ),
  });

  // polishing cloth draped over the pedestal edge
  const cloth = artPanel(scene, "cup-polish-cloth", group, {
    width: 0.28,
    height: 0.3,
    columns: 8,
    rows: 8,
    relief: 0.02,
    reliefShape: (u, v) => Math.sin(u * Math.PI * 3) * Math.sin(v * Math.PI),
    at: [0.29, 0.305, 0.1],
    turn: [Math.PI / 2.1, 0.3, 0],
    material: materials.flat("polish-cloth", "#b9c2cc", { roughness: 0.92 }),
  });
  cloth.isPickable = false;

  // championship photograph leaning against the pedestal
  const photo = buildFramedPiece(scene, materials, group, {
    name: "cup-photo",
    width: 0.34,
    height: 0.26,
    depth: 0.03,
    moulding: 0.024,
    mat: 0.02,
    art: materials.art(
      "cup-photo",
      680,
      520,
      (ctx, w, h) => paintVintageHockeyPhoto(ctx, w, h, 41, "The night it came home"),
      { roughness: 0.5 }
    ),
  });
  photo.group.parent = group;
  photo.group.position.set(0.5, 0.16, 0.24);
  photo.group.rotation.set(-0.22, 0.7, 0);

  buildBrassPlaque(scene, materials, group, {
    name: "cup",
    width: 0.3,
    height: 0.09,
    at: [0, 0.2, 0.315],
    lines: ["CHAMPIONS", "DO NOT LIFT BY THE BOWL"],
  });

  contactShadow(scene, cache, group, { x: 0, z: 0, radius: 0.55, strength: 1 });
  contactShadow(scene, cache, group, {
    x: -0.62,
    z: -0.16,
    radius: 0.42,
    strength: 0.8,
  });

  const worldAnchor = new Vector3(at[0], at[1] + 0.95, at[2]);
  const spot = new SpotLight(
    "cup-spot",
    new Vector3(at[0] - 0.3, at[1] + 2.5, at[2] - 0.5),
    new Vector3(0, -1, 0),
    Math.PI / 3.4,
    8,
    scene
  );
  spot.setDirectionToTarget(worldAnchor);
  spot.diffuse = new Color3(1, 0.95, 0.86);
  spot.intensity = 14;
  spot.range = 5;
  spot.includedOnlyMeshes = group.getChildMeshes();
  cache.lights.push(spot);

  return { group, cup, focus: worldAnchor, meshes: cupMeshes };
}

/* --------------------------------------------------------------------------
   THE OFFICE DOOR
   -------------------------------------------------------------------------- */

function buildOfficeDoor(scene, materials, parent, cache) {
  const group = new TransformNode("office-door", scene);
  group.parent = parent;
  group.position.set(0, 0, HALL.doorZ);

  const doorWidth = 1.16;
  const doorHeight = 2.32;
  const casing = 0.1;

  // wall the door is set into, built as two returns and a header
  const sideWidth = (HALL.width - doorWidth - casing * 2) / 2;
  [-1, 1].forEach((side) => {
    solid(scene, `office-door-return${side}`, group, {
      width: sideWidth,
      height: HALL.height,
      depth: 0.22,
      at: [side * (doorWidth / 2 + casing + sideWidth / 2), HALL.height / 2, 0.11],
      material: materials.plaster(),
      collide: true,
    });
  });
  solid(scene, "office-door-header", group, {
    width: HALL.width,
    height: HALL.height - doorHeight - casing,
    depth: 0.22,
    at: [0, doorHeight + casing + (HALL.height - doorHeight - casing) / 2, 0.11],
    material: materials.plaster(),
    collide: true,
  });

  // walnut casing with a moulded profile
  [
    [casing, doorHeight + casing, -(doorWidth / 2 + casing / 2), (doorHeight + casing) / 2],
    [casing, doorHeight + casing, doorWidth / 2 + casing / 2, (doorHeight + casing) / 2],
    [doorWidth + casing * 2, casing, 0, doorHeight + casing / 2],
  ].forEach(([w, h, x, y], index) => {
    solid(scene, `office-door-casing${index}`, group, {
      width: w,
      height: h,
      depth: 0.06,
      at: [x, y, 0.03],
      material: materials.walnutDark(),
    });
    // inner bead
    solid(scene, `office-door-bead${index}`, group, {
      width: w * 0.32,
      height: h * (index === 2 ? 0.3 : 1),
      depth: 0.024,
      at: [x * 0.86, y, 0.072],
      material: materials.brass(),
    });
  });

  // pediment over the casing
  solid(scene, "office-door-pediment", group, {
    width: doorWidth + casing * 3.2,
    height: 0.06,
    depth: 0.1,
    at: [0, doorHeight + casing * 1.6, 0.05],
    material: materials.walnutDark(),
  });

  // the leaf itself, on a hinge node so it can swing
  const hinge = new TransformNode("office-door-hinge", scene);
  hinge.parent = group;
  hinge.position.set(-doorWidth / 2, 0, 0);

  const leaf = solid(scene, "office-door-leaf", hinge, {
    width: doorWidth,
    height: doorHeight,
    depth: 0.055,
    at: [doorWidth / 2, doorHeight / 2, 0],
    material: materials.walnut(),
    collide: true,
  });

  // raised panels and stiles
  const panels = [
    [0.4, 0.78, -0.22, 0.62],
    [0.4, 0.78, 0.22, 0.62],
    [0.4, 0.66, -0.22, 1.6],
    [0.4, 0.66, 0.22, 1.6],
  ];
  panels.forEach(([w, h, x, y], index) => {
    solid(scene, `office-door-panel${index}`, hinge, {
      width: w,
      height: h,
      depth: 0.016,
      at: [doorWidth / 2 + x, y, -0.032],
      material: materials.walnutDark(),
    });
    solid(scene, `office-door-panel-bead${index}`, hinge, {
      width: w + 0.03,
      height: 0.012,
      depth: 0.02,
      at: [doorWidth / 2 + x, y + h / 2 + 0.012, -0.03],
      material: materials.walnutDark(),
    });
  });

  // brass hardware
  const lever = new TransformNode("office-door-lever", scene);
  lever.parent = hinge;
  lever.position.set(doorWidth - 0.11, 1.05, -0.04);
  rod(scene, "office-door-rose", lever, {
    diameter: 0.075,
    height: 0.014,
    sides: 18,
    at: [0, 0, 0],
    turn: [Math.PI / 2, 0, 0],
    material: materials.brass(),
  });
  rod(scene, "office-door-lever-arm", lever, {
    diameter: 0.019,
    height: 0.125,
    sides: 12,
    at: [-0.05, 0, -0.03],
    turn: [0, 0, Math.PI / 2],
    material: materials.brass(),
  });
  rod(scene, "office-door-lever-return", lever, {
    diameter: 0.019,
    height: 0.04,
    sides: 12,
    at: [-0.105, 0, -0.048],
    turn: [Math.PI / 2, 0, 0],
    material: materials.brass(),
  });
  rod(scene, "office-door-escutcheon", lever, {
    diameter: 0.032,
    height: 0.012,
    sides: 14,
    at: [0, -0.11, 0],
    turn: [Math.PI / 2, 0, 0],
    material: materials.brass(),
  });

  // kickplate and hinges
  solid(scene, "office-door-kickplate", hinge, {
    width: doorWidth - 0.08,
    height: 0.19,
    depth: 0.012,
    at: [doorWidth / 2, 0.12, -0.034],
    material: materials.brass(),
  });
  [0.34, 1.16, 1.98].forEach((y, index) => {
    solid(scene, `office-door-hinge-leaf${index}`, hinge, {
      width: 0.03,
      height: 0.11,
      depth: 0.062,
      at: [0.014, y, 0],
      material: materials.brass(),
    });
  });

  // understated signage
  const sign = materials.art(
    "office-door-sign",
    620,
    170,
    (ctx, w, h) =>
      paintEtchedBrassPlate(ctx, w, h, [
        { text: HALL_FUN_LABELS.officeSign, scale: 1.05 },
        "Executive Suite",
      ]),
    { metallic: 0.72, roughness: 0.28, environmentIntensity: 0.75, bump: 0.3 }
  );
  solid(scene, "office-door-sign-body", hinge, {
    width: 0.34,
    height: 0.095,
    depth: 0.01,
    at: [doorWidth / 2, 1.72, -0.032],
    material: materials.brass(),
  });
  artPanel(scene, "office-door-sign-face", hinge, {
    width: 0.32,
    height: 0.085,
    at: [doorWidth / 2, 1.72, -0.038],
    turn: [0, Math.PI, 0],
    material: sign,
  });

  // warm light leaking around the edges — this is the transition point
  const leakMaterial = materials.warmLeak();
  const leaks = [
    [doorWidth - 0.02, 0.014, doorWidth / 2, 0.007],
    [0.012, doorHeight - 0.04, doorWidth - 0.005, doorHeight / 2],
    [0.012, doorHeight - 0.04, 0.005, doorHeight / 2],
    [doorWidth - 0.02, 0.01, doorWidth / 2, doorHeight - 0.005],
  ];
  const leakMeshes = leaks.map(([w, h, x, y], index) =>
    solid(scene, `office-door-leak${index}`, hinge, {
      width: w,
      height: h,
      depth: 0.006,
      at: [x, y, 0.026],
      material: leakMaterial,
      receiveShadows: false,
    })
  );

  // spill on the runner in front of the sill
  const spill = artPanel(scene, "office-door-spill", group, {
    width: HALL.runnerWidth * 0.94,
    height: 1.1,
    at: [0, 0.012, -0.56],
    turn: [-Math.PI / 2, 0, 0],
    material: materials.art(
      "door-spill",
      256,
      256,
      (ctx, w, h) => {
        ctx.clearRect(0, 0, w, h);
        const gradient = ctx.createLinearGradient(0, 0, 0, h);
        gradient.addColorStop(0, "rgba(255,196,120,0.5)");
        gradient.addColorStop(0.55, "rgba(255,180,100,0.16)");
        gradient.addColorStop(1, "rgba(255,170,90,0)");
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, w, h);
        const fade = ctx.createLinearGradient(0, 0, w, 0);
        fade.addColorStop(0, "rgba(0,0,0,1)");
        fade.addColorStop(0.16, "rgba(0,0,0,0)");
        fade.addColorStop(0.84, "rgba(0,0,0,0)");
        fade.addColorStop(1, "rgba(0,0,0,1)");
        ctx.globalCompositeOperation = "destination-out";
        ctx.fillStyle = fade;
        ctx.fillRect(0, 0, w, h);
        ctx.globalCompositeOperation = "source-over";
      },
      { clamp: true, hasAlpha: true }
    ),
  });
  spill.material.albedoTexture.hasAlpha = true;
  spill.material.useAlphaFromAlbedoTexture = true;
  spill.material.transparencyMode = 2;
  spill.material.emissiveTexture = spill.material.albedoTexture;
  spill.material.emissiveIntensity = 1.1;
  spill.isPickable = false;

  const doorLight = new PointLight(
    "office-door-glow",
    new Vector3(0, 1.1, HALL.doorZ - 0.5),
    scene
  );
  doorLight.diffuse = new Color3(1, 0.78, 0.52);
  doorLight.intensity = 2.4;
  doorLight.range = 4.4;
  doorLight.includedOnlyMeshes = [
    ...group.getChildMeshes(),
  ];
  cache.lights.push(doorLight);

  return {
    group,
    hinge,
    leaf,
    leakMeshes,
    light: doorLight,
    open(t) {
      hinge.rotation.y = lerp(0, -1.46, t);
    },
  };
}

/* --------------------------------------------------------------------------
   THE JOKE, TAKEN SERIOUSLY
   -------------------------------------------------------------------------- */

function buildDartStation(scene, materials, parent, cache) {
  const mount = wallMount(scene, parent, 1, HALL.tkachukZ, 0);
  const group = new TransformNode("dart-station", scene);
  group.parent = mount;

  // the protest poster, unframed and slightly crooked on purpose
  const posterMaterial = materials.art(
    "tkachuk-poster",
    768,
    1024,
    paintTkachukProtestPoster,
    { roughness: 0.88, emissive: 0.22 }
  );
  const poster = artPanel(scene, "tkachuk-poster", group, {
    width: 1.12,
    height: 1.46,
    at: [0.02, 1.92, 0.012],
    turn: [0, 0, -0.035],
    material: posterMaterial,
  });
  poster.isPickable = true;
  // it is taped, not mounted — a thin curl at the bottom corner
  solid(scene, "tkachuk-poster-curl", group, {
    width: 0.14,
    height: 0.1,
    depth: 0.004,
    at: [0.4, 1.28, 0.03],
    turn: [0.4, 0, -0.5],
    material: posterMaterial,
  });

  // target board below it
  const boardGroup = new TransformNode("dart-board", scene);
  boardGroup.parent = group;
  boardGroup.position.set(0, 1.02, 0);

  solid(scene, "dart-board-backer", boardGroup, {
    width: 0.66,
    height: 0.66,
    depth: 0.05,
    at: [0, 0, 0.025],
    material: materials.leather(),
  });
  [
    [0.7, 0.03, 0, 0.345],
    [0.7, 0.03, 0, -0.345],
    [0.03, 0.66, 0.345, 0],
    [0.03, 0.66, -0.345, 0],
  ].forEach(([w, h, x, y], index) => {
    solid(scene, `dart-board-trim${index}`, boardGroup, {
      width: w,
      height: h,
      depth: 0.06,
      at: [x, y, 0.03],
      material: materials.walnutDark(),
    });
  });

  const targetFace = artPanel(scene, "dart-board-face", boardGroup, {
    width: 0.62,
    height: 0.62,
    at: [0, 0, 0.052],
    material: materials.art(
      "dart-target",
      768,
      768,
      paintDartTargetPoster,
      { roughness: 0.8 }
    ),
  });
  targetFace.isPickable = true;

  // dart tray on a small walnut shelf
  const shelf = solid(scene, "dart-tray-shelf", group, {
    width: 0.44,
    height: 0.03,
    depth: 0.16,
    at: [0, 0.62, 0.08],
    material: materials.walnutDark(),
  });
  void shelf;
  [-1, 1].forEach((side) => {
    solid(scene, `dart-tray-bracket${side}`, group, {
      width: 0.028,
      height: 0.12,
      depth: 0.1,
      at: [side * 0.19, 0.56, 0.05],
      turn: [-0.5, 0, 0],
      material: materials.brass(),
    });
  });
  const tray = solid(scene, "dart-tray", group, {
    width: 0.34,
    height: 0.035,
    depth: 0.11,
    at: [0, 0.652, 0.08],
    material: materials.blackMetal(),
  });
  void tray;
  solid(scene, "dart-tray-felt", group, {
    width: 0.31,
    height: 0.006,
    depth: 0.09,
    at: [0, 0.671, 0.08],
    material: materials.runner(),
  });

  buildBrassPlaque(scene, materials, group, {
    name: "dart",
    width: 0.28,
    height: 0.062,
    at: [0, 0.5, 0.02],
    lines: ["HOUSE RULES — THREE PER VISIT"],
  });

  /*
    Dart pool. Darts are recycled so repeated throws never grow the scene, and
    each landing gets a small offset and roll so the board never looks stamped.
  */
  const dartMaterialBody = materials.flat("dart-body", "#1b1d22", {
    metallic: 0.7,
    roughness: 0.34,
  });
  const dartMaterialFlight = materials.flat("dart-flight", "#b32036", {
    roughness: 0.7,
  });
  const dartTip = materials.steel();

  const darts = [];
  const DART_POOL = 18;
  for (let i = 0; i < DART_POOL; i += 1) {
    const dart = new TransformNode(`dart-${i}`, scene);
    dart.parent = parent;
    dart.setEnabled(false);

    rod(scene, `dart-${i}-tip`, dart, {
      top: 0.001,
      bottom: 0.006,
      height: 0.048,
      sides: 8,
      at: [0, 0, 0.024],
      turn: [Math.PI / 2, 0, 0],
      material: dartTip,
    });
    rod(scene, `dart-${i}-barrel`, dart, {
      top: 0.009,
      bottom: 0.007,
      height: 0.062,
      sides: 10,
      at: [0, 0, -0.031],
      turn: [Math.PI / 2, 0, 0],
      material: dartMaterialBody,
    });
    rod(scene, `dart-${i}-shaft`, dart, {
      diameter: 0.005,
      height: 0.04,
      sides: 6,
      at: [0, 0, -0.082],
      turn: [Math.PI / 2, 0, 0],
      material: dartTip,
    });
    [0, Math.PI / 2].forEach((roll, index) => {
      artPanel(scene, `dart-${i}-flight${index}`, dart, {
        width: 0.036,
        height: 0.05,
        at: [0, 0, -0.115],
        turn: [0, Math.PI / 2, roll],
        material: dartMaterialFlight,
      });
    });

    darts.push(dart);
  }

  cache.darts = {
    pool: darts,
    next: 0,
    board: targetFace,
    poster,
    thrown: 0,
  };

  const boardLight = new PointLight(
    "dart-board-light",
    new Vector3(HALL.width / 2 - 0.55, 1.95, HALL.tkachukZ),
    scene
  );
  boardLight.diffuse = new Color3(1, 0.82, 0.7);
  boardLight.intensity = 7.5;
  boardLight.range = 5.5;
  boardLight.includedOnlyMeshes = [
    ...group.getChildMeshes(),
    ...darts.flatMap((dart) => dart.getChildMeshes()),
  ];
  cache.lights.push(boardLight);

  return { group, mount, board: targetFace, darts: cache.darts };
}

function throwDart(scene, cache, camera) {
  const store = cache.darts;
  if (!store) return false;

  const origin = camera.position.clone();
  const forward = camera.getForwardRay(1).direction.clone();
  // a little human inaccuracy, and a little more of it every throw
  const drift = Math.min(0.05, 0.012 + store.thrown * 0.004);
  forward.x += (Math.random() - 0.5) * drift;
  forward.y += (Math.random() - 0.5) * drift;
  forward.normalize();

  const ray = new Ray(origin, forward, 14);
  const hit = scene.pickWithRay(ray, (mesh) => mesh?.isPickable === true);

  const dart = store.pool[store.next % store.pool.length];
  store.next += 1;
  store.thrown += 1;

  const landing = hit?.hit
    ? hit.pickedPoint.clone()
    : origin.add(forward.scale(6));

  // stand off the surface so the barrel is not buried
  landing.subtractInPlace(forward.scale(0.03));

  const start = origin.add(forward.scale(0.5)).add(new Vector3(0, -0.08, 0));
  dart.setEnabled(true);
  dart.position.copyFrom(start);
  dart.rotation.set(0, Math.atan2(forward.x, forward.z), 0);

  const flightTime = 200 + Math.random() * 90;
  const startedAt = performance.now();
  const spin = (Math.random() - 0.5) * 1.4;

  const observer = scene.onBeforeRenderObservable.add(() => {
    const t = clamp((performance.now() - startedAt) / flightTime, 0, 1);
    const eased = easeOutCubic(t);
    dart.position.set(
      lerp(start.x, landing.x, eased),
      lerp(start.y, landing.y, eased) - Math.sin(t * Math.PI) * 0.035,
      lerp(start.z, landing.z, eased)
    );
    dart.rotation.z = lerp(spin, spin * 0.15, eased);
    if (t >= 1) {
      scene.onBeforeRenderObservable.remove(observer);
      dart.rotation.x = (Math.random() - 0.5) * 0.22;
      dart.rotation.z = (Math.random() - 0.5) * 0.5;
    }
  });

  return Boolean(
    hit?.hit &&
      (hit.pickedMesh === store.board || hit.pickedMesh === store.poster)
  );
}

/* --------------------------------------------------------------------------
   CORRIDOR SHELL
   -------------------------------------------------------------------------- */

function buildCorridorShell(scene, materials, parent, cache) {
  const shell = new TransformNode("corridor-shell", scene);
  shell.parent = parent;

  const length = HALL.doorZ - HALL.startZ;
  const centreZ = (HALL.doorZ + HALL.startZ) / 2;
  const wainscot = 1.06;
  const rail = 0.06;
  const base = 0.16;
  const crown = 0.14;

  // floor
  const floor = MeshBuilder.CreateGround(
    "corridor-floor",
    { width: HALL.width, height: length + 1.4 },
    scene
  );
  place(floor, shell, {
    at: [0, 0, centreZ - 0.4],
    material: materials.stone(),
    collide: true,
    pickable: true,
  });

  // the runner that pulls the eye toward the office
  const runner = MeshBuilder.CreateGround(
    "corridor-runner",
    { width: HALL.runnerWidth, height: length + 0.2 },
    scene
  );
  place(runner, shell, {
    at: [0, 0.012, centreZ - 0.1],
    material: materials.runner(),
  });
  // bound edge, slightly raised
  [-1, 1].forEach((side) => {
    solid(scene, `corridor-runner-edge${side}`, shell, {
      width: 0.035,
      height: 0.014,
      depth: length + 0.2,
      at: [side * (HALL.runnerWidth / 2 - 0.014), 0.012, centreZ - 0.1],
      material: materials.walnutDark(),
    });
  });

  // ceiling
  const ceiling = solid(scene, "corridor-ceiling", shell, {
    width: HALL.width,
    height: 0.14,
    depth: length + 1.4,
    at: [0, HALL.height + 0.07, centreZ - 0.4],
    material: materials.plaster(),
    collide: true,
  });
  void ceiling;

  // walls in segments: better texel density and better culling
  const segments = 7;
  const segmentLength = (length + 1.4) / segments;
  for (let i = 0; i < segments; i += 1) {
    const z = HALL.startZ - 1.4 + segmentLength * (i + 0.5);
    [-1, 1].forEach((side) => {
      const x = side * (HALL.width / 2 + 0.06);

      solid(scene, `corridor-wall${side}-${i}`, shell, {
        width: 0.12,
        height: HALL.height,
        depth: segmentLength,
        at: [x, HALL.height / 2, z],
        material: materials.plaster(),
        collide: true,
        // pickable so stray darts find plaster instead of hanging in the air
        pickable: true,
      });

      // walnut wainscot panelling
      solid(scene, `corridor-wainscot${side}-${i}`, shell, {
        width: 0.05,
        height: wainscot,
        depth: segmentLength,
        at: [side * (HALL.width / 2 - 0.024), wainscot / 2, z],
        material: materials.walnut(),
      });

      // recessed panels within the wainscot
      const panelCount = 2;
      for (let p = 0; p < panelCount; p += 1) {
        const pz = z - segmentLength / 2 + segmentLength * ((p + 0.5) / panelCount);
        solid(scene, `corridor-panel${side}-${i}-${p}`, shell, {
          width: 0.02,
          height: wainscot - 0.34,
          depth: segmentLength / panelCount - 0.16,
          at: [side * (HALL.width / 2 - 0.052), wainscot / 2 - 0.03, pz],
          material: materials.walnutDark(),
        });
      }

      // chair rail
      solid(scene, `corridor-rail${side}-${i}`, shell, {
        width: 0.07,
        height: rail,
        depth: segmentLength,
        at: [side * (HALL.width / 2 - 0.032), wainscot + rail / 2, z],
        material: materials.walnutDark(),
      });

      // baseboard
      solid(scene, `corridor-base${side}-${i}`, shell, {
        width: 0.075,
        height: base,
        depth: segmentLength,
        at: [side * (HALL.width / 2 - 0.034), base / 2, z],
        material: materials.walnutDark(),
      });
      solid(scene, `corridor-base-cap${side}-${i}`, shell, {
        width: 0.09,
        height: 0.014,
        depth: segmentLength,
        at: [side * (HALL.width / 2 - 0.04), base, z],
        material: materials.walnutDark(),
      });

      // crown moulding
      solid(scene, `corridor-crown${side}-${i}`, shell, {
        width: 0.09,
        height: crown,
        depth: segmentLength,
        at: [side * (HALL.width / 2 - 0.04), HALL.height - crown / 2, z],
        material: materials.walnutDark(),
      });
      solid(scene, `corridor-crown-bead${side}-${i}`, shell, {
        width: 0.11,
        height: 0.014,
        depth: segmentLength,
        at: [side * (HALL.width / 2 - 0.05), HALL.height - crown, z],
        material: materials.brass(),
      });
    });

    // coffered ceiling beams
    solid(scene, `corridor-beam-${i}`, shell, {
      width: HALL.width,
      height: 0.09,
      depth: 0.13,
      at: [0, HALL.height - 0.045, z - segmentLength / 2],
      material: materials.walnutDark(),
    });
  }

  // end wall behind the player, so turning round is not a void
  solid(scene, "corridor-end-wall", shell, {
    width: HALL.width + 0.24,
    height: HALL.height,
    depth: 0.14,
    at: [0, HALL.height / 2, HALL.startZ - 1.4],
    material: materials.plaster(),
    collide: true,
    pickable: true,
  });
  solid(scene, "corridor-end-wainscot", shell, {
    width: HALL.width,
    height: wainscot,
    depth: 0.04,
    at: [0, wainscot / 2, HALL.startZ - 1.34],
    material: materials.walnut(),
  });
  solid(scene, "corridor-end-rail", shell, {
    width: HALL.width,
    height: rail,
    depth: 0.06,
    at: [0, wainscot + rail / 2, HALL.startZ - 1.33],
    material: materials.walnutDark(),
  });
  // brushed steel lift doors on the back wall
  [-1, 1].forEach((side) => {
    solid(scene, `corridor-lift-door${side}`, shell, {
      width: 0.62,
      height: 2.16,
      depth: 0.04,
      at: [side * 0.32, 1.08, HALL.startZ - 1.32],
      material: materials.steel(),
    });
  });
  solid(scene, "corridor-lift-surround", shell, {
    width: 1.42,
    height: 0.07,
    depth: 0.07,
    at: [0, 2.2, HALL.startZ - 1.31],
    material: materials.brass(),
  });

  // ceiling services: recessed housings, vents, sprinkler heads
  const lights = [];
  const recessedZ = [];
  for (let i = 0; i < 8; i += 1) {
    const z = HALL.startZ - 0.4 + i * ((length + 0.2) / 8);
    recessedZ.push(z);

    solid(scene, `corridor-recess-housing-${i}`, shell, {
      width: 0.34,
      height: 0.05,
      depth: 0.34,
      at: [0, HALL.height - 0.03, z],
      material: materials.blackMetal(),
    });
    const lens = artPanel(scene, `corridor-recess-lens-${i}`, shell, {
      width: 0.26,
      height: 0.26,
      at: [0, HALL.height - 0.056, z],
      turn: [Math.PI / 2, 0, 0],
      material: materials.lampLens(),
    });
    lens.isPickable = false;

    // sprinkler head and a small vent, offset so the ceiling is not a pattern
    rod(scene, `corridor-sprinkler-${i}`, shell, {
      diameter: 0.02,
      height: 0.05,
      sides: 8,
      at: [0.62, HALL.height - 0.04, z + 0.5],
      material: materials.brass(),
    });
  }

  // practical ceiling lights spaced along the run
  [0.1, 0.28, 0.46, 0.64, 0.82].forEach((t, index) => {
    const light = new PointLight(
      `corridor-practical-${index}`,
      new Vector3(0, HALL.height - 0.2, HALL.startZ + length * t),
      scene
    );
    light.diffuse = new Color3(1, 0.9, 0.76);
    light.specular = new Color3(1, 0.95, 0.86);
    light.intensity = 5.4;
    light.range = 12;
    lights.push(light);
    cache.lights.push(light);
  });

  const hallHemi = new HemisphericLight(
    "corridor-hemi",
    new Vector3(0.08, 1, 0.18),
    scene
  );
  hallHemi.intensity = 1.4;
  hallHemi.diffuse = new Color3(1, 0.95, 0.88);
  hallHemi.groundColor = new Color3(0.42, 0.38, 0.34);
  lights.push(hallHemi);
  cache.lights.push(hallHemi);

  // HVAC grilles, louvres modelled rather than painted
  [-1, 1].forEach((side) => {
    [-16.8, -9.9, -2.4].forEach((z, index) => {
      const vent = new TransformNode(`corridor-vent${side}-${index}`, scene);
      vent.parent = shell;
      vent.position.set(side * (HALL.width / 2 - 0.03), HALL.height - 0.36, z);
      vent.rotation.y = side < 0 ? Math.PI / 2 : -Math.PI / 2;
      solid(scene, `corridor-vent${side}-${index}-frame`, vent, {
        width: 0.42,
        height: 0.22,
        depth: 0.02,
        at: [0, 0, 0.01],
        material: materials.blackMetal(),
      });
      for (let l = 0; l < 6; l += 1) {
        solid(scene, `corridor-vent${side}-${index}-louvre${l}`, vent, {
          width: 0.38,
          height: 0.018,
          depth: 0.026,
          at: [0, 0.085 - l * 0.032, 0.022],
          turn: [-0.5, 0, 0],
          material: materials.steel(),
        });
      }
    });
  });

  // wall sconces between the exhibits
  [-18.6, -12.8, -7.0, -1.4].forEach((z, index) => {
    [-1, 1].forEach((side) => {
      const sconce = new TransformNode(`corridor-sconce${side}-${index}`, scene);
      sconce.parent = shell;
      sconce.position.set(side * (HALL.width / 2 - 0.03), 2.12, z);
      sconce.rotation.y = side < 0 ? Math.PI / 2 : -Math.PI / 2;

      solid(scene, `corridor-sconce${side}-${index}-plate`, sconce, {
        width: 0.1,
        height: 0.26,
        depth: 0.02,
        at: [0, 0, 0.01],
        material: materials.brass(),
      });
      rod(scene, `corridor-sconce${side}-${index}-arm`, sconce, {
        diameter: 0.018,
        height: 0.14,
        sides: 10,
        at: [0, -0.02, 0.07],
        turn: [Math.PI / 2.6, 0, 0],
        material: materials.brass(),
      });
      const shade = rod(scene, `corridor-sconce${side}-${index}-shade`, sconce, {
        top: 0.11,
        bottom: 0.075,
        height: 0.16,
        sides: 16,
        at: [0, 0.11, 0.13],
        material: materials.flat("sconce-shade", "#d8cbaa", {
          roughness: 0.8,
          emissive: "#ffcf94",
          emissiveIntensity: 2.1,
        }),
      });
      shade.isPickable = false;

      const sconceLight = new PointLight(
        `corridor-sconce-light${side}-${index}`,
        new Vector3(side * (HALL.width / 2 - 0.22), 2.12, z),
        scene
      );
      sconceLight.diffuse = new Color3(1, 0.88, 0.7);
      sconceLight.intensity = 2.4;
      sconceLight.range = 6.2;
      cache.lights.push(sconceLight);
    });
  });

  cache.recessedZ = recessedZ;
  return { shell, lights };
}

/* --------------------------------------------------------------------------
   ASSEMBLY
   -------------------------------------------------------------------------- */

function buildMemorabiliaHall(scene, materials) {
  const root = new TransformNode("puckception-hall", scene);
  const cache = { lights: [], interactables: [] };

  buildCorridorShell(scene, materials, root, cache);

  const register = (entry) => {
    cache.interactables.push(entry);
    return entry;
  };

  /* Hero exhibits -------------------------------------------------------- */

  const karlsson = buildKarlssonExhibit(scene, materials, root, cache);
  register({
    id: "karlsson",
    label: "Autographed game-worn sweater",
    card: EXHIBIT_CARDS.karlsson,
    point: karlsson.focus,
    radius: 2.5,
  });

  const ovechkin = buildOvechkinExhibit(scene, materials, root, cache);
  register({
    id: "ovechkin",
    label: "Rookie-era Capitals sweater",
    card: EXHIBIT_CARDS.ovechkin,
    point: ovechkin.focus,
    radius: 2.5,
  });

  /* Ticket stubs on a narrow entrance ledge ----------------------------- */

  const stubLedge = new TransformNode("ticket-ledge", scene);
  stubLedge.parent = root;
  stubLedge.position.set(-HALL.width / 2 + 0.22, 0, HALL.startZ + 1.55);
  solid(scene, "ticket-ledge-board", stubLedge, {
    width: 0.18,
    height: 0.028,
    depth: 0.62,
    at: [0, 1.08, 0],
    material: materials.walnutDark(),
  });
  [-1, 1].forEach((side) => {
    solid(scene, `ticket-ledge-bracket${side}`, stubLedge, {
      width: 0.02,
      height: 0.12,
      depth: 0.08,
      at: [0.02, 1.02, side * 0.22],
      turn: [0, 0, 0.55],
      material: materials.brass(),
    });
  });
  [-0.2, 0.0, 0.2].forEach((z, index) => {
    const stub = artPanel(scene, `ticket-stub-${index}`, stubLedge, {
      width: 0.14,
      height: 0.08,
      at: [0.01, 1.118, z],
      turn: [-Math.PI / 2, 0, (hashRandom(index * 9.1) - 0.5) * 0.4],
      material: materials.art(
        `ticket-stub-${index}`,
        420,
        240,
        (ctx, w, h) =>
          paintVintageHockeyPhoto(
            ctx,
            w,
            h,
            70 + index * 11,
            ["SEC 12", "PRESS", "GAME 1"][index]
          ),
        { roughness: 0.86 }
      ),
    });
    void stub;
  });

  /* Archive wall of framed pieces ---------------------------------------- */

  const photoCaptions = [
    "Overtime, 1971",
    "The old barn",
    "Morning skate",
    "Third period",
    "Warm-up, away sweaters",
    "Outdoor game",
    "Game 7 stub",
    "Press door",
    "Sec 112 · Row C",
  ];

  const framedPlan = [
    { side: -1, z: -21.45, y: 1.7, w: 0.38, h: 0.48, kind: "photo", seed: 41 },
    { side: -1, z: -20.75, y: 1.34, w: 0.26, h: 0.2, kind: "photo", seed: 43 },
    { side: -1, z: -16.85, y: 1.78, w: 0.5, h: 0.36, kind: "program" },
    { side: -1, z: -15.55, y: 1.5, w: 0.34, h: 0.42, kind: "photo", seed: 47 },
    { side: -1, z: -14.35, y: 1.86, w: 0.46, h: 0.3, kind: "credentials" },
    { side: 1, z: -21.2, y: 1.68, w: 0.42, h: 0.52, kind: "photo", seed: 53 },
    { side: 1, z: -20.4, y: 1.32, w: 0.24, h: 0.18, kind: "photo", seed: 59, tilt: 0.09 },
    { side: 1, z: -16.35, y: 1.82, w: 0.52, h: 0.4, kind: "newspaper" },
    { side: 1, z: -15.1, y: 1.46, w: 0.36, h: 0.44, kind: "photo", seed: 61 },
    { side: 1, z: -13.85, y: 1.74, w: 0.4, h: 0.28, kind: "banner" },
    { side: -1, z: -12.6, y: 1.72, w: 0.42, h: 0.52, kind: "photo", seed: 3 },
    { side: -1, z: -12.0, y: 1.42, w: 0.34, h: 0.28, kind: "photo", seed: 7 },
    { side: -1, z: -8.6, y: 1.9, w: 0.56, h: 0.42, kind: "credentials" },
    { side: -1, z: -7.7, y: 1.5, w: 0.4, h: 0.5, kind: "photo", seed: 11 },
    { side: -1, z: -6.2, y: 1.78, w: 0.6, h: 0.78, kind: "newspaper" },
    { side: -1, z: -5.1, y: 1.44, w: 0.3, h: 0.24, kind: "photo", seed: 19, tilt: 0.11 },
    { side: -1, z: -4.2, y: 1.82, w: 0.68, h: 0.44, kind: "rink" },
    { side: -1, z: -2.6, y: 1.6, w: 0.44, h: 0.3, kind: "pins" },
    { side: -1, z: -1.7, y: 1.86, w: 0.38, h: 0.48, kind: "photo", seed: 23 },
    { side: 1, z: -12.9, y: 1.66, w: 0.46, h: 0.34, kind: "photo", seed: 29 },
    { side: 1, z: -12.2, y: 2.02, w: 0.32, h: 0.4, kind: "photo", seed: 31 },
    { side: 1, z: -11.1, y: 1.5, w: 0.5, h: 0.62, kind: "program" },
    { side: 1, z: -7.6, y: 1.88, w: 0.62, h: 0.44, kind: "banner" },
    { side: 1, z: -6.5, y: 1.42, w: 0.36, h: 0.44, kind: "photo", seed: 37 },
    { side: 1, z: -5.4, y: 1.78, w: 0.52, h: 0.36, kind: "credentials" },
    { side: 1, z: -1.9, y: 1.52, w: 0.34, h: 0.42, kind: "note" },
  ];

  framedPlan.forEach((item, index) => {
    const mount = wallMount(scene, root, item.side, item.z, 0);
    let art;

    if (item.kind === "photo") {
      art = materials.art(
        `photo-${index}`,
        Math.round(item.w * 900),
        Math.round(item.h * 900),
        (ctx, w, h) =>
          paintVintageHockeyPhoto(
            ctx,
            w,
            h,
            item.seed,
            photoCaptions[index % photoCaptions.length]
          ),
        { roughness: 0.52, environmentIntensity: 0.1 }
      );
    } else if (item.kind === "credentials") {
      art = materials.art(
        `credentials-${index}`,
        900,
        640,
        paintCredentialBoard,
        { roughness: 0.7 }
      );
    } else if (item.kind === "newspaper") {
      art = materials.art(
        `frontpage-${index}`,
        820,
        1080,
        paintChampionshipFrontPage,
        { roughness: 0.82 }
      );
    } else if (item.kind === "rink") {
      art = materials.art(`rink-${index}`, 1024, 660, paintRinkDiagram, {
        roughness: 0.72,
      });
    } else if (item.kind === "pins") {
      art = materials.art(`pins-${index}`, 900, 620, paintPinBoard, {
        roughness: 0.3,
        metallic: 0.4,
        environmentIntensity: 0.5,
      });
    } else if (item.kind === "program") {
      art = materials.art(`program-${index}`, 720, 900, paintProgramCover, {
        roughness: 0.8,
      });
    } else if (item.kind === "banner") {
      art = materials.art(
        `banner-${index}`,
        900,
        640,
        (ctx, w, h) => paintMiniBanner(ctx, w, h, "LEAGUE", "TITLE", "#1a1f2e"),
        { roughness: 0.9 }
      );
    } else {
      art = materials.art(`note-${index}`, 720, 900, paintScratchedNote, {
        roughness: 0.88,
      });
    }

    const piece = buildFramedPiece(scene, materials, mount, {
      name: `archive-${index}`,
      width: item.w,
      height: item.h,
      art,
      tilt: item.tilt || (hashRandom(index * 5.3) - 0.5) * 0.012,
      depth: item.kind === "pins" ? 0.05 : 0.038,
      moulding: item.kind === "newspaper" ? 0.03 : 0.034,
    });
    piece.group.position.y = item.y;
  });

  register({
    id: "photos",
    label: "Archive wall",
    card: EXHIBIT_CARDS.photos,
    point: new Vector3(-HALL.width / 2 + 0.2, 1.7, -6.2),
    radius: 2.1,
  });

  /* Pennants on the right, hung from a brass rail ------------------------ */

  const pennantMount = wallMount(scene, root, 1, -10.3, 0);
  const pennantRail = rod(scene, "pennant-rail", pennantMount, {
    diameter: 0.02,
    height: 1.5,
    sides: 12,
    at: [0, 2.42, 0.07],
    turn: [0, 0, Math.PI / 2],
    material: materials.brass(),
  });
  void pennantRail;
  ["OTT", "1967", "CUP"].forEach((label, index) => {
    const tone = ["#7a1626", "#1a2b52", "#204028"][index];
    const pennantMaterial = materials.art(
      `pennant-${index}`,
      640,
      240,
      (ctx, w, h) => paintPennant(ctx, w, h, label, tone),
      { roughness: 0.92 }
    );
    const pennant = createFacePanel(scene, `pennant-${index}`, {
      width: 0.44,
      height: 0.18,
      columns: 10,
      rows: 4,
      relief: 0.02,
      reliefShape: (u) => Math.sin(u * Math.PI * 2.4),
    });
    pennant.parent = pennantMount;
    pennant.position.set(-0.5 + index * 0.5, 2.3, 0.075);
    pennant.rotation.z = (hashRandom(index * 7.1) - 0.5) * 0.09;
    pennant.material = pennantMaterial;
  });

  /* Signed pucks in cases on a walnut console --------------------------- */

  const consoleGroup = new TransformNode("hall-console", scene);
  consoleGroup.parent = root;
  consoleGroup.position.set(-HALL.width / 2 + 0.3, 0, -8.5);

  solid(scene, "hall-console-top", consoleGroup, {
    width: 0.5,
    height: 0.04,
    depth: 1.5,
    at: [0, 0.86, 0],
    material: materials.walnut(),
    collide: true,
  });
  solid(scene, "hall-console-apron", consoleGroup, {
    width: 0.44,
    height: 0.09,
    depth: 1.42,
    at: [0, 0.79, 0],
    material: materials.walnutDark(),
  });
  [[-0.6], [0.6]].forEach(([z], index) => {
    [-1, 1].forEach((side) => {
      rod(scene, `hall-console-leg-${index}-${side}`, consoleGroup, {
        top: 0.03,
        bottom: 0.04,
        height: 0.76,
        sides: 10,
        at: [side * 0.18, 0.38, z],
        material: materials.walnutDark(),
      });
    });
  });
  contactShadow(scene, cache, consoleGroup, { x: 0, z: 0, radius: 0.9, strength: 0.7 });

  buildDisplayCase(scene, materials, consoleGroup, {
    name: "pucks-a",
    at: [0, 0.9, -0.42],
    width: 0.36,
    height: 0.2,
    depth: 0.22,
    pucks: 2,
  });
  buildDisplayCase(scene, materials, consoleGroup, {
    name: "pucks-b",
    at: [0, 0.9, 0.28],
    width: 0.24,
    height: 0.18,
    depth: 0.2,
    pucks: 1,
  });
  buildStopwatchAndWhistle(scene, materials, consoleGroup, { at: [0.02, 0.9, 0.62] });
  buildTapeRolls(scene, materials, consoleGroup, { at: [-0.02, 0.9, -0.66] });

  // the plaque that lies about the console being untouchable
  buildBrassPlaque(scene, materials, wallMount(scene, root, -1, -8.5, 0), {
    name: "do-not-touch",
    width: 0.24,
    height: 0.062,
    at: [0, 1.14, 0.02],
    lines: [HALL_FUN_LABELS.doNotTouch],
  });

  register({
    id: "equipment",
    label: "Signed pucks and timing kit",
    card: EXHIBIT_CARDS.equipment,
    point: new Vector3(-HALL.width / 2 + 0.5, 1.0, -8.5),
    radius: 1.9,
  });

  /* Sticks, pads and gloves leaning in the corner ----------------------- */

  const equipmentCorner = new TransformNode("equipment-corner", scene);
  equipmentCorner.parent = root;
  equipmentCorner.position.set(HALL.width / 2 - 0.34, 0, -6.0);

  buildGoaliePads(scene, materials, equipmentCorner, {
    at: [0.06, 0.04, 0.5],
    turn: [-0.14, 0.24, 0],
  });
  buildStick(scene, materials, equipmentCorner, {
    name: "wood",
    at: [-0.04, 0, -0.1],
    turn: [-0.1, 0.3, 0.16],
    length: 1.66,
    wooden: true,
  });
  buildStick(scene, materials, equipmentCorner, {
    name: "composite-a",
    at: [0.04, 0, -0.24],
    turn: [-0.09, -0.2, 0.11],
    length: 1.7,
  });
  buildStick(scene, materials, equipmentCorner, {
    name: "composite-b",
    at: [0.12, 0, -0.38],
    turn: [-0.12, 0.12, 0.2],
    length: 1.68,
  });
  buildStick(scene, materials, equipmentCorner, {
    name: "broken",
    at: [-0.1, 0.02, -0.62],
    turn: [-0.34, 0.6, 0.5],
    length: 1.6,
    broken: true,
  });
  buildSkateGuards(scene, materials, equipmentCorner, {
    at: [-0.12, 0.02, 0.92],
    turn: [0, 0.4, 0],
  });
  buildHockeyGlove(scene, materials, equipmentCorner, {
    name: "left",
    at: [-0.18, 0.9, 0.18],
    turn: [0.1, 0.5, 0.06],
  });
  buildHockeyGlove(scene, materials, equipmentCorner, {
    name: "right",
    at: [-0.18, 0.9, -0.06],
    turn: [0.1, -0.4, -0.06],
    tone: materials.tape(),
  });
  // small shelf holding the gloves off the floor
  solid(scene, "glove-shelf", equipmentCorner, {
    width: 0.34,
    height: 0.032,
    depth: 0.62,
    at: [-0.16, 0.88, 0.06],
    material: materials.walnutDark(),
  });
  [-1, 1].forEach((side) => {
    solid(scene, `glove-shelf-bracket${side}`, equipmentCorner, {
      width: 0.028,
      height: 0.16,
      depth: 0.14,
      at: [-0.02, 0.8, side * 0.24],
      turn: [0, 0, 0.5],
      material: materials.brass(),
    });
  });
  contactShadow(scene, cache, equipmentCorner, {
    x: 0,
    z: 0.2,
    radius: 1.0,
    strength: 0.8,
  });

  // an invisible blocker so the player cannot walk through the pile
  const equipmentBlocker = solid(scene, "equipment-blocker", equipmentCorner, {
    width: 0.62,
    height: 1.3,
    depth: 1.9,
    at: [0, 0.65, 0.2],
    collide: true,
  });
  equipmentBlocker.isVisible = false;

  buildEquipmentBag(scene, materials, root, {
    at: [HALL.width / 2 - 0.46, 0, -4.5],
    turn: [0, 0.34, 0],
  });
  contactShadow(scene, cache, root, {
    x: HALL.width / 2 - 0.46,
    z: -4.5,
    radius: 0.7,
    strength: 0.9,
  });
  buildPuckBucket(scene, materials, root, {
    at: [-HALL.width / 2 + 0.36, 0, -3.5],
  });
  contactShadow(scene, cache, root, {
    x: -HALL.width / 2 + 0.36,
    z: -3.5,
    radius: 0.45,
    strength: 0.95,
  });
  buildTrashCan(scene, materials, root, {
    at: [-HALL.width / 2 + 0.4, 0, -11.4],
  });
  contactShadow(scene, cache, root, {
    x: -HALL.width / 2 + 0.4,
    z: -11.4,
    radius: 0.42,
    strength: 0.95,
  });
  // the stick that explains the dent
  buildStick(scene, materials, root, {
    name: "bin-witness",
    at: [-HALL.width / 2 + 0.62, 0, -11.7],
    turn: [-0.16, 0.5, 0.2],
    length: 1.58,
  });
  // wooden stick left by the lift, as if someone just walked in
  buildStick(scene, materials, root, {
    name: "lift-rest",
    at: [HALL.width / 2 - 0.4, 0, HALL.startZ + 0.9],
    turn: [-0.14, -0.42, 0.18],
    length: 1.62,
    wooden: true,
  });

  buildClipboard(scene, materials, root, {
    at: [-HALL.width / 2 + 0.3, 0.902, -8.02],
    turn: [0, 0.3, 0],
  });
  buildCoffeeCup(scene, materials, root, {
    at: [-HALL.width / 2 + 0.42, 0.9, -7.86],
  });

  // pucks that have quietly migrated across the floor
  const strayPucks = [
    [-0.72, -20.8],
    [0.84, -19.15],
    [-1.08, -16.35],
    [0.58, -14.7],
    [-0.86, -12.2],
    [0.74, -10.6],
    [-1.2, -6.8],
    [0.98, -8.2],
    [0.2, -2.2],
    [-0.62, -0.4],
  ];
  strayPucks.forEach(([x, z], index) => {
    buildPuck(scene, materials, root, {
      name: `stray-${index}`,
      at: [x, 0.013, z],
      turn: [0, hashRandom(index * 3.1) * 3, 0],
    });
    contactShadow(scene, cache, root, { x, z, radius: 0.09, strength: 0.8 });
  });

  // one puck lodged where a puck cannot possibly have reached
  const lodged = buildPuck(scene, materials, root, {
    name: "lodged",
    at: [0.9, HALL.height - 0.12, -9.2],
    turn: [0.4, 0.7, 1.35],
  });
  void lodged;
  solid(scene, "lodged-puck-crack", root, {
    width: 0.16,
    height: 0.012,
    depth: 0.16,
    at: [0.9, HALL.height - 0.02, -9.2],
    material: materials.blackMetal(),
  });

  /* The whiteboard nobody is allowed to erase --------------------------- */

  const whiteboardMount = wallMount(scene, root, -1, -3.4, 0);
  const whiteboard = buildFramedPiece(scene, materials, whiteboardMount, {
    name: "whiteboard",
    width: 1.02,
    height: 0.62,
    depth: 0.042,
    moulding: 0.028,
    mat: 0,
    glass: false,
    art: materials.art(
      "whiteboard",
      1280,
      780,
      paintWhiteboardBreakout,
      { roughness: 0.28, environmentIntensity: 0.24 }
    ),
    frameMaterial: materials.steel(),
  });
  whiteboard.group.position.y = 1.62;
  // marker tray
  solid(scene, "whiteboard-tray", whiteboardMount, {
    width: 0.7,
    height: 0.02,
    depth: 0.06,
    at: [0, 1.27, 0.05],
    material: materials.steel(),
  });
  ["#1f2b52", "#8c1e2a", "#1b6b3a"].forEach((tone, index) => {
    rod(scene, `whiteboard-marker${index}`, whiteboardMount, {
      diameter: 0.016,
      height: 0.11,
      sides: 8,
      at: [-0.2 + index * 0.14, 1.29, 0.05],
      turn: [0, 0, Math.PI / 2],
      material: materials.flat(`marker-${index}`, tone, { roughness: 0.5 }),
    });
  });
  register({
    id: "whiteboard",
    label: "Coaching whiteboard",
    card: EXHIBIT_CARDS.whiteboard,
    point: new Vector3(-HALL.width / 2 + 0.2, 1.62, -3.4),
    radius: 2.0,
  });

  /* The gag, and the darts --------------------------------------------- */

  const dartStation = buildDartStation(scene, materials, root, cache);
  register({
    id: "tkachuk",
    label: "Fuck Brady Tkachuk",
    hint: "Throw a dart",
    card: EXHIBIT_CARDS.tkachuk,
    point: new Vector3(HALL.width / 2 - 0.3, 1.55, HALL.tkachukZ),
    radius: 3.2,
    action: "dart",
  });
  void dartStation;

  /* Championship display, tucked beside the door ----------------------- */

  const cupDisplay = buildChampionshipCup(scene, materials, root, cache, {
    at: [-HALL.width / 2 + 0.68, 0, -0.5],
  });
  register({
    id: "cup",
    label: "Championship trophy",
    card: EXHIBIT_CARDS.cup,
    point: cupDisplay.focus,
    radius: 2.4,
  });
  const cupBlocker = solid(scene, "cup-blocker", root, {
    width: 1.5,
    height: 1.5,
    depth: 1.3,
    at: [-HALL.width / 2 + 0.6, 0.75, -0.55],
    collide: true,
  });
  cupBlocker.isVisible = false;

  /* Masks by the door -------------------------------------------------- */

  const maskWall = wallMount(scene, root, 1, 0.1, 0);
  // walnut mounting board with brass hooks
  solid(scene, "mask-board", maskWall, {
    width: 1.24,
    height: 0.66,
    depth: 0.03,
    at: [0, 1.82, 0.015],
    material: materials.walnut(),
  });
  [
    [1.28, 0.026, 0, 2.16],
    [1.28, 0.026, 0, 1.48],
  ].forEach(([w, h, x, y], index) => {
    solid(scene, `mask-board-trim${index}`, maskWall, {
      width: w,
      height: h,
      depth: 0.044,
      at: [x, y, 0.022],
      material: materials.walnutDark(),
    });
  });
  [-0.42, 0.0, 0.42].forEach((x, index) => {
    rod(scene, `mask-hook${index}`, maskWall, {
      diameter: 0.014,
      height: 0.09,
      sides: 10,
      at: [x, 2.02, 0.05],
      turn: [Math.PI / 2.2, 0, 0],
      material: materials.brass(),
    });
  });

  buildFibreglassMask(scene, materials, maskWall, {
    at: [-0.42, 1.86, 0.16],
    turn: [0.16, 0, 0.04],
  });
  buildCageMask(scene, materials, maskWall, {
    at: [0.0, 1.88, 0.16],
    turn: [0.12, 0, -0.03],
  });
  buildModernMask(scene, materials, maskWall, {
    at: [0.42, 1.85, 0.17],
    turn: [0.14, 0, 0.05],
  });

  buildBrassPlaque(scene, materials, maskWall, {
    name: "masks",
    width: 0.52,
    height: 0.09,
    at: [0, 1.42, 0.03],
    lines: ["FIBREGLASS · CAGE · MODERN SHELL"],
  });
  register({
    id: "masks",
    label: "Goaltending masks",
    card: EXHIBIT_CARDS.masks,
    point: new Vector3(HALL.width / 2 - 0.35, 1.86, 0.1),
    radius: 2.2,
  });

  // the employee plaque
  buildBrassPlaque(scene, materials, wallMount(scene, root, 1, -1.6, 0), {
    name: "trade-request",
    width: 0.42,
    height: 0.1,
    at: [0, 1.22, 0.02],
    lines: [HALL_FUN_LABELS.tradeRequest],
  });

  // a tiny framed photograph left deliberately crooked
  const crookedMount = wallMount(scene, root, -1, -1.15, 0);
  const crooked = buildFramedPiece(scene, materials, crookedMount, {
    name: "crooked",
    width: 0.16,
    height: 0.2,
    depth: 0.03,
    moulding: 0.02,
    mat: 0.016,
    art: materials.art(
      "crooked-photo",
      340,
      420,
      (ctx, w, h) => paintVintageHockeyPhoto(ctx, w, h, 53, ""),
      { roughness: 0.5 }
    ),
    tilt: 0.14,
  });
  crooked.group.position.y = 1.34;

  /* The door ------------------------------------------------------------ */

  const door = buildOfficeDoor(scene, materials, root, cache);
  register({
    id: "door",
    label: "Enter hockey operations",
    hint: "Enter",
    card: EXHIBIT_CARDS.door,
    point: new Vector3(0, 1.2, HALL.doorZ - 0.1),
    radius: 2.4,
    action: "door",
  });

  return {
    root,
    cache,
    door,
    interactables: cache.interactables,
    lights: cache.lights,
    exhibits: { karlsson, ovechkin, cup: cupDisplay },
    setEnabled(enabled) {
      root.setEnabled(enabled);
      cache.lights.forEach((light) => light.setEnabled(enabled));
    },
    /*
      Called once the office has taken over. Geometry, the corridor's own
      lights and its textures all go, which is the point — that budget is
      wanted for the hub.
    */
    dispose() {
      cache.lights.forEach((light) => light.dispose());
      cache.lights.length = 0;
      root.dispose(false, true);
    },
  };
}

/* ============================================================================
   FIRST-PERSON CONTROLLER
   ==========================================================================

   Collision is solved analytically against the corridor box and a short list
   of prop volumes rather than through the physics engine. The corridor is a
   known shape, so this is both cheaper and completely predictable — no
   clipping through displays and no camera drift.
*/

const HALL_BLOCKERS = Object.freeze([
  { x: 1.35, z: -21.1, hw: 0.28, hd: 0.22 },
  { x: -1.45, z: -8.5, hw: 0.36, hd: 0.82 },
  { x: 1.4, z: -5.85, hw: 0.44, hd: 1.15 },
  { x: 1.29, z: -4.5, hw: 0.46, hd: 0.34 },
  { x: -1.39, z: -3.5, hw: 0.26, hd: 0.26 },
  { x: -1.35, z: -11.4, hw: 0.3, hd: 0.3 },
  { x: -1.07, z: -0.55, hw: 0.76, hd: 0.8 },
]);

function resolveHallCollision(x, z, radius, doorOpen) {
  const limitX = HALL.width / 2 - radius - 0.06;
  let nextX = clamp(x, -limitX, limitX);
  let nextZ = clamp(
    z,
    HALL.startZ - 1.4 + radius + 0.12,
    doorOpen ? HALL.doorZ + 4 : HALL.doorZ - radius - 0.1
  );

  for (let i = 0; i < HALL_BLOCKERS.length; i += 1) {
    const blocker = HALL_BLOCKERS[i];
    const dx = nextX - blocker.x;
    const dz = nextZ - blocker.z;
    const overlapX = blocker.hw + radius - Math.abs(dx);
    const overlapZ = blocker.hd + radius - Math.abs(dz);
    if (overlapX > 0 && overlapZ > 0) {
      // push out along the shallower axis so sliding feels natural
      if (overlapX < overlapZ) {
        nextX += Math.sign(dx || 1) * overlapX;
      } else {
        nextZ += Math.sign(dz || 1) * overlapZ;
      }
      nextX = clamp(nextX, -limitX, limitX);
    }
  }

  return { x: nextX, z: nextZ };
}

function createHallController({
  scene,
  camera,
  canvas,
  interactables,
  onPromptChange,
  onActivate,
  onFirstMove,
  audio,
  reducedMotion,
}) {
  const state = {
    enabled: true,
    lookScale: 1,
    yaw: 0,
    pitch: 0.04,
    velocity: { x: 0, z: 0 },
    bobPhase: 0,
    bobHeight: 0,
    stepPhase: 0,
    doorOpen: false,
    prompt: null,
    moved: false,
    pointerLocked: false,
    dragging: false,
  };

  const keys = new Set();
  const RADIUS = 0.34;
  const WALK = reducedMotion ? 1.85 : 2.55;
  const RUN = reducedMotion ? 2.4 : 3.7;
  const ACCEL = 14;
  const DAMP = 11;
  const LOOK = 0.0021;

  camera.position.set(0, HALL.eyeHeight, HALL.startZ + 0.35);
  camera.rotation.set(state.pitch, 0, 0);
  camera.fov = 0.94;
  camera.minZ = 0.08;
  camera.maxZ = 90;
  camera.inertia = 0;
  camera.inputs?.clear?.();

  const isMoveKey = (code) =>
    [
      "KeyW",
      "KeyA",
      "KeyS",
      "KeyD",
      "ArrowUp",
      "ArrowDown",
      "ArrowLeft",
      "ArrowRight",
    ].includes(code);

  const onKeyDown = (event) => {
    if (!state.enabled) return;
    if (event.code === "Escape") return;
    if (event.target && /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName)) {
      return;
    }
    if (isMoveKey(event.code) || event.code === "ShiftLeft" || event.code === "ShiftRight") {
      keys.add(event.code);
      if (isMoveKey(event.code)) {
        event.preventDefault();
        if (!state.moved) {
          state.moved = true;
          onFirstMove?.();
        }
      }
    }
    if (event.code === "KeyE" || event.code === "Enter" || event.code === "Space") {
      event.preventDefault();
      if (state.prompt) {
        onActivate?.(state.prompt);
      }
    }
  };

  const onKeyUp = (event) => {
    keys.delete(event.code);
  };

  const applyLook = (dx, dy) => {
    const scale = LOOK * state.lookScale;
    state.yaw += dx * scale;
    state.pitch = clamp(state.pitch + dy * scale, -1.24, 1.24);
  };

  const onPointerMove = (event) => {
    if (!state.enabled) return;
    if (state.pointerLocked) {
      applyLook(event.movementX || 0, event.movementY || 0);
      return;
    }
    if (state.dragging) {
      applyLook(event.movementX || 0, event.movementY || 0);
    }
  };

  const onPointerDown = (event) => {
    if (!state.enabled || event.button !== 0) return;
    if (!state.pointerLocked) {
      // Pointer lock is the good path; drag-look is the fallback if it is denied.
      const request = canvas.requestPointerLock?.();
      if (request && typeof request.catch === "function") {
        request.catch(() => {
          state.dragging = true;
        });
      }
      state.dragging = !document.pointerLockElement;
      return;
    }
    if (state.prompt) {
      onActivate?.(state.prompt);
    }
  };

  const onPointerUp = () => {
    state.dragging = false;
  };

  const onPointerLockChange = () => {
    state.pointerLocked = document.pointerLockElement === canvas;
    if (state.pointerLocked) {
      state.dragging = false;
    }
  };

  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
  canvas.addEventListener("pointerdown", onPointerDown);
  window.addEventListener("pointerup", onPointerUp);
  window.addEventListener("pointermove", onPointerMove);
  document.addEventListener("pointerlockchange", onPointerLockChange);

  const forward = new Vector3();
  const right = new Vector3();

  const tick = () => {
    const dt = Math.min(0.05, scene.getEngine().getDeltaTime() / 1000);
    if (dt <= 0) return;

    camera.rotation.set(state.pitch, state.yaw, 0);

    const sin = Math.sin(state.yaw);
    const cos = Math.cos(state.yaw);
    forward.set(sin, 0, cos);
    right.set(cos, 0, -sin);

    let inputX = 0;
    let inputZ = 0;
    if (state.enabled) {
      if (keys.has("KeyW") || keys.has("ArrowUp")) inputZ += 1;
      if (keys.has("KeyS") || keys.has("ArrowDown")) inputZ -= 1;
      if (keys.has("KeyD") || keys.has("ArrowRight")) inputX += 1;
      if (keys.has("KeyA") || keys.has("ArrowLeft")) inputX -= 1;
    }

    const magnitude = Math.hypot(inputX, inputZ);
    const speed = keys.has("ShiftLeft") || keys.has("ShiftRight") ? RUN : WALK;

    let desiredX = 0;
    let desiredZ = 0;
    if (magnitude > 0) {
      const nx = inputX / magnitude;
      const nz = inputZ / magnitude;
      desiredX = (forward.x * nz + right.x * nx) * speed;
      desiredZ = (forward.z * nz + right.z * nx) * speed;
    }

    const blend = 1 - Math.exp(-(magnitude > 0 ? ACCEL : DAMP) * dt);
    state.velocity.x += (desiredX - state.velocity.x) * blend;
    state.velocity.z += (desiredZ - state.velocity.z) * blend;

    const travelled = Math.hypot(state.velocity.x, state.velocity.z);
    if (travelled < 0.02) {
      state.velocity.x = 0;
      state.velocity.z = 0;
    }

    const resolved = resolveHallCollision(
      camera.position.x + state.velocity.x * dt,
      camera.position.z + state.velocity.z * dt,
      RADIUS,
      state.doorOpen
    );

    // head bob and footsteps scale with actual travel, not with key state
    if (travelled > 0.08 && !reducedMotion) {
      state.bobPhase += dt * (4.4 + travelled * 1.25);
      state.stepPhase += dt * (1.55 + travelled * 0.42);
      if (state.stepPhase >= 1) {
        state.stepPhase -= 1;
        const onRunner = Math.abs(resolved.x) < HALL.runnerWidth / 2;
        audio?.footstep(onRunner ? "carpet" : "stone", clamp(travelled / RUN, 0.4, 1));
      }
    } else {
      state.bobPhase += dt * 1.05;
    }

    const bobTarget =
      reducedMotion
        ? 0
        : Math.sin(state.bobPhase * 2) * 0.016 * clamp(travelled / WALK, 0, 1) +
          Math.sin(state.bobPhase * 0.6) * 0.004;
    state.bobHeight += (bobTarget - state.bobHeight) * (1 - Math.exp(-9 * dt));

    const sway = reducedMotion
      ? 0
      : Math.sin(state.bobPhase) * 0.006 * clamp(travelled / WALK, 0, 1);

    camera.position.set(
      resolved.x,
      HALL.eyeHeight + state.bobHeight,
      resolved.z
    );
    camera.rotation.set(state.pitch + sway * 0.35, state.yaw + sway * 0.4, 0);

    audio?.setDoorProximity(
      clamp(1 - (HALL.doorZ - resolved.z) / 9.4, 0, 1)
    );

    // interaction prompt: nearest thing in range and roughly in front
    let best = null;
    let bestScore = Infinity;
    for (let i = 0; i < interactables.length; i += 1) {
      const entry = interactables[i];
      const dx = entry.point.x - camera.position.x;
      const dy = entry.point.y - camera.position.y;
      const dz = entry.point.z - camera.position.z;
      const distance = Math.hypot(dx, dy, dz);
      if (distance > entry.radius) continue;
      const facing = (dx * forward.x + dz * forward.z) / Math.max(distance, 0.001);
      if (facing < 0.42) continue;
      const score = distance * (1.6 - facing);
      if (score < bestScore) {
        bestScore = score;
        best = entry;
      }
    }

    const nextId = best?.id || null;
    if (nextId !== (state.prompt?.id || null)) {
      state.prompt = best;
      onPromptChange?.(best);
    }
  };

  let observer = scene.onBeforeRenderObservable.add(tick);

  const detach = () => {
    if (observer) {
      scene.onBeforeRenderObservable.remove(observer);
      observer = null;
    }
  };

  return {
    state,
    setEnabled(enabled) {
      state.enabled = enabled;
      state.lookScale = enabled ? 1 : 0.22;
      if (!enabled) {
        state.velocity.x = 0;
        state.velocity.z = 0;
        keys.clear();
      }
    },
    setDoorOpen(open) {
      state.doorOpen = open;
    },
    /*
      Hands the camera back. The cinematic drives position and target directly,
      so the walk loop has to stop writing to it entirely.
    */
    suspend() {
      state.enabled = false;
      keys.clear();
      state.velocity.x = 0;
      state.velocity.z = 0;
      state.dragging = false;
      detach();
      if (document.pointerLockElement === canvas) {
        document.exitPointerLock?.();
      }
    },
    releasePointer() {
      if (document.pointerLockElement === canvas) {
        document.exitPointerLock?.();
      }
    },
    getPrompt() {
      return state.prompt;
    },
    dispose() {
      detach();
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      canvas.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointermove", onPointerMove);
      document.removeEventListener("pointerlockchange", onPointerLockChange);
      if (document.pointerLockElement === canvas) {
        document.exitPointerLock?.();
      }
    },
  };
}

/* ============================================================================
   HALLWAY SOUND
   ==========================================================================

   All of it is synthesised, so nothing new has to be downloaded before the
   corridor can be entered: a slightly-too-comfortable lounge loop, HVAC and
   electrical room tone beneath it, footsteps that know what they are walking
   on, and a convolution room the introduction can sit inside.
*/

function createHallAudio() {
  const AudioContextClass =
    typeof window !== "undefined"
      ? window.AudioContext || window.webkitAudioContext
      : null;

  if (!AudioContextClass) {
    return {
      unlock() {},
      start() {},
      stop() {},
      footstep() {},
      setDoorProximity() {},
      duck() {},
      doorSwing() {},
      roomBloom() {},
      dispose() {},
    };
  }

  const ctx = new AudioContextClass();

  const master = ctx.createGain();
  master.gain.value = 0.0001;
  master.connect(ctx.destination);

  const musicBus = ctx.createGain();
  musicBus.gain.value = 0.34;
  const musicDuck = ctx.createGain();
  musicDuck.gain.value = 1;
  musicBus.connect(musicDuck);
  musicDuck.connect(master);

  const ambienceBus = ctx.createGain();
  ambienceBus.gain.value = 0.5;
  ambienceBus.connect(master);

  const sfxBus = ctx.createGain();
  sfxBus.gain.value = 0.6;
  sfxBus.connect(master);

  // small-corridor impulse response (shorter IR keeps init off the critical path)
  const reverb = ctx.createConvolver();
  const irLength = Math.floor(ctx.sampleRate * 0.45);
  const ir = ctx.createBuffer(2, irLength, ctx.sampleRate);
  for (let channel = 0; channel < 2; channel += 1) {
    const data = ir.getChannelData(channel);
    for (let i = 0; i < irLength; i += 1) {
      const t = i / irLength;
      data[i] = (Math.random() * 2 - 1) * Math.pow(1 - t, 3.1) * 0.6;
    }
    // early reflections off a narrow hard-floored hallway
    [0.009, 0.017, 0.026, 0.041, 0.062].forEach((delay, index) => {
      const sample = Math.floor(delay * ctx.sampleRate);
      if (sample < irLength) {
        data[sample] += (index % 2 === 0 ? 0.55 : -0.42) / (index + 1);
      }
    });
  }
  reverb.buffer = ir;
  const reverbReturn = ctx.createGain();
  reverbReturn.gain.value = 0.42;
  reverb.connect(reverbReturn);
  reverbReturn.connect(master);

  function noiseBuffer(seconds) {
    const length = Math.floor(ctx.sampleRate * seconds);
    const buffer = ctx.createBuffer(1, length, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    let last = 0;
    for (let i = 0; i < length; i += 1) {
      const white = Math.random() * 2 - 1;
      last = (last + 0.02 * white) / 1.02;
      data[i] = last * 3.2;
    }
    return buffer;
  }

  const brown = noiseBuffer(4);

  /* HVAC ---------------------------------------------------------------- */
  const hvac = ctx.createBufferSource();
  hvac.buffer = brown;
  hvac.loop = true;
  const hvacFilter = ctx.createBiquadFilter();
  hvacFilter.type = "lowpass";
  hvacFilter.frequency.value = 170;
  hvacFilter.Q.value = 0.6;
  const hvacGain = ctx.createGain();
  hvacGain.gain.value = 0.5;
  hvac.connect(hvacFilter);
  hvacFilter.connect(hvacGain);
  hvacGain.connect(ambienceBus);

  const hvacBody = ctx.createOscillator();
  hvacBody.type = "sine";
  hvacBody.frequency.value = 51;
  const hvacBodyGain = ctx.createGain();
  hvacBodyGain.gain.value = 0.05;
  hvacBody.connect(hvacBodyGain);
  hvacBodyGain.connect(ambienceBus);

  /* Electrical / fluorescent room tone ---------------------------------- */
  const roomTone = ctx.createBufferSource();
  roomTone.buffer = brown;
  roomTone.loop = true;
  const roomToneFilter = ctx.createBiquadFilter();
  roomToneFilter.type = "bandpass";
  roomToneFilter.frequency.value = 2100;
  roomToneFilter.Q.value = 1.1;
  const roomToneGain = ctx.createGain();
  roomToneGain.gain.value = 0.05;
  roomTone.connect(roomToneFilter);
  roomToneFilter.connect(roomToneGain);
  roomToneGain.connect(ambienceBus);

  const ballast = ctx.createOscillator();
  ballast.type = "sawtooth";
  ballast.frequency.value = 120;
  const ballastFilter = ctx.createBiquadFilter();
  ballastFilter.type = "lowpass";
  ballastFilter.frequency.value = 480;
  const ballastGain = ctx.createGain();
  ballastGain.gain.value = 0.012;
  ballast.connect(ballastFilter);
  ballastFilter.connect(ballastGain);
  ballastGain.connect(ambienceBus);

  /* Distant building noise ---------------------------------------------- */
  const distant = ctx.createBufferSource();
  distant.buffer = brown;
  distant.loop = true;
  const distantFilter = ctx.createBiquadFilter();
  distantFilter.type = "lowpass";
  distantFilter.frequency.value = 620;
  const distantGain = ctx.createGain();
  distantGain.gain.value = 0.06;
  const distantLfo = ctx.createOscillator();
  distantLfo.frequency.value = 0.06;
  const distantLfoGain = ctx.createGain();
  distantLfoGain.gain.value = 0.035;
  distantLfo.connect(distantLfoGain);
  distantLfoGain.connect(distantGain.gain);
  distant.connect(distantFilter);
  distantFilter.connect(distantGain);
  distantGain.connect(reverb);
  distantGain.connect(ambienceBus);

  /* Lounge music ---------------------------------------------------------
     Rhodes-ish electric piano over a lazy walking bass. Comfortable, and
     just slightly wrong for a corridor full of hockey relics.
  */
  const chords = [
    { root: 87.31, voices: [349.23, 440.0, 523.25, 659.26] },
    { root: 73.42, voices: [293.66, 349.23, 440.0, 587.33] },
    { root: 116.54, voices: [349.23, 466.16, 554.37, 698.46] },
    { root: 98.0, voices: [392.0, 493.88, 587.33, 739.99] },
  ];
  const melody = [659.26, 587.33, 523.25, 493.88, 440.0, 523.25, 587.33, 440.0];

  const musicTremolo = ctx.createGain();
  musicTremolo.gain.value = 1;
  musicTremolo.connect(musicBus);
  const tremoloLfo = ctx.createOscillator();
  tremoloLfo.frequency.value = 0.18;
  const tremoloDepth = ctx.createGain();
  tremoloDepth.gain.value = 0.11;
  tremoloLfo.connect(tremoloDepth);
  tremoloDepth.connect(musicTremolo.gain);

  const musicSend = ctx.createGain();
  musicSend.gain.value = 0.2;
  musicTremolo.connect(musicSend);
  musicSend.connect(reverb);

  function voice(frequency, at, duration, level, type = "triangle") {
    const osc = ctx.createOscillator();
    osc.type = type;
    osc.frequency.value = frequency;
    const bell = ctx.createOscillator();
    bell.type = "sine";
    bell.frequency.value = frequency * 2.004;
    const gain = ctx.createGain();
    const bellGain = ctx.createGain();
    const tone = ctx.createBiquadFilter();
    tone.type = "lowpass";
    tone.frequency.value = 1900;

    gain.gain.setValueAtTime(0.0001, at);
    gain.gain.exponentialRampToValueAtTime(level, at + 0.05);
    gain.gain.exponentialRampToValueAtTime(0.0001, at + duration);
    bellGain.gain.setValueAtTime(0.0001, at);
    bellGain.gain.exponentialRampToValueAtTime(level * 0.28, at + 0.02);
    bellGain.gain.exponentialRampToValueAtTime(0.0001, at + duration * 0.42);

    osc.connect(gain);
    bell.connect(bellGain);
    gain.connect(tone);
    bellGain.connect(tone);
    tone.connect(musicTremolo);

    osc.start(at);
    bell.start(at);
    osc.stop(at + duration + 0.1);
    bell.stop(at + duration + 0.1);
  }

  function brush(at, level) {
    const source = ctx.createBufferSource();
    source.buffer = brown;
    source.playbackRate.value = 3.4;
    const filter = ctx.createBiquadFilter();
    filter.type = "highpass";
    filter.frequency.value = 5200;
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(level, at);
    gain.gain.exponentialRampToValueAtTime(0.0001, at + 0.16);
    source.connect(filter);
    filter.connect(gain);
    gain.connect(musicTremolo);
    source.start(at, Math.random() * 2, 0.2);
    source.stop(at + 0.22);
  }

  const BEAT = 60 / 62;
  let musicCursor = 0;
  let musicBar = 0;
  let scheduler = null;

  function scheduleMusic() {
    const horizon = ctx.currentTime + 2.5;
    while (musicCursor < horizon) {
      const chord = chords[musicBar % chords.length];
      const barLength = BEAT * 4;

      chord.voices.forEach((frequency, index) => {
        voice(
          frequency,
          musicCursor + index * 0.035,
          barLength * 0.94,
          0.045
        );
      });
      voice(chord.root, musicCursor, barLength * 0.9, 0.07, "sine");
      voice(chord.root * 1.5, musicCursor + BEAT * 2.5, BEAT * 1.2, 0.045, "sine");

      voice(
        melody[musicBar % melody.length],
        musicCursor + BEAT * 1.5,
        BEAT * 1.6,
        0.03
      );

      for (let beat = 0; beat < 4; beat += 1) {
        brush(musicCursor + beat * BEAT, beat % 2 === 0 ? 0.012 : 0.02);
      }

      musicCursor += barLength;
      musicBar += 1;
    }
  }

  /* Door proximity bed -------------------------------------------------- */
  const doorPad = ctx.createGain();
  doorPad.gain.value = 0.0001;
  doorPad.connect(master);
  const doorPadFilter = ctx.createBiquadFilter();
  doorPadFilter.type = "lowpass";
  doorPadFilter.frequency.value = 420;
  doorPadFilter.connect(doorPad);
  [58.27, 87.31, 130.81].forEach((frequency, index) => {
    const osc = ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.value = frequency;
    const gain = ctx.createGain();
    gain.gain.value = 0.12 / (index + 1);
    osc.connect(gain);
    gain.connect(doorPadFilter);
    osc.start();
  });

  let started = false;
  let proximity = 0;
  let gestureHooked = false;

  const api = {
    unlock() {
      if (ctx.state === "suspended") {
        ctx.resume().catch(() => {});
      }
    },

    start() {
      if (started) return;
      started = true;
      api.unlock();

      /*
        Browsers will not let audio begin without a gesture. Rather than nag,
        the corridor simply comes alive the moment the player touches anything.
      */
      if (!gestureHooked && ctx.state === "suspended") {
        gestureHooked = true;
        const onGesture = () => {
          api.unlock();
          if (ctx.state !== "suspended") {
            window.removeEventListener("pointerdown", onGesture);
            window.removeEventListener("keydown", onGesture);
          }
        };
        window.addEventListener("pointerdown", onGesture);
        window.addEventListener("keydown", onGesture);
      }
      const now = ctx.currentTime;
      [hvac, roomTone, distant].forEach((source) => {
        try {
          source.start(now, Math.random() * 2);
        } catch (_error) {
          /* already started */
        }
      });
      [hvacBody, ballast, tremoloLfo, distantLfo].forEach((osc) => {
        try {
          osc.start(now);
        } catch (_error) {
          /* already started */
        }
      });
      master.gain.cancelScheduledValues(now);
      master.gain.setValueAtTime(0.0001, now);
      master.gain.exponentialRampToValueAtTime(0.85, now + 2.6);
      musicCursor = now + 0.4;
      scheduleMusic();
      scheduler = window.setInterval(scheduleMusic, 700);
    },

    stop(fadeSeconds = 1.4) {
      if (!started) return;
      const now = ctx.currentTime;
      master.gain.cancelScheduledValues(now);
      master.gain.setValueAtTime(Math.max(master.gain.value, 0.0002), now);
      master.gain.exponentialRampToValueAtTime(0.0001, now + fadeSeconds);
      if (scheduler != null) {
        window.clearInterval(scheduler);
        scheduler = null;
      }
    },

    footstep(surface, force = 1) {
      if (!started) return;
      const now = ctx.currentTime;
      const source = ctx.createBufferSource();
      source.buffer = brown;
      source.playbackRate.value = surface === "carpet" ? 1.5 : 2.6;
      const filter = ctx.createBiquadFilter();
      const gain = ctx.createGain();

      if (surface === "carpet") {
        filter.type = "lowpass";
        filter.frequency.value = 900;
        gain.gain.setValueAtTime(0.16 * force, now);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.13);
      } else {
        filter.type = "bandpass";
        filter.frequency.value = 1500 + Math.random() * 500;
        filter.Q.value = 0.9;
        gain.gain.setValueAtTime(0.2 * force, now);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.2);
      }

      const send = ctx.createGain();
      send.gain.value = surface === "carpet" ? 0.1 : 0.4;

      source.connect(filter);
      filter.connect(gain);
      gain.connect(sfxBus);
      gain.connect(send);
      send.connect(reverb);
      source.start(now, Math.random() * 2, 0.3);
      source.stop(now + 0.32);
    },

    setDoorProximity(value) {
      if (!started) return;
      const next = clamp(value, 0, 1);
      if (Math.abs(next - proximity) < 0.02) return;
      proximity = next;
      const now = ctx.currentTime;
      // the lounge loop steps back and the room warms as the office nears
      musicBus.gain.setTargetAtTime(0.34 - next * 0.19, now, 0.6);
      doorPad.gain.setTargetAtTime(0.0001 + next * 0.19, now, 0.7);
      hvacGain.gain.setTargetAtTime(0.5 - next * 0.2, now, 0.8);
      reverbReturn.gain.setTargetAtTime(0.42 - next * 0.16, now, 0.8);
    },

    duck(amount, seconds = 0.4) {
      const now = ctx.currentTime;
      musicDuck.gain.cancelScheduledValues(now);
      musicDuck.gain.setTargetAtTime(clamp(amount, 0.04, 1), now, seconds);
    },

    doorSwing() {
      if (!started) return;
      const now = ctx.currentTime;
      // latch
      const click = ctx.createOscillator();
      click.type = "square";
      click.frequency.setValueAtTime(1500, now);
      click.frequency.exponentialRampToValueAtTime(220, now + 0.05);
      const clickGain = ctx.createGain();
      clickGain.gain.setValueAtTime(0.16, now);
      clickGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.09);
      click.connect(clickGain);
      clickGain.connect(sfxBus);
      clickGain.connect(reverb);
      click.start(now);
      click.stop(now + 0.12);

      // sweep of the leaf through the air
      const sweep = ctx.createBufferSource();
      sweep.buffer = brown;
      sweep.playbackRate.value = 0.8;
      const sweepFilter = ctx.createBiquadFilter();
      sweepFilter.type = "lowpass";
      sweepFilter.frequency.setValueAtTime(300, now + 0.06);
      sweepFilter.frequency.linearRampToValueAtTime(1200, now + 0.9);
      const sweepGain = ctx.createGain();
      sweepGain.gain.setValueAtTime(0.0001, now + 0.06);
      sweepGain.gain.linearRampToValueAtTime(0.1, now + 0.4);
      sweepGain.gain.exponentialRampToValueAtTime(0.0001, now + 1.3);
      sweep.connect(sweepFilter);
      sweepFilter.connect(sweepGain);
      sweepGain.connect(sfxBus);
      sweepGain.connect(reverb);
      sweep.start(now + 0.06, 0, 1.4);
      sweep.stop(now + 1.5);
    },

    /*
      A short breath of the room, played alongside the spoken introduction so
      the line belongs to the office instead of to the browser.
    */
    roomBloom(level = 0.14, seconds = 1.1) {
      const now = ctx.currentTime;
      const source = ctx.createBufferSource();
      source.buffer = brown;
      source.playbackRate.value = 1.1;
      const filter = ctx.createBiquadFilter();
      filter.type = "bandpass";
      filter.frequency.value = 700;
      filter.Q.value = 0.5;
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.exponentialRampToValueAtTime(level, now + 0.12);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + seconds);
      source.connect(filter);
      filter.connect(gain);
      gain.connect(reverb);
      source.start(now, Math.random() * 2, seconds + 0.2);
      source.stop(now + seconds + 0.3);

      const sub = ctx.createOscillator();
      sub.type = "sine";
      sub.frequency.value = 62;
      const subGain = ctx.createGain();
      subGain.gain.setValueAtTime(0.0001, now);
      subGain.gain.exponentialRampToValueAtTime(level * 0.9, now + 0.2);
      subGain.gain.exponentialRampToValueAtTime(0.0001, now + seconds * 1.4);
      sub.connect(subGain);
      subGain.connect(master);
      sub.start(now);
      sub.stop(now + seconds * 1.5);
    },

    dispose() {
      if (scheduler != null) {
        window.clearInterval(scheduler);
        scheduler = null;
      }
      try {
        ctx.close();
      } catch (_error) {
        /* already closed */
      }
    },
  };

  return api;
}

/*
  Broadcast-style delivery for the arrival line. Voice selection prefers a
  measured English narration voice and never targets a specific person.
*/
function speakBroadcastLine(text, audio) {
  if (typeof window === "undefined" || !window.speechSynthesis) {
    audio?.roomBloom(0.16, 1.4);
    return Promise.resolve(false);
  }

  const synth = window.speechSynthesis;

  const pickVoice = () => {
    const voices = synth.getVoices() || [];
    if (!voices.length) return null;
    const english = voices.filter((voice) =>
      /^en(-|_|$)/i.test(voice.lang || "")
    );
    const pool = english.length ? english : voices;
    const preferred = [
      /guy/i,
      /david/i,
      /mark/i,
      /daniel/i,
      /alex/i,
      /google (uk|us) english/i,
    ];
    for (let i = 0; i < preferred.length; i += 1) {
      const match = pool.find((voice) => preferred[i].test(voice.name || ""));
      if (match) return match;
    }
    return pool[0];
  };

  return new Promise((resolve) => {
    const speak = () => {
      try {
        synth.cancel();
      } catch (_error) {
        /* ignore */
      }

      const utterance = new SpeechSynthesisUtterance(text);
      const voice = pickVoice();
      if (voice) {
        utterance.voice = voice;
        utterance.lang = voice.lang || "en-US";
      }
      // measured, authoritative, unhurried — the pacing of a broadcast open
      utterance.rate = 0.84;
      utterance.pitch = 0.78;
      utterance.volume = 1;

      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        window.clearTimeout(guard);
        audio?.duck(1, 1.4);
        resolve(true);
      };

      utterance.onstart = () => {
        audio?.duck(0.18, 0.35);
        audio?.roomBloom(0.15, 1.3);
      };
      utterance.onboundary = () => {
        audio?.roomBloom(0.05, 0.5);
      };
      utterance.onend = finish;
      utterance.onerror = finish;

      // never let a silent speech engine stall the cinematic
      const guard = window.setTimeout(finish, 6500);

      audio?.duck(0.18, 0.3);
      audio?.roomBloom(0.15, 1.3);
      synth.speak(utterance);
    };

    if (!(synth.getVoices() || []).length) {
      let handled = false;
      const onVoices = () => {
        if (handled) return;
        handled = true;
        synth.removeEventListener?.("voiceschanged", onVoices);
        speak();
      };
      synth.addEventListener?.("voiceschanged", onVoices);
      window.setTimeout(onVoices, 700);
      return;
    }

    speak();
  });
}

function configureRendering({
  scene,
  camera,
  accentPrimary,
  accentSecondary,
}) {
  scene.clearColor = new Color4(0.22, 0.2, 0.18, 1);
  scene.fogMode = Scene.FOGMODE_NONE;
  scene.environmentTexture = null;
  scene.environmentIntensity = 0;
  scene.ambientColor = new Color3(0.62, 0.6, 0.56);

  const image = scene.imageProcessingConfiguration;
  image.toneMappingEnabled = false;
  image.contrast = 1.08;
  image.exposure = 1.85;
  image.vignetteEnabled = false;

  const ambient = new HemisphericLight(
    "office-ambient",
    new Vector3(0.15, 1, 0.22),
    scene
  );
  ambient.intensity = 2.2;
  ambient.diffuse = new Color3(0.92, 0.9, 0.86);
  ambient.groundColor = new Color3(0.28, 0.26, 0.24);

  const key = new DirectionalLight(
    "office-key",
    new Vector3(-0.28, -0.82, -0.48),
    scene
  );
  key.position = new Vector3(2.8, 7.4, 0.4);
  key.diffuse = new Color3(1, 0.92, 0.82);
  key.intensity = 2.85;

  const fill = new DirectionalLight(
    "office-fill",
    new Vector3(0.55, -0.35, 0.42),
    scene
  );
  fill.diffuse = new Color3(0.72, 0.8, 0.92);
  fill.intensity = 1.15;

  const coolRim =
    new SpotLight(
      "office-rim",
      new Vector3(
        2.4,
        3.4,
        5.4
      ),
      new Vector3(
        -0.42,
        -0.40,
        -0.80
      ),
      Math.PI / 2.1,
      2,
      scene
    );

  coolRim.diffuse =
    new Color3(
      0.72,
      0.82,
      0.96
    );

  coolRim.intensity = 1.2;

  const deskWarm = new PointLight(
    "desk-practical",
    new Vector3(0.18, 1.55, 2.92),
    scene
  );
  deskWarm.diffuse = new Color3(1, 0.9, 0.72);
  deskWarm.intensity = 1.4;
  deskWarm.range = 12;

  const hallCool = new PointLight(
    "hall-practical",
    new Vector3(0, 2.6, -6.2),
    scene
  );
  hallCool.diffuse = new Color3(0.9, 0.93, 1);
  hallCool.intensity = 1.2;
  hallCool.range = 22;

  const primary = colorFromHex(accentPrimary, "#c9a86a");
  const secondary = colorFromHex(accentSecondary, "#9aa5b1");

  const teamFill = new PointLight(
    "team-fill",
    new Vector3(0, 2.05, 5.7),
    scene
  );
  teamFill.diffuse = primary;
  teamFill.specular = secondary;
  teamFill.intensity = 0.6;
  teamFill.range = 8;

  const deskKey = new SpotLight(
    "desk-key",
    new Vector3(0.3, 2.2, 1.8),
    new Vector3(-0.15, -0.85, 0.45),
    Math.PI / 2.6,
    12,
    scene
  );
  deskKey.diffuse = new Color3(1, 0.93, 0.8);
  deskKey.intensity = 2.6;

  return {
    pipeline: null,
    ambient,
    key,
    fill,
    coolRim,
    deskWarm,
    deskKey,
    hallCool,
    teamFill,
    shadowGenerator: null,
    glow: { intensity: 0 },
  };
}

function placePracticalLights(lighting, assets) {
  if (!lighting) {
    return;
  }

  const desk = assets?.desk ? computeAssetBounds(assets.desk) : null;
  const hallway = assets?.hallway ? computeAssetBounds(assets.hallway) : null;
  const human = assets?.seatedExecutive
    ? computeAssetBounds(assets.seatedExecutive)
    : null;

  if (desk) {
    lighting.deskWarm.position.set(
      desk.center.x,
      desk.max.y + 0.42,
      desk.center.z - 0.1
    );
    lighting.deskWarm.intensity = 1.8;
    lighting.deskWarm.range = 5.5;
    if (lighting.deskKey) {
      lighting.deskKey.position.set(
        desk.center.x + 0.55,
        desk.max.y + 1.28,
        desk.center.z - 1.15
      );
      lighting.deskKey.setDirectionToTarget(
        new Vector3(desk.center.x, desk.max.y, desk.center.z)
      );
    }
    lighting.key.position.set(
      desk.center.x + 2.4,
      desk.max.y + 5.6,
      desk.center.z - 2.2
    );
  }

  if (hallway) {
    lighting.hallCool.position.set(
      hallway.center.x,
      hallway.min.y + 2.35,
      hallway.min.z + hallway.size.z * 0.32
    );
  }

  if (human) {
    lighting.coolRim.position.set(
      human.center.x + 1.55,
      human.max.y + 0.55,
      human.center.z + 1.35
    );
    lighting.coolRim.direction = human.center
      .subtract(lighting.coolRim.position)
      .normalize();
    lighting.teamFill.position.set(
      human.center.x,
      human.max.y + 0.28,
      human.center.z
    );
    lighting.teamFill.intensity = 0.7;
  }
}

function registerShadowCasters(
  lighting,
  assets
) {
  const generator =
    lighting?.shadowGenerator;

  if (!generator) {
    return;
  }

  const casterKeys = new Set([
    "desk",
    "chair",
    "seatedExecutive",
    "standingExecutive",
    "hockeyStick",
    "trophy",
    "props",
    "clipboard",
    "contract",
  ]);

  Object.entries(assets)
    .filter(([, asset]) => asset)
    .forEach(([key, asset]) => {
      const casts = casterKeys.has(key);
      (asset.meshes || []).forEach((mesh) => {
        if (!mesh || typeof mesh.receiveShadows === "undefined") {
          return;
        }
        mesh.receiveShadows = casts;
        if (casts && mesh.getTotalVertices?.() > 0) {
          generator.addShadowCaster(mesh, false);
        }
      });
    });
}


/* ============================================================================
   SCENE ASSEMBLY
   ========================================================================== */

/*
  Office assets load in two tiers.

  The corridor is procedural, so nothing here has to finish before the player
  can walk. Tier one is what the office reveal genuinely cannot be staged
  without — the room, the desk and the document. Tier two is the dressing, and
  it resolves quietly while the player is still in the hallway.

  `assets` is a live object: the camera director and the light placement both
  read it lazily, so a shot that needs the trophy simply becomes available the
  moment the trophy lands.
*/
async function loadOfficeAssets({
  scene,
  mode,
  team,
  accentPrimary,
  onTier,
}) {
  const assets = {};
  const errors = [];

  async function safeLoad(
    key,
    options
  ) {
    try {
      assets[key] =
        await loadGlbAsset({
          scene,
          ...options,
        });

      return assets[key];
    } catch (error) {
      console.error(
        `Unable to load ${key}`,
        error
      );

      errors.push({
        key,
        error,
      });

      return null;
    }
  }

  /* Tier one — the office reveal cannot be staged without these. */
  await Promise.all([
    safeLoad("office", {
      name: "dark-office",
      url: darkOfficeGlb,
      calibration: ASSET_CALIBRATION.office,
    }),
    safeLoad("desk", {
      name: "executive-desk",
      url: executiveDeskGlb,
      calibration: ASSET_CALIBRATION.desk,
      materialOptions: {
        roughnessFloor: 0.24,
        roughnessCeiling: 0.84,
      },
    }),
    safeLoad("contract", {
      name: "physical-contract",
      url: contractGlb,
      calibration: ASSET_CALIBRATION.contract,
    }),
  ]);

  if (assets.office) {
    hideNamedMeshes(assets.office, [
      "blackout",
      "curtain",
      "cortina",
      "escritotio",
      "asiento",
      "cojin",
      "sky",
      "dome",
      "background",
    ]);
  }

  onTier?.("officeEssentials");

  /*
    Tier two — everything that dresses the room. Awaited by the caller only at
    the point the cinematic actually needs it.
  */
  const dressing = Promise.all([
    safeLoad("chair", {
      name: "executive-chair",
      url: leatherChairGlb,
      calibration: ASSET_CALIBRATION.chair,
    }),
    safeLoad("seatedExecutive", {
      name: "seated-executive",
      url: manSittingGlb,
      calibration: ASSET_CALIBRATION.seatedExecutive,
    }),
    safeLoad("trophy", {
      name: "trophy-cup",
      url: trophyCupGlb,
      calibration: ASSET_CALIBRATION.trophy,
    }),
    safeLoad("hockeyStick", {
      name: "hockey-stick",
      url: hockeyStickGlb,
      calibration: ASSET_CALIBRATION.hockeyStick,
    }),
    safeLoad("props", {
      name: "office-props",
      url: officePropsGlb,
      calibration: ASSET_CALIBRATION.props,
    }),
    safeLoad("clipboard", {
      name: "clipboard",
      url: clipboardGlb,
      calibration: ASSET_CALIBRATION.clipboard,
    }),
  ]).then(() => {
    /*
      Play authored animations when they exist.
      The code never assumes the GLB is rigged.
    */
    playBestAnimation(
      assets.seatedExecutive,
      [
        "sit",
        "seated",
        "idle",
        "breath",
      ],
      {
        loop: true,
        speedRatio: 0.9,
      }
    );

    onTier?.("officeDressing");
    return assets;
  });

  // referenced by an optional shot only; never fetched during the intro
  void manDressedInSuitGlb;
  void ASSET_CALIBRATION.standingExecutive;

  const doors = createEntranceDoors(scene, assets.hallway);

  let logoPlane = null;

  if (
    mode ===
      CINEMATIC_MODE.APPOINTMENT &&
    team
  ) {
    logoPlane =
      createTeamLogoPlane({
        scene,
        team,
        position:
          new Vector3(
            0,
            2.12,
            6.10
          ),
        width: 1.62,
        height: 1.62,
      });
  }

  const particles =
    createDustParticles(
      scene,
      accentPrimary
    );

  return {
    assets,
    errors,
    logoPlane,
    particles,
    doors,
    dressing,
  };
}


/* ============================================================================
   SIGNING SURFACE
   ========================================================================== */

function createSigningSurface({
  scene,
  desk,
  team,
  gmName,
  playerUniverse,
  injuriesEnabled,
  contractDate,
}) {
  const texture =
    new DynamicTexture(
      "gm-signing-texture",
      {
        width:
          CONTRACT_TEXTURE.width,
        height:
          CONTRACT_TEXTURE.height,
      },
      scene,
      true
    );

  texture.wrapU =
    Texture.CLAMP_ADDRESSMODE;

  texture.wrapV =
    Texture.CLAMP_ADDRESSMODE;

  drawContractTexture({
    texture,
    team,
    gmName,
    playerUniverse,
    injuriesEnabled,
    contractDate,
    executed: false,
  });

  const material =
    new StandardMaterial(
      "gm-signing-material",
      scene
    );

  material.diffuseColor =
    Color3.White();

  material.emissiveColor =
    new Color3(
      0.18,
      0.17,
      0.14
    );

  material.specularColor =
    new Color3(
      0.03,
      0.03,
      0.03
    );

  material.backFaceCulling =
    false;

  material.diffuseTexture =
    texture;

  const plane =
    MeshBuilder.CreatePlane(
      "gm-signing-plane",
      {
        width: 0.42,
        height: 0.58,
        sideOrientation:
          Mesh.DOUBLESIDE,
      },
      scene
    );

  const deskBounds = desk
    ? computeAssetBounds(desk)
    : null;

  plane.position = new Vector3(
    deskBounds ? deskBounds.center.x + 0.08 : 0.08,
    deskBounds ? deskBounds.max.y + 0.016 : 0.79,
    deskBounds ? deskBounds.center.z - 0.16 : 2.78
  );

  plane.rotation = new Vector3(
    -Math.PI / 2,
    0.06,
    0
  );

  plane.material =
    material;

  plane.isPickable = true;
  plane.setEnabled(true);

  return {
    plane,
    texture,
    material,
  };
}


/* ============================================================================
   FULLSCREEN GLB CINEMATIC
   ========================================================================== */

function ExecutiveCinematic({
  mode,
  team,
  gmName,
  playerUniverse,
  injuriesEnabled,
  accentPrimary,
  accentSecondary,
  contractDate,
  onComplete,
  onSkipIntro,
  onAssetTier,
  onFloorFailure,
  sceneControlRef,
  overlayActive,
}) {
  const canvasRef =
    useRef(null);

  const cancelledRef =
    useRef(false);

  const executeRef =
    useRef(null);

  const signatureReadyRef =
    useRef(false);

  const controlRef =
    useRef(null);

  const [
    stage,
    setStage,
  ] = useState(
    CINEMATIC_STAGE.LOADING
  );

  /* Hallway HUD state. Deliberately small — a reticle, a prompt, a card. */
  const [
    hallPhase,
    setHallPhase,
  ] = useState(
    HALL_PHASE.BOOTING
  );

  const [
    prompt,
    setPrompt,
  ] = useState(null);

  const [
    card,
    setCard,
  ] = useState(null);

  const [
    hintVisible,
    setHintVisible,
  ] = useState(false);

  const [
    narration,
    setNarration,
  ] = useState("");

  const [
    dartTally,
    setDartTally,
  ] = useState(0);

  const [
    blackout,
    setBlackout,
  ] = useState(true);

  const [
    sceneReady,
    setSceneReady,
  ] = useState(false);

  const [
    signatureReady,
    setSignatureReady,
  ] = useState(false);

  const [
    executing,
    setExecuting,
  ] = useState(false);

  const [
    welcome,
    setWelcome,
  ] = useState(false);

  const [
    assetIssue,
    setAssetIssue,
  ] = useState("");

  const [
    officeFailed,
    setOfficeFailed,
  ] = useState(false);

  const appointment =
    mode ===
    CINEMATIC_MODE.APPOINTMENT;

  const stageCopy =
    CINEMATIC_STAGE_COPY[stage] ||
    "Executive floor";

  useEffect(() => {
    const canvas =
      canvasRef.current;

    if (!canvas) {
      return undefined;
    }

    cancelledRef.current =
      false;

    signatureReadyRef.current =
      false;

    let engine = null;
    let scene = null;
    let pointerObserver = null;
    let resizeHandler = null;

    let signatureUnlocked =
      false;

    let signatureDrawing =
      false;

    let signatureDistance =
      0;

    let lastSignaturePoint =
      null;

    let signingSurface = null;
    let officeBuild = null;
    let lighting = null;
    let cameraDirector = null;
    let postSequenceStarted = false;

    let hall = null;
    let hallController = null;
    let hallAudio = null;
    let lowPower = false;
    let renderPaused = false;

    const renderScene = () => {
      if (renderPaused) {
        return;
      }
      if (scene && !scene.isDisposed) {
        scene.render();
      }
    };

    const setPausedRender = (paused) => {
      if (renderPaused === paused) {
        return;
      }
      renderPaused = paused;
      if (!engine) {
        return;
      }
      if (paused) {
        engine.stopRenderLoop();
        if (scene && !scene.isDisposed) {
          scene.render();
        }
      } else {
        engine.runRenderLoop(renderScene);
      }
    };

    const reducedMotion =
      window.matchMedia?.(
        "(prefers-reduced-motion: reduce)"
      )?.matches;

    const speed =
      reducedMotion
        ? 0.36
        : 1;

    const ms = (value) =>
      Math.max(
        100,
        Math.round(
          value * speed
        )
      );

    const run = async () => {
      try {
        engine =
          new Engine(
            canvas,
            true,
            {
              preserveDrawingBuffer:
                false,
              stencil: true,
              antialias: true,
              powerPreference:
                "high-performance",
            },
            true
          );

        engine.setHardwareScalingLevel(
          window.devicePixelRatio >
            1.5
            ? 1.15
            : 1
        );

        scene =
          new Scene(engine);

        const camera =
          new UniversalCamera(
            "franchise-office-camera",
            new Vector3(
              0,
              HALL.eyeHeight,
              HALL.startZ + 0.35
            ),
            scene
          );
        
        camera.minZ = 0.12;
        camera.maxZ = 220;
        camera.fov = 0.84;
        camera.inertia = 0;
        camera.rotation.z = 0;
        
        camera.setTarget(
          new Vector3(
            0,
            1.42,
            1.8
          )
        );

        scene.activeCamera =
          camera;

        lighting =
          configureRendering({
            scene,
            camera,
            accentPrimary,
            accentSecondary,
          });

        const tagAssets = (build) => {
          Object.entries(
            build.assets
          ).forEach(([key, asset]) => {
            if (!asset) {
              return;
            }

            (asset.meshes || []).forEach((mesh) => {
              if (!mesh) {
                return;
              }

              mesh.metadata = {
                ...(mesh.metadata || {}),
                setupAssetKey: key,
              };

              mesh.isPickable = true;
            });
          });
        };

        const setOfficeVisible = (build, visible) => {
          Object.values(build.assets).forEach((asset) => {
            asset?.root?.setEnabled(visible);
          });
          build.doors?.left?.setEnabled?.(visible);
          build.doors?.right?.setEnabled?.(visible);
          if (build.particles) {
            if (visible) {
              build.particles.start?.();
            } else {
              build.particles.stop?.();
            }
          }
        };

        const OFFICE_LIGHT_KEYS = [
          "ambient",
          "key",
          "fill",
          "coolRim",
          "deskWarm",
          "deskKey",
          "hallCool",
          "teamFill",
        ];

        const setOfficeLightsEnabled = (enabled) => {
          OFFICE_LIGHT_KEYS.forEach((name) => {
            lighting?.[name]?.setEnabled?.(enabled);
          });
        };

        /*
          ==========================================================
          OPENING HALLWAY
          ==========================================================

          The corridor is procedural, so it is standing and walkable in the
          time it takes to build geometry — no download gates the first frame.
          Every office GLB is fetched behind the player's back while they walk.
        */
        if (!appointment) {
          setOfficeLightsEnabled(false);

          // corridor grade: dark, warm practicals, real reflections on metal
          scene.environmentIntensity = 1;
          scene.clearColor = new Color4(0.055, 0.052, 0.048, 1);
          scene.ambientColor = new Color3(0.36, 0.34, 0.31);
          scene.imageProcessingConfiguration.exposure = 1.46;
          scene.imageProcessingConfiguration.contrast = 1.08;
          scene.imageProcessingConfiguration.vignetteEnabled = true;
          scene.imageProcessingConfiguration.vignetteWeight = 0.9;
          scene.imageProcessingConfiguration.vignetteColor = new Color4(
            0,
            0,
            0,
            0
          );

          const hallMaterials = createHallMaterials(scene);
          hall = buildMemorabiliaHall(scene, hallMaterials);
          hallAudio = createHallAudio();

          let requestDoor = null;
          const doorRequested = new Promise((resolve) => {
            requestDoor = resolve;
          });

          const closeCard = () => {
            setCard(null);
            hallController?.setEnabled(true);
          };

          hallController = createHallController({
            scene,
            camera,
            canvas,
            interactables: hall.interactables,
            reducedMotion,
            audio: hallAudio,
            onFirstMove: () => setHintVisible(false),
            onPromptChange: (entry) => {
              setPrompt(
                entry
                  ? {
                      id: entry.id,
                      label: entry.label,
                      hint: entry.hint || "Inspect",
                    }
                  : null
              );
            },
            onActivate: (entry) => {
              hallAudio?.unlock();

              if (entry.action === "door") {
                requestDoor?.();
                return;
              }

              if (entry.action === "dart") {
                const onTarget = throwDart(scene, hall.cache, camera);
                setDartTally((count) => count + (onTarget ? 1 : 0));
                hallAudio?.footstep("stone", 0.5);
                return;
              }

              if (!entry.card) {
                return;
              }

              // soft-lock rather than hard-stop: the player closes and walks on
              hallController?.setEnabled(false);
              setCard({ ...entry.card, id: entry.id });
            },
          });

          controlRef.current = {
            closeCard,
            unlockAudio: () => hallAudio?.unlock(),
            setLowPower: (enabled) => {
              if (lowPower === enabled) return;
              lowPower = enabled;
              engine?.setHardwareScalingLevel(enabled ? 1.7 : 1);
            },
            setPausedRender,
            duckAudio: (amount) => hallAudio?.duck(amount, 0.5),
            stopAudio: () => hallAudio?.stop(1.6),
            releaseInput: () => {
              hallController?.suspend?.();
              if (document.pointerLockElement) {
                document.exitPointerLock?.();
              }
            },
          };

          /*
            The corridor shots are authored coordinates, so a director exists
            from the first frame — the door approach never has to wait for a
            download to be able to move the camera.
          */
          cameraDirector = createCameraDirector({
            scene,
            camera,
            assets: {},
            logoPlane: null,
            cancelledRef,
          });

          scene.metadata = {
            purpose: "puckception-opening-hallway",
            glbFirst: false,
            physicalUnits: "meters",
          };

          engine.runRenderLoop(renderScene);

          resizeHandler = () => engine?.resize();
          window.addEventListener("resize", resizeHandler);
          window.requestAnimationFrame(() => engine?.resize());

          // one frame of geometry on screen before the curtain lifts
          await scene.whenReadyAsync();

          if (cancelledRef.current) {
            return;
          }

          setSceneReady(true);
          setStage(CINEMATIC_STAGE.HALLWAY);
          setHallPhase(HALL_PHASE.EXPLORING);
          onAssetTier?.("hallway");

          await sleep(ms(120), cancelledRef);
          setBlackout(false);
          setHintVisible(true);
          hallAudio.start();

          // the hint retires on its own if the player just stands still
          window.setTimeout(() => setHintVisible(false), 9000);

          /*
            Everything the office needs, fetched during the walk. Nothing here
            is awaited until the player actually asks for the door.
          */
          const officeLoad = loadOfficeAssets({
            scene,
            mode,
            team,
            accentPrimary,
            onTier: onAssetTier,
          })
            .then((build) => {
              if (cancelledRef.current) {
                return null;
              }

              officeBuild = build;
              tagAssets(build);
              setOfficeVisible(build, false);

              cameraDirector = createCameraDirector({
                scene,
                camera,
                assets: build.assets,
                logoPlane: build.logoPlane,
                cancelledRef,
              });

              build.dressing.then(() => {
                if (cancelledRef.current || scene.isDisposed) {
                  return;
                }
                tagAssets(build);
                setOfficeVisible(build, false);
                placePracticalLights(lighting, build.assets);
                registerShadowCasters(lighting, build.assets);
                if (build.errors.length) {
                  setAssetIssue(
                    build.errors.map(({ key }) => key).join(", ")
                  );
                }
              });

              return build;
            })
            .catch((loadError) => {
              console.error("Office assets failed", loadError);
              setAssetIssue("Parts of the office could not load.");
              return null;
            });

          await doorRequested;

          if (cancelledRef.current) {
            return;
          }

          /*
            ------------------------------------------------------------
            DOOR → OFFICE CINEMATIC
            ------------------------------------------------------------
          */
          setHallPhase(HALL_PHASE.DOOR);
          setPrompt(null);
          setCard(null);
          setHintVisible(false);
          hallController.suspend();
          hallController.releasePointer();
          setStage(CINEMATIC_STAGE.OFFICE_ENTRY);

          hallAudio.doorSwing();
          hallAudio.duck(0.4, 0.8);

          // the leaf swings and the camera steps up to the threshold together
          await Promise.all([
            tween({
              duration: ms(1150),
              cancelledRef,
              easing: easeOutCubic,
              onUpdate: (t) => {
                hall.door.open(t);
                hall.door.light.intensity = lerp(2.4, 5.6, t);
              },
            }),
            (async () => {
              await sleep(ms(120), cancelledRef);
              await cameraDirector.focus("doorApproach", {
                duration: ms(1000),
                walking: true,
              });
            })(),
          ]);

          if (cancelledRef.current) {
            return;
          }

          // if the player sprinted here, the wait happens now, behind a door
          const build = await officeLoad;
          await build?.dressing;

          if (cancelledRef.current) {
            return;
          }

          if (!build) {
            /*
              The office could not be assembled. The corridor is real and
              already standing, so it becomes the backdrop and the agreement
              opens against the open doorway instead of a black screen.
            */
            hallController.dispose();
            hallController = null;
            setHallPhase(HALL_PHASE.SETTLED);
            hallAudio.duck(0.6, 1.2);
            await cameraDirector.focus("doorThreshold", {
              duration: ms(1400),
              walking: true,
            });
            onComplete?.();
            return;
          }

          // cross the threshold, then hand the room over to the office grade
          await cameraDirector.focus("doorThreshold", {
            duration: ms(1500),
            walking: true,
          });

          if (cancelledRef.current) {
            return;
          }

          setHallPhase(HALL_PHASE.OFFICE);
          setStage(CINEMATIC_STAGE.MEETING);

          setOfficeVisible(build, true);
          setOfficeLightsEnabled(true);
          placePracticalLights(lighting, build.assets);
          registerShadowCasters(lighting, build.assets);
          hall.setEnabled(false);

          await tween({
            duration: ms(700),
            cancelledRef,
            easing: easeInOutCubic,
            onUpdate: (t) => {
              scene.imageProcessingConfiguration.exposure = lerp(1.04, 1.62, t);
              scene.imageProcessingConfiguration.contrast = lerp(1.18, 1.09, t);
              scene.imageProcessingConfiguration.vignetteWeight = lerp(
                2.6,
                1.4,
                t
              );
              scene.clearColor = new Color4(
                lerp(0.017, 0.06, t),
                lerp(0.019, 0.056, t),
                lerp(0.024, 0.05, t),
                1
              );
            },
          });

          if (cancelledRef.current) {
            return;
          }

          hallAudio.setDoorProximity(1);
          hallAudio.duck(0.5, 0.8);

          // establish the room
          await cameraDirector.focus("officeReveal", {
            duration: ms(2200),
            walking: true,
          });

          if (cancelledRef.current) {
            return;
          }

          /*
            The arrival line, played against the room rather than over it.
          */
          setNarration("Welcome to Puckception.");
          const spoken = speakBroadcastLine(
            "Welcome to Puckception.",
            hallAudio
          );

          await Promise.all([
            spoken,
            cameraDirector.focus("officeAddress", {
              duration: ms(2600),
              breathing: true,
            }),
          ]);

          if (cancelledRef.current) {
            return;
          }

          await sleep(ms(500), cancelledRef);
          setNarration("");
          hallAudio.duck(0.72, 1.2);

          if (cancelledRef.current) {
            return;
          }

          /*
            ------------------------------------------------------------
            THE CONTRACT ON THE DESK
            ------------------------------------------------------------
          */
          setStage(CINEMATIC_STAGE.CONTRACT);

          await cameraDirector.focus("deskApproach", {
            duration: ms(1700),
          });

          if (cancelledRef.current) {
            return;
          }

          await cameraDirector.focus("contractReveal", {
            duration: ms(1600),
            breathing: true,
          });

          if (cancelledRef.current) {
            return;
          }

          setHallPhase(HALL_PHASE.SETTLED);

          // the corridor is finished with; free its GPU budget for the hub
          hallController.dispose();
          hallController = null;
          hall.dispose();
          hall = null;

          controlRef.current = {
            ...(controlRef.current || {}),
            closeCard: () => {},
            setLowPower: (enabled) => {
              if (lowPower === enabled) return;
              lowPower = enabled;
              engine?.setHardwareScalingLevel(enabled ? 1.7 : 1);
            },
            setPausedRender,
            releaseInput: () => {
              if (document.pointerLockElement) {
                document.exitPointerLock?.();
              }
            },
          };

          // the setup interface now takes over the document
          onComplete?.();
          return;
        }

        officeBuild =
          await loadOfficeAssets({
            scene,
            mode,
            team,
            accentPrimary,
            onTier: onAssetTier,
          });

        await officeBuild.dressing;

        tagAssets(officeBuild);

        cameraDirector = createCameraDirector({
          scene,
          camera,
          assets: officeBuild.assets,
          logoPlane: officeBuild.logoPlane,
          cancelledRef,
        });

        placePracticalLights(lighting, officeBuild.assets);
        registerShadowCasters(
          lighting,
          officeBuild.assets
        );

        cameraDirector.snap("hallwayStart");

        if (
          officeBuild.errors.length
        ) {
          setAssetIssue(
            officeBuild.errors
              .map(
                ({ key }) => key
              )
              .join(", ")
          );
        }

        if (appointment) {
          try {
            signingSurface =
              createSigningSurface({
                scene,
                desk: officeBuild.assets.desk,
                team,
                gmName,
                playerUniverse,
                injuriesEnabled,
                contractDate,
              });
          } catch (signingError) {
            console.error(
              "Appointment signing surface failed",
              signingError
            );
            signingSurface = null;
          }

          if (!signingSurface) {
            setOfficeFailed(true);
            setSceneReady(true);
            setBlackout(false);
            setAssetIssue(
              "The 3D contract could not load."
            );
            return;
          }
        }

        scene.metadata = {
          purpose:
            appointment
              ? "franchise-appointment-glb-cinematic"
              : "franchise-intro-glb-cinematic",
          teamCode:
            appointment
              ? team?.code || ""
              : "",
          glbFirst: true,
          physicalUnits: "meters",
        };

        engine.runRenderLoop(renderScene);

        resizeHandler = () =>
          engine?.resize();

        window.addEventListener(
          "resize",
          resizeHandler
        );

        window.requestAnimationFrame(
          () =>
            engine?.resize()
        );

        /*
          Pointer / signature input.
        */
        if (appointment) {
          pointerObserver =
            scene.onPointerObservable.add(
              (pointerInfo) => {
                if (
                  !signatureUnlocked ||
                  !signingSurface
                ) {
                  return;
                }

                if (
                  pointerInfo.type !==
                    PointerEventTypes.POINTERDOWN &&
                  pointerInfo.type !==
                    PointerEventTypes.POINTERMOVE &&
                  pointerInfo.type !==
                    PointerEventTypes.POINTERUP
                ) {
                  return;
                }

                const pickInfo =
                  scene.pick(
                    scene.pointerX,
                    scene.pointerY,
                    (mesh) =>
                      mesh ===
                      signingSurface.plane,
                    false,
                    camera
                  );

                if (
                  !pickInfo?.hit ||
                  pickInfo.pickedMesh !==
                    signingSurface.plane
                ) {
                  if (
                    pointerInfo.type ===
                    PointerEventTypes.POINTERUP
                  ) {
                    signatureDrawing =
                      false;

                    lastSignaturePoint =
                      null;
                  }

                  return;
                }

                const uv =
                  pickInfo.getTextureCoordinates?.();

                if (!uv) {
                  return;
                }

                const x =
                  uv.x *
                  CONTRACT_TEXTURE.width;

                const y =
                  (1 - uv.y) *
                  CONTRACT_TEXTURE.height;

                const zone =
                  SIGNATURE_ZONE;

                const inside =
                  x >= zone.x &&
                  x <=
                    zone.x +
                      zone.width &&
                  y >= zone.y &&
                  y <=
                    zone.y +
                      zone.height;

                const ctx =
                  signingSurface.texture.getContext();

                if (
                  pointerInfo.type ===
                  PointerEventTypes.POINTERDOWN
                ) {
                  if (!inside) {
                    return;
                  }

                  signatureDrawing =
                    true;

                  lastSignaturePoint =
                    {
                      x,
                      y,
                    };

                  ctx.fillStyle =
                    "#13223a";

                  ctx.beginPath();
                  ctx.arc(
                    x,
                    y,
                    3.1,
                    0,
                    Math.PI * 2
                  );
                  ctx.fill();

                  signingSurface.texture.update(
                    true
                  );

                  pointerInfo.event?.preventDefault?.();

                  return;
                }

                if (
                  pointerInfo.type ===
                    PointerEventTypes.POINTERMOVE &&
                  signatureDrawing
                ) {
                  const px =
                    clamp(
                      x,
                      zone.x,
                      zone.x +
                        zone.width
                    );

                  const py =
                    clamp(
                      y,
                      zone.y,
                      zone.y +
                        zone.height
                    );

                  if (
                    !lastSignaturePoint
                  ) {
                    lastSignaturePoint =
                      {
                        x: px,
                        y: py,
                      };

                    return;
                  }

                  const dx =
                    px -
                    lastSignaturePoint.x;

                  const dy =
                    py -
                    lastSignaturePoint.y;

                  const distance =
                    Math.sqrt(
                      dx * dx +
                        dy * dy
                    );

                  ctx.strokeStyle =
                    "#13223a";

                  ctx.lineWidth = 6.0;
                  ctx.lineCap = "round";
                  ctx.lineJoin = "round";

                  ctx.beginPath();
                  ctx.moveTo(
                    lastSignaturePoint.x,
                    lastSignaturePoint.y
                  );
                  ctx.lineTo(
                    px,
                    py
                  );
                  ctx.stroke();

                  signingSurface.texture.update(
                    true
                  );

                  signatureDistance +=
                    distance;

                  lastSignaturePoint =
                    {
                      x: px,
                      y: py,
                    };

                  if (
                    signatureDistance >=
                      160 &&
                    !signatureReadyRef.current
                  ) {
                    signatureReadyRef.current =
                      true;

                    setSignatureReady(
                      true
                    );
                  }

                  pointerInfo.event?.preventDefault?.();

                  return;
                }

                if (
                  pointerInfo.type ===
                  PointerEventTypes.POINTERUP
                ) {
                  signatureDrawing =
                    false;

                  lastSignaturePoint =
                    null;

                  if (
                    signatureDistance >=
                      160 &&
                    !signatureReadyRef.current
                  ) {
                    signatureReadyRef.current =
                      true;

                    setSignatureReady(
                      true
                    );
                  }

                  pointerInfo.event?.preventDefault?.();
                }
              }
            );
        }

        setSceneReady(true);
        await sleep(ms(80), cancelledRef);
        setBlackout(false);

        await sleep(
          ms(220),
          cancelledRef
        );

        if (
          cancelledRef.current
        ) {
          return;
        }

        /*
          ------------------------------------------------------------
          APPOINTMENT CINEMATIC
          ------------------------------------------------------------
        */

        lighting.teamFill.intensity =
          0.95;

        lighting.glow.intensity =
          0.42;

        setStage(
          CINEMATIC_STAGE.HALLWAY
        );

        await cameraDirector.focus(
          "hallwayMid",
          {
            duration: ms(1800),
            walking: true,
          }
        );

        if (
          cancelledRef.current
        ) {
          return;
        }

        setStage(
          CINEMATIC_STAGE.OFFICE_ENTRY
        );

        await openEntranceDoors(
          officeBuild.doors,
          cancelledRef,
          ms(900)
        );

        await cameraDirector.focus(
          "officeWide",
          {
            duration: ms(1800),
          }
        );

        if (
          cancelledRef.current
        ) {
          return;
        }

        setStage(
          CINEMATIC_STAGE.MEETING
        );

        await cameraDirector.focus(
          "gmHero",
          {
            duration: ms(1600),
          }
        );

        if (
          cancelledRef.current
        ) {
          return;
        }

        await sleep(
          ms(700),
          cancelledRef
        );

        if (
          cancelledRef.current
        ) {
          return;
        }

        setStage(
          CINEMATIC_STAGE.CONTRACT
        );

        /*
          Slide the physical GLB contract from the executive's side
          toward the user.

          The GLB remains the physical prop.
          The high-resolution DynamicTexture surface appears afterward
          only for readable interaction.
        */
        const physicalContract =
          officeBuild.assets.contract;

        const physicalClipboard =
          officeBuild.assets.clipboard;

        const contractStart =
          physicalContract?.root
            ?.position
            ?.clone();

        const clipboardStart =
          physicalClipboard?.root
            ?.position
            ?.clone();

        await tween({
          duration: ms(1400),
          cancelledRef,
          easing: easeOutQuint,
          onUpdate: (t) => {
            if (
              physicalContract &&
              contractStart
            ) {
              physicalContract.root.position.z =
                lerp(
                  contractStart.z,
                  2.52,
                  t
                );

              physicalContract.root.position.x =
                lerp(
                  contractStart.x,
                  0.08,
                  t
                );
            }

            if (
              physicalClipboard &&
              clipboardStart
            ) {
              physicalClipboard.root.position.x =
                lerp(
                  clipboardStart.x,
                  -0.88,
                  t
                );
            }
          },
        });

        if (
          cancelledRef.current
        ) {
          return;
        }

        camera.minZ = 0.025;

        await cameraDirector.focus(
          "contract",
          {
            duration: ms(1400),
            easing: easeOutCubic,
          }
        );

        if (
          cancelledRef.current
        ) {
          return;
        }

        signingSurface.plane.setEnabled(
          true
        );

        signatureUnlocked =
          true;

        canvas.style.cursor =
          "crosshair";

        setStage(
          CINEMATIC_STAGE.SIGNING
        );

        /*
          Sequence continues only after user clicks Execute.
        */
        executeRef.current =
          async () => {
            if (
              postSequenceStarted ||
              !signatureReadyRef.current ||
              cancelledRef.current
            ) {
              return;
            }

            postSequenceStarted = true;
            setExecuting(true);

            signatureUnlocked =
              false;

            signatureDrawing =
              false;

            canvas.style.cursor =
              "default";

            setStage(
              CINEMATIC_STAGE.SIGNED
            );

            drawContractTexture({
              texture:
                signingSurface.texture,
              team,
              gmName,
              playerUniverse,
              injuriesEnabled,
              contractDate,
              executed: true,
            });

            await sleep(
              ms(650),
              cancelledRef
            );

            if (
              cancelledRef.current
            ) {
              return;
            }

            const signedStart =
              signingSurface.plane.position.clone();

            await tween({
              duration: ms(700),
              cancelledRef,
              easing: easeInOutCubic,
              onUpdate: (t) => {
                signingSurface.plane.position.y =
                  lerp(
                    signedStart.y,
                    signedStart.y - 0.04,
                    t
                  );

                const scale =
                  lerp(
                    1,
                    0.86,
                    t
                  );

                signingSurface.plane.scaling.set(
                  scale,
                  scale,
                  scale
                );
              },
            });

            signingSurface.plane.setEnabled(
              false
            );

            if (
              cancelledRef.current
            ) {
              return;
            }

            /*
              Use the actual standing-suit GLB for the handshake beat.
              We do NOT manufacture a box/sphere character.
            */
            setStage(
              CINEMATIC_STAGE.HANDSHAKE
            );

            setBlackout(true);

            await sleep(
              ms(320),
              cancelledRef
            );

            if (
              cancelledRef.current
            ) {
              return;
            }

            setAssetEnabled(
              officeBuild.assets
                .seatedExecutive,
              false
            );

            setAssetEnabled(
              officeBuild.assets
                .standingExecutive,
              true
            );

            playBestAnimation(
              officeBuild.assets
                .standingExecutive,
              [
                "handshake",
                "greet",
                "gesture",
                "wave",
                "idle",
              ],
              {
                loop: false,
                speedRatio: 1,
              }
            );

            camera.minZ = 0.08;

            await cameraDirector.focus(
              "standingExecutive",
              {
                duration: ms(650),
                easing: easeOutCubic,
              }
            );

            const handshakeTarget =
              cameraDirector.targetFor(
                "standingExecutive"
              ) ||
              camera.position.add(
                camera
                  .getForwardRay(1)
                  .direction.scale(3)
              );

            setBlackout(false);

            await sleep(
              ms(180),
              cancelledRef
            );

            if (
              cancelledRef.current
            ) {
              return;
            }

            /*
              Physical-feeling handshake camera impulse.
              No cartoon first-person arm is generated.
            */
            const base =
              camera.position.clone();

            await tween({
              duration: ms(1250),
              cancelledRef,
              easing: easeInOutCubic,
              onUpdate: (_t, raw) => {
                const envelope =
                  Math.sin(
                    raw * Math.PI
                  );

                const vertical =
                  Math.sin(
                    raw *
                      Math.PI *
                      5
                  ) *
                  0.018 *
                  envelope;

                const forward =
                  Math.sin(
                    raw * Math.PI
                  ) *
                  0.055;

                camera.position.set(
                  base.x,
                  base.y +
                    vertical,
                  base.z +
                    forward
                );

                camera.rotation.z = 0;

                camera.setTarget(
                  handshakeTarget
                );
              },
            });

            if (
              cancelledRef.current
            ) {
              return;
            }

            setStage(
              CINEMATIC_STAGE.WELCOME
            );

            setWelcome(true);

            const startingIntensity =
              lighting.teamFill
                .intensity;

            await Promise.all([
              cameraDirector.focus(
                "branding",
                {
                  duration: ms(1700),
                }
              ),

              tween({
                duration: ms(1700),
                cancelledRef,
                easing: easeOutCubic,
                onUpdate: (t) => {
                  lighting.teamFill.intensity =
                    lerp(
                      startingIntensity,
                      2.1,
                      t
                    );

                  lighting.glow.intensity =
                    lerp(
                      0.28,
                      0.56,
                      t
                    );
                },
              }),
            ]);

            if (
              cancelledRef.current
            ) {
              return;
            }

            await sleep(
              ms(2300),
              cancelledRef
            );

            if (
              cancelledRef.current
            ) {
              return;
            }

            setBlackout(true);

            await sleep(
              ms(850),
              cancelledRef
            );

            if (
              cancelledRef.current
            ) {
              return;
            }

            onComplete?.();
          };
      } catch (error) {
        console.error(
          "GLB executive cinematic failed",
          error
        );

        const message =
          error?.message ||
          "The cinematic could not initialize.";

        setAssetIssue(message);
        setSceneReady(true);
        setBlackout(false);

        if (appointment) {
          setOfficeFailed(true);
          return;
        }

        // no walkable floor: fall back to the flat setup, with the menu bed
        onFloorFailure?.();
        onComplete?.();
      }
    };

    run();

    return () => {
      cancelledRef.current =
        true;

      executeRef.current =
        null;

      controlRef.current = null;

      if (hallController) {
        hallController.dispose();
        hallController = null;
      }

      if (hallAudio) {
        hallAudio.dispose();
        hallAudio = null;
      }

      if (typeof window !== "undefined") {
        try {
          window.speechSynthesis?.cancel();
        } catch (_speechError) {
          /* nothing to cancel */
        }
      }

      if (
        scene &&
        pointerObserver
      ) {
        scene.onPointerObservable.remove(
          pointerObserver
        );
      }

      if (resizeHandler) {
        window.removeEventListener(
          "resize",
          resizeHandler
        );
      }

      if (
        scene &&
        !scene.isDisposed
      ) {
        scene.dispose();
      }

      if (
        engine &&
        !engine.isDisposed
      ) {
        engine.dispose();
      }
    };
  }, [
    mode,
    team,
    gmName,
    playerUniverse,
    injuriesEnabled,
    accentPrimary,
    accentSecondary,
    contractDate,
    onComplete,
    onAssetTier,
    onFloorFailure,
  ]);

  useEffect(() => {
    if (!sceneControlRef) {
      return undefined;
    }

    sceneControlRef.current = controlRef;

    return () => {
      sceneControlRef.current = null;
    };
  }, [sceneControlRef]);

  /*
    While the agreement is on screen the room only has to sit there. Dropping
    the render resolution hands that budget to the hub warm-up instead.
  */
  useEffect(() => {
    controlRef.current?.setLowPower?.(Boolean(overlayActive));
    controlRef.current?.setPausedRender?.(Boolean(overlayActive));

    if (overlayActive) {
      controlRef.current?.releaseInput?.();
      if (document.pointerLockElement) {
        document.exitPointerLock?.();
      }
      controlRef.current?.duckAudio?.(0.5);
      setCard(null);
    }
  }, [overlayActive]);

  const closeCard = useCallback(() => {
    controlRef.current?.closeCard?.();
    setCard(null);
  }, []);

  useEffect(() => {
    if (!card) {
      return undefined;
    }

    const onKeyDown = (event) => {
      if (
        event.key === "Escape" ||
        event.code === "KeyE" ||
        event.code === "Enter" ||
        event.code === "Space"
      ) {
        event.preventDefault();
        event.stopPropagation();
        closeCard();
      }
    };

    window.addEventListener("keydown", onKeyDown, true);

    return () =>
      window.removeEventListener("keydown", onKeyDown, true);
  }, [card, closeCard]);

  const executeAppointment =
    useCallback(() => {
      if (
        !signatureReady ||
        executing
      ) {
        return;
      }

      executeRef.current?.();
    }, [
      signatureReady,
      executing,
    ]);

  const confirmFallbackAppointment =
    useCallback(() => {
      if (executing) {
        return;
      }

      setExecuting(true);
      onComplete?.();
    }, [
      executing,
      onComplete,
    ]);

  /*
    Escape releases the mouse while exploring — it must not throw the player
    out of the hallway. It only skips once the cinematic has taken over.
  */
  useEffect(() => {
    if (
      appointment ||
      hallPhase === HALL_PHASE.EXPLORING ||
      hallPhase === HALL_PHASE.SETTLED
    ) {
      return undefined;
    }

    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        (onSkipIntro || onComplete)?.();
      }
    };

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
    appointment,
    hallPhase,
    onComplete,
    onSkipIntro,
  ]);

  return (
    <section
      className={`setup-cinematic ${
        hallPhase === HALL_PHASE.EXPLORING
          ? "setup-cinematic--free"
          : ""
      } ${
        hallPhase === HALL_PHASE.EXPLORING && !card
          ? "setup-cinematic--roaming"
          : ""
      } ${
        overlayActive
          ? "setup-cinematic--overlay"
          : ""
      }`}
      aria-label={
        appointment
          ? `${team?.name || "NHL"} executive appointment`
          : "Puckception executive floor"
      }
    >
      <canvas
        ref={canvasRef}
        className="setup-cinematic-canvas"
      />

      <div
        className={`setup-blackout ${
          blackout
            ? "is-black"
            : ""
        }`}
        aria-hidden="true"
      />

      <div
        className="setup-letterbox setup-letterbox--top"
        aria-hidden="true"
      />

      <div
        className="setup-letterbox setup-letterbox--bottom"
        aria-hidden="true"
      />

      {!officeFailed &&
      !overlayActive &&
      hallPhase !== HALL_PHASE.EXPLORING ? (
      <div className="setup-cinematic-status">
        <span>
          NHL Executive Floor
        </span>

        <strong>
          {stageCopy}
        </strong>
      </div>
      ) : null}

      {/* ---------- opening hallway HUD ---------- */}

      {hallPhase === HALL_PHASE.EXPLORING && !overlayActive ? (
        <>
          <div
            className={`hall-reticle ${
              prompt ? "is-live" : ""
            }`}
            aria-hidden="true"
          >
            <span />
          </div>

          <div
            className={`hall-prompt ${
              prompt && !card ? "is-shown" : ""
            }`}
            aria-hidden={!prompt || Boolean(card)}
          >
            <kbd>E</kbd>
            <em>{prompt?.hint || "Inspect"}</em>
            <span>{prompt?.label || ""}</span>
          </div>

          <div
            className={`hall-hint ${
              hintVisible && !card ? "is-shown" : ""
            }`}
          >
            <p>
              <kbd>W</kbd>
              <kbd>A</kbd>
              <kbd>S</kbd>
              <kbd>D</kbd>
              <small>or arrows to walk</small>
            </p>

            <p>
              <kbd>Mouse</kbd>
              <small>to look — click once to hold the room</small>
            </p>

            <p>
              <kbd>E</kbd>
              <small>to inspect what the reticle finds</small>
            </p>
          </div>

          {dartTally > 0 ? (
            <div className="hall-dart-tally" aria-hidden="true">
              <strong>{dartTally}</strong>
              <span>on target</span>
            </div>
          ) : null}
        </>
      ) : null}

      {card ? (
        <div className="hall-card-layer">
          <article className="hall-card">
            <header>
              <small>{card.kicker}</small>

              <h3>{card.title}</h3>

              <p>{card.subtitle}</p>
            </header>

            <div className="hall-card-body">
              {(card.lines || []).map((line, index) => (
                <p key={index}>{line}</p>
              ))}
            </div>

            <footer>
              <span>{card.footer}</span>

              <button
                type="button"
                onClick={closeCard}
                autoFocus
              >
                Close
              </button>
            </footer>
          </article>
        </div>
      ) : null}

      {narration ? (
        <div className="hall-narration">
          <span aria-hidden="true" />

          <p>{narration}</p>
        </div>
      ) : null}

      {appointment && team ? (
        <div className="setup-cinematic-club">
          <span>
            {team.code}
          </span>

          <strong>
            {team.name}
          </strong>
        </div>
      ) : null}

      {!appointment &&
      !overlayActive &&
      hallPhase !== HALL_PHASE.SETTLED ? (
        <button
          type="button"
          className="setup-skip-intro"
          onClick={
            onSkipIntro ||
            onComplete
          }
        >
          {hallPhase === HALL_PHASE.EXPLORING
            ? "Skip to setup"
            : "Skip cinematic"}
        </button>
      ) : null}

      {appointment && officeFailed ? (
        <div className="setup-appointment-fallback">
          <TeamLogo
            team={team}
            size={72}
          />

          <span>
            Appointment confirm
          </span>

          <h2>
            {team?.name || "NHL club"}
          </h2>

          <p>
            The 3D office could not load. Confirm to take the job as{" "}
            {gmName?.trim() || "General Manager"}.
          </p>

          <button
            type="button"
            onClick={
              confirmFallbackAppointment
            }
            disabled={executing}
          >
            Execute Appointment for{" "}
            {team?.name || "this club"}
          </button>
        </div>
      ) : null}

      {stage ===
      CINEMATIC_STAGE.SIGNING ? (
        <div
          className={`setup-signing-prompt ${
            signatureReady
              ? "is-ready"
              : ""
          }`}
        >
          <strong>
            {signatureReady
              ? "Signature captured"
              : "Sign the appointment"}
          </strong>

          <p>
            {signatureReady
              ? "Execute the agreement when the signature is complete."
              : "Draw directly on the physical document."}
          </p>

          {signatureReady ? (
            <button
              type="button"
              onClick={
                executeAppointment
              }
              disabled={executing}
            >
              Execute Appointment
            </button>
          ) : null}
        </div>
      ) : null}

      {welcome ? (
        <div className="setup-team-welcome">
          <TeamLogo
            team={team}
            size={92}
          />

          <span>
            General Manager
          </span>

          <strong>
            {gmName?.trim() ||
              "General Manager"}
          </strong>

          <h2>
            {team?.name}
          </h2>
        </div>
      ) : null}

      {!sceneReady && !officeFailed ? (
        <div className="setup-cinematic-loader">
          <span />

          <strong>
            Opening the executive floor
          </strong>
        </div>
      ) : null}

      {assetIssue && !officeFailed && !overlayActive ? (
        <div className="setup-asset-warning">
          <strong>
            Asset loading notice
          </strong>

          <span>
            {assetIssue}
          </span>
        </div>
      ) : null}
    </section>
  );
}


/* ============================================================================
   FRANCHISE LOADING
   ========================================================================== */

/*
  The loading state is a translucent treatment over the office the player is
  already standing in, not an opaque screen. There is no timer and no invented
  percentage: each resource category simply reports what it is doing, in plain
  language, and the facts keep turning for as long as the work honestly takes.
*/
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
  const facts =
    useMemo(
      () =>
        shuffleArray(
          NHL_FUN_FACTS
        ),
      []
    );

  const [
    factIndex,
    setFactIndex,
  ] = useState(0);

  useEffect(() => {
    const id =
      window.setInterval(
        () => {
          setFactIndex(
            (current) =>
              (current + 1) %
              facts.length
          );
        },
        8500
      );

    return () =>
      window.clearInterval(id);
  }, [facts.length]);

  const failed = Boolean(error);

  const categories =
    useMemo(
      () =>
        Object.entries(
          HUB_WARMUP_LABELS
        ).map(([key, label]) => ({
          key,
          label,
          status: warmup?.[key] || "waiting",
        })),
      [warmup]
    );

  const settled =
    categories.filter(
      ({ status }) => status === "ready"
    ).length;

  return (
    <div
      className="setup-loading-screen"
      role="status"
      aria-live="polite"
    >
      <div className="setup-loading-panel">
        <small>
          Franchise Mode
        </small>

        <h2>
          {selected?.name ||
            "NHL"}
        </h2>

        <p>
          {failed
            ? error
            : playerUniverse ===
              "real_nhl"
            ? "Opening the real NHL player universe."
            : `${
                gmName?.trim() ||
                "General Manager"
              } is taking over hockey operations.`}
        </p>

        {!failed ? (
          <div className="setup-loading-meta">
            <span>
              {injuriesEnabled
                ? "Injuries on"
                : "Injuries off"}
            </span>

            <span>
              {playerUniverse ===
              "real_nhl"
                ? "Real NHL"
                : "Generated"}
            </span>
          </div>
        ) : null}

        {!failed ? (
          <>
            <ul className="setup-loading-tasks">
              {categories.map(
                ({ key, label, status }) => (
                  <li
                    key={key}
                    className={`is-${status}`}
                  >
                    <i aria-hidden="true" />

                    <span>{label}</span>

                    <em>
                      {status === "ready"
                        ? "Ready"
                        : status === "loading"
                        ? "Arriving"
                        : "Queued"}
                    </em>
                  </li>
                )
              )}
            </ul>

            <div
              className={`setup-loading-bar ${
                settled === categories.length
                  ? "is-complete"
                  : ""
              }`}
              aria-hidden="true"
            >
              <span />
            </div>
          </>
        ) : null}

        {failed ? (
          <div className="setup-loading-actions">
            {typeof onRetry === "function" ? (
              <button
                type="button"
                className="setup-loading-retry"
                onClick={onRetry}
                disabled={loading && !failed}
              >
                Retry
              </button>
            ) : null}
            {typeof onBack === "function" ? (
              <button
                type="button"
                className="setup-loading-back"
                onClick={onBack}
              >
                Back to setup
              </button>
            ) : null}
          </div>
        ) : null}

        {!failed ? (
          <blockquote>
            {facts[factIndex]}
          </blockquote>
        ) : null}
      </div>
    </div>
  );
}


/* ============================================================================
   ROOT SCREEN
   ========================================================================== */

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
    teamsLoading,
    error,
    setError,
    hubWarmup,
    primeHubAssets,
  } = useGameUI();

  /*
    The corridor grows its own soundtrack, so the menu theme stays quiet for
    the whole of this screen.
  */
  const sceneControlRef = useRef(null);

  const [
    appStage,
    setAppStage,
  ] = useState(
    APP_STAGE.INTRO
  );

  const [
    signatureReady,
    setSignatureReady,
  ] = useState(false);

  const [
    pickedTeamCode,
    setPickedTeamCode,
  ] = useState("");

  const [
    statusText,
    setStatusText,
  ] = useState(
    "Executive introduction."
  );

  const [
    floorFailed,
    setFloorFailed,
  ] = useState(false);

  useSetupStageMusic(floorFailed);

  useEffect(() => {
    if (appStage !== APP_STAGE.CONFIGURE) {
      return;
    }
    loadTeams();
  }, [appStage, loadTeams]);

  /*
    Progressive priority. The corridor is on screen first; the hub's expensive
    resources begin arriving the moment the player can walk, and the rest
    follows as the cinematic and the agreement consume real time.
  */
  const handleAssetTier = useCallback(
    (tier) => {
      if (tier === "hallway") {
        primeHubAssets(HUB_WARMUP_STAGES.ENVIRONMENT);
        return;
      }

      if (tier === "officeEssentials") {
        window.setTimeout(() => {
          primeHubAssets(HUB_WARMUP_STAGES.CRESTS);
        }, 1800);
        return;
      }

      if (tier === "officeDressing") {
        primeHubAssets(HUB_WARMUP_STAGES.OPERATIONS);
      }
    },
    [primeHubAssets]
  );

  useEffect(() => {
    if (appStage === APP_STAGE.INTRO) {
      return;
    }

    // the agreement is a long, quiet window; use all of it
    primeHubAssets(HUB_WARMUP_STAGES.ENVIRONMENT);
    primeHubAssets(HUB_WARMUP_STAGES.CRESTS);
    primeHubAssets(HUB_WARMUP_STAGES.OPERATIONS);
  }, [appStage, primeHubAssets]);

  const orderedTeams =
    useMemo(
      () =>
        buildOrderedTeams(
          teams.length
            ? teams
            : buildDefaultFranchiseTeamList()
        ),
      [teams]
    );

  const orderedIndex =
    useMemo(
      () => {
        if (pickedTeamCode) {
          const byCode = orderedTeams.findIndex(
            (item) => item.code === pickedTeamCode
          );
          if (byCode >= 0) {
            return byCode;
          }
        }
        return findOrderedIndexFromSetupIndex(
          orderedTeams,
          setupTeamIndex
        );
      },
      [
        orderedTeams,
        pickedTeamCode,
        setupTeamIndex,
      ]
    );

  const selected =
    orderedTeams[
      orderedIndex
    ] || null;

  const selectedCode =
    selected?.code || "";

  const [
    accentPrimary,
    accentSecondary,
  ] = teamAccentForCode(
    selectedCode
  );

  const contractDate =
    useMemo(
      () =>
        formatContractDate(),
      []
    );

  useEffect(() => {
    if (!pickedTeamCode) {
      return;
    }
    const match = orderedTeams.find(
      (item) => item.code === pickedTeamCode
    );
    if (
      match &&
      match.index !== setupTeamIndex
    ) {
      setSetupTeamIndex(match.index);
    }
  }, [
    pickedTeamCode,
    orderedTeams,
    setupTeamIndex,
    setSetupTeamIndex,
  ]);

  const setTeamByOrderedIndex =
    useCallback(
      (nextIndex) => {
        if (
          !orderedTeams.length
        ) {
          return;
        }

        const safeIndex =
          (
            (
              nextIndex %
              orderedTeams.length
            ) +
            orderedTeams.length
          ) %
          orderedTeams.length;

        const team =
          orderedTeams[
            safeIndex
          ];

        if (!team) {
          return;
        }

        setPickedTeamCode(
          team.code || ""
        );

        setSetupTeamIndex(
          team.index
        );

        setStatusText(
          `${team.name} selected.`
        );
      },
      [orderedTeams, setSetupTeamIndex]
    );

  const finishIntro =
    useCallback(() => {
      sceneControlRef.current?.current?.releaseInput?.();
      sceneControlRef.current?.current?.setPausedRender?.(true);
      sceneControlRef.current?.current?.duckAudio?.(0.35);
      setAppStage(
        APP_STAGE.CONFIGURE
      );

      setStatusText(
        "Select a club and configure the franchise."
      );
    }, []);

  const reportFloorFailure =
    useCallback(() => {
      setFloorFailed(true);
    }, []);

  const finishAppointment =
    useCallback(async () => {
      setAppStage(
        APP_STAGE.STARTING
      );

      // the room settles down while the franchise is assembled
      sceneControlRef.current?.current?.duckAudio?.(0.28);

      setStatusText(
        `Appointment executed. Opening ${selected?.name || "franchise"} hockey operations.`
      );

      try {
        const result =
          await Promise.resolve(
            beginFranchise()
          );

        if (result && result.ok === false) {
          setStatusText(
            result.error ||
              "Franchise start failed. Retry or return to setup."
          );
        }
      } catch (startError) {
        console.error(
          "Unable to begin franchise",
          startError
        );

        setStatusText(
          "Franchise start failed. Retry or return to setup."
        );
      }
    }, [
      beginFranchise,
      selected,
    ]);

  /*
    During the generic intro there is intentionally no team accent.
    Once configuration begins, the selected team can own the accent.
  */
  const visualPrimary =
    appStage === APP_STAGE.INTRO
      ? "#c9a86a"
      : accentPrimary;

  const visualSecondary =
    appStage === APP_STAGE.INTRO
      ? "#9aa5b1"
      : accentSecondary;

  return (
    <div
      className="nhlcal-root setup-root"
      style={{
        "--team-accent":
          visualPrimary,
        "--team-accent-2":
          visualSecondary,
      }}
    >
      {/*
        The floor stays mounted for the whole screen. The agreement is signed
        on the desk it was revealed on, and the loading state that follows has
        a real room behind it rather than a black rectangle.
      */}
      <ExecutiveCinematic
        mode={
          CINEMATIC_MODE.INTRO
        }
        team={null}
        gmName=""
        playerUniverse="generated"
        injuriesEnabled={false}
        accentPrimary="#c9a86a"
        accentSecondary="#9aa5b1"
        contractDate={
          contractDate
        }
        onComplete={
          finishIntro
        }
        onSkipIntro={
          finishIntro
        }
        onAssetTier={
          handleAssetTier
        }
        onFloorFailure={
          reportFloorFailure
        }
        sceneControlRef={
          sceneControlRef
        }
        overlayActive={
          appStage !== APP_STAGE.INTRO
        }
      />

      {appStage ===
      APP_STAGE.CONFIGURE ? (
        <main className="setup-config-layout setup-config-layout--desk">
          <div className="setup-config-topline">
            <strong>
              Franchise Agreement
            </strong>

            <small>
              Complete the schedule, then sign and execute
            </small>
          </div>

          <div className="setup-config-grid">
            <ConfigurationPanel
              selected={
                selected
              }
              gmName={
                gmName
              }
              setGmName={
                setGmName
              }
              playerUniverse={
                playerUniverse
              }
              setPlayerUniverse={
                setPlayerUniverse
              }
              injuriesEnabled={
                injuriesEnabled
              }
              setInjuriesEnabled={
                setInjuriesEnabled
              }
              onAccept={
                finishAppointment
              }
              loading={
                loading
              }
              error={
                error
              }
              contractDate={
                contractDate
              }
              signatureReady={
                signatureReady
              }
              setSignatureReady={
                setSignatureReady
              }
            />

            <TeamSelection
              teams={
                orderedTeams
              }
              selectedIndex={
                orderedIndex
              }
              onSelect={
                setTeamByOrderedIndex
              }
              loading={
                teamsLoading
              }
            />
          </div>

          <p
            className="setup-sr-status"
            aria-live="polite"
          >
            {statusText}
          </p>
        </main>
      ) : null}

      {appStage ===
      APP_STAGE.STARTING ? (
        <SetupLoadingScreen
          selected={
            selected
          }
          gmName={
            gmName
          }
          injuriesEnabled={
            injuriesEnabled
          }
          playerUniverse={
            playerUniverse
          }
          error={
            error
          }
          loading={
            loading
          }
          warmup={
            hubWarmup
          }
          onRetry={
            finishAppointment
          }
          onBack={() => {
            setError(null);
            setAppStage(
              APP_STAGE.CONFIGURE
            );
            setStatusText(
              "Select a club and configure the franchise."
            );
          }}
        />
      ) : null}

      <style>
        {SETUP_SCREEN_CSS}
      </style>
    </div>
  );
}


/* ============================================================================
   SINGLE-FILE STYLING
   ========================================================================== */

const SETUP_SCREEN_CSS = `
.nhlcal-root.setup-root {
  --setup-font:
    var(
      --font-ops-ui,
      Inter,
      ui-sans-serif,
      system-ui,
      -apple-system,
      BlinkMacSystemFont,
      "Segoe UI",
      sans-serif
    );

  --setup-bg: #080a0e;
  --setup-panel: #12151c;
  --setup-panel-strong: #0d1016;
  --setup-text: #f0ede6;
  --setup-muted: rgba(229, 225, 216, 0.62);
  --setup-line: rgba(201, 168, 106, 0.28);
  --setup-gold: #c9a86a;

  position: fixed;
  z-index: 23000;
  inset: 0;

  display: flex;
  flex-direction: column;
  grid-template-columns: none;
  grid-template-rows: none;

  width: 100vw;
  height: 100dvh;
  min-height: 100dvh;

  overflow: hidden;

  font-family:
    var(--setup-font);

  color:
    var(--setup-text);

  background: transparent;
}

.setup-root *,
.setup-root *::before,
.setup-root *::after {
  box-sizing: border-box;
}

.setup-root button,
.setup-root input,
.setup-root fieldset,
.setup-root legend {
  font-family:
    var(--setup-font);
}

.setup-root button {
  -webkit-tap-highlight-color:
    transparent;
}

.setup-root button:focus-visible,
.setup-root input:focus-visible {
  outline: 2px solid
    color-mix(
      in srgb,
      var(--team-accent) 80%,
      #fff
    );
  outline-offset: 2px;
}


/* --------------------------------------------------------------------------
   CONFIGURATION
   -------------------------------------------------------------------------- */

.setup-config-layout {
  position: relative;

  width: 100%;
  height: 100%;
  flex: 1;
  min-height: 0;

  display: grid;

  grid-template-rows:
    auto
    minmax(0, 1fr);

  gap: 2px;

  padding: 2px;

  overflow: hidden;

  animation:
    setupConfigArrive
    640ms
    cubic-bezier(.2, .72, .2, 1)
    both;
}

.setup-config-layout::before {
  content: "";

  position: absolute;
  inset: 0;

  pointer-events: none;

  opacity: 0.09;

  background:
    repeating-linear-gradient(
      155deg,
      rgba(255, 255, 255, 0.035)
        0 1px,
      transparent
        1px 20px
    );
}

.setup-config-topline {
  position: relative;
  z-index: 2;

  min-height: 28px;

  display: grid;

  grid-template-columns:
    auto
    1fr;

  align-items: center;

  gap: 12px;

  padding: 0 2px;

  border-bottom: none;

  text-transform: uppercase;

  letter-spacing:
    0.13em;
}

.setup-config-topline span,
.setup-config-topline small {
  font-size: 10px;
  font-weight: 800;
  color: var(--setup-muted);
}

.setup-config-topline strong {
  font-size:
    clamp(
      13px,
      1.3vw,
      17px
    );

  font-weight: 900;

  color: var(--setup-gold);
}

.setup-config-topline small {
  justify-self: end;

  text-align: right;

  color: var(--setup-gold);
}

.setup-config-grid {
  position: relative;
  z-index: 2;

  min-height: 0;
  height: 100%;

  display: grid;
  grid-template-columns: minmax(360px, 0.4fr) minmax(0, 0.6fr);
  gap: 2px;
}

.setup-config-grid::before {
  content: none;
}

.setup-team-selector,
.setup-config-card {
  min-height: 0;
  border: none;
  background: transparent;
  box-shadow: none;
  color: var(--setup-text);
}

.setup-team-selector {
  position: relative;
  z-index: 1;

  display: grid;
  grid-template-rows: auto minmax(0, 1fr);

  overflow: hidden;
  padding: 2px 2px 2px 8px;
  background: transparent;
}

.setup-team-picker-fallback {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(108px, 1fr));
  gap: 6px;
  overflow: auto;
  max-height: min(52vh, 420px);
  padding: 4px 2px 8px;
}

.setup-team-picker-fallback button {
  border: 1px solid rgba(201, 168, 106, 0.22);
  border-radius: 4px;
  background: rgba(8, 10, 14, 0.72);
  color: rgba(236, 232, 224, 0.88);
  padding: 8px 10px;
  font-size: 11px;
  font-weight: 800;
  text-align: left;
  cursor: pointer;
}

.setup-team-picker-fallback button.is-selected {
  border-color: rgba(201, 168, 106, 0.72);
  background: rgba(201, 168, 106, 0.14);
  color: #f3e6c8;
}

.setup-panel-heading {
  position: relative;
  top: auto;
  left: auto;
  z-index: 2;
  min-height: 0;
  pointer-events: none;

  display: flex;

  flex-direction: column;

  justify-content: center;

  gap: 2px;

  padding: 2px 0 4px;

  border-bottom: none;
}

.setup-panel-heading span {
  font-size: 9px;
  font-weight: 800;

  letter-spacing:
    0.18em;

  text-transform:
    uppercase;

  color: var(--setup-muted);
}

.setup-panel-heading strong {
  font-size: 15px;
  font-weight: 900;

  letter-spacing:
    0.07em;

  text-transform:
    uppercase;

  color: var(--setup-gold);
}

.setup-team-tools {
  display: grid;

  grid-template-columns:
    minmax(0, 1fr)
    auto;

  gap: 10px;

  align-items: end;

  padding:
    10px 16px 0;
}

.setup-team-search {
  display: grid;

  gap: 6px;
}

.setup-team-search > span {
  font-size: 9px;
  font-weight: 900;

  letter-spacing:
    0.16em;

  text-transform:
    uppercase;

  color:
    var(--setup-muted);
}

.setup-team-search input {
  width: 100%;

  min-height: 40px;

  padding:
    0 12px;

  border:
    1px solid
    rgba(255, 255, 255, 0.22);

  background:
    rgba(255, 255, 255, 0.10);

  color:
    #f7f4ee;

  font-size: 13px;
  font-weight: 700;
}

.setup-conference-filter {
  display: grid;

  grid-template-columns:
    repeat(3, minmax(52px, auto));

  gap: 6px;
}

.setup-conference-filter button {
  min-height: 40px;

  padding:
    0 12px;

  border:
    1px solid
    rgba(255, 255, 255, 0.22);

  background:
    rgba(255, 255, 255, 0.08);

  color:
    var(--setup-text);

  font-size: 10px;
  font-weight: 900;

  letter-spacing:
    0.1em;

  text-transform:
    uppercase;

  cursor: pointer;
}

.setup-conference-filter button.is-selected {
  border-color:
    color-mix(
      in srgb,
      var(--team-accent) 65%,
      rgba(255,255,255,.18)
    );

  background:
    color-mix(
      in srgb,
      var(--team-accent) 10%,
      rgba(0,0,0,.28)
    );

  color:
    var(--setup-text);
}

.setup-team-empty {
  margin: 0;
  padding: 12px 4px;

  font-size: 13px;
  line-height: 1.45;

  color:
    var(--setup-text);

  background:
    rgba(255, 255, 255, 0.06);

  border:
    1px dashed
    rgba(255, 255, 255, 0.22);

  border-radius: 4px;
}

.setup-team-columns {
  min-height: 0;

  display: grid;

  grid-template-columns:
    1fr 1fr;

  gap: 14px;

  padding:
    14px;

  overflow:
    auto;

  background:
    rgba(255, 255, 255, 0.04);
}

.setup-team-columns:has(> div:only-child),
.setup-team-columns:has(> .setup-team-other) {
  grid-template-columns: 1fr;
}

.setup-team-columns:has(> div:nth-child(3)) {
  grid-template-columns: 1fr 1fr;
}

.setup-team-other {
  grid-column: 1 / -1;
}

.setup-team-columns > div {
  min-width: 0;
}

.setup-team-columns h3 {
  margin:
    0 0 10px;

  font-size: 11px;
  font-weight: 900;

  letter-spacing:
    0.14em;

  text-transform:
    uppercase;

  color:
    rgba(
      247,
      244,
      238,
      0.78
    );
}

.setup-team-grid {
  display: grid;

  grid-template-columns:
    repeat(
      auto-fill,
      minmax(148px, 1fr)
    );

  gap: 8px;

  padding: 12px 16px 16px;

  overflow: auto;
}

.setup-team-tile {
  min-width: 0;
  min-height: 64px;

  display: grid;

  grid-template-columns:
    48px
    minmax(0, 1fr);

  align-items: center;

  gap: 8px;

  padding:
    8px 10px;

  border:
    1px solid
    rgba(201, 168, 106, 0.28);

  border-radius: 4px;

  background:
    rgba(8, 10, 14, 0.72);

  color:
    var(--setup-text);

  text-align: left;

  cursor: pointer;

  transition:
    border-color 150ms ease,
    background 150ms ease,
    transform 150ms ease;
}

.setup-team-tile:hover {
  transform:
    translateY(-1px);

  border-color:
    rgba(
      255,
      255,
      255,
      0.38
    );

  background:
    rgba(
      255,
      255,
      255,
      0.16
    );
}

.setup-team-tile.is-selected {
  border-color:
    color-mix(
      in srgb,
      var(--team-accent) 78%,
      rgba(255,255,255,.28)
    );

  background:
    linear-gradient(
      90deg,
      color-mix(
        in srgb,
        var(--team-accent) 28%,
        transparent
      ),
      rgba(255,255,255,.10)
    );

  box-shadow:
    inset 3px 0 0
      var(--team-accent);
}

.setup-team-tile > span {
  min-width: 0;

  display: grid;

  gap: 2px;
}

.setup-team-tile strong {
  font-size: 12px;
  font-weight: 900;

  letter-spacing:
    0.11em;
}

.setup-team-tile small {
  overflow: hidden;

  text-overflow:
    ellipsis;

  white-space:
    nowrap;

  font-size: 10px;
  font-weight: 650;

  color:
    var(--setup-muted);
}

.setup-team-logo {
  object-fit: contain;

  filter:
    drop-shadow(
      0 7px 14px
      rgba(0,0,0,.46)
    );
}

.setup-deed-sheet {
  position: relative;
  left: auto;
  top: auto;
  bottom: auto;
  z-index: 3;
  width: auto;
  min-height: 0;
  overflow: auto;
  padding: 0;
  background: transparent;
}

.setup-deed-paper {
  position: relative;
  min-height: 100%;
  display: flex;
  flex-direction: column;
  padding: 8px 10px 6px;
  color: var(--setup-text);
  background: transparent;
  box-shadow: none;
  transform: none;
}

.setup-deed-kicker {
  margin: 0 0 6px;
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

.setup-deed-title span {
  display: block;
  margin-top: 4px;
  font-size: 0.72em;
}

.setup-deed-club {
  display: grid;
  grid-template-columns: 56px minmax(0, 1fr);
  gap: 12px;
  align-items: center;
  margin: 0 0 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--setup-line);
}

.setup-deed-club img,
.setup-deed-club em {
  width: 56px;
  height: 56px;
  object-fit: contain;
  background: transparent;
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
  color: var(--setup-text);
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

.setup-deed-gm {
  display: grid;
  gap: 6px;
  margin: 0 0 12px;
}

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
  border-radius: 0;
  background: transparent;
  color: var(--setup-text);
  font-size: 22px;
  font-weight: 900;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.setup-deed-gm input::placeholder {
  color: var(--setup-muted);
  text-transform: none;
  font-weight: 700;
}

.setup-deed-options {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin: 0 0 12px;
}

.setup-deed-options fieldset {
  margin: 0;
  padding: 0;
  border: 0;
  min-width: 0;
}

.setup-deed-options fieldset > div {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 8px;
}

.setup-token {
  appearance: none;
  display: grid;
  justify-items: center;
  gap: 6px;
  padding: 8px 4px 6px;
  border: 0;
  background: transparent;
  color: var(--setup-text);
  cursor: pointer;
  touch-action: manipulation;
  user-select: none;
}

.setup-token-orb {
  width: 42px;
  height: 42px;
  border-radius: 50%;
  border: 1px solid rgba(201, 168, 106, 0.28);
  background:
    radial-gradient(
      circle at 32% 28%,
      #6a5a40,
      #1c1710 62%,
      #070605 100%
    );
  box-shadow:
    0 10px 14px rgba(0, 0, 0, 0.5),
    inset -6px -8px 12px rgba(0, 0, 0, 0.45),
    inset 4px 5px 8px rgba(255, 236, 190, 0.12);
}

.setup-token.is-on .setup-token-orb {
  border-color: rgba(201, 168, 106, 0.7);
  background:
    radial-gradient(
      circle at 32% 28%,
      #f3e2b0,
      #c9a86a 42%,
      #6a4e1c 100%
    );
  box-shadow:
    0 12px 16px rgba(0, 0, 0, 0.5),
    inset -5px -7px 10px rgba(80, 50, 10, 0.45),
    inset 5px 6px 8px rgba(255, 255, 255, 0.28);
}

.setup-token strong {
  font-size: 10px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.setup-deed-meta {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px 16px;
  margin: 0 0 12px;
}

.setup-deed-meta dd {
  margin: 4px 0 0;
  font-size: 14px;
  font-weight: 900;
  text-transform: uppercase;
  color: var(--setup-text);
}

.setup-deed-legal {
  margin: 0 0 12px;
  font-size: 12px;
  line-height: 1.45;
  color: var(--setup-muted);
}

.setup-deed-sign {
  display: grid;
  gap: 6px;
  margin: 0 0 12px;
  flex: 1 1 auto;
}

.setup-signature-frame {
  position: relative;
  min-height: 96px;
  height: 14vh;
}

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

.setup-signature-pad path {
  fill: #e8d5a0;
}

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

.setup-deed-paper .setup-accept-btn {
  width: 100%;
  margin-top: auto;
}

.setup-club-ball-grid {
  display: grid;
  grid-template-columns: repeat(8, minmax(0, 1fr));
  align-content: start;
  gap: 10px 8px;
  min-height: 0;
  height: 100%;
  overflow: auto;
  padding: 8px 4px 12px;
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
  touch-action: manipulation;
  user-select: none;
}

.setup-club-ball-orb {
  width: clamp(44px, 6.2vw, 62px);
  height: clamp(44px, 6.2vw, 62px);
  display: grid;
  place-items: center;
  border-radius: 50%;
  border: 1px solid rgba(201, 168, 106, 0.28);
  background:
    radial-gradient(
      circle at 32% 28%,
      #6a5a40,
      #1c1710 62%,
      #070605 100%
    );
  box-shadow:
    0 10px 14px rgba(0, 0, 0, 0.5),
    inset -6px -8px 12px rgba(0, 0, 0, 0.45),
    inset 4px 5px 8px rgba(255, 236, 190, 0.12);
}

.setup-club-ball-orb img,
.setup-club-ball-orb em {
  width: 68%;
  height: 68%;
  object-fit: contain;
  pointer-events: none;
}

.setup-club-ball-orb em {
  display: grid;
  place-items: center;
  font-style: normal;
  font-size: 11px;
  font-weight: 900;
  color: #f3e6c8;
}

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
  box-shadow:
    0 0 0 2px rgba(201, 168, 106, 0.35),
    0 12px 16px rgba(0, 0, 0, 0.5),
    inset -5px -7px 10px rgba(80, 50, 10, 0.45),
    inset 5px 6px 8px rgba(255, 255, 255, 0.22);
}

.setup-club-ball.is-selected strong {
  color: var(--setup-gold);
}

.setup-club-ball:hover .setup-club-ball-orb,
.setup-club-ball:focus-visible .setup-club-ball-orb {
  border-color: rgba(201, 168, 106, 0.55);
}

.setup-club-ball-wrap {
  display: none;
}

.setup-club-ball-wrap canvas {
  display: none;
}

.setup-logo-coin {
  display: grid;
  place-items: center;

  border-radius: 50%;

  background: transparent;

  transform:
    perspective(420px)
    rotateX(18deg)
    rotateY(-12deg);

  transform-style: preserve-3d;
}

.setup-team-tile:hover .setup-logo-coin,
.setup-team-tile.is-selected .setup-logo-coin {
  transform:
    perspective(420px)
    rotateX(8deg)
    rotateY(16deg)
    translateZ(10px);
}

.setup-logo-coin .setup-team-logo {
  border-radius: 50%;

  background: transparent;
}

.setup-config-card {
  min-height: 0;

  display: flex;

  flex-direction: column;

  gap: 16px;

  padding: 0;

  overflow-y: auto;
}

.setup-config-club {
  display: grid;

  grid-template-columns:
    auto
    minmax(0, 1fr);

  align-items: center;

  gap: 14px;
}

.setup-config-club > div {
  min-width: 0;
}

.setup-config-club span {
  font-size: 9px;
  font-weight: 850;

  letter-spacing:
    0.17em;

  text-transform:
    uppercase;

  color:
    color-mix(
      in srgb,
      var(--team-accent) 58%,
      var(--setup-muted)
    );
}

.setup-config-club h2 {
  margin:
    5px 0 4px;

  font-size:
    clamp(
      18px,
      2vw,
      26px
    );

  font-weight: 900;

  line-height: 1.06;

  text-transform:
    uppercase;

  letter-spacing:
    0.025em;
}

.setup-config-club p {
  margin: 0;

  font-size: 11px;

  line-height: 1.45;

  color:
    var(--setup-muted);
}

.setup-config-rule {
  height: 1px;

  background:
    linear-gradient(
      90deg,
      var(--team-accent),
      transparent
    );

  opacity: 0.44;
}

.setup-field {
  display: grid;

  gap: 7px;
}

.setup-field > span,
.setup-fieldset legend,
.setup-signing-note > span {
  font-size: 10px;
  font-weight: 900;

  letter-spacing:
    0.14em;

  text-transform:
    uppercase;

  color: var(--setup-muted);
}

.setup-field input {
  width: 100%;

  min-height: 44px;

  padding:
    0 12px;

  border:
    1px solid
    var(--setup-line);

  outline: none;

  background: rgba(0, 0, 0, 0.35);

  color: var(--setup-text);

  font-size: 13px;
  font-weight: 700;
}

.setup-field input:focus {
  border-color:
    color-mix(
      in srgb,
      var(--team-accent) 62%,
      rgba(255,255,255,.22)
    );

  box-shadow:
    0 0 0 1px
      color-mix(
        in srgb,
        var(--team-accent) 18%,
        transparent
      );
}

.setup-field input::placeholder {
  color:
    rgba(
      247,
      244,
      238,
      0.45
    );
}

.setup-fieldset {
  min-width: 0;

  margin: 0;

  padding: 0;

  border: 0;
}

.setup-fieldset legend {
  margin-bottom: 7px;
}

.setup-choice-grid {
  display: grid;

  grid-template-columns:
    1fr 1fr;

  gap: 8px;
}

.setup-choice-grid button,
.setup-binary button {
  border:
    1px solid
    var(--setup-line);

  background: rgba(0, 0, 0, 0.28);

  color: var(--setup-text);

  cursor: pointer;
}

.setup-choice-grid button {
  min-height: 72px;

  display: grid;

  gap: 3px;

  align-content: center;

  padding:
    10px 12px;

  text-align: left;
}

.setup-choice-grid strong {
  font-size: 11px;
  font-weight: 900;

  letter-spacing:
    0.05em;

  text-transform:
    uppercase;
}

.setup-choice-grid small {
  font-size: 10px;
  line-height: 1.35;
}

.setup-choice-grid button.is-selected,
.setup-binary button.is-selected {
  border-color:
    color-mix(
      in srgb,
      var(--team-accent) 65%,
      rgba(255,255,255,.18)
    );

  background:
    color-mix(
      in srgb,
      var(--team-accent) 10%,
      rgba(0,0,0,.28)
    );

  color:
    var(--setup-text);
}

.setup-binary {
  width: 200px;

  display: grid;

  grid-template-columns:
    1fr 1fr;

  gap: 7px;
}

.setup-binary button {
  min-height: 38px;

  font-size: 11px;
  font-weight: 900;

  letter-spacing:
    0.08em;

  text-transform:
    uppercase;
}

.setup-field-hint {
  margin: 8px 0 0;

  font-size: 11px;

  line-height: 1.4;

  color:
    var(--setup-muted);
}

.setup-signing-note {
  margin-top: auto;

  padding:
    12px 13px;

  border-left:
    2px solid
    var(--team-accent);

  background:
    color-mix(
      in srgb,
      var(--team-accent) 12%,
      rgba(0, 0, 0, 0.45)
    );
}

.setup-signing-note p {
  margin:
    4px 0 0;

  font-size: 11px;

  line-height: 1.4;

  color:
    var(--setup-muted);
}

.setup-error {
  padding:
    10px 12px;

  border:
    1px solid
    rgba(
      255,
      96,
      109,
      0.42
    );

  background:
    rgba(
      100,
      10,
      18,
      0.35
    );

  color:
    #ffd4d8;

  font-size: 11px;
}

.setup-accept-btn {
  min-height: 56px;

  display: grid;

  place-items: center;

  gap: 2px;

  padding:
    8px 16px;

  border:
    1px solid
    color-mix(
      in srgb,
      var(--team-accent) 64%,
      rgba(255,255,255,.18)
    );

  background:
    linear-gradient(
      180deg,
      color-mix(
        in srgb,
        var(--team-accent) 15%,
        rgba(13,15,20,.98)
      ),
      rgba(7,9,13,.98)
    );

  color:
    #f0e8db;

  cursor: pointer;

  transition:
    transform 150ms ease,
    box-shadow 150ms ease,
    border-color 150ms ease;
}

.setup-accept-btn:hover:not(:disabled) {
  transform:
    translateY(-1px);

  border-color:
    color-mix(
      in srgb,
      var(--team-accent) 82%,
      #efe5d3
    );

  box-shadow:
    0 14px 34px
      rgba(0,0,0,.42),
    0 0 28px
      color-mix(
        in srgb,
        var(--team-accent) 14%,
        transparent
      );
}

.setup-accept-btn:disabled {
  opacity: 0.38;

  cursor: not-allowed;
}

.setup-accept-btn span {
  font-size: 13px;
  font-weight: 900;

  letter-spacing:
    0.08em;

  text-transform:
    uppercase;
}

.setup-accept-btn small {
  font-size: 9px;
  font-weight: 750;

  letter-spacing:
    0.12em;

  text-transform:
    uppercase;

  color:
    rgba(
      229,
      221,
      207,
      0.53
    );
}


/* --------------------------------------------------------------------------
   CINEMATIC
   -------------------------------------------------------------------------- */

.setup-cinematic {
  position: fixed;

  z-index: 24000;

  inset: 0;

  overflow: hidden;

  isolation: isolate;

  background:
    #05060a;

  color:
    #f0ede7;

  font-family:
    var(--setup-font);
}

.setup-cinematic--overlay,
.setup-cinematic--overlay .setup-cinematic-canvas {
  pointer-events: none;
}

.setup-cinematic-canvas {
  position: absolute;

  z-index: 1;

  inset: 0;

  display: block;

  width: 100%;
  height: 100%;

  outline: none;

  touch-action: none;

  background:
    #05060a;
}

/* first person: the reticle is the pointer */
.setup-cinematic--roaming .setup-cinematic-canvas {
  cursor: none;
}

.setup-blackout {
  position: absolute;

  z-index: 90;

  inset: 0;

  pointer-events: none;

  opacity: 0;

  background:
    #000;

  transition:
    opacity
    760ms
    cubic-bezier(
      .2,
      .72,
      .2,
      1
    );
}

.setup-blackout.is-black {
  opacity: 1;
}

.setup-letterbox {
  position: absolute;

  z-index: 30;

  left: 0;
  right: 0;

  height: 10px;

  pointer-events: none;

  background:
    rgba(
      0,
      0,
      0,
      0.35
    );

  transition:
    height
    700ms
    cubic-bezier(.2, .72, .2, 1);
}

/* free movement is gameplay, not a cinematic — the bars retract */
.setup-cinematic--free .setup-letterbox {
  height: 0;
}

.setup-letterbox--top {
  top: 0;
}

.setup-letterbox--bottom {
  bottom: 0;
}

.setup-cinematic-status {
  position: absolute;

  z-index: 40;

  top:
    clamp(
      36px,
      5.5vh,
      64px
    );

  left:
    clamp(
      24px,
      3.2vw,
      54px
    );

  display: grid;

  gap: 4px;

  pointer-events: none;

  text-shadow:
    0 3px 18px
    rgba(0,0,0,.90);
}

.setup-cinematic-status span,
.setup-cinematic-club span {
  font-size: 9px;
  font-weight: 900;

  letter-spacing:
    0.20em;

  text-transform:
    uppercase;

  color:
    color-mix(
      in srgb,
      var(--team-accent) 68%,
      #d8d2c7
    );
}

.setup-cinematic-status strong,
.setup-cinematic-club strong {
  font-size: 12px;
  font-weight: 850;

  letter-spacing:
    0.08em;

  text-transform:
    uppercase;

  color:
    rgba(
      240,
      236,
      229,
      0.78
    );
}

.setup-cinematic-club {
  position: absolute;

  z-index: 40;

  top:
    clamp(
      36px,
      5.5vh,
      64px
    );

  right:
    clamp(
      24px,
      3.2vw,
      54px
    );

  display: grid;

  justify-items: end;

  gap: 4px;

  text-align: right;

  pointer-events: none;

  text-shadow:
    0 3px 18px
    rgba(0,0,0,.90);
}

.setup-skip-intro {
  position: absolute;

  z-index: 42;

  right:
    clamp(
      22px,
      3vw,
      48px
    );

  bottom:
    clamp(
      34px,
      5vh,
      62px
    );

  min-height: 40px;

  padding:
    0 16px;

  border:
    1px solid
    rgba(
      255,
      255,
      255,
      0.22
    );

  background:
    rgba(
      4,
      5,
      7,
      0.78
    );

  color:
    #f1ece3;

  font-size: 11px;
  font-weight: 900;

  letter-spacing:
    0.14em;

  text-transform:
    uppercase;

  cursor: pointer;

  backdrop-filter:
    blur(8px);
}

.setup-skip-intro:hover {
  border-color:
    rgba(
      255,
      255,
      255,
      0.38
    );

  color:
    #fff;
}

.setup-appointment-fallback {
  position: absolute;

  z-index: 80;

  left: 50%;
  top: 50%;

  transform:
    translate(-50%, -50%);

  width:
    min(420px, calc(100vw - 32px));

  display: grid;

  justify-items: center;

  gap: 10px;

  padding: 28px 24px;

  border:
    1px solid
    rgba(255, 255, 255, 0.12);

  background:
    rgba(8, 10, 14, 0.94);

  text-align: center;

  box-shadow:
    0 24px 70px
    rgba(0, 0, 0, 0.48);
}

.setup-appointment-fallback span {
  font-size: 10px;
  font-weight: 900;

  letter-spacing: 0.16em;

  text-transform: uppercase;

  color:
    var(--setup-gold);
}

.setup-appointment-fallback h2 {
  margin: 0;

  font-size:
    clamp(22px, 3vw, 32px);

  font-weight: 900;

  letter-spacing: 0.04em;

  text-transform: uppercase;
}

.setup-appointment-fallback p {
  margin: 0;

  max-width: 32rem;

  font-size: 13px;

  line-height: 1.45;

  color:
    var(--setup-muted);
}

.setup-appointment-fallback button {
  margin-top: 8px;

  min-height: 44px;

  padding: 0 18px;

  border: 0;

  background:
    var(--setup-gold);

  color:
    #14110c;

  font-size: 12px;
  font-weight: 900;

  letter-spacing: 0.08em;

  text-transform: uppercase;

  cursor: pointer;
}

.setup-appointment-fallback button:disabled {
  opacity: 0.45;

  cursor: wait;
}

.setup-signing-prompt {
  position: absolute;

  z-index: 50;

  left: 50%;

  bottom:
    clamp(
      38px,
      5.5vh,
      68px
    );

  width:
    min(
      520px,
      calc(100vw - 36px)
    );

  transform:
    translateX(-50%);

  display: grid;

  justify-items: center;

  gap: 5px;

  padding:
    12px 16px;

  border-top:
    1px solid
    rgba(
      255,
      255,
      255,
      0.13
    );

  background:
    linear-gradient(
      90deg,
      transparent,
      rgba(
        0,
        0,
        0,
        0.76
      ) 18%,
      rgba(
        0,
        0,
        0,
        0.88
      ) 50%,
      rgba(
        0,
        0,
        0,
        0.76
      ) 82%,
      transparent
    );

  text-align: center;

  pointer-events: none;
}

.setup-signing-prompt strong {
  font-size:
    clamp(
      14px,
      1.4vw,
      19px
    );

  font-weight: 900;

  letter-spacing:
    0.07em;

  text-transform:
    uppercase;
}

.setup-signing-prompt p {
  margin: 0;

  font-size: 10px;

  color:
    rgba(
      233,
      228,
      218,
      0.58
    );
}

.setup-signing-prompt button {
  pointer-events: auto;

  min-width: 290px;
  min-height: 44px;

  margin-top: 5px;

  border:
    1px solid
    color-mix(
      in srgb,
      var(--team-accent) 72%,
      rgba(255,255,255,.25)
    );

  background:
    color-mix(
      in srgb,
      var(--team-accent) 13%,
      rgba(8,10,14,.95)
    );

  color:
    #eee8dc;

  font-size: 11px;
  font-weight: 900;

  letter-spacing:
    0.09em;

  text-transform:
    uppercase;

  cursor: pointer;
}

.setup-signing-prompt button:hover {
  border-color:
    color-mix(
      in srgb,
      var(--team-accent) 88%,
      #efe6d6
    );
}

.setup-team-welcome {
  position: absolute;

  z-index: 48;

  top: 50%;

  left:
    clamp(
      34px,
      7vw,
      110px
    );

  transform:
    translateY(-50%);

  width:
    min(
      570px,
      54vw
    );

  display: grid;

  justify-items: start;

  gap: 4px;

  pointer-events: none;

  text-shadow:
    0 4px 24px
    rgba(0,0,0,.92);

  animation:
    setupWelcomeIn
    780ms
    cubic-bezier(
      .17,
      .77,
      .2,
      1
    )
    both;
}

.setup-team-welcome span {
  margin-top: 8px;

  font-size: 10px;
  font-weight: 900;

  letter-spacing:
    0.18em;

  text-transform:
    uppercase;

  color:
    color-mix(
      in srgb,
      var(--team-accent) 66%,
      #d9d2c5
    );
}

.setup-team-welcome strong {
  font-size: 18px;
  font-weight: 900;

  line-height: 1.1;
}

.setup-team-welcome h2 {
  margin:
    5px 0 0;

  max-width:
    690px;

  font-size:
    clamp(
      34px,
      5.7vw,
      76px
    );

  font-weight: 950;

  line-height: 0.94;

  letter-spacing:
    -0.025em;

  text-transform:
    uppercase;
}

.setup-cinematic-loader {
  position: absolute;

  z-index: 96;

  top: 50%;
  left: 50%;

  transform:
    translate(
      -50%,
      -50%
    );

  display: grid;

  justify-items: center;

  gap: 12px;

  pointer-events: none;
}

.setup-cinematic-loader > span {
  width: 38px;
  height: 38px;

  border:
    2px solid
    rgba(
      255,
      255,
      255,
      0.10
    );

  border-top-color:
    var(--team-accent);

  border-right-color:
    var(--team-accent-2);

  border-radius:
    50%;

  animation:
    setupSpin
    900ms
    linear
    infinite;
}

.setup-cinematic-loader strong {
  font-size: 9px;
  font-weight: 900;

  letter-spacing:
    0.18em;

  text-transform:
    uppercase;

  color:
    rgba(
      234,
      229,
      219,
      0.58
    );
}

.setup-asset-warning {
  position: absolute;

  z-index: 65;

  left: 50%;

  top:
    clamp(
      78px,
      9vh,
      104px
    );

  transform:
    translateX(-50%);

  max-width:
    min(
      620px,
      calc(100vw - 36px)
    );

  display: flex;

  align-items: center;

  gap: 8px;

  padding:
    7px 10px;

  border:
    1px solid
    rgba(
      220,
      165,
      88,
      0.24
    );

  background:
    rgba(
      12,
      10,
      8,
      0.72
    );

  color:
    rgba(
      232,
      220,
      202,
      0.68
    );

  font-size: 9px;

  backdrop-filter:
    blur(8px);
}

.setup-asset-warning strong {
  flex: 0 0 auto;

  text-transform:
    uppercase;

  letter-spacing:
    0.10em;
}

.setup-asset-warning span {
  overflow: hidden;

  text-overflow:
    ellipsis;

  white-space:
    nowrap;
}


/* --------------------------------------------------------------------------
   STARTING / LOADING
   -------------------------------------------------------------------------- */

.setup-loading-screen {
  position: fixed;

  z-index: 25000;

  inset: 0;

  display: grid;

  place-items: center;

  padding: 22px;

  background:
    radial-gradient(
      circle at 50% 30%,
      color-mix(
        in srgb,
        var(--team-accent) 10%,
        transparent
      ),
      transparent 38%
    ),
    radial-gradient(
      ellipse 90% 80% at 50% 55%,
      rgba(5, 7, 10, 0.28),
      rgba(5, 7, 10, 0.08) 70%,
      transparent 100%
    );
}

.setup-loading-panel {
  width:
    min(
      620px,
      100%
    );

  display: grid;

  justify-items: center;

  gap: 8px;

  padding:
    30px;

  border:
    1px solid
    rgba(
      255,
      255,
      255,
      0.09
    );

  background:
    rgba(
      7,
      9,
      13,
      0.62
    );

  text-align: center;

  box-shadow:
    0 28px 90px
    rgba(0,0,0,.58);

  backdrop-filter: blur(4px);
}

.setup-loading-ring {
  width: 52px;
  height: 52px;

  margin-bottom: 8px;

  border:
    3px solid
    rgba(
      255,
      255,
      255,
      0.09
    );

  border-top-color:
    var(--team-accent);

  border-right-color:
    var(--team-accent-2);

  border-radius:
    50%;

  animation:
    setupSpin
    900ms
    linear
    infinite;
}

.setup-loading-panel > small {
  font-size: 9px;
  font-weight: 900;

  letter-spacing:
    0.18em;

  text-transform:
    uppercase;

  color:
    var(--setup-muted);
}

.setup-loading-panel h2 {
  margin: 0;

  font-size:
    clamp(
      26px,
      4vw,
      46px
    );

  font-weight: 950;

  text-transform:
    uppercase;
}

.setup-loading-panel p {
  margin: 0;

  color:
    var(--setup-muted);

  font-size: 12px;
}

.setup-loading-meta {
  display: flex;

  flex-wrap: wrap;

  justify-content: center;

  gap: 7px;

  margin:
    10px 0;
}

.setup-loading-meta span {
  padding:
    6px 9px;

  border:
    1px solid
    rgba(
      255,
      255,
      255,
      0.08
    );

  font-size: 9px;
  font-weight: 800;

  letter-spacing:
    0.09em;

  text-transform:
    uppercase;

  color:
    rgba(
      234,
      229,
      218,
      0.62
    );
}

.setup-loading-panel blockquote {
  width: 100%;

  margin:
    8px 0 0;

  padding:
    12px 14px;

  border-left:
    2px solid
    var(--team-accent);

  background:
    rgba(
      255,
      255,
      255,
      0.02
    );

  color:
    rgba(
      235,
      229,
      217,
      0.70
    );

  font-size: 11px;

  line-height: 1.45;

  text-align: left;
}

.setup-loading-slow {
  margin: 4px 0 0;

  max-width: 28rem;

  font-size: 12px;

  line-height: 1.4;

  color:
    var(--setup-gold);
}

.setup-loading-actions {
  display: flex;

  flex-wrap: wrap;

  justify-content: center;

  gap: 8px;

  margin-top: 8px;
}

.setup-loading-retry,
.setup-loading-back {
  min-height: 40px;

  padding: 0 16px;

  font-size: 11px;
  font-weight: 900;

  letter-spacing: 0.1em;

  text-transform: uppercase;

  cursor: pointer;
}

.setup-loading-retry {
  border: 0;

  background:
    var(--setup-gold);

  color:
    #14110c;
}

.setup-loading-retry:disabled {
  opacity: 0.45;

  cursor: wait;
}

.setup-loading-back {
  border:
    1px solid
    rgba(255, 255, 255, 0.16);

  background:
    transparent;

  color:
    var(--setup-text);
}


/* --------------------------------------------------------------------------
   ACCESSIBILITY / ANIMATION
   -------------------------------------------------------------------------- */

.setup-sr-status {
  position: absolute;

  width: 1px;
  height: 1px;

  padding: 0;
  margin: -1px;

  overflow: hidden;

  clip:
    rect(
      0,
      0,
      0,
      0
    );

  white-space:
    nowrap;

  border: 0;
}

@keyframes setupSpin {
  to {
    transform:
      rotate(360deg);
  }
}

@keyframes setupConfigArrive {
  from {
    opacity: 0;
    filter: blur(8px);
  }

  to {
    opacity: 1;
    filter: blur(0);
  }
}

@keyframes setupWelcomeIn {
  from {
    opacity: 0;

    transform:
      translateY(
        calc(
          -50% + 18px
        )
      );

    filter:
      blur(3px);
  }

  to {
    opacity: 1;

    transform:
      translateY(-50%);

    filter:
      blur(0);
  }
}


/* --------------------------------------------------------------------------
   RESPONSIVE
   -------------------------------------------------------------------------- */

@media (max-width: 1100px) {
  .setup-config-grid {
    grid-template-columns: minmax(300px, 0.42fr) minmax(0, 0.58fr);
  }

  .setup-deed-sheet {
    width: auto;
  }

  .setup-team-selector {
    padding-left: 8px;
  }

  .setup-club-ball-grid {
    grid-template-columns: repeat(6, minmax(0, 1fr));
  }
}

@media (max-width: 860px) {
  .nhlcal-root.setup-root {
    height: auto;
    min-height: 100dvh;

    overflow: auto;
  }

  .setup-config-layout {
    min-height: 100dvh;

    overflow: visible;
  }

  .setup-config-topline {
    grid-template-columns:
      1fr auto;
  }

  .setup-config-topline small {
    grid-column: 1 / -1;
    justify-self: start;
    text-align: left;
  }

  .setup-team-tools {
    grid-template-columns: 1fr;
  }

  .setup-config-grid {
    grid-template-columns: 1fr;
    overflow: visible;
    min-height: 920px;
  }

  .setup-deed-sheet {
    position: relative;
    width: 100%;
    background: none;
    padding-right: 0;
  }

  .setup-team-selector {
    position: relative;
    padding-left: 8px;
    min-height: 520px;
  }

  .setup-deed-paper {
    transform: none;
  }

  .setup-team-selector {
    max-height: none;
    padding-left: 8px;
    -webkit-mask-image: none;
    mask-image: none;
  }

  .setup-club-float-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  .setup-team-columns {
    overflow: visible;
  }

  .setup-team-grid {
    grid-template-columns:
      repeat(
        2,
        minmax(0,1fr)
      );
  }

  .setup-team-welcome {
    left: 24px;
    right: 24px;

    width: auto;
  }
}

@media (max-width: 600px) {
  .setup-config-layout {
    padding:
      12px;
  }

  .setup-config-topline {
    min-height: 46px;

    gap: 10px;
  }

  .setup-team-columns {
    grid-template-columns:
      1fr;
  }

  .setup-choice-grid {
    grid-template-columns:
      1fr;
  }

  .setup-config-card:not(.setup-deed-panel) {
    padding:
      14px;
  }

  .setup-cinematic-status,
  .setup-cinematic-club {
    top: 32px;
  }

  .setup-cinematic-status {
    left: 15px;
  }

  .setup-cinematic-club {
    right: 15px;
  }

  .setup-signing-prompt {
    bottom: 32px;
  }
}

/* --------------------------------------------------------------------------
   OPENING HALLWAY HUD

   Everything here is restrained on purpose: a reticle that only wakes up when
   something is in range, one prompt line, a hint that retires itself, and a
   small memorabilia card. No permanent instructions across the screen.
   -------------------------------------------------------------------------- */

.hall-reticle {
  position: absolute;
  z-index: 60;

  top: 50%;
  left: 50%;

  width: 26px;
  height: 26px;

  margin: -13px 0 0 -13px;

  display: grid;
  place-items: center;

  pointer-events: none;
}

.hall-reticle::before,
.hall-reticle::after {
  content: "";

  position: absolute;

  border-radius: 50%;

  transition:
    opacity 220ms ease,
    transform 260ms cubic-bezier(.2,.72,.2,1),
    border-color 220ms ease;
}

.hall-reticle::before {
  width: 4px;
  height: 4px;

  background:
    rgba(244, 238, 226, 0.62);

  box-shadow:
    0 0 4px rgba(0, 0, 0, 0.8);
}

.hall-reticle::after {
  width: 22px;
  height: 22px;

  border:
    1px solid
    rgba(244, 238, 226, 0);

  transform: scale(0.6);
}

.hall-reticle.is-live::before {
  background:
    rgba(255, 226, 168, 0.95);
}

.hall-reticle.is-live::after {
  border-color:
    rgba(255, 214, 140, 0.5);

  transform: scale(1);
}

.hall-prompt {
  position: absolute;
  z-index: 62;

  left: 50%;
  bottom: 15%;

  display: flex;
  align-items: center;
  gap: 9px;

  padding: 8px 14px 8px 9px;

  border:
    1px solid
    rgba(201, 168, 106, 0.22);

  background:
    linear-gradient(
      180deg,
      rgba(16, 15, 14, 0.82),
      rgba(9, 9, 10, 0.9)
    );

  backdrop-filter: blur(6px);

  box-shadow:
    0 14px 40px rgba(0, 0, 0, 0.55);

  opacity: 0;

  transform:
    translate(-50%, 10px);

  pointer-events: none;

  transition:
    opacity 200ms ease,
    transform 260ms cubic-bezier(.2,.72,.2,1);
}

.hall-prompt.is-shown {
  opacity: 1;

  transform:
    translate(-50%, 0);
}

.hall-prompt kbd {
  min-width: 24px;

  padding: 4px 0;

  border:
    1px solid
    rgba(255, 226, 168, 0.4);

  border-radius: 3px;

  background:
    rgba(255, 226, 168, 0.09);

  color: #ffe2a8;

  font-family: inherit;
  font-size: 10px;
  font-weight: 900;

  text-align: center;
}

.hall-prompt em {
  font-style: normal;
  font-size: 10px;
  font-weight: 900;

  letter-spacing: 0.14em;
  text-transform: uppercase;

  color: #ffe2a8;
}

.hall-prompt span {
  font-size: 11px;

  color:
    rgba(240, 234, 222, 0.72);
}

.hall-hint {
  position: absolute;
  z-index: 61;

  left: 50%;
  bottom: 46px;

  display: grid;
  gap: 7px;

  padding: 14px 20px;

  border:
    1px solid
    rgba(255, 255, 255, 0.07);

  background:
    linear-gradient(
      180deg,
      rgba(14, 14, 16, 0.7),
      rgba(8, 8, 10, 0.82)
    );

  backdrop-filter: blur(8px);

  box-shadow:
    0 20px 60px rgba(0, 0, 0, 0.5);

  opacity: 0;

  transform:
    translate(-50%, 14px);

  pointer-events: none;

  transition:
    opacity 900ms ease,
    transform 900ms cubic-bezier(.2,.72,.2,1);
}

.hall-hint.is-shown {
  opacity: 1;

  transform:
    translate(-50%, 0);
}

.hall-hint p {
  margin: 0;

  display: flex;
  align-items: center;
  gap: 5px;
}

.hall-hint kbd {
  min-width: 22px;

  padding: 3px 5px;

  border:
    1px solid
    rgba(255, 255, 255, 0.16);

  border-radius: 3px;

  background:
    rgba(255, 255, 255, 0.05);

  color:
    rgba(244, 239, 229, 0.92);

  font-family: inherit;
  font-size: 9px;
  font-weight: 900;

  text-align: center;
}

.hall-hint small {
  margin-left: 4px;

  font-size: 10px;

  letter-spacing: 0.04em;

  color:
    rgba(236, 230, 218, 0.5);
}

.hall-dart-tally {
  position: absolute;
  z-index: 61;

  right: 34px;
  bottom: 46px;

  display: flex;
  align-items: baseline;
  gap: 6px;

  padding: 7px 12px;

  border-left:
    2px solid
    rgba(179, 32, 54, 0.7);

  background:
    rgba(10, 10, 12, 0.6);

  pointer-events: none;
}

.hall-dart-tally strong {
  font-size: 17px;
  font-weight: 950;

  color: #ffd9a0;
}

.hall-dart-tally span {
  font-size: 9px;
  font-weight: 800;

  letter-spacing: 0.16em;
  text-transform: uppercase;

  color:
    rgba(236, 230, 218, 0.5);
}

/* Memorabilia card — deliberately small. */

.hall-card-layer {
  position: absolute;
  z-index: 70;

  inset: 0;

  display: grid;
  place-items: center;

  padding: 24px;

  background:
    radial-gradient(
      ellipse at center,
      rgba(0, 0, 0, 0.18),
      rgba(0, 0, 0, 0.62)
    );

  animation:
    hallCardLayerIn
    260ms ease
    both;
}

.hall-card {
  width:
    min(430px, 100%);

  display: grid;
  gap: 12px;

  padding: 22px 24px 18px;

  border:
    1px solid
    rgba(201, 168, 106, 0.3);

  background:
    linear-gradient(
      180deg,
      rgba(24, 21, 18, 0.97),
      rgba(13, 12, 12, 0.98)
    );

  box-shadow:
    0 34px 90px rgba(0, 0, 0, 0.7),
    inset 0 1px 0 rgba(255, 226, 168, 0.08);

  animation:
    hallCardIn
    380ms cubic-bezier(.2,.72,.2,1)
    both;
}

.hall-card header {
  display: grid;
  gap: 4px;

  padding-bottom: 12px;

  border-bottom:
    1px solid
    rgba(201, 168, 106, 0.18);
}

.hall-card header small {
  font-size: 9px;
  font-weight: 900;

  letter-spacing: 0.18em;
  text-transform: uppercase;

  color: var(--setup-gold);
}

.hall-card header h3 {
  margin: 0;

  font-size: 22px;
  font-weight: 950;

  letter-spacing: 0.01em;

  color: #f5f0e6;
}

.hall-card header p {
  margin: 0;

  font-size: 11px;
  font-weight: 700;

  letter-spacing: 0.05em;

  color:
    rgba(236, 230, 218, 0.6);
}

.hall-card-body {
  display: grid;
  gap: 9px;

  max-height: 38vh;

  overflow-y: auto;
}

.hall-card-body p {
  margin: 0;

  font-size: 12px;

  line-height: 1.55;

  color:
    rgba(238, 232, 220, 0.82);
}

.hall-card footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;

  padding-top: 12px;

  border-top:
    1px solid
    rgba(255, 255, 255, 0.06);
}

.hall-card footer span {
  font-size: 9px;
  font-weight: 800;

  letter-spacing: 0.14em;
  text-transform: uppercase;

  color:
    rgba(236, 230, 218, 0.42);
}

.hall-card footer button {
  min-height: 32px;

  padding: 0 18px;

  border:
    1px solid
    rgba(201, 168, 106, 0.45);

  background:
    rgba(201, 168, 106, 0.12);

  color: #ffe2a8;

  font-family: inherit;
  font-size: 10px;
  font-weight: 900;

  letter-spacing: 0.16em;
  text-transform: uppercase;

  cursor: pointer;

  transition:
    background 200ms ease,
    border-color 200ms ease;
}

.hall-card footer button:hover {
  background:
    rgba(201, 168, 106, 0.22);

  border-color:
    rgba(201, 168, 106, 0.7);
}

@keyframes hallCardLayerIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes hallCardIn {
  from {
    opacity: 0;
    transform:
      translateY(14px)
      scale(0.98);
  }
  to {
    opacity: 1;
    transform: none;
  }
}

/* Arrival line caption, so the room speaks even with audio off. */

.hall-narration {
  position: absolute;
  z-index: 68;

  left: 50%;
  bottom: 12%;

  display: flex;
  align-items: center;
  gap: 12px;

  padding: 10px 20px 10px 16px;

  border-left:
    2px solid
    var(--setup-gold);

  background:
    linear-gradient(
      90deg,
      rgba(12, 11, 10, 0.78),
      rgba(12, 11, 10, 0)
    );

  transform: translateX(-50%);

  pointer-events: none;

  animation:
    hallNarrationIn
    620ms cubic-bezier(.2,.72,.2,1)
    both;
}

.hall-narration span {
  width: 7px;
  height: 7px;

  border-radius: 50%;

  background: var(--setup-gold);

  animation:
    hallNarrationPulse
    1.6s ease-in-out
    infinite;
}

.hall-narration p {
  margin: 0;

  font-size:
    clamp(13px, 1.4vw, 17px);

  font-weight: 800;

  letter-spacing: 0.05em;

  color:
    rgba(246, 241, 231, 0.94);

  text-shadow:
    0 2px 14px rgba(0, 0, 0, 0.8);
}

@keyframes hallNarrationIn {
  from {
    opacity: 0;
    transform:
      translateX(-50%)
      translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateX(-50%);
  }
}

@keyframes hallNarrationPulse {
  0%, 100% { opacity: 0.35; }
  50% { opacity: 1; }
}


/* --------------------------------------------------------------------------
   AGREEMENT ON THE DESK

   The configuration layer sits above the office rather than replacing it. The
   room stays visible around and behind the paper.
   -------------------------------------------------------------------------- */

.setup-config-layout--desk {
  position: fixed;
  z-index: 24500;

  inset: 0;

  height: auto;

  padding:
    clamp(14px, 3.2vh, 34px)
    clamp(14px, 4vw, 64px)
    clamp(14px, 3vh, 30px);

  grid-template-rows:
    auto
    minmax(0, 1fr);

  gap: clamp(8px, 1.4vh, 16px);

  isolation: isolate;
  transform: translateZ(0);

  background:
    radial-gradient(
      ellipse 70% 55% at 50% 78%,
      rgba(6, 6, 8, 0.42),
      rgba(6, 6, 8, 0.12) 52%,
      transparent 76%
    );

  animation:
    setupDeskArrive
    280ms ease-out
    both;
}

.setup-config-layout--desk::before {
  opacity: 0;
}

.setup-config-layout--desk .setup-config-topline {
  padding: 0 6px;
}

/*
  The paper itself. Warm stock, a bound left edge, and just enough lift off the
  desk that it reads as an object sitting on wood.
*/
.setup-config-layout--desk .setup-config-grid {
  padding:
    clamp(10px, 1.6vh, 20px)
    clamp(12px, 1.6vw, 24px);

  gap: clamp(10px, 1.4vw, 24px);

  border:
    1px solid
    rgba(201, 168, 106, 0.24);

  border-left:
    3px solid
    rgba(201, 168, 106, 0.5);

  border-radius: 2px;

  background:
    linear-gradient(
      178deg,
      rgba(28, 24, 20, 0.78),
      rgba(14, 12, 11, 0.82)
    );

  box-shadow:
    0 40px 110px rgba(0, 0, 0, 0.5),
    0 2px 0 rgba(255, 226, 168, 0.05) inset,
    0 -18px 40px rgba(0, 0, 0, 0.32) inset;
}

/* paper-clip and a stamp, so the sheet belongs to a physical file */
.setup-config-layout--desk .setup-config-grid::before {
  content: "";

  position: absolute;
  z-index: 3;

  top: -9px;
  left: 34px;

  width: 15px;
  height: 34px;

  border:
    2px solid
    rgba(196, 199, 206, 0.5);

  border-radius: 8px 8px 3px 3px;

  border-bottom-color:
    rgba(196, 199, 206, 0.16);

  pointer-events: none;
}

@keyframes setupDeskArrive {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}


/* --------------------------------------------------------------------------
   LOADING, REWORKED

   Translucent over the office, no timers, no invented percentage. The
   categories say what they are doing and the facts keep turning.
   -------------------------------------------------------------------------- */

.setup-loading-screen {
  background:
    radial-gradient(
      circle at 50% 34%,
      color-mix(
        in srgb,
        var(--team-accent) 10%,
        transparent
      ),
      transparent 42%
    ),
    linear-gradient(
      180deg,
      rgba(4, 5, 8, 0.18),
      rgba(4, 5, 8, 0.32)
    );

  backdrop-filter: none;
}

.setup-loading-panel {
  width:
    min(560px, 100%);

  gap: 7px;

  border-color:
    rgba(201, 168, 106, 0.22);

  background:
    linear-gradient(
      180deg,
      rgba(15, 14, 14, 0.7),
      rgba(8, 8, 10, 0.74)
    );

  box-shadow:
    0 36px 110px rgba(0, 0, 0, 0.5);

  backdrop-filter: blur(4px);
}

.setup-loading-tasks {
  width: 100%;

  margin: 6px 0 2px;
  padding: 0;

  list-style: none;

  display: grid;
  gap: 1px;
}

.setup-loading-tasks li {
  display: grid;

  grid-template-columns:
    14px
    1fr
    auto;

  align-items: center;

  gap: 10px;

  padding: 8px 2px;

  border-bottom:
    1px solid
    rgba(255, 255, 255, 0.04);

  text-align: left;
}

.setup-loading-tasks li:last-child {
  border-bottom: none;
}

.setup-loading-tasks i {
  width: 7px;
  height: 7px;

  margin-left: 3px;

  border-radius: 50%;

  background:
    rgba(255, 255, 255, 0.14);

  transition:
    background 400ms ease,
    box-shadow 400ms ease;
}

.setup-loading-tasks li.is-loading i {
  background: var(--team-accent);

  animation:
    setupTaskPulse
    1.3s ease-in-out
    infinite;
}

.setup-loading-tasks li.is-ready i {
  background: #6fbf8a;

  box-shadow:
    0 0 10px rgba(111, 191, 138, 0.55);
}

.setup-loading-tasks span {
  font-size: 11px;
  font-weight: 700;

  color:
    rgba(238, 232, 220, 0.78);
}

.setup-loading-tasks em {
  font-style: normal;
  font-size: 9px;
  font-weight: 800;

  letter-spacing: 0.14em;
  text-transform: uppercase;

  color:
    rgba(236, 230, 218, 0.4);
}

.setup-loading-tasks li.is-ready em {
  color:
    rgba(111, 191, 138, 0.8);
}

.setup-loading-tasks li.is-loading em {
  color: var(--setup-gold);
}

/*
  Indeterminate on purpose. A sweep says work is happening without claiming a
  number the application cannot honestly measure.
*/
.setup-loading-bar {
  position: relative;

  width: 100%;
  height: 2px;

  overflow: hidden;

  background:
    rgba(255, 255, 255, 0.06);
}

.setup-loading-bar span {
  position: absolute;
  inset: 0 auto 0 0;

  width: 38%;

  background:
    linear-gradient(
      90deg,
      transparent,
      var(--team-accent),
      var(--team-accent-2),
      transparent
    );

  animation:
    setupLoadingSweep
    1.9s
    cubic-bezier(.5, 0, .5, 1)
    infinite;
}

.setup-loading-bar.is-complete span {
  width: 100%;

  animation: none;

  background:
    linear-gradient(
      90deg,
      var(--team-accent),
      var(--team-accent-2)
    );
}

@keyframes setupLoadingSweep {
  from { transform: translateX(-100%); }
  to { transform: translateX(300%); }
}

@keyframes setupTaskPulse {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 1; }
}


@media (max-width: 900px) {
  .setup-config-layout--desk .setup-config-grid {
    grid-template-columns: minmax(0, 1fr);

    overflow-y: auto;
  }

  .setup-club-ball-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  .hall-hint {
    bottom: 30px;

    padding: 11px 14px;
  }

  .hall-dart-tally {
    right: 16px;
    bottom: 30px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .setup-blackout,
  .setup-team-welcome,
  .setup-team-tile,
  .setup-accept-btn {
    transition-duration:
      1ms !important;

    animation-duration:
      1ms !important;
  }

  .hall-card,
  .hall-card-layer,
  .hall-narration,
  .setup-config-layout--desk {
    animation-duration:
      1ms !important;
  }

  .hall-narration span,
  .setup-loading-tasks li.is-loading i {
    animation: none !important;
  }
}
`;
