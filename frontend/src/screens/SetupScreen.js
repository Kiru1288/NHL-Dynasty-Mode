import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { useGameUI } from "../game/GameUIContext";
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
} from "@babylonjs/core";
import "@babylonjs/loaders/glTF";

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

import darkOfficeGlb
  from "../pictures/office_pics/dark_office.glb?url";

import officeHallwayGlb
  from "../pictures/office_pics/office_hallway.glb?url";

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

/*
  These two assets were already part of the earlier setup work.
  They stay in the root pictures directory unless you move them.
*/
import contractGlb
  from "../pictures/contract.glb?url";

import clipboardGlb
  from "../pictures/ps1_style_patient_sheet_with_clipboard.glb?url";


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
  hallway: {
    measure: "depth",
    target: 14.0,
    position: new Vector3(0, 0, -8.4),
    yaw: 0,
  },

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
    kind: "ROOM",
    asset: "hallway",
    coverage: "wide",
    cameraDepth: 0.04,
    lookDepth: 0.62,
    eyeHeight: 1.62,
    lookHeight: 1.38,
    side: 0.28,
    fov: 0.92,
  },
  hallwayMid: {
    kind: "ROOM",
    asset: "hallway",
    coverage: "medium",
    cameraDepth: 0.34,
    lookDepth: 0.92,
    eyeHeight: 1.58,
    lookHeight: 1.36,
    side: 0.18,
    fov: 0.82,
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
   APPOINTMENT DEED AND 3D CLUB BALLS
   ========================================================================== */

function useSetupStageMusic() {
  useEffect(() => {
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
  }, []);
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

function TeamSelection({
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
}

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

async function loadOfficeAssets({
  scene,
  mode,
  team,
  accentPrimary,
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

  await Promise.all([
    safeLoad("hallway", {
      name: "office-hallway",
      url: officeHallwayGlb,
      calibration: ASSET_CALIBRATION.hallway,
    }),
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
    safeLoad("standingExecutive", {
      name: "standing-executive",
      url: manDressedInSuitGlb,
      calibration: ASSET_CALIBRATION.standingExecutive,
      enabled: false,
    }),
    safeLoad("hockeyStick", {
      name: "hockey-stick",
      url: hockeyStickGlb,
      calibration: ASSET_CALIBRATION.hockeyStick,
    }),
    safeLoad("trophy", {
      name: "trophy-cup",
      url: trophyCupGlb,
      calibration: ASSET_CALIBRATION.trophy,
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

  const doors = createEntranceDoors(scene, assets.hallway);

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

  if (
    assets.standingExecutive
  ) {
    playBestAnimation(
      assets.standingExecutive,
      [
        "idle",
        "stand",
        "breath",
      ],
      {
        loop: true,
        speedRatio: 0.9,
      }
    );
  }

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
}) {
  const canvasRef =
    useRef(null);

  const cancelledRef =
    useRef(false);

  const executeRef =
    useRef(null);

  const signatureReadyRef =
    useRef(false);

  const [
    stage,
    setStage,
  ] = useState(
    CINEMATIC_STAGE.LOADING
  );

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
              1.62,
              -12.4
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

        officeBuild =
          await loadOfficeAssets({
            scene,
            mode,
            team,
            accentPrimary,
          });
        Object.entries(
          officeBuild.assets
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

        engine.runRenderLoop(() => {
          if (
            scene &&
            !scene.isDisposed
          ) {
            scene.render();
          }
        });

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
          INTRO CINEMATIC
          ------------------------------------------------------------
        */
        if (!appointment) {
          setStage(CINEMATIC_STAGE.HALLWAY);

          await openEntranceDoors(
            officeBuild.doors,
            cancelledRef,
            ms(900)
          );

          if (cancelledRef.current) {
            return;
          }

          await cameraDirector.focus("hallwayMid", {
            duration: ms(2600),
            walking: true,
          });

          if (cancelledRef.current) {
            return;
          }

          setStage(CINEMATIC_STAGE.OFFICE_ENTRY);

          await cameraDirector.focus("officeWide", {
            duration: ms(2400),
            walking: true,
          });

          if (cancelledRef.current) {
            return;
          }

          await cameraDirector.focus("deskThreeQuarter", {
            duration: ms(1800),
          });

          if (cancelledRef.current) {
            return;
          }

          setStage(CINEMATIC_STAGE.MEETING);

          await cameraDirector.focus("gmHero", {
            duration: ms(1700),
          });

          if (cancelledRef.current) {
            return;
          }

          await cameraDirector.focus("stickProp", {
            duration: ms(1100),
          });

          if (cancelledRef.current) {
            return;
          }

          await cameraDirector.focus("trophyProp", {
            duration: ms(1000),
          });

          if (cancelledRef.current) {
            return;
          }

          await cameraDirector.focus("officeHero", {
            duration: ms(2100),
          });

          if (cancelledRef.current) {
            return;
          }

          await sleep(ms(900), cancelledRef);

          if (cancelledRef.current) {
            return;
          }

          setBlackout(true);
          await sleep(ms(420), cancelledRef);
          onComplete?.();
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

        onComplete?.();
      }
    };

    run();

    return () => {
      cancelledRef.current =
        true;

      executeRef.current =
        null;

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
  ]);

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

  useEffect(() => {
    if (appointment) {
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
    onComplete,
    onSkipIntro,
  ]);

  return (
    <section
      className="setup-cinematic"
      aria-label={
        appointment
          ? `${team?.name || "NHL"} executive appointment`
          : "NHL executive floor introduction"
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

      {!officeFailed ? (
      <div className="setup-cinematic-status">
        <span>
          NHL Executive Floor
        </span>

        <strong>
          {stageCopy}
        </strong>
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

      {!appointment ? (
        <button
          type="button"
          className="setup-skip-intro"
          onClick={
            onSkipIntro ||
            onComplete
          }
        >
          Skip cinematic
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
            Loading executive floor
          </strong>
        </div>
      ) : null}

      {assetIssue && !officeFailed ? (
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

function SetupLoadingScreen({
  selected,
  gmName,
  injuriesEnabled,
  playerUniverse,
  error,
  loading,
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

  const [
    waitedTooLong,
    setWaitedTooLong,
  ] = useState(false);

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
        10000
      );

    return () =>
      window.clearInterval(id);
  }, [facts.length]);

  useEffect(() => {
    const id = window.setTimeout(
      () => setWaitedTooLong(true),
      12000
    );

    return () =>
      window.clearTimeout(id);
  }, []);

  const failed = Boolean(error);

  return (
    <div
      className="setup-loading-screen"
      role="status"
      aria-live="polite"
    >
      <div className="setup-loading-panel">
        {!failed ? (
          <span className="setup-loading-ring" />
        ) : null}

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
            ? "Preparing the real NHL player universe."
            : `${
                gmName?.trim() ||
                "General Manager"
              } is entering hockey operations.`}
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

        {waitedTooLong && !failed ? (
          <p className="setup-loading-slow">
            This is taking longer than expected. You can wait, retry, or return to setup.
          </p>
        ) : null}

        {failed || waitedTooLong ? (
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
  } = useGameUI();

  useSetupStageMusic();

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
    statusText,
    setStatusText,
  ] = useState(
    "Executive introduction."
  );

  useEffect(() => {
    /*
      Team loading starts immediately but DOES NOT block the intro cinematic.
      The intro is generic and intentionally does not reveal a selected club.
    */
    loadTeams();
  }, [loadTeams]);

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
    if (
      selected?.index != null &&
      selected.index !==
        setupTeamIndex
    ) {
      setSetupTeamIndex(
        selected.index
      );
    }
  }, [
    selected,
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

        setSetupTeamIndex(
          team.index
        );

        setStatusText(
          `${team.name} selected.`
        );
      },
      [
        orderedTeams,
        setSetupTeamIndex,
      ]
    );

  const finishIntro =
    useCallback(() => {
      setAppStage(
        APP_STAGE.CONFIGURE
      );

      setStatusText(
        "Select a club and configure the franchise."
      );
    }, []);

  const finishAppointment =
    useCallback(async () => {
      setAppStage(
        APP_STAGE.STARTING
      );

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
      {appStage ===
      APP_STAGE.INTRO ? (
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
        />
      ) : null}

      {appStage ===
      APP_STAGE.CONFIGURE ? (
        <main className="setup-config-layout">
          <div className="setup-config-topline">
            <strong>
              Executive Appointment
            </strong>

            <small>
              Choose club, name, and sign the deed
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

  background:
    radial-gradient(
      circle at 50% -8%,
      color-mix(
        in srgb,
        var(--team-accent, #c9a86a) 22%,
        transparent
      ),
      transparent 42%
    ),
    linear-gradient(
      180deg,
      #14110c,
      #080a0e 55%,
      #050607
    );
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

.setup-club-ball-wrap {
  position: relative;
  inset: auto;
  min-height: 0;
  height: 100%;
  width: 100%;
  background: transparent;
}

.setup-club-ball-wrap canvas {
  display: block;
  width: 100% !important;
  height: 100% !important;
  background: transparent !important;
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
    #2a2c31;

  color:
    #f0ede7;

  font-family:
    var(--setup-font);
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
    #2a2c31;
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
        var(--team-accent) 13%,
        transparent
      ),
      transparent 34%
    ),
    #05070a;
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
      0.84
    );

  text-align: center;

  box-shadow:
    0 28px 90px
    rgba(0,0,0,.58);
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
}
`;
