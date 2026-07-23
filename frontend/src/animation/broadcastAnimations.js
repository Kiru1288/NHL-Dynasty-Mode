import React, { useEffect, useMemo, useRef } from "react";
import { Html, useGLTF } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

/**
 * broadcastAnimations.js
 *
 * Reusable animation utilities for 3D broadcast / press conference hosts.
 *
 * This file does NOT:
 * - connect to the backend
 * - generate AI scripts
 * - generate real lip sync
 *
 * This file DOES:
 * - animate 3D GLB hosts
 * - handle idle breathing
 * - handle active speaker movement
 * - handle score reactions
 * - handle camera focus
 * - expose score helper functions used by WorldJuniorsMenu.js
 */

/* -------------------------------------------------------------------------- */
/* Shared helpers                                                             */
/* -------------------------------------------------------------------------- */

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function safeArray(value, fallback = []) {
  return Array.isArray(value) ? value : fallback;
}

function safeNum(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function safeStr(value, fallback = "") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function normalizeKey(value) {
  return safeStr(value)
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "");
}

function lerpArray3(current, target, amount) {
  current[0] = THREE.MathUtils.lerp(current[0], target[0], amount);
  current[1] = THREE.MathUtils.lerp(current[1], target[1], amount);
  current[2] = THREE.MathUtils.lerp(current[2], target[2], amount);
  return current;
}

function cloneGltfScene(scene) {
  const cloned = scene.clone(true);

  cloned.traverse((node) => {
    if (!node) return;

    if (node.isMesh) {
      node.castShadow = true;
      node.receiveShadow = true;

      if (node.material) {
        node.material = node.material.clone();

        if ("roughness" in node.material) {
          node.material.roughness = clamp(node.material.roughness ?? 0.55, 0.35, 0.85);
        }

        if ("metalness" in node.material) {
          node.material.metalness = clamp(node.material.metalness ?? 0.05, 0, 0.35);
        }
      }
    }
  });

  return cloned;
}

function getEmotionSettings(emotion) {
  const key = normalizeKey(emotion);

  const base = {
    talkSpeed: 1,
    talkEnergy: 1,
    nodAmount: 1,
    leanAmount: 1,
    reactionAmount: 1,
    idleEnergy: 1,
  };

  if (key.includes("excited") || key.includes("hype") || key.includes("celebrate")) {
    return {
      ...base,
      talkSpeed: 1.25,
      talkEnergy: 1.25,
      nodAmount: 1.25,
      leanAmount: 1.15,
      reactionAmount: 1.35,
      idleEnergy: 1.1,
    };
  }

  if (key.includes("angry") || key.includes("critical") || key.includes("heated")) {
    return {
      ...base,
      talkSpeed: 1.18,
      talkEnergy: 1.18,
      nodAmount: 1.1,
      leanAmount: 1.25,
      reactionAmount: 1.25,
      idleEnergy: 1,
    };
  }

  if (key.includes("calm") || key.includes("analytical") || key.includes("serious")) {
    return {
      ...base,
      talkSpeed: 0.88,
      talkEnergy: 0.75,
      nodAmount: 0.7,
      leanAmount: 0.75,
      reactionAmount: 0.75,
      idleEnergy: 0.8,
    };
  }

  if (key.includes("surprised") || key.includes("shock")) {
    return {
      ...base,
      talkSpeed: 1.08,
      talkEnergy: 1.1,
      nodAmount: 0.95,
      leanAmount: 1.35,
      reactionAmount: 1.45,
      idleEnergy: 1,
    };
  }

  return base;
}

/* -------------------------------------------------------------------------- */
/* Score helpers - REQUIRED by WorldJuniorsMenu.js                            */
/* -------------------------------------------------------------------------- */

export function buildBroadcastScoreContext(game = null) {
  if (!game) {
    return {
      away: "Away",
      home: "Home",
      awayScore: null,
      homeScore: null,
      status: "scheduled",
      stage: "",
      label: "",
      overtime: false,
    };
  }

  const away =
    game.away ??
    game.away_team ??
    game.awayTeam ??
    game.away_country ??
    game.awayCountry ??
    "Away";

  const home =
    game.home ??
    game.home_team ??
    game.homeTeam ??
    game.home_country ??
    game.homeCountry ??
    "Home";

  const awayScore =
    game.awayScore ??
    game.away_score ??
    game.awayGoals ??
    game.away_goals ??
    game.score?.away ??
    null;

  const homeScore =
    game.homeScore ??
    game.home_score ??
    game.homeGoals ??
    game.home_goals ??
    game.score?.home ??
    null;

  const status =
    game.status ??
    game.game_status ??
    game.gameStatus ??
    game.result_status ??
    "scheduled";

  const stage =
    game.stage ??
    game.round ??
    game.game_type ??
    game.gameType ??
    game.pool ??
    game.group ??
    "";

  const label =
    game.label ??
    game.tag ??
    game.headline ??
    game.title ??
    game.storyline ??
    "";

  return {
    away: safeStr(away, "Away"),
    home: safeStr(home, "Home"),
    awayScore,
    homeScore,
    status: safeStr(status, "scheduled"),
    stage: safeStr(stage, ""),
    label: safeStr(label, ""),
    overtime:
      Boolean(game.overtime || game.ot || game.went_to_ot || game.wentToOt) ||
      normalizeKey(label).includes("ot"),
  };
}

export function buildSimpleScoreLine(scoreContext) {
  if (!scoreContext) return "Score pending";

  const away = safeStr(scoreContext.away, "Away");
  const home = safeStr(scoreContext.home, "Home");

  const awayScore =
    scoreContext.awayScore === null || scoreContext.awayScore === undefined
      ? "—"
      : scoreContext.awayScore;

  const homeScore =
    scoreContext.homeScore === null || scoreContext.homeScore === undefined
      ? "—"
      : scoreContext.homeScore;

  const status = safeStr(scoreContext.status, "scheduled");
  const stage = safeStr(scoreContext.stage, "");

  return `${away} ${awayScore} - ${homeScore} ${home}${stage ? ` · ${stage}` : ""}${
    status ? ` · ${status}` : ""
  }`;
}

function getScoreEnergy(scoreContext) {
  if (!scoreContext) return 0;

  const awayScore = safeNum(
    scoreContext.awayScore ??
      scoreContext.away_score ??
      scoreContext.awayGoals ??
      scoreContext.away_goals,
    0
  );

  const homeScore = safeNum(
    scoreContext.homeScore ??
      scoreContext.home_score ??
      scoreContext.homeGoals ??
      scoreContext.home_goals,
    0
  );

  const totalGoals = awayScore + homeScore;
  const goalDiff = Math.abs(homeScore - awayScore);

  let energy = 0;

  if (totalGoals >= 8) energy += 0.35;
  else if (totalGoals >= 6) energy += 0.25;
  else if (totalGoals >= 4) energy += 0.15;

  if (goalDiff <= 1 && totalGoals > 0) energy += 0.3;
  if (normalizeKey(scoreContext.status).includes("final")) energy += 0.1;
  if (normalizeKey(scoreContext.status).includes("live")) energy += 0.2;
  if (normalizeKey(scoreContext.stage).includes("gold")) energy += 0.2;
  if (normalizeKey(scoreContext.stage).includes("semi")) energy += 0.15;
  if (normalizeKey(scoreContext.stage).includes("medal")) energy += 0.15;
  if (scoreContext.overtime || scoreContext.ot || normalizeKey(scoreContext.label).includes("ot")) energy += 0.25;

  return clamp(energy, 0, 1);
}

/* -------------------------------------------------------------------------- */
/* Default host layout                                                        */
/* -------------------------------------------------------------------------- */

export const DEFAULT_BROADCAST_HOSTS = Object.freeze([
  {
    id: "host_1",
    label: "Left Analyst",
    role: "Analyst",
    position: [-2.08, 0.12, 0],
    rotation: [0, 0.38, 0],
    scale: [1.08, 1.08, 1.08],
    cameraPosition: [-0.62, 2.1, 6.15],
    cameraTarget: [-1.32, 0.35, 0],
    personality: "reactive",
    gestureOffset: 0.15,
  },
  {
    id: "host_2",
    label: "Center Anchor",
    role: "Anchor",
    position: [0, 0.18, -0.05],
    rotation: [0, 0, 0],
    scale: [1.14, 1.14, 1.14],
    cameraPosition: [0, 2.05, 6.05],
    cameraTarget: [0, 0.42, -0.02],
    personality: "steady",
    gestureOffset: 0,
  },
  {
    id: "host_3",
    label: "Right Scout",
    role: "Scout",
    position: [2.08, 0.12, 0],
    rotation: [0, -0.38, 0],
    scale: [1.08, 1.08, 1.08],
    cameraPosition: [0.62, 2.1, 6.15],
    cameraTarget: [1.32, 0.35, 0],
    personality: "measured",
    gestureOffset: -0.15,
  },
]);

export const DEFAULT_CAMERA_HOME = Object.freeze({
  position: [0, 2.05, 6.4],
  target: [0, 0.45, 0],
});

export const DEFAULT_ANIMATION_PRESET = Object.freeze({
  idleBreathAmount: 0.012,
  idleBreathSpeed: 1.15,
  idleSwayAmount: 0.01,
  idleSwaySpeed: 0.8,

  speakingBobAmount: 0.034,
  speakingBobSpeed: 8.25,
  speakingLeanAmount: 0.028,
  speakingTurnAmount: 0.035,
  speakingPulseAmount: 0.045,

  listeningNodAmount: 0.008,
  listeningNodSpeed: 1.35,

  reactionPulseAmount: 0.04,
  reactionDecay: 0.92,

  cameraLerp: 0.045,
  cameraTargetLerp: 0.065,
});

/* -------------------------------------------------------------------------- */
/* Scene material/lighting helpers                                            */
/* -------------------------------------------------------------------------- */

export function BroadcastHostSpotlight({
  hostId,
  activeSpeakerId,
  position = [0, 0, 0],
  intensity = 0.55,
  color = "#dbeafe",
}) {
  const isActive = activeSpeakerId === hostId;
  const lightRef = useRef(null);

  useFrame((state) => {
    if (!lightRef.current) return;

    const t = state.clock.elapsedTime;
    const pulse = isActive ? 0.2 + Math.sin(t * 5.5) * 0.08 : 0;

    lightRef.current.intensity = THREE.MathUtils.lerp(
      lightRef.current.intensity,
      isActive ? intensity + pulse : intensity * 0.25,
      0.08
    );
  });

  return (
    <pointLight
      ref={lightRef}
      position={[position[0], position[1] + 1.45, position[2] + 1.25]}
      intensity={isActive ? intensity : intensity * 0.25}
      color={color}
      distance={3.75}
      decay={2}
    />
  );
}

export function BroadcastDeskScorePulse({
  scoreContext,
  position = [0, -0.25, 1.05],
  color = "#ef4444",
}) {
  const scoreEnergy = getScoreEnergy(scoreContext);
  const lightRef = useRef(null);

  useFrame((state) => {
    if (!lightRef.current) return;

    const t = state.clock.elapsedTime;
    const heartbeat = Math.max(0, Math.sin(t * (2.5 + scoreEnergy * 5)));
    const target = scoreEnergy > 0 ? 0.18 + heartbeat * scoreEnergy * 0.5 : 0.05;

    lightRef.current.intensity = THREE.MathUtils.lerp(
      lightRef.current.intensity,
      target,
      0.05
    );
  });

  return (
    <pointLight
      ref={lightRef}
      position={position}
      intensity={0.05}
      color={color}
      distance={5}
      decay={2}
    />
  );
}

/* -------------------------------------------------------------------------- */
/* AnimatedHost                                                               */
/* -------------------------------------------------------------------------- */

export function AnimatedHost({
  modelUrl,
  hostId = "host_1",
  speakerLabel,
  role,
  isSpeaking = false,
  isListening = true,
  isActive = false,
  activeSpeakerId,
  position = [0, 0, 0],
  rotation = [0, 0, 0],
  scale = [1, 1, 1],
  emotion = "neutral",
  scoreContext = null,
  showNameplate = true,
  showActiveRing = true,
  animationPreset = DEFAULT_ANIMATION_PRESET,
  onModelReady,
}) {
  const groupRef = useRef(null);
  const modelRef = useRef(null);
  const activeRingRef = useRef(null);
  const reactionRef = useRef(0);
  const lastSpeakingRef = useRef(false);
  const lastScoreKeyRef = useRef("");

  const gltf = useGLTF(modelUrl);

  const clonedScene = useMemo(() => {
    if (!gltf?.scene) return null;
    return cloneGltfScene(gltf.scene);
  }, [gltf]);

  const resolvedLabel = speakerLabel || hostId;
  const resolvedRole = role || "Broadcast Desk";
  const scoreEnergy = useMemo(() => getScoreEnergy(scoreContext), [scoreContext]);
  const emotionSettings = useMemo(() => getEmotionSettings(emotion), [emotion]);

  useEffect(() => {
    if (!clonedScene || typeof onModelReady !== "function") return;
    onModelReady(clonedScene);
  }, [clonedScene, onModelReady]);

  useEffect(() => {
    if (isSpeaking && !lastSpeakingRef.current) {
      reactionRef.current = Math.max(reactionRef.current, 0.35);
    }

    lastSpeakingRef.current = isSpeaking;
  }, [isSpeaking]);

  useEffect(() => {
    if (!scoreContext) return;

    const scoreKey = [
      scoreContext.awayScore ?? scoreContext.away_score ?? "",
      scoreContext.homeScore ?? scoreContext.home_score ?? "",
      scoreContext.status ?? "",
      scoreContext.stage ?? "",
    ].join("-");

    if (scoreKey && scoreKey !== lastScoreKeyRef.current) {
      reactionRef.current = Math.max(reactionRef.current, 0.25 + scoreEnergy * 0.55);
      lastScoreKeyRef.current = scoreKey;
    }
  }, [scoreContext, scoreEnergy]);

  useFrame((state, delta) => {
    if (!groupRef.current) return;

    const t = state.clock.elapsedTime;
    const safeDelta = Math.min(delta, 0.05);
    const preset = animationPreset || DEFAULT_ANIMATION_PRESET;

    const speakingWeight = isSpeaking ? 1 : 0;
    const listeningWeight = !isSpeaking && isListening ? 1 : 0;
    const activeWeight = isActive || activeSpeakerId === hostId ? 1 : 0;

    reactionRef.current *= Math.pow(preset.reactionDecay, safeDelta * 60);
    if (reactionRef.current < 0.005) reactionRef.current = 0;

    const reaction = reactionRef.current;
    const energy = clamp(
      0.75 +
        speakingWeight * 0.65 * emotionSettings.talkEnergy +
        scoreEnergy * 0.35 +
        reaction * 0.7,
      0.5,
      2.15
    );

    const idleBreath =
      Math.sin(t * preset.idleBreathSpeed * emotionSettings.idleEnergy) *
      preset.idleBreathAmount;

    const idleSway =
      Math.sin(t * preset.idleSwaySpeed + position[0]) *
      preset.idleSwayAmount *
      emotionSettings.idleEnergy;

    const speakingBob =
      speakingWeight *
      Math.sin(t * preset.speakingBobSpeed * emotionSettings.talkSpeed) *
      preset.speakingBobAmount *
      energy;

    const speakingLean =
      speakingWeight *
      Math.sin(t * 3.15 * emotionSettings.talkSpeed + position[0]) *
      preset.speakingLeanAmount *
      emotionSettings.leanAmount;

    const speakingTurn =
      speakingWeight *
      Math.sin(t * 4.3 * emotionSettings.talkSpeed + position[0]) *
      preset.speakingTurnAmount *
      emotionSettings.nodAmount;

    const listeningNod =
      listeningWeight *
      Math.sin(t * preset.listeningNodSpeed + position[0] * 0.75) *
      preset.listeningNodAmount;

    const reactionKick =
      Math.sin(t * 8.5 + position[0]) *
      reaction *
      preset.reactionPulseAmount *
      emotionSettings.reactionAmount;

    groupRef.current.position.x =
      position[0] + idleSway + speakingLean * 0.45 + reactionKick * 0.2;

    groupRef.current.position.y =
      position[1] + idleBreath + speakingBob + Math.abs(reactionKick) * 0.3;

    groupRef.current.position.z =
      position[2] + speakingWeight * Math.sin(t * 2.2) * 0.006 - reaction * 0.01;

    groupRef.current.rotation.x =
      rotation[0] + listeningNod + speakingBob * 0.25 + reactionKick * 0.4;

    groupRef.current.rotation.y =
      rotation[1] + speakingTurn + idleSway * 0.8 + reactionKick * 0.35;

    groupRef.current.rotation.z =
      rotation[2] + speakingLean * 0.25 + idleSway * 0.15;

    const pulseScale =
      1 +
      speakingWeight *
        (0.006 + Math.max(0, Math.sin(t * 7.5)) * preset.speakingPulseAmount * 0.08) +
      reaction * 0.012;

    groupRef.current.scale.set(
      scale[0] * pulseScale,
      scale[1] * (pulseScale + speakingBob * 0.05),
      scale[2] * pulseScale
    );

    if (activeRingRef.current) {
      const ringTargetScale = isSpeaking
        ? 1.18 + Math.sin(t * 5) * 0.05
        : activeWeight
          ? 1.06
          : 0.92;

      activeRingRef.current.scale.x = THREE.MathUtils.lerp(
        activeRingRef.current.scale.x,
        ringTargetScale,
        0.08
      );

      activeRingRef.current.scale.y = THREE.MathUtils.lerp(
        activeRingRef.current.scale.y,
        ringTargetScale,
        0.08
      );

      activeRingRef.current.scale.z = THREE.MathUtils.lerp(
        activeRingRef.current.scale.z,
        ringTargetScale,
        0.08
      );

      const material = activeRingRef.current.material;

      if (material) {
        material.opacity = THREE.MathUtils.lerp(
          material.opacity,
          isSpeaking ? 0.35 : activeWeight ? 0.18 : 0.04,
          0.08
        );
      }
    }
  });

  if (!clonedScene) return null;

  return (
    <group ref={groupRef} position={position} rotation={rotation} scale={scale}>
      {showActiveRing ? (
        <mesh
          ref={activeRingRef}
          position={[0, -0.63, 0.34]}
          rotation={[-Math.PI / 2, 0, 0]}
        >
          <ringGeometry args={[0.44, 0.57, 48]} />
          <meshBasicMaterial
            color={isSpeaking ? "#bfdbfe" : "#f8fafc"}
            transparent
            opacity={0.08}
            depthWrite={false}
          />
        </mesh>
      ) : null}

      <primitive ref={modelRef} object={clonedScene} />

      {showNameplate ? (
        <Html
          position={[0, -0.82, 0.92]}
          center
          distanceFactor={8}
          transform
          occlude={false}
          style={{
            pointerEvents: "none",
            userSelect: "none",
          }}
        >
          <div
            style={{
              minWidth: 112,
              padding: "6px 10px",
              borderRadius: 999,
              border: isSpeaking
                ? "1px solid rgba(219, 234, 254, 0.85)"
                : "1px solid rgba(255, 255, 255, 0.22)",
              background: isSpeaking
                ? "linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 64, 175, 0.78))"
                : "rgba(2, 6, 23, 0.72)",
              boxShadow: isSpeaking
                ? "0 0 24px rgba(59, 130, 246, 0.45)"
                : "0 10px 24px rgba(0, 0, 0, 0.22)",
              color: "#f8fafc",
              fontFamily:
                "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
              textAlign: "center",
              transform: "translateY(0)",
              opacity: isSpeaking || isActive ? 1 : 0.72,
            }}
          >
            <div
              style={{
                fontSize: 10,
                lineHeight: "11px",
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                color: isSpeaking ? "#bfdbfe" : "rgba(226, 232, 240, 0.72)",
                marginBottom: 2,
                whiteSpace: "nowrap",
              }}
            >
              {isSpeaking ? "Speaking" : resolvedRole}
            </div>

            <div
              style={{
                fontSize: 12,
                lineHeight: "14px",
                fontWeight: 800,
                whiteSpace: "nowrap",
                maxWidth: 132,
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              {resolvedLabel}
            </div>
          </div>
        </Html>
      ) : null}
    </group>
  );
}

/* -------------------------------------------------------------------------- */
/* BroadcastCameraRig                                                         */
/* -------------------------------------------------------------------------- */

const MANUAL_CAMERA_PRESETS = Object.freeze({
  wide: {
    position: [0, 2.05, 6.4],
    target: [0, 0.42, 0],
  },
  desk: {
    position: [0, 1.72, 4.35],
    target: [0, 0.18, 0.45],
  },
  score: {
    position: [0.35, 2.18, 5.35],
    target: [0, 0.52, 0.15],
  },
});

export function BroadcastCameraRig({
  activeSpeakerId,
  hosts = DEFAULT_BROADCAST_HOSTS,
  enabled = true,
  home = DEFAULT_CAMERA_HOME,
  lerp = DEFAULT_ANIMATION_PRESET.cameraLerp,
  targetLerp = DEFAULT_ANIMATION_PRESET.cameraTargetLerp,
  subtle = true,
  autoCamera = true,
  selectedCamera = "wide",
}) {
  const { camera } = useThree();
  const lookTargetRef = useRef(new THREE.Vector3(...home.target));
  const targetPositionRef = useRef([...home.position]);
  const targetLookRef = useRef([...home.target]);

  const hostMap = useMemo(() => {
    const map = new Map();

    safeArray(hosts).forEach((host) => {
      if (host?.id) map.set(host.id, host);
    });

    return map;
  }, [hosts]);

  useEffect(() => {
    if (!enabled) {
      targetPositionRef.current = [...home.position];
      targetLookRef.current = [...home.target];
      return;
    }

    if (!autoCamera) {
      const preset = MANUAL_CAMERA_PRESETS[selectedCamera];

      if (preset) {
        targetPositionRef.current = [...preset.position];
        targetLookRef.current = [...preset.target];
        return;
      }

      const hostKey =
        selectedCamera === "avery"
          ? "host_1"
          : selectedCamera === "mason"
            ? "host_2"
            : selectedCamera === "rex"
              ? "host_3"
              : activeSpeakerId;

      const manualHost = hostMap.get(hostKey);

      if (manualHost) {
        targetPositionRef.current = [...(manualHost.cameraPosition || home.position)];
        targetLookRef.current = [
          ...(manualHost.cameraTarget || manualHost.position || home.target),
        ];
        return;
      }
    }

    const activeHost = hostMap.get(activeSpeakerId);

    if (!activeHost) {
      targetPositionRef.current = [...home.position];
      targetLookRef.current = [...home.target];
      return;
    }

    const focusPosition = activeHost.cameraPosition || home.position;
    const focusTarget = activeHost.cameraTarget || activeHost.position || home.target;

    if (subtle) {
      targetPositionRef.current = [
        THREE.MathUtils.lerp(home.position[0], focusPosition[0], 0.55),
        THREE.MathUtils.lerp(home.position[1], focusPosition[1], 0.45),
        THREE.MathUtils.lerp(home.position[2], focusPosition[2], 0.42),
      ];

      targetLookRef.current = [
        THREE.MathUtils.lerp(home.target[0], focusTarget[0], 0.65),
        THREE.MathUtils.lerp(home.target[1], focusTarget[1], 0.45),
        THREE.MathUtils.lerp(home.target[2], focusTarget[2], 0.55),
      ];
    } else {
      targetPositionRef.current = [...focusPosition];
      targetLookRef.current = [...focusTarget];
    }
  }, [activeSpeakerId, autoCamera, enabled, home, hostMap, selectedCamera, subtle]);

  useFrame(() => {
    if (!enabled || !camera) return;

    camera.position.x = THREE.MathUtils.lerp(
      camera.position.x,
      targetPositionRef.current[0],
      lerp
    );

    camera.position.y = THREE.MathUtils.lerp(
      camera.position.y,
      targetPositionRef.current[1],
      lerp
    );

    camera.position.z = THREE.MathUtils.lerp(
      camera.position.z,
      targetPositionRef.current[2],
      lerp
    );

    const nextLook = lookTargetRef.current.toArray();
    lerpArray3(nextLook, targetLookRef.current, targetLerp);
    lookTargetRef.current.set(nextLook[0], nextLook[1], nextLook[2]);

    camera.lookAt(lookTargetRef.current);
  });

  return null;
}

/* -------------------------------------------------------------------------- */
/* Broadcast line helper                                                      */
/* -------------------------------------------------------------------------- */

export function buildLevelOneBroadcastLines({
  game = null,
  recap = "",
  headline = "",
  stakes = "",
  userProspectNote = "",
  defaultDurationMs = 5800,
} = {}) {
  const score = buildBroadcastScoreContext(game);
  const scoreLine = buildSimpleScoreLine(score);

  const safeHeadline =
    headline ||
    game?.headline ||
    game?.title ||
    game?.storyline ||
    `${score.away} against ${score.home}`;

  const safeRecap =
    recap ||
    game?.recap ||
    game?.game_recap ||
    game?.gameRecap ||
    game?.summary ||
    game?.postgame ||
    game?.post_game ||
    game?.postGame ||
    game?.broadcast_recap ||
    game?.broadcastRecap ||
    "";

  const safeStakes =
    stakes ||
    game?.stakes ||
    game?.implication ||
    game?.implications ||
    game?.tagline ||
    "";

  const lines = [
    {
      speakerId: "host_2",
      speakerName: "Center Anchor",
      emotion: "serious",
      durationMs: defaultDurationMs,
      text: `Welcome back to the desk. ${scoreLine}. ${safeHeadline}`,
      scoreContext: score,
    },
    {
      speakerId: "host_1",
      speakerName: "Left Analyst",
      emotion: safeRecap ? "excited" : "analytical",
      durationMs: defaultDurationMs + 500,
      text:
        safeRecap ||
        "The game result is available, but the full recap has not been attached yet. The desk is reading the score, the matchup, and the tournament context from the sim.",
      scoreContext: score,
    },
    {
      speakerId: "host_3",
      speakerName: "Right Scout",
      emotion: userProspectNote ? "analytical" : "calm",
      durationMs: defaultDurationMs,
      text:
        userProspectNote ||
        safeStakes ||
        "From a scouting and tournament perspective, the important part is how this result changes momentum, pressure, and the next matchup.",
      scoreContext: score,
    },
  ];

  return lines.filter((line) => safeStr(line.text).trim().length > 0);
}

/* -------------------------------------------------------------------------- */
/* Broadcast host group                                                       */
/* -------------------------------------------------------------------------- */

export function BroadcastHostTrio({
  modelUrl,
  hosts = DEFAULT_BROADCAST_HOSTS,
  activeSpeakerId = "host_2",
  currentLine = null,
  scoreContext = null,
  showNameplates = true,
  showActiveRings = true,
  onModelReady,
}) {
  return (
    <>
      {safeArray(hosts).map((host) => {
        const isSpeaking = activeSpeakerId === host.id;
        const emotion = currentLine?.speakerId === host.id ? currentLine?.emotion : "neutral";

        return (
          <React.Fragment key={host.id}>
            <BroadcastHostSpotlight
              hostId={host.id}
              activeSpeakerId={activeSpeakerId}
              position={host.position}
            />

            <AnimatedHost
              modelUrl={modelUrl}
              hostId={host.id}
              speakerLabel={
                currentLine?.speakerId === host.id ? currentLine?.speakerName : host.label
              }
              role={host.role}
              isSpeaking={isSpeaking}
              isActive={isSpeaking}
              isListening={!isSpeaking}
              activeSpeakerId={activeSpeakerId}
              position={host.position}
              rotation={host.rotation}
              scale={host.scale}
              emotion={emotion}
              scoreContext={scoreContext || currentLine?.scoreContext}
              showNameplate={showNameplates}
              showActiveRing={showActiveRings}
              onModelReady={onModelReady}
            />
          </React.Fragment>
        );
      })}
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* Small 3D caption helper                                                    */
/* -------------------------------------------------------------------------- */

export function BroadcastFloatingCaption({
  currentLine,
  scoreContext,
  position = [0, -1.08, 1.28],
  visible = true,
}) {
  if (!visible) return null;

  const speakerName = currentLine?.speakerName || "Broadcast Desk";

  const text =
    currentLine?.text ||
    (scoreContext ? buildSimpleScoreLine(scoreContext) : "Broadcast line pending.");

  return (
    <Html
      position={position}
      center
      distanceFactor={7}
      transform
      occlude={false}
      style={{
        pointerEvents: "none",
        userSelect: "none",
      }}
    >
      <div
        style={{
          width: 360,
          maxWidth: 360,
          padding: "10px 14px",
          borderRadius: 18,
          border: "1px solid rgba(219, 234, 254, 0.28)",
          background:
            "linear-gradient(135deg, rgba(2, 6, 23, 0.92), rgba(15, 23, 42, 0.84))",
          boxShadow: "0 18px 40px rgba(0, 0, 0, 0.34)",
          color: "#f8fafc",
          fontFamily:
            "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
          textAlign: "left",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: 5,
            color: "#bfdbfe",
            fontSize: 10,
            fontWeight: 900,
            letterSpacing: "0.13em",
            textTransform: "uppercase",
          }}
        >
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: 999,
              background: "#60a5fa",
              boxShadow: "0 0 14px rgba(96, 165, 250, 0.9)",
            }}
          />
          {speakerName}
        </div>

        <div
          style={{
            fontSize: 12,
            lineHeight: "17px",
            fontWeight: 650,
            color: "rgba(248, 250, 252, 0.94)",
          }}
        >
          {text}
        </div>
      </div>
    </Html>
  );
}

/* -------------------------------------------------------------------------- */
/* Preload helper                                                             */
/* -------------------------------------------------------------------------- */

export function preloadBroadcastHostModel(modelUrl) {
  if (!modelUrl) return;
  useGLTF.preload(modelUrl);
}

const broadcastAnimations = {
  DEFAULT_BROADCAST_HOSTS,
  DEFAULT_CAMERA_HOME,
  DEFAULT_ANIMATION_PRESET,
  AnimatedHost,
  BroadcastCameraRig,
  BroadcastHostTrio,
  BroadcastHostSpotlight,
  BroadcastDeskScorePulse,
  BroadcastFloatingCaption,
  buildBroadcastScoreContext,
  buildSimpleScoreLine,
  buildLevelOneBroadcastLines,
  preloadBroadcastHostModel,
};

export default broadcastAnimations;