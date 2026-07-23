import React, { Component, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import PlayerHeadshot from "../PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../../utils/playerHeadshots";
import PS1PlayerModel from "./PS1PlayerModel";
import {
  derivePortraitAppearance,
  getPS1FaceSrc,
  getPortraitCamera,
  getTeamPortraitColors,
  mapPortraitSizeToHeadshot,
} from "./ps1PortraitUtils";
import "./PS1PlayerPortrait.css";

/**
 * PS1-style 3D player portrait for profile/detail modals.
 *
 * Uses React Three Fiber with a flat face plane (not UV face mapping) for a
 * retro pasted-photo look. Roster boards and trade cards keep CSS headshots
 * to avoid dozens of live Canvas instances.
 *
 * Falls back to the existing CSS headshot if WebGL or the portrait scene fails.
 */

class PortraitErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch() {
    this.setState({ hasError: true });
  }

  render() {
    if (this.state.hasError) return this.props.fallback;
    return this.props.children;
  }
}

function PortraitFallback({ player, size, className }) {
  return (
    <div className={`ps1-portrait ps1-portrait--fallback ${className}`.trim()} data-size={size}>
      <PlayerHeadshot
        player={player}
        size={mapPortraitSizeToHeadshot(size)}
        variant="card"
      />
    </div>
  );
}

function PortraitScene({ player, faceSrc, teamColors, animate }) {
  return (
    <>
      <color attach="background" args={["#070b14"]} />
      <ambientLight intensity={1} />
      <PS1PlayerModel player={player} faceSrc={faceSrc} teamColors={teamColors} animate={animate} />
    </>
  );
}

export default function PS1PlayerPortrait({
  player,
  size = "profile",
  faceSrc: faceSrcProp,
  teamColors: teamColorsProp,
  className = "",
  animate = true,
  lazy = false,
  placeholder = null,
}) {
  const [webglFailed, setWebglFailed] = useState(false);
  const [shouldMount, setShouldMount] = useState(!lazy);
  const lazyHostRef = useRef(null);

  useEffect(() => {
    if (!lazy || shouldMount) return undefined;

    const node = lazyHostRef.current;
    if (!node) return undefined;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setShouldMount(true);
          observer.disconnect();
        }
      },
      { rootMargin: "160px" }
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, [lazy, shouldMount]);

  const resolvedPlayer = useMemo(() => ensurePlayerHeadshotFields(player || {}), [player]);
  const resolvedFaceSrc = useMemo(
    () => getPS1FaceSrc(resolvedPlayer, faceSrcProp),
    [resolvedPlayer, faceSrcProp]
  );
  const teamColors = useMemo(
    () => {
      const base = teamColorsProp || getTeamPortraitColors(resolvedPlayer);
      const appearance = derivePortraitAppearance(resolvedPlayer);
      return {
        ...base,
        skin: appearance.skin || base.skin,
        hair: appearance.hair !== "transparent" ? appearance.hair : base.hair,
      };
    },
    [teamColorsProp, resolvedPlayer]
  );

  const camera = useMemo(() => getPortraitCamera(size), [size]);

  if (webglFailed) {
    return <PortraitFallback player={resolvedPlayer} size={size} className={className} />;
  }

  if (lazy && !shouldMount) {
    return (
      <div
        ref={lazyHostRef}
        className={`ps1-portrait ps1-portrait--lazy ${className}`.trim()}
        data-size={size}
        aria-hidden="true"
      >
        {placeholder}
      </div>
    );
  }

  const fallback = <PortraitFallback player={resolvedPlayer} size={size} className={className} />;

  return (
    <div
      ref={lazy ? lazyHostRef : null}
      className={`ps1-portrait ${className}`.trim()}
      data-size={size}
      data-portrait="ps1"
      aria-hidden="true"
    >
      <PortraitErrorBoundary fallback={fallback}>
        <Canvas
          dpr={[1, 1]}
          gl={{
            antialias: false,
            alpha: true,
            powerPreference: "default",
          }}
          camera={{
            position: camera.position,
            fov: camera.fov,
            near: 0.1,
            far: 10,
          }}
          onCreated={({ gl }) => {
            gl.setClearColor(new THREE.Color("#070b14"), 1);
            gl.outputColorSpace = THREE.SRGBColorSpace;
          }}
          onError={() => setWebglFailed(true)}
        >
          <Suspense fallback={null}>
            <PortraitScene
              player={resolvedPlayer}
              faceSrc={resolvedFaceSrc}
              teamColors={teamColors}
              animate={animate}
            />
          </Suspense>
        </Canvas>
      </PortraitErrorBoundary>
    </div>
  );
}
