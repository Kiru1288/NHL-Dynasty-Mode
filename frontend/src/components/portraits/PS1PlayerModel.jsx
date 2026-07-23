import React, { useEffect, useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { derivePortraitAppearance } from "./ps1PortraitUtils";

/**
 * Head-and-neck PS1 portrait — no torso/shoulders.
 * Face is a flat plane (pasted PNG when available) or simple procedural eyes/mouth.
 */

function applyPS1TextureSettings(texture) {
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.generateMipmaps = false;
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function useFaceTexture(faceSrc) {
  const [texture, setTexture] = useState(null);

  useEffect(() => {
    if (!faceSrc) {
      setTexture(null);
      return undefined;
    }

    let alive = true;
    let loadedTexture = null;
    const loader = new THREE.TextureLoader();

    loader.load(
      faceSrc,
      (tex) => {
        if (!alive) {
          tex.dispose();
          return;
        }
        loadedTexture = applyPS1TextureSettings(tex);
        setTexture(loadedTexture);
      },
      undefined,
      () => {
        if (alive) setTexture(null);
      }
    );

    return () => {
      alive = false;
      if (loadedTexture) loadedTexture.dispose();
    };
  }, [faceSrc]);

  return texture;
}

function HairCap({ style, color, headY = 0.1 }) {
  if (style === "bald" || style === "mask" || !color || color === "transparent") {
    return null;
  }

  if (style === "buzz") {
    return (
      <mesh position={[0, headY + 0.34, -0.02]}>
        <sphereGeometry args={[0.44, 6, 4, 0, Math.PI * 2, 0, Math.PI * 0.35]} />
        <meshBasicMaterial color={color} />
      </mesh>
    );
  }

  if (style === "afro") {
    return (
      <mesh position={[0, headY + 0.38, -0.04]}>
        <sphereGeometry args={[0.52, 7, 5]} />
        <meshBasicMaterial color={color} />
      </mesh>
    );
  }

  if (style === "flow" || style === "long") {
    return (
      <group position={[0, headY + 0.28, -0.04]}>
        <mesh>
          <sphereGeometry args={[0.46, 6, 4, 0, Math.PI * 2, 0, Math.PI * 0.62]} />
          <meshBasicMaterial color={color} />
        </mesh>
        <mesh position={[-0.22, -0.08, -0.06]} rotation={[0, 0.2, 0.15]}>
          <boxGeometry args={[0.14, 0.32, 0.12]} />
          <meshBasicMaterial color={color} />
        </mesh>
        <mesh position={[0.22, -0.08, -0.06]} rotation={[0, -0.2, -0.15]}>
          <boxGeometry args={[0.14, 0.32, 0.12]} />
          <meshBasicMaterial color={color} />
        </mesh>
      </group>
    );
  }

  if (style === "swept" || style === "part") {
    return (
      <mesh position={[0.06, headY + 0.32, -0.03]} rotation={[0.08, style === "swept" ? -0.22 : 0.12, 0]}>
        <boxGeometry args={[0.52, 0.22, 0.38]} />
        <meshBasicMaterial color={color} />
      </mesh>
    );
  }

  if (style === "curly" || style === "messy") {
    return (
      <mesh position={[0, headY + 0.34, -0.04]}>
        <sphereGeometry args={[0.5, 8, 6, 0, Math.PI * 2, 0, Math.PI * 0.55]} />
        <meshBasicMaterial color={color} />
      </mesh>
    );
  }

  if (style === "spiky") {
    return (
      <group position={[0, headY + 0.3, -0.02]}>
        {[-0.18, -0.06, 0.06, 0.18].map((x, i) => (
          <mesh key={i} position={[x, 0.08 + (i % 2) * 0.04, 0]} rotation={[0.2, 0, x * 0.6]}>
            <boxGeometry args={[0.1, 0.18, 0.1]} />
            <meshBasicMaterial color={color} />
          </mesh>
        ))}
      </group>
    );
  }

  // crop, grey, beard stub, default
  return (
    <mesh position={[0, headY + 0.32, -0.02]} rotation={[0.1, 0, 0]}>
      <sphereGeometry args={[0.46, 6, 4, 0, Math.PI * 2, 0, Math.PI * 0.5]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
}

function ProceduralFaceFeatures({ skin, skinShadow, eyeSpacing, mouthWidth, headY = 0.1 }) {
  return (
    <group position={[0, headY, 0.47]}>
      <mesh position={[-eyeSpacing, 0.06, 0]}>
        <sphereGeometry args={[0.045, 4, 4]} />
        <meshBasicMaterial color="#1a1a1a" />
      </mesh>
      <mesh position={[eyeSpacing, 0.06, 0]}>
        <sphereGeometry args={[0.045, 4, 4]} />
        <meshBasicMaterial color="#1a1a1a" />
      </mesh>
      <mesh position={[0, -0.1, 0]}>
        <boxGeometry args={[mouthWidth, 0.025, 0.02]} />
        <meshBasicMaterial color={skinShadow || skin} />
      </mesh>
      <mesh position={[0, 0.02, -0.01]}>
        <boxGeometry args={[0.14, 0.06, 0.02]} />
        <meshBasicMaterial color={skinShadow || skin} transparent opacity={0.35} />
      </mesh>
    </group>
  );
}

export default function PS1PlayerModel({ player, faceSrc, teamColors = {}, animate = true }) {
  const rootRef = useRef(null);
  const faceTexture = useFaceTexture(faceSrc);
  const appearance = useMemo(() => derivePortraitAppearance(player || {}), [player]);

  const colors = useMemo(
    () => ({
      skin: appearance.skin || teamColors.skin || "#c58b5f",
      skinShadow: appearance.skinShadow || "#8d5b3d",
      hair: appearance.hair || teamColors.hair || "#111927",
      hairStyle: appearance.hairStyle || "crop",
    }),
    [appearance, teamColors]
  );

  const headY = 0.1;

  useFrame((state) => {
    if (!animate || !rootRef.current) return;
    const t = state.clock.elapsedTime;
    rootRef.current.rotation.y = Math.sin(t * 0.35) * 0.04 + (appearance.headTilt || 0);
    rootRef.current.position.y = Math.sin(t * 0.9) * 0.008;
  });

  return (
    <group ref={rootRef} position={[0, -0.05, 0]}>
      {/* Upper neck only — frame crops below this */}
      <mesh position={[0, headY - 0.38, 0]}>
        <cylinderGeometry args={[0.16, 0.2, 0.22, 5]} />
        <meshBasicMaterial color={colors.skinShadow} />
      </mesh>
      <mesh position={[0, headY - 0.24, 0]}>
        <cylinderGeometry args={[0.18, 0.16, 0.18, 5]} />
        <meshBasicMaterial color={colors.skin} />
      </mesh>

      {/* Low-poly head */}
      <mesh position={[0, headY, 0]}>
        <sphereGeometry args={[0.46, 8, 6]} />
        <meshBasicMaterial color={colors.skin} />
      </mesh>

      <HairCap style={colors.hairStyle} color={colors.hair} headY={headY} />

      {/* Pasted face texture or flat skin plane */}
      <mesh position={[0, headY, 0.44]}>
        <planeGeometry args={[0.54, 0.66]} />
        {faceTexture ? (
          <meshBasicMaterial map={faceTexture} transparent toneMapped={false} />
        ) : (
          <meshBasicMaterial color={colors.skin} />
        )}
      </mesh>

      {!faceTexture ? (
        <ProceduralFaceFeatures
          skin={colors.skin}
          skinShadow={colors.skinShadow}
          eyeSpacing={appearance.eyeSpacing}
          mouthWidth={appearance.mouthWidth}
          headY={headY}
        />
      ) : null}

      {colors.hairStyle === "beard" ? (
        <mesh position={[0, headY - 0.12, 0.38]}>
          <boxGeometry args={[0.36, 0.18, 0.12]} />
          <meshBasicMaterial color={colors.hair} transparent opacity={0.85} />
        </mesh>
      ) : null}
    </group>
  );
}
