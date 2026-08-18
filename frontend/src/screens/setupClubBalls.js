import React, {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Decal } from "@react-three/drei";
import { MathUtils, SRGBColorSpace, TextureLoader } from "three";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";

const COLS = 8;
const RADIUS = 0.42;
const GAP_X = 1.12;
const GAP_Y = 1.14;
const DECAL_OFFSETS = [
  { position: [0, 0, 0.2], rotation: [0, 0, 0] },
  { position: [0.2, 0, 0], rotation: [0, Math.PI / 2, 0] },
  { position: [0, 0, -0.2], rotation: [0, Math.PI, 0] },
  { position: [-0.2, 0, 0], rotation: [0, -Math.PI / 2, 0] },
];

function useLogoTexture(src) {
  const [map, setMap] = useState(null);

  useEffect(() => {
    if (!src) {
      setMap(null);
      return undefined;
    }

    let disposed = false;
    const loader = new TextureLoader();
    loader.setCrossOrigin("anonymous");
    loader.load(
      src,
      (texture) => {
        if (disposed) {
          texture.dispose();
          return;
        }
        texture.colorSpace = SRGBColorSpace;
        texture.anisotropy = 8;
        setMap(texture);
      },
      undefined,
      () => {
        if (!disposed) {
          setMap(null);
        }
      }
    );

    return () => {
      disposed = true;
    };
  }, [src]);

  return map;
}

function FitClubGrid({ cols, rows }) {
  const { camera, size } = useThree();

  useLayoutEffect(() => {
    const width = (cols - 1) * GAP_X + RADIUS * 2.2;
    const height = (rows - 1) * GAP_Y + RADIUS * 2.2;
    const aspect = size.width / Math.max(size.height, 1);
    const halfW = width / 2;
    const halfH = height / 2;

    camera.position.set(0, 0, 12);
    camera.near = 0.1;
    camera.far = 40;

    if (halfW / aspect >= halfH) {
      camera.left = -halfW;
      camera.right = halfW;
      camera.top = halfW / aspect;
      camera.bottom = -halfW / aspect;
    } else {
      camera.top = halfH;
      camera.bottom = -halfH;
      camera.left = -halfH * aspect;
      camera.right = halfH * aspect;
    }

    camera.updateProjectionMatrix();
  }, [camera, size.width, size.height, cols, rows]);

  return null;
}

function ClubBall({
  src,
  position,
  selected,
  onSelect,
}) {
  const meshRef = useRef(null);
  const map = useLogoTexture(src);

  useFrame((_, delta) => {
    const mesh = meshRef.current;
    if (!mesh) {
      return;
    }
    mesh.rotation.y += delta * (selected ? 2.8 : 1.85);
    mesh.rotation.x = MathUtils.lerp(
      mesh.rotation.x,
      selected ? -0.12 : -0.06,
      0.08
    );
    mesh.scale.setScalar(
      MathUtils.lerp(mesh.scale.x, selected ? 1.16 : 1, Math.min(1, delta * 8))
    );
  });

  return (
    <mesh
      ref={meshRef}
      position={position}
      onClick={(event) => {
        event.stopPropagation();
        onSelect();
      }}
      onPointerOver={() => {
        document.body.style.cursor = "pointer";
      }}
      onPointerOut={() => {
        document.body.style.cursor = "auto";
      }}
    >
      <sphereGeometry args={[RADIUS, 48, 48]} />
      <meshStandardMaterial
        color={selected ? "#3a2f1c" : "#16130f"}
        metalness={0.42}
        roughness={0.28}
      />
      {map
        ? DECAL_OFFSETS.map((decal, index) => (
            <Decal
              key={index}
              position={decal.position}
              rotation={decal.rotation}
              scale={0.78}
              map={map}
              polygonOffset
              polygonOffsetFactor={-1}
            />
          ))
        : null}
    </mesh>
  );
}

function ClubBallField({
  teams,
  selectedIndex,
  onSelect,
}) {
  const selectRef = useRef(onSelect);
  selectRef.current = onSelect;
  const rows = Math.ceil(teams.length / COLS) || 1;

  const layout = useMemo(
    () =>
      teams.map((team, index) => {
        const col = index % COLS;
        const row = Math.floor(index / COLS);
        return {
          team,
          index,
          position: [
            (col - (COLS - 1) / 2) * GAP_X,
            ((rows - 1) / 2 - row) * GAP_Y,
            0,
          ],
          src:
            team.logo ||
            resolveFranchiseTeamLogo(
              team.raw || team,
              team.name || team.code
            ),
        };
      }),
    [teams, rows]
  );

  return (
    <>
      <FitClubGrid cols={COLS} rows={rows} />
      <ambientLight intensity={0.78} />
      <hemisphereLight
        color="#f4ead4"
        groundColor="#1a1610"
        intensity={0.9}
      />
      <directionalLight
        position={[4.5, 6.5, 8]}
        intensity={2.35}
        color="#fff4e0"
      />
      <pointLight
        position={[-5, 2.5, 5]}
        intensity={1.1}
        color="#c9a86a"
      />
      {layout.map((item) => (
        <ClubBall
          key={item.team.code || item.index}
          src={item.src}
          position={item.position}
          selected={item.index === selectedIndex}
          onSelect={() => selectRef.current?.(item.index)}
        />
      ))}
    </>
  );
}

export function ClubBallBoard({
  teams,
  selectedIndex,
  onSelect,
}) {
  return (
    <div className="setup-club-ball-wrap">
      <Canvas
        orthographic
        dpr={[1, 2]}
        gl={{
          antialias: true,
          alpha: true,
          premultipliedAlpha: false,
        }}
        style={{ background: "transparent" }}
        camera={{ position: [0, 0, 12], zoom: 1 }}
        onCreated={({ gl }) => {
          gl.setClearColor(0x000000, 0);
        }}
      >
        <ClubBallField
          teams={teams}
          selectedIndex={selectedIndex}
          onSelect={onSelect}
        />
      </Canvas>
    </div>
  );
}
