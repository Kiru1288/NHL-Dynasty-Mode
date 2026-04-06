import React, { useEffect, useState } from "react";
import { GAME_H, GAME_W } from "../../game/constants";

/**
 * Fixed-resolution scene scaled to fit viewport (PS2-style pipeline).
 */
export function GameCanvas({ children }) {
  const [scale, setScale] = useState(1);

  useEffect(() => {
    function fit() {
      const sx = window.innerWidth / GAME_W;
      const sy = window.innerHeight / GAME_H;
      setScale(Math.min(sx, sy, 1) * 0.99);
    }
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, []);

  return (
    <div className="game-root">
      <div
        className="game-canvas"
        style={{
          width: GAME_W,
          height: GAME_H,
          transform: `scale(${scale})`,
        }}
      >
        {children}
      </div>
    </div>
  );
}
