import React from "react";

/**
 * Full-viewport shell (no fixed-resolution scaling).
 */
export function GameCanvas({ children }) {
  return (
    <div className="game-root">
      <div className="game-canvas">{children}</div>
    </div>
  );
}
