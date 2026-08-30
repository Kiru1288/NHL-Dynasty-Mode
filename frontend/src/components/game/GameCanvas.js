import React from "react";

/**
 * Full-viewport shell. Window fit is applied globally via fluid UI scale.
 */
export function GameCanvas({ children }) {
  return (
    <div className="game-root">
      <div className="game-canvas">{children}</div>
    </div>
  );
}
