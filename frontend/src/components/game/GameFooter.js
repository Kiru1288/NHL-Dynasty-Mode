import React from "react";

export function GameFooter({ hints }) {
  const text = hints || "↑↓ SELECT  ·  ENTER CONFIRM  ·  ← → ADJUST  ·  ESC BACK";
  return (
    <footer className="game-footer">
      <div className="game-footer__bar">{text}</div>
    </footer>
  );
}
