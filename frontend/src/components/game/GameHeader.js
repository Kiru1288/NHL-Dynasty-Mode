import React from "react";

export function GameHeader({ teamName, sectionTitle, showEspn }) {
  return (
    <header className="game-header">
      <div className="game-header__teamMark">
        <span className="game-header__logoSlot" aria-hidden />
        <span className="game-header__teamText">{teamName || "—"}</span>
      </div>
      <div className="game-header__section">{sectionTitle}</div>
      <div className="game-header__brand">
        {showEspn !== false && <span className="game-header__espn">ESPN</span>}
        <span className="game-header__tag">NHL FRANCHISE</span>
      </div>
    </header>
  );
}
