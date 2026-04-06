import React from "react";

export function Header() {
  return (
    <header className="app-header">
      <div className="app-header__brand">
        <span className="app-header__mark">NHL</span>
        <span className="app-header__title">Franchise Command</span>
      </div>
      <div className="app-header__tag">Interactive GM · Decision-driven calendar</div>
    </header>
  );
}
