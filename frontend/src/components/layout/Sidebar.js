import React from "react";
import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Command Center" },
  { to: "/setup", label: "New franchise" },
  { to: "/league", label: "League intel" },
  { to: "/team", label: "Club desk" },
  { to: "/prospects", label: "Prospects" },
  { to: "/narrative", label: "Storylines" },
];

export function Sidebar() {
  return (
    <aside className="app-sidebar">
      <nav className="app-sidebar__nav">
        {links.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            end={l.to === "/"}
            className={({ isActive }) =>
              "app-sidebar__link" + (isActive ? " app-sidebar__link--active" : "")
            }
          >
            {l.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
