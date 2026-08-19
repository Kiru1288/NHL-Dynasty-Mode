import React, { useMemo, useRef } from "react";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";

export const ClubBallBoard = React.memo(function ClubBallBoard({
  teams,
  selectedIndex,
  onSelect,
}) {
  const selectRef = useRef(onSelect);
  selectRef.current = onSelect;

  const items = useMemo(
    () =>
      (teams || []).map((team, index) => ({
        index,
        code: team.code || team.abbr || team.abbreviation || "",
        name: team.name || team.code || "NHL club",
        src:
          team.logo ||
          resolveFranchiseTeamLogo(
            team.raw || team,
            team.name || team.code
          ),
      })),
    [teams]
  );

  return (
    <div className="setup-club-ball-grid" role="listbox" aria-label="Choose your club">
      {items.map((item) => {
        const selected = item.index === selectedIndex;
        return (
          <button
            key={item.code || item.index}
            type="button"
            role="option"
            aria-selected={selected}
            className={selected ? "setup-club-ball is-selected" : "setup-club-ball"}
            title={item.name}
            onClick={() => selectRef.current?.(item.index)}
          >
            <span className="setup-club-ball-orb" aria-hidden="true">
              {item.src ? (
                <img src={item.src} alt="" draggable={false} />
              ) : (
                <em>{item.code || "NHL"}</em>
              )}
            </span>
            <strong>{item.code || "NHL"}</strong>
          </button>
        );
      })}
    </div>
  );
});
