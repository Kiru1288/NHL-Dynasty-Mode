import React, { useEffect, useState } from "react";
import { getPlayerDevelopmentOutlook } from "../../services/franchiseService";

// F15: expected growth range next season + the levers that move it.
export default function DevelopmentOutlook({ playerId }) {
  const [data, setData] = useState(null);
  useEffect(() => {
    let live = true;
    setData(null);
    if (!playerId) return undefined;
    getPlayerDevelopmentOutlook(playerId)
      .then((d) => live && setData(d))
      .catch(() => live && setData(null));
    return () => {
      live = false;
    };
  }, [playerId]);
  if (!data || !data.ok) return null;
  const sign = (v) => `${Number(v) > 0 ? "+" : ""}${Number(v || 0).toFixed(1)}`;
  return (
    <div className="dvo">
      <div className="dvo-head">
        <span className="dvo-eyebrow">Development outlook</span>
        <strong>
          {sign(data.expected)} OVR <em>({sign(data.range_low)} to {sign(data.range_high)})</em>
        </strong>
      </div>
      <ul className="dvo-levers">
        {(data.levers || []).map((l) => (
          <li key={l.key} className={`is-${l.tone}`} title={l.effect}>
            <span>{l.label}</span>
            <b>{l.value}</b>
            <em>{l.effect}</em>
          </li>
        ))}
      </ul>
    </div>
  );
}
