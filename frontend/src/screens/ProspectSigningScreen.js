import React, { useCallback, useEffect, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { getProspectRightsDesk } from "../services/franchiseService";
import { ProspectRightsEventMenu } from "../events/offseasonEventMenus";

/** Year-round prospect signing desk: ELC offers for drafted kids you hold the rights to. */
export default function ProspectSigningScreen() {
  const { franchiseState, setScreen, refreshFranchise } = useGameUI();
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setError("");
    try {
      const res = await getProspectRightsDesk();
      setPayload(res?.prospect_rights || res || {});
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Could not load your prospects.");
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const leave = useCallback(() => {
    if (typeof refreshFranchise === "function") refreshFranchise();
    if (typeof setScreen === "function") setScreen(SCREENS.ROSTER || SCREENS.HUB);
  }, [refreshFranchise, setScreen]);

  if (error) {
    return (
      <div className="game-screen" style={{ padding: 32 }}>
        <p>{error}</p>
        <button type="button" onClick={leave}>Back</button>
      </div>
    );
  }
  if (!payload) {
    return <div className="game-screen" style={{ padding: 32 }}>Loading prospects…</div>;
  }
  return (
    <ProspectRightsEventMenu
      key={String(payload?.prospects?.length ?? 0)}
      franchiseState={{ ...(franchiseState || {}), prospect_rights: payload }}
      eventData={{ prospect_rights: payload }}
      onContinue={leave}
      onBack={leave}
    />
  );
}
