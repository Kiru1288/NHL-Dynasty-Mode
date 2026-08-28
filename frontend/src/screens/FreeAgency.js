import React, { useCallback, useEffect, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import FreeAgencyMenu from "../events/freeAgency/FreeAgencyMenu";
import { getFreeAgencyDesk } from "../services/franchiseService";

/**
 * Standalone Free Agency Wire — same UI as the offseason timeline desk,
 * available from Hub without mutating offseason_stage.
 */
export default function FreeAgency() {
  const { franchiseState, setScreen, setFranchiseState, refreshFranchise } = useGameUI();
  const [desk, setDesk] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const loadDesk = useCallback(async () => {
    if (franchiseState?.free_agency_market?.free_agents?.length) {
      setDesk(franchiseState.free_agency_market);
      setLoading(false);
      return;
    }
    setLoading(true);
    setError("");
    try {
      const res = await getFreeAgencyDesk();
      const market = res?.free_agency_market || res?.market || null;
      if (market) {
        setDesk(market);
        setFranchiseState((prev) => {
          if (prev?.free_agency_market === market) return prev;
          return {
            ...(prev || {}),
            free_agency_market: market,
            free_agents: market.free_agents || res?.free_agents || prev?.free_agents,
          };
        });
      }
    } catch (e) {
      setError(String(e?.message || e || "Free agency desk unavailable"));
    } finally {
      setLoading(false);
    }
  }, [franchiseState?.free_agency_market, setFranchiseState]);

  useEffect(() => {
    loadDesk();
  }, [loadDesk]);

  const onBack = useCallback(() => {
    setScreen(SCREENS.HUB);
  }, [setScreen]);

  if (loading && !desk && !franchiseState?.free_agency_market) {
    return (
      <div className="game-screen" style={{ padding: "2rem", color: "#c8d4e0" }}>
        Opening Free Agency Wire…
      </div>
    );
  }

  if (error && !franchiseState?.free_agency_market && !desk) {
    return (
      <div className="game-screen" style={{ padding: "2rem", color: "#c8d4e0" }}>
        <p>{error}</p>
        <button type="button" onClick={onBack}>
          Back to Hub
        </button>
      </div>
    );
  }

  return (
    <div className="game-screen free-agency-screen" style={{ height: "100%", minHeight: 0 }}>
      <FreeAgencyMenu
        franchiseState={franchiseState}
        eventData={{
          free_agency_market: desk || franchiseState?.free_agency_market,
          free_agents: desk?.free_agents || franchiseState?.free_agents,
        }}
        standalone
        onBack={onBack}
        onContinue={onBack}
        ctaLabel="Back to Hub"
      />
    </div>
  );
}
