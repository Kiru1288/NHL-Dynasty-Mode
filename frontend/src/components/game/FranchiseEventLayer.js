import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useGameUI } from "../../game/GameUIContext";
import { SCREENS } from "../../game/constants";
import FranchiseEventOverlay, {
  getCurrentFranchiseEvent,
} from "../../events/FranchiseEventOverlay";
import {
  FRANCHISE_EVENT_LAYER_CLASSES as CLS,
  FRANCHISE_EVENT_TRANSITION_HALF_MS,
  FRANCHISE_EVENT_TRANSITION_MS,
  transitionDelay,
} from "../../events/shared/franchiseEventTransition";

function playoffsAreComplete(franchiseState) {
  return Boolean(
    franchiseState?.playoffs_done ||
      franchiseState?.flags?.playoffs_done ||
      franchiseState?.flags?.playoffs_simulated
  );
}

function eventIdentity(event, phase, stage) {
  const key = event?.key || "";
  return `${key}|${phase}|${stage}`;
}

/**
 * Global cinematic franchise events (playoffs + offseason).
 * Sits above ShowcasePopupLayer so menus appear instead of raw JSON popups.
 *
 * Players can leave the timeline to Hub World (trades, stats, office) without
 * advancing the stage; reopen via Hub phase CTA / openFranchiseEvent.
 */
export function FranchiseEventLayer() {
  const {
    franchiseState,
    onContinueOffseason,
    onReopenOffseasonStage,
    onGenerateNextSeason,
    onEnterPlayoffs,
    onAdvanceSeasonPhase,
    franchiseEventForceOpen,
    setFranchiseEventForceOpen,
    franchisePhaseTransition,
    setScreen,
  } = useGameUI();

  const [dismissed, setDismissed] = useState(false);
  const [pinnedOpen, setPinnedOpen] = useState(false);
  const [motion, setMotion] = useState("hidden");
  const [renderState, setRenderState] = useState(franchiseState);

  const event = useMemo(
    () => getCurrentFranchiseEvent(franchiseState),
    [franchiseState]
  );

  const eventKey = event?.key;
  const phase = String(
    franchiseState?.season_phase || franchiseState?.phase || ""
  ).toLowerCase();
  const stage = String(franchiseState?.offseason_stage || "").toLowerCase();
  const identity = eventIdentity(event, phase, stage);

  const stickyEventRef = useRef(event);
  if (event) stickyEventRef.current = event;

  const phaseAllowsAuto =
    ["playoff_ready", "post_cup", "offseason", "playoffs"].includes(phase) ||
    (phase === "complete" && playoffsAreComplete(franchiseState));
  const inSeasonCinematic = ["opening_night", "trade_deadline"].includes(eventKey);

  const shouldShow =
    Boolean(event || stickyEventRef.current) &&
    !dismissed &&
    (pinnedOpen || phaseAllowsAuto || inSeasonCinematic);

  const shownEvent = event || stickyEventRef.current;
  const prevIdentityRef = useRef(null);
  const swapTimerRef = useRef(null);
  const exitTimerRef = useRef(null);

  const clearSwapTimer = useCallback(() => {
    if (swapTimerRef.current) {
      window.clearTimeout(swapTimerRef.current);
      swapTimerRef.current = null;
    }
  }, []);

  const clearExitTimer = useCallback(() => {
    if (exitTimerRef.current) {
      window.clearTimeout(exitTimerRef.current);
      exitTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      clearSwapTimer();
      clearExitTimer();
    };
  }, [clearSwapTimer, clearExitTimer]);

  useEffect(() => {
    if (!franchiseEventForceOpen) return;
    setDismissed(false);
    setPinnedOpen(true);
    setFranchiseEventForceOpen(false);
  }, [franchiseEventForceOpen, setFranchiseEventForceOpen]);

  useEffect(() => {
    if (!shouldShow || !shownEvent) {
      if (motion !== "hidden" && motion !== "exiting") {
        setMotion("hidden");
        prevIdentityRef.current = null;
      }
      return undefined;
    }

    if (motion === "hidden" || motion === "exiting") {
      setRenderState(franchiseState);
      setMotion("entering");
      const delay = transitionDelay(FRANCHISE_EVENT_TRANSITION_MS);
      if (!delay) {
        setMotion("visible");
        prevIdentityRef.current = identity;
        return undefined;
      }
      const t = window.setTimeout(() => {
        setMotion("visible");
        prevIdentityRef.current = identity;
      }, delay);
      return () => window.clearTimeout(t);
    }

    if (
      motion === "visible" &&
      prevIdentityRef.current &&
      prevIdentityRef.current !== identity
    ) {
      clearSwapTimer();
      setMotion("swapping-out");
      const half = transitionDelay(FRANCHISE_EVENT_TRANSITION_HALF_MS);
      swapTimerRef.current = window.setTimeout(() => {
        setRenderState(franchiseState);
        setMotion("swapping-in");
        swapTimerRef.current = window.setTimeout(() => {
          setMotion("visible");
          prevIdentityRef.current = identity;
          swapTimerRef.current = null;
        }, half);
      }, half);
    } else if (motion === "visible" && !prevIdentityRef.current) {
      prevIdentityRef.current = identity;
    }

    return clearSwapTimer;
  }, [
    shouldShow,
    franchiseState,
    identity,
    motion,
    clearSwapTimer,
  ]);

  useEffect(() => {
    if (motion === "swapping-out" || motion === "exiting") return;
    setRenderState(franchiseState);
  }, [franchiseState, motion]);

  const handleLeaveToHub = useCallback(() => {
    clearExitTimer();
    const delay = transitionDelay(FRANCHISE_EVENT_TRANSITION_MS);
    if (!delay) {
      setDismissed(true);
      setPinnedOpen(false);
      stickyEventRef.current = null;
      setMotion("hidden");
      prevIdentityRef.current = null;
      if (typeof setScreen === "function") setScreen(SCREENS.HUB);
      return;
    }
    setMotion("exiting");
    exitTimerRef.current = window.setTimeout(() => {
      setDismissed(true);
      setPinnedOpen(false);
      stickyEventRef.current = null;
      setMotion("hidden");
      prevIdentityRef.current = null;
      exitTimerRef.current = null;
      if (typeof setScreen === "function") setScreen(SCREENS.HUB);
    }, delay);
  }, [clearExitTimer, setScreen]);

  if (!shouldShow || !shownEvent) return null;

  const motionClass =
    motion === "entering"
      ? CLS.entering
      : motion === "visible"
        ? CLS.visible
        : motion === "exiting"
          ? CLS.exiting
          : motion === "swapping-out"
            ? CLS.swappingOut
            : motion === "swapping-in"
              ? CLS.swappingIn
              : "";

  const phaseBlurActive = Boolean(franchisePhaseTransition?.active);
  const phaseBlurLabel = String(franchisePhaseTransition?.label || "").trim();

  return (
    <div
      className={`${CLS.root} register-ops ${motionClass}${phaseBlurActive ? ` ${CLS.phaseBlur}` : ""}`}
      data-register="ops"
      role="presentation"
      aria-busy={phaseBlurActive ? "true" : undefined}
    >
      <div className={`${CLS.overlay} ${motionClass}`}>
        <div className={`${CLS.content} ${motionClass}`}>
          <FranchiseEventOverlay
            franchiseState={renderState || franchiseState}
            onClose={handleLeaveToHub}
            onContinueOffseason={onContinueOffseason}
            onReopenOffseasonStage={onReopenOffseasonStage}
            onGenerateNextSeason={onGenerateNextSeason}
            onEnterPlayoffs={onEnterPlayoffs}
            onAdvancePhase={onAdvanceSeasonPhase}
          />
        </div>
        {phaseBlurActive ? (
          <div className="franchise-event-phase-blur" role="status" aria-live="polite">
            <div className="franchise-event-phase-blur__veil" aria-hidden />
            {phaseBlurLabel ? (
              <p className="franchise-event-phase-blur__label">{phaseBlurLabel}</p>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
