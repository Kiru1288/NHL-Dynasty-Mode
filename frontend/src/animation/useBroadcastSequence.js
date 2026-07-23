// frontend/src/animation/useBroadcastSequence.js

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/**
 * useBroadcastSequence.js
 *
 * Reusable broadcast sequence hook.
 *
 * This file does NOT:
 * - connect to the backend
 * - generate AI scripts
 * - generate voice audio
 * - perform 3D animation directly
 *
 * This file DOES:
 * - decide which host is speaking
 * - advance through broadcast lines
 * - expose caption text
 * - expose active speaker id
 * - support pause/play/restart/next/previous
 * - safely handle empty backend data
 *
 * IMPORTANT:
 * This file exports BOTH:
 * - named export: useBroadcastSequence
 * - default export: useBroadcastSequence
 *
 * That means both of these imports work:
 *
 * import useBroadcastSequence from "../../animation/useBroadcastSequence";
 * import { useBroadcastSequence } from "../../animation/useBroadcastSequence";
 */

const FALLBACK_LINE = Object.freeze({
  id: "fallback-line",
  speakerId: "host_2",
  speakerName: "Center Anchor",
  emotion: "calm",
  text: "The broadcast desk is waiting for the next game update from the sim.",
  durationMs: 5200,
  scoreContext: null,
  meta: {
    fallback: true,
  },
});

const DEFAULT_OPTIONS = Object.freeze({
  autoPlay: true,
  loop: true,
  defaultDurationMs: 5800,
  minDurationMs: 2600,
  maxDurationMs: 14000,
  restartOnLinesChange: true,
  pauseWhenHidden: true,
});

function safeArray(value, fallback = []) {
  return Array.isArray(value) ? value : fallback;
}

function safeStr(value, fallback = "") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function safeNum(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalizeSpeakerId(value, fallback = "host_2") {
  const raw = safeStr(value, fallback).trim();
  return raw || fallback;
}

function normalizeLineText(line) {
  return safeStr(
    line?.text ??
      line?.line ??
      line?.caption ??
      line?.body ??
      line?.summary ??
      line?.dialogue ??
      line?.script,
    ""
  ).trim();
}

function createLineId(line, index) {
  return safeStr(
    line?.id ??
      line?.lineId ??
      line?.line_id ??
      line?.key ??
      `${normalizeSpeakerId(
        line?.speakerId ??
          line?.speaker_id ??
          line?.hostId ??
          line?.host_id ??
          "host_2"
      )}-${index}`,
    `line-${index}`
  );
}

function normalizeBroadcastLine(line, index, options) {
  const defaultDurationMs = safeNum(
    options?.defaultDurationMs,
    DEFAULT_OPTIONS.defaultDurationMs
  );

  const minDurationMs = safeNum(
    options?.minDurationMs,
    DEFAULT_OPTIONS.minDurationMs
  );

  const maxDurationMs = safeNum(
    options?.maxDurationMs,
    DEFAULT_OPTIONS.maxDurationMs
  );

  if (!line || typeof line !== "object") {
    return {
      ...FALLBACK_LINE,
      id: `fallback-${index}`,
      durationMs: defaultDurationMs,
      meta: {
        ...FALLBACK_LINE.meta,
        originalIndex: index,
      },
    };
  }

  const speakerId = normalizeSpeakerId(
    line.speakerId ??
      line.speaker_id ??
      line.hostId ??
      line.host_id ??
      line.characterId ??
      line.character_id,
    "host_2"
  );

  const speakerName = safeStr(
    line.speakerName ??
      line.speaker_name ??
      line.hostName ??
      line.host_name ??
      line.characterName ??
      line.character_name,
    speakerId
  );

  const rawDuration =
    line.durationMs ??
    line.duration_ms ??
    line.duration ??
    line.seconds ??
    line.timeMs ??
    line.time_ms;

  let durationMs = safeNum(rawDuration, defaultDurationMs);

  // If caller accidentally passes seconds instead of milliseconds,
  // convert seconds to milliseconds.
  if (durationMs > 0 && durationMs < 80) {
    durationMs *= 1000;
  }

  durationMs = clamp(durationMs, minDurationMs, maxDurationMs);

  const text = normalizeLineText(line);

  return {
    ...line,
    id: createLineId(line, index),
    speakerId,
    speakerName,
    emotion: safeStr(line.emotion ?? line.tone ?? line.mood, "neutral"),
    text: text || FALLBACK_LINE.text,
    durationMs,
    scoreContext:
      line.scoreContext ??
      line.score_context ??
      line.gameContext ??
      line.game_context ??
      line.context ??
      null,
    meta: {
      ...(line.meta || {}),
      originalIndex: index,
      fallback: !text,
    },
  };
}

function normalizeLines(lines, options) {
  const normalized = safeArray(lines)
    .map((line, index) => normalizeBroadcastLine(line, index, options))
    .filter((line) => safeStr(line.text).trim().length > 0);

  if (!normalized.length) {
    return [
      {
        ...FALLBACK_LINE,
        durationMs: safeNum(options?.defaultDurationMs, FALLBACK_LINE.durationMs),
      },
    ];
  }

  return normalized;
}

function getLineSignatureFromNormalizedLines(lines) {
  return safeArray(lines)
    .map((line) => {
      return `${line.id}:${line.speakerId}:${line.text}:${line.durationMs}`;
    })
    .join("|");
}

function getNextIndex(currentIndex, total, loop) {
  if (total <= 0) return 0;

  if (currentIndex + 1 < total) {
    return currentIndex + 1;
  }

  return loop ? 0 : total - 1;
}

function getPreviousIndex(currentIndex, total, loop) {
  if (total <= 0) return 0;

  if (currentIndex - 1 >= 0) {
    return currentIndex - 1;
  }

  return loop ? total - 1 : 0;
}

function canUseBrowserTimer() {
  return typeof window !== "undefined" && typeof window.setTimeout === "function";
}

function clearBrowserTimeout(timerId) {
  if (!timerId) return;

  if (typeof window !== "undefined" && typeof window.clearTimeout === "function") {
    window.clearTimeout(timerId);
  } else {
    clearTimeout(timerId);
  }
}

function setBrowserTimeout(callback, delay) {
  if (typeof window !== "undefined" && typeof window.setTimeout === "function") {
    return window.setTimeout(callback, delay);
  }

  return setTimeout(callback, delay);
}

export function useBroadcastSequence(lines = [], userOptions = {}) {
  const options = useMemo(() => {
    return {
      ...DEFAULT_OPTIONS,
      ...(userOptions || {}),
    };
  }, [userOptions]);

  const normalizedLines = useMemo(() => {
    return normalizeLines(lines, options);
  }, [lines, options]);

  const lineSignature = useMemo(() => {
    return getLineSignatureFromNormalizedLines(normalizedLines);
  }, [normalizedLines]);

  const [currentLineIndex, setCurrentLineIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(Boolean(options.autoPlay));
  const [hasCompletedOnce, setHasCompletedOnce] = useState(false);
  const [sequenceVersion, setSequenceVersion] = useState(0);

  const timerRef = useRef(null);
  const startedAtRef = useRef(Date.now());
  const pausedAtRef = useRef(null);
  const remainingMsRef = useRef(null);
  const lastSignatureRef = useRef(lineSignature);

  const totalLines = normalizedLines.length;

  const safeIndex = clamp(
    currentLineIndex,
    0,
    Math.max(0, totalLines - 1)
  );

  const currentLine = normalizedLines[safeIndex] || FALLBACK_LINE;

  const activeSpeakerId = currentLine?.speakerId || FALLBACK_LINE.speakerId;
  const activeSpeakerName = currentLine?.speakerName || FALLBACK_LINE.speakerName;
  const activeEmotion = currentLine?.emotion || "neutral";
  const currentCaption = currentLine?.text || FALLBACK_LINE.text;
  const currentDurationMs = safeNum(
    currentLine?.durationMs,
    options.defaultDurationMs
  );
  const scoreContext = currentLine?.scoreContext || null;

  const progress = useMemo(() => {
    if (totalLines <= 0) return 0;
    return (safeIndex + 1) / totalLines;
  }, [safeIndex, totalLines]);

  const isFirstLine = safeIndex === 0;
  const isLastLine = safeIndex === totalLines - 1;

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearBrowserTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const goToLine = useCallback(
    (index, shouldPlay = isPlaying) => {
      clearTimer();

      const nextIndex = clamp(
        safeNum(index, 0),
        0,
        Math.max(0, totalLines - 1)
      );

      setCurrentLineIndex(nextIndex);
      setIsPlaying(Boolean(shouldPlay));
      setSequenceVersion((value) => value + 1);

      startedAtRef.current = Date.now();
      pausedAtRef.current = null;
      remainingMsRef.current = null;
    },
    [clearTimer, isPlaying, totalLines]
  );

  const pause = useCallback(() => {
    clearTimer();

    const elapsed = Date.now() - startedAtRef.current;
    const remaining = clamp(currentDurationMs - elapsed, 0, currentDurationMs);

    remainingMsRef.current = remaining;
    pausedAtRef.current = Date.now();

    setIsPlaying(false);
  }, [clearTimer, currentDurationMs]);

  const play = useCallback(() => {
    if (!totalLines) return;

    startedAtRef.current = Date.now();
    pausedAtRef.current = null;

    setIsPlaying(true);
  }, [totalLines]);

  const restart = useCallback(
    (shouldPlay = options.autoPlay) => {
      clearTimer();

      setCurrentLineIndex(0);
      setIsPlaying(Boolean(shouldPlay));
      setHasCompletedOnce(false);
      setSequenceVersion((value) => value + 1);

      startedAtRef.current = Date.now();
      pausedAtRef.current = null;
      remainingMsRef.current = null;
    },
    [clearTimer, options.autoPlay]
  );

  const stop = useCallback(() => {
    clearTimer();

    setCurrentLineIndex(0);
    setIsPlaying(false);
    setHasCompletedOnce(false);
    setSequenceVersion((value) => value + 1);

    startedAtRef.current = Date.now();
    pausedAtRef.current = null;
    remainingMsRef.current = null;
  }, [clearTimer]);

  const nextLine = useCallback(
    (shouldPlay = isPlaying) => {
      clearTimer();

      const completed = safeIndex === totalLines - 1;
      const nextIndex = getNextIndex(safeIndex, totalLines, options.loop);

      if (completed) {
        setHasCompletedOnce(true);
      }

      setCurrentLineIndex(nextIndex);
      setIsPlaying(Boolean(shouldPlay && (options.loop || !completed)));
      setSequenceVersion((value) => value + 1);

      startedAtRef.current = Date.now();
      pausedAtRef.current = null;
      remainingMsRef.current = null;
    },
    [clearTimer, isPlaying, options.loop, safeIndex, totalLines]
  );

  const previousLine = useCallback(
    (shouldPlay = isPlaying) => {
      clearTimer();

      const previousIndex = getPreviousIndex(
        safeIndex,
        totalLines,
        options.loop
      );

      setCurrentLineIndex(previousIndex);
      setIsPlaying(Boolean(shouldPlay));
      setSequenceVersion((value) => value + 1);

      startedAtRef.current = Date.now();
      pausedAtRef.current = null;
      remainingMsRef.current = null;
    },
    [clearTimer, isPlaying, options.loop, safeIndex, totalLines]
  );

  const toggle = useCallback(() => {
    if (isPlaying) {
      pause();
    } else {
      play();
    }
  }, [isPlaying, pause, play]);

  useEffect(() => {
    const signatureChanged = lastSignatureRef.current !== lineSignature;

    if (!signatureChanged) return;

    lastSignatureRef.current = lineSignature;

    if (options.restartOnLinesChange) {
      restart(options.autoPlay);
      return;
    }

    setCurrentLineIndex((index) => {
      return clamp(index, 0, Math.max(0, normalizedLines.length - 1));
    });

    setSequenceVersion((value) => value + 1);
  }, [
    lineSignature,
    normalizedLines.length,
    options.autoPlay,
    options.restartOnLinesChange,
    restart,
  ]);

  useEffect(() => {
    if (!isPlaying) return undefined;
    if (!totalLines) return undefined;
    if (!canUseBrowserTimer()) return undefined;

    clearTimer();

    const waitMs = remainingMsRef.current ?? currentDurationMs;

    remainingMsRef.current = null;
    startedAtRef.current = Date.now();

    timerRef.current = setBrowserTimeout(() => {
      const completed = safeIndex === totalLines - 1;

      if (completed && !options.loop) {
        setHasCompletedOnce(true);
        setIsPlaying(false);
        return;
      }

      nextLine(true);
    }, waitMs);

    return () => {
      clearTimer();
    };
  }, [
    clearTimer,
    currentDurationMs,
    isPlaying,
    nextLine,
    options.loop,
    safeIndex,
    totalLines,
    sequenceVersion,
  ]);

  useEffect(() => {
    if (!options.pauseWhenHidden) return undefined;
    if (typeof document === "undefined") return undefined;

    const handleVisibilityChange = () => {
      if (document.hidden && isPlaying) {
        pause();
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [isPlaying, options.pauseWhenHidden, pause]);

  useEffect(() => {
    return () => {
      clearTimer();
    };
  }, [clearTimer]);

  const currentLineForAnimation = useMemo(() => {
    return {
      ...currentLine,
      speakerId: activeSpeakerId,
      speakerName: activeSpeakerName,
      emotion: activeEmotion,
      text: currentCaption,
      durationMs: currentDurationMs,
      scoreContext,
      isPlaying,
      isFirstLine,
      isLastLine,
      lineIndex: safeIndex,
      totalLines,
      progress,
      sequenceVersion,
    };
  }, [
    activeEmotion,
    activeSpeakerId,
    activeSpeakerName,
    currentCaption,
    currentDurationMs,
    currentLine,
    isFirstLine,
    isLastLine,
    isPlaying,
    progress,
    safeIndex,
    scoreContext,
    sequenceVersion,
    totalLines,
  ]);

  return {
    lines: normalizedLines,
    totalLines,

    currentLineIndex: safeIndex,
    currentLine,
    currentLineForAnimation,

    activeSpeakerId,
    activeSpeakerName,
    activeEmotion,

    currentCaption,
    currentDurationMs,
    scoreContext,

    isPlaying,
    isPaused: !isPlaying,
    isFirstLine,
    isLastLine,
    hasCompletedOnce,
    progress,
    sequenceVersion,

    play,
    pause,
    toggle,
    restart,
    stop,
    nextLine,
    previousLine,
    goToLine,
  };
}

export default useBroadcastSequence;