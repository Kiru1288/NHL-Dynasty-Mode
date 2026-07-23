/** Ceremony reveal timing and animation class tokens. */

export const AWARDS_REVEAL = {
  introMs: 900,
  slideMs: 680,
  trophyMs: 520,
  winnerMs: 420,
  statStaggerMs: 70,
  finalistStaggerMs: 90,
};

export function revealClasses(revealed, phase = "all") {
  const base = revealed ? "an-revealed" : "an-pending";
  return `an-animate ${base} an-phase-${phase}`;
}

export function slideTransitionKey(slide) {
  return slide?.id || slide?.awardKey || "empty";
}
