/** GM-facing agent card helpers (matches backend player_agent_engine tiers). */

export function resolveAgentDealDifficulty(agent = {}) {
  const preset = String(agent.deal_difficulty || "").toLowerCase();
  if (preset === "hard" || preset === "easy" || preset === "medium") {
    return {
      tier: preset,
      label:
        agent.deal_difficulty_label ||
        (preset === "hard" ? "Hard negotiator" : preset === "easy" ? "Easy negotiator" : "Balanced negotiator"),
    };
  }
  const neg = String(agent.negotiation || "").toLowerCase();
  const patience = Number(agent.patience);
  if (neg === "demanding" || neg === "aggressive" || (Number.isFinite(patience) && patience < 0.34)) {
    return { tier: "hard", label: "Hard negotiator" };
  }
  if (neg === "patient" || neg === "stable" || (Number.isFinite(patience) && patience >= 0.66)) {
    return { tier: "easy", label: "Easy negotiator" };
  }
  return { tier: "medium", label: "Balanced negotiator" };
}

export function agentDifficultyIcon(tier) {
  if (tier === "hard") return "!!";
  if (tier === "easy") return "✓";
  return "≈";
}

export function agentDifficultyTone(tier) {
  if (tier === "hard") return "gold";
  if (tier === "easy") return "green";
  return "cyan";
}
