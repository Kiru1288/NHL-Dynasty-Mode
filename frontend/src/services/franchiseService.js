import { api } from "./api";

export async function listTeams() {
  const { data } = await api.get("/api/franchise/teams");
  return data.teams || [];
}

export async function startFranchise(payload) {
  const { data } = await api.post("/api/franchise/start", payload);
  return data;
}

export async function getFranchiseState() {
  const { data } = await api.get("/api/franchise/state");
  return data;
}

export async function advanceDay() {
  const { data } = await api.post("/api/franchise/advance");
  return data;
}

export async function submitDecision(decisionId, choiceId) {
  const { data } = await api.post("/api/franchise/decision", {
    decision_id: decisionId,
    choice_id: choiceId,
  });
  return data;
}
