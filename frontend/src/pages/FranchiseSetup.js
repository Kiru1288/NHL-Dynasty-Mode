import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Panel } from "../components/ui/Panel";
import { Button } from "../components/ui/Button";
import { listTeams, startFranchise } from "../services/franchiseService";
import {
  setFranchiseSessionId,
  clearFranchiseSession,
  formatFranchiseApiError,
} from "../services/api";

export function FranchiseSetup() {
  const navigate = useNavigate();
  const [teams, setTeams] = useState([]);
  const [teamQuery, setTeamQuery] = useState("");
  const [coachName, setCoachName] = useState("");
  const [archetype, setArchetype] = useState("balanced");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    clearFranchiseSession();
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const t = await listTeams();
        if (!cancelled) {
          setTeams(t);
          if (t.length) {
            setTeamQuery((prev) => prev || t[0].team_id);
          }
        }
      } catch (e) {
        if (!cancelled) setError(formatFranchiseApiError(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function onStart(e) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await startFranchise({
        team_query: teamQuery,
        head_coach_name: coachName || "Head Coach",
        coach_archetype: archetype,
      });
      setFranchiseSessionId(res.session_id);
      navigate("/", { replace: true });
    } catch (err) {
      setError(formatFranchiseApiError(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h1 className="page-title">Franchise Setup</h1>
      <p className="page-sub">Select your market and bench boss. The league will not move until you advance days.</p>

      <Panel title="Start credentials" subtitle="You enter as GM — one calendar day at a time.">
        <form onSubmit={onStart}>
          <div className="form-grid">
            <label style={{ gridColumn: "span 2" }}>
              Club
              <select value={teamQuery} onChange={(e) => setTeamQuery(e.target.value)} required>
                {teams.map((t) => (
                  <option key={t.team_id} value={t.team_id}>
                    {t.name} ({t.team_id})
                  </option>
                ))}
              </select>
            </label>
            <label style={{ gridColumn: "span 2" }}>
              Head coach name
              <input
                value={coachName}
                onChange={(e) => setCoachName(e.target.value)}
                placeholder="Your hire"
                required
              />
            </label>
            <label style={{ gridColumn: "span 2" }}>
              Coach archetype
              <select value={archetype} onChange={(e) => setArchetype(e.target.value)}>
                <option value="balanced">Balanced</option>
                <option value="development">Development / teacher</option>
                <option value="defense_first">Defense-first structure</option>
                <option value="aggressive">Aggressive attack</option>
                <option value="players_coach">Players&apos; coach / culture</option>
              </select>
            </label>
          </div>
          {error && <div className="alert alert--err">{error}</div>}
          <Button type="submit" disabled={loading || teams.length === 0}>
            {loading ? "Starting…" : "Begin franchise"}
          </Button>
        </form>
      </Panel>
    </div>
  );
}
