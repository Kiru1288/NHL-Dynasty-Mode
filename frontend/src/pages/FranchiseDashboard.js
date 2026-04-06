import React, { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Panel } from "../components/ui/Panel";
import { Button } from "../components/ui/Button";
import { Table } from "../components/ui/Table";
import { Loader } from "../components/ui/Loader";
import {
  advanceDay,
  getFranchiseState,
  submitDecision,
} from "../services/franchiseService";
import {
  api,
  clearFranchiseSession,
  getFranchiseSessionId,
  isNetworkError,
} from "../services/api";

function goSetup() {
  clearFranchiseSession();
}

export function FranchiseDashboard() {
  const navigate = useNavigate();
  const [state, setState] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [advancing, setAdvancing] = useState(false);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const s = await getFranchiseState();
      setState(s);
    } catch (e) {
      if (e.response?.status === 404 || e.response?.status === 400) {
        clearFranchiseSession();
        navigate("/setup", { replace: true });
        return;
      }
      setError(isNetworkError(e) ? `API offline (${api.defaults.baseURL})` : e.message);
    } finally {
      setLoading(false);
    }
  }, [navigate]);

  useEffect(() => {
    if (!getFranchiseSessionId()) {
      navigate("/setup", { replace: true });
      return;
    }
    refresh();
  }, [navigate, refresh]);

  async function onAdvance() {
    if (!state?.flags?.can_advance) return;
    setAdvancing(true);
    setError(null);
    try {
      const res = await advanceDay();
      setState(res.state);
    } catch (e) {
      const d = e.response?.data?.detail;
      setError(typeof d === "string" ? d : JSON.stringify(d || e.message));
    } finally {
      setAdvancing(false);
    }
  }

  async function onResolve(decisionId, choiceId) {
    setError(null);
    try {
      const res = await submitDecision(decisionId, choiceId);
      setState(res.state);
    } catch (e) {
      const d = e.response?.data?.detail;
      setError(typeof d === "string" ? d : JSON.stringify(d || e.message));
    }
  }

  if (loading && !state) {
    return <Loader label="Loading franchise state…" />;
  }

  const rec = state?.team?.record;
  const pending = state?.pending_decisions || [];

  return (
    <div className="franchise-dashboard">
      <div className="franchise-topbar">
        <div>
          <div className="franchise-topbar__team">{state?.team?.name || "—"}</div>
          <div className="franchise-topbar__sub">
            HC {state?.team?.coach} · {state?.team?.coach_archetype?.replace(/_/g, " ")}
          </div>
        </div>
        <div className="franchise-topbar__meta">
          <div>
            <span className="franchise-topbar__label">Record</span>
            <span className="franchise-topbar__value">
              {rec ? `${rec.w}-${rec.l}-${rec.otl}` : "0-0-0"} ({rec?.pts ?? 0} pts)
            </span>
          </div>
          <div>
            <span className="franchise-topbar__label">Cap signal</span>
            <span className="franchise-topbar__value">{state?.team?.cap_pressure}</span>
          </div>
          <div>
            <span className="franchise-topbar__label">Calendar</span>
            <span className="franchise-topbar__value">{state?.calendar_summary}</span>
          </div>
        </div>
        <div className="franchise-topbar__actions">
          <Link
            to="/setup"
            onClick={goSetup}
            className="ui-btn ui-btn--secondary"
            style={{ display: "inline-block" }}
          >
            New franchise
          </Link>
        </div>
      </div>

      {error && <div className="alert alert--err">{error}</div>}

      <div className="franchise-grid">
        <div className="franchise-col franchise-col--main">
          <Panel title="Control room feed" subtitle="Narrative + league events for your session.">
            <pre className="log-view" style={{ maxHeight: "min(38vh, 360px)" }}>
              {(state?.timeline || []).join("\n") || "No entries yet."}
            </pre>
            <div style={{ marginTop: "1rem", display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
              <Button onClick={onAdvance} disabled={!state?.flags?.can_advance || advancing}>
                {advancing ? "Advancing…" : "Advance day"}
              </Button>
              <Button variant="secondary" onClick={refresh} disabled={advancing}>
                Refresh state
              </Button>
            </div>
            {pending.length > 0 && (
              <p className="page-sub" style={{ color: "var(--accent)", marginBottom: 0 }}>
                Resolve decisions below before the calendar moves.
              </p>
            )}
            {state?.phase === "complete" && (
              <p className="page-sub">Season complete. Start a new franchise from Setup.</p>
            )}
          </Panel>

          {pending.length > 0 && (
            <Panel title="Pending decisions" subtitle="Choose an option — consequences hit morale and room dynamics.">
              {pending.map((d) => (
                <div key={d.id} className="decision-card">
                  <h3 className="decision-card__title">{d.title}</h3>
                  <p className="decision-card__desc">{d.description}</p>
                  <div className="decision-card__opts">
                    {(d.options || []).map((o) => (
                      <Button key={o.id} variant="secondary" onClick={() => onResolve(d.id, o.id)}>
                        {o.label}
                      </Button>
                    ))}
                  </div>
                </div>
              ))}
            </Panel>
          )}
        </div>

        <div className="franchise-col franchise-col--side">
          <Panel title="Notifications">
            <ul className="note-list">
              {(state?.notifications || []).map((n, i) => (
                <li key={i}>{n}</li>
              ))}
            </ul>
          </Panel>
          <Panel title="Roster (club)">
            <Table
              columns={[
                { key: "name", label: "Player" },
                { key: "position", label: "Pos" },
                { key: "ovr", label: "OVR" },
                { key: "morale", label: "Morale" },
              ]}
              rows={state?.roster || []}
            />
          </Panel>
          <Panel title="Standings">
            <Table
              columns={[
                { key: "name", label: "Team" },
                { key: "gp", label: "GP" },
                { key: "w", label: "W" },
                { key: "l", label: "L" },
                { key: "pts", label: "PTS" },
              ]}
              rows={(state?.standings || []).slice(0, 16)}
            />
          </Panel>
        </div>
      </div>
    </div>
  );
}
