import React, { useCallback, useEffect, useState } from "react";
import { getCapGrowth, signTvDeal } from "../../services/franchiseService";

// F11: why the salary cap is moving next season (factor breakdown), how past caps
// were set, and the national TV deal negotiation that opens every 10 seasons.
export default function CapGrowthPanel() {
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const load = useCallback(async () => {
    try {
      setData(await getCapGrowth());
    } catch (e) {
      setErr(String(e?.message || "Cap model unavailable"));
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const sign = async (option) => {
    setBusy(true);
    setErr("");
    try {
      await signTvDeal(option);
      await load();
    } catch (e) {
      setErr(String(e?.response?.data?.detail || e?.message || "Could not sign the deal"));
    } finally {
      setBusy(false);
    }
  };

  if (!data) return err ? <p className="cl-empty">{err}</p> : null;
  const model = data.model || {};
  const factors = Array.isArray(model.factors) ? model.factors : [];
  const tv = data.tv_deal || {};
  const history = Array.isArray(data.history) ? data.history.slice().reverse() : [];
  const pct = (v) => `${Number(v) > 0 ? "+" : ""}${Number(v || 0).toFixed(2)}%`;

  return (
    <section className="cgp">
      <header className="cgp-head">
        <h3>Why the cap is moving</h3>
        <span className={`cgp-total ${Number(model.growth_pct) < 0 ? "is-down" : "is-up"}`}>
          {pct(model.growth_pct)} model growth
        </span>
      </header>
      {model.headline ? <p className="cgp-headline">{model.headline}</p> : null}
      {data.projection_source && data.projection_source !== "model" ? (
        <p className="cgp-note">Next season's cap is already announced; the model drives the seasons after it.</p>
      ) : null}
      <ul className="cgp-factors">
        {factors.slice(0, 14).map((f) => (
          <li key={f.key} className={`is-${f.direction}`}>
            <b>{pct(f.impact_pct)}</b>
            <span>{f.label}</span>
            <em>{f.detail}</em>
          </li>
        ))}
      </ul>

      <div className="cgp-tv">
        <h4>
          National TV deal: {tv.current?.label || "—"}
          {tv.deal_year ? ` · year ${tv.deal_year} of ${tv.current?.term || 10}` : ""}
        </h4>
        {tv.open ? (
          <div className="cgp-tv-open">
            <p>The national deal is up. Your vote decides the league's media deal for the next decade.</p>
            {(tv.options || []).map((o) => (
              <button key={o.key} type="button" disabled={busy} onClick={() => sign(o.key)}>
                <strong>{o.label}</strong>
                <span>{o.blurb}</span>
              </button>
            ))}
          </div>
        ) : null}
      </div>

      {history.length ? (
        <div className="cgp-history">
          <h4>Past cap decisions</h4>
          {history.map((h) => (
            <details key={h.season}>
              <summary>
                {h.season}: ${Number(h.new_cap || 0).toFixed(1)}M ({pct(h.change_pct)}) — {h.headline}
              </summary>
              <ul className="cgp-factors">
                {(h.factors || []).slice(0, 8).map((f) => (
                  <li key={f.key} className={`is-${f.direction}`}>
                    <b>{pct(f.impact_pct)}</b>
                    <span>{f.label}</span>
                    <em>{f.detail}</em>
                  </li>
                ))}
              </ul>
            </details>
          ))}
        </div>
      ) : null}
      {err ? <p className="cl-banner is-error">{err}</p> : null}
    </section>
  );
}
