/** Layered crest editor: pick a layer, edit it, reorder / duplicate / hide / remove. */
import React, { useState } from "react";
import { CREST_PRESETS, CrestSVG, crestPreset } from "./crest";

const MAX_LAYERS = 16;

function ColorField({ label, value, onChange, allowNone = false }) {
  const none = value === "none";
  return (
    <label className="lh-field">
      <span>{label}</span>
      <span className="lh-color">
        <input type="color" value={none ? "#000000" : value || "#000000"} disabled={none} onChange={(e) => onChange(e.target.value.toUpperCase())} />
        {allowNone ? (
          <button type="button" className={`lh-chip ${none ? "is-on" : ""}`} onClick={() => onChange(none ? "#FFFFFF" : "none")}>
            {none ? "none" : "fill"}
          </button>
        ) : (
          <code>{value}</code>
        )}
      </span>
    </label>
  );
}

function Slider({ label, value, min, max, step = 1, onChange }) {
  return (
    <label className="lh-field">
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={Number(value)} onChange={(e) => onChange(Number(e.target.value))} />
      <code>{Number(value).toFixed(step < 1 ? 2 : 0)}</code>
    </label>
  );
}

export default function CrestEditor({ crest, colors, options, initials, onChange }) {
  const layers = crest?.layers || [];
  const [sel, setSel] = useState(layers.length ? layers[layers.length - 1].id : null);
  const active = layers.find((l) => l.id === sel) || null;

  const setLayers = (next) => onChange({ ...crest, layers: next });
  const update = (patch) => setLayers(layers.map((l) => (l.id === sel ? { ...l, ...patch } : l)));
  const newId = () => `l${Date.now().toString(36).slice(-5)}${Math.floor(Math.random() * 99)}`;
  const add = (kind) => {
    if (layers.length >= MAX_LAYERS) return;
    const id = newId();
    const base = { id, kind, fill: colors.secondary, stroke: "none", stroke_width: 0, x: 0, y: 0, scale: kind === "shape" ? 0.7 : 0.8, rotation: 0, visible: true };
    const extra = kind === "shape" ? { shape: "circle" } : kind === "symbol" ? { symbol: "star" } : kind === "text" ? { text: initials || "A", font: "block", fill: colors.neutral } : {};
    setLayers([...layers, { ...base, ...extra }]);
    setSel(id);
  };
  const move = (dir) => {
    const i = layers.findIndex((l) => l.id === sel);
    const j = i + dir;
    if (i < 0 || j < 0 || j >= layers.length) return;
    const next = [...layers];
    [next[i], next[j]] = [next[j], next[i]];
    setLayers(next);
  };
  const duplicate = () => {
    if (!active || layers.length >= MAX_LAYERS) return;
    const id = newId();
    const i = layers.findIndex((l) => l.id === sel);
    const next = [...layers];
    next.splice(i + 1, 0, { ...active, id, x: Number(active.x || 0) + 6, y: Number(active.y || 0) + 6 });
    setLayers(next);
    setSel(id);
  };
  const remove = () => {
    if (!active || layers.length <= 1) return;
    const next = layers.filter((l) => l.id !== sel);
    setLayers(next);
    setSel(next[next.length - 1]?.id || null);
  };

  return (
    <div className="lh-crest-editor">
      <div className="lh-crest-stage">
        <CrestSVG crest={crest} size={230} />
        <div className="lh-presets">
          {CREST_PRESETS.map((p) => (
            <button key={p.id} type="button" className="lh-chip" onClick={() => {
              const next = crestPreset(p.id, colors, initials);
              if (next) {
                onChange(next);
                setSel(next.layers[next.layers.length - 1].id);
              }
            }}>
              {p.label}
            </button>
          ))}
        </div>
      </div>
      <div className="lh-layer-list" role="listbox" aria-label="Crest layers">
        <div className="lh-layer-head">
          <span>Layers {layers.length}/{MAX_LAYERS}</span>
          <span className="lh-row-actions">
            {["shape", "symbol", "text", "stripe"].map((k) => (
              <button key={k} type="button" className="lh-chip" disabled={layers.length >= MAX_LAYERS} onClick={() => add(k)}>+ {k}</button>
            ))}
          </span>
        </div>
        {[...layers].reverse().map((l) => (
          <div key={l.id} role="option" aria-selected={l.id === sel} className={`lh-layer ${l.id === sel ? "is-sel" : ""} ${l.visible === false ? "is-hidden" : ""}`} onClick={() => setSel(l.id)}>
            <span className="lh-swatch" style={{ background: l.fill === "none" ? "transparent" : l.fill, borderColor: l.stroke === "none" ? "transparent" : l.stroke }} />
            <span className="lh-layer-name">{l.kind === "text" ? `“${l.text}”` : l.shape || l.symbol || l.kind}</span>
            <button type="button" className="lh-icon-btn" title={l.visible === false ? "Show" : "Hide"} onClick={(e) => {
              e.stopPropagation();
              setLayers(layers.map((x) => (x.id === l.id ? { ...x, visible: x.visible === false } : x)));
            }}>{l.visible === false ? "show" : "hide"}</button>
          </div>
        ))}
        <div className="lh-row-actions">
          <button type="button" className="lh-chip" onClick={() => move(1)} disabled={!active}>Forward</button>
          <button type="button" className="lh-chip" onClick={() => move(-1)} disabled={!active}>Back</button>
          <button type="button" className="lh-chip" onClick={duplicate} disabled={!active || layers.length >= MAX_LAYERS}>Duplicate</button>
          <button type="button" className="lh-chip lh-chip--warn" onClick={remove} disabled={!active || layers.length <= 1}>Remove</button>
        </div>
      </div>
      {active ? (
        <div className="lh-layer-props">
          {active.kind === "shape" && (
            <label className="lh-field"><span>Shape</span>
              <select value={active.shape} onChange={(e) => update({ shape: e.target.value })}>
                {(options?.crest_shapes || []).map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </label>
          )}
          {active.kind === "symbol" && (
            <label className="lh-field"><span>Symbol</span>
              <select value={active.symbol} onChange={(e) => update({ symbol: e.target.value })}>
                {(options?.crest_symbols || []).filter((s) => s !== "none").map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </label>
          )}
          {active.kind === "text" && (
            <>
              <label className="lh-field"><span>Text</span>
                <input maxLength={6} value={active.text || ""} onChange={(e) => update({ text: e.target.value.replace(/[^A-Za-z0-9 .&'-]/g, "") })} />
              </label>
              <label className="lh-field"><span>Font</span>
                <select value={active.font} onChange={(e) => update({ font: e.target.value })}>
                  {(options?.number_fonts || []).map((s) => <option key={s} value={s}>{s}</option>)}
                </select>
              </label>
            </>
          )}
          {active.id === layers[0]?.id && active.kind === "shape" && (
            <label className="lh-field"><span>Silhouette</span>
              <select value={crest.shape} onChange={(e) => onChange({ ...crest, shape: e.target.value })}>
                {(options?.crest_shapes || []).map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </label>
          )}
          <ColorField label="Fill" value={active.fill} allowNone onChange={(v) => update({ fill: v })} />
          <ColorField label="Outline" value={active.stroke} allowNone onChange={(v) => update({ stroke: v })} />
          <Slider label="Outline width" value={active.stroke_width} min={0} max={14} onChange={(v) => update({ stroke_width: v })} />
          <Slider label="Scale" value={active.scale} min={0.1} max={2.5} step={0.05} onChange={(v) => update({ scale: v })} />
          <Slider label="X" value={active.x} min={-60} max={60} onChange={(v) => update({ x: v })} />
          <Slider label="Y" value={active.y} min={-60} max={60} onChange={(v) => update({ y: v })} />
          <Slider label="Rotate" value={active.rotation} min={-180} max={180} onChange={(v) => update({ rotation: v })} />
          <div className="lh-row-actions">
            <button type="button" className="lh-chip" onClick={() => update({ x: 0, y: 0 })}>Centre</button>
            <button type="button" className="lh-chip" onClick={() => update({ rotation: 0, scale: 1 })}>Reset size</button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
