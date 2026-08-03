import React from "react";

export function Table({
  columns,
  rows,
  onRowClick,
  selectedKey,
  rowKey = (r, i) => i,
  emptyLabel = "No rows loaded",
  emptyKicker = "OPS · STANDBY",
  loading = false,
  loadingRows = 6,
}) {
  return (
    <div className="ui-table-wrap">
      <table className="ui-table">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} scope="col">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {loading ? (
            // Placeholder rows match real row geometry so nothing shifts on arrival.
            Array.from({ length: loadingRows }).map((_, i) => (
              <tr key={`skeleton-${i}`} aria-hidden="true">
                <td colSpan={columns.length} style={{ padding: 0 }}>
                  <div className="fcn-skeleton-row" />
                </td>
              </tr>
            ))
          ) : rows.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="ui-table__empty">
                <div className="ops-state" style={{ maxWidth: "100%", border: 0, padding: "8px 0", background: "transparent" }}>
                  <span className="ops-state__kicker">{emptyKicker}</span>
                  <p className="ops-state__title" style={{ fontSize: "0.8rem" }}>
                    {emptyLabel}
                  </p>
                </div>
              </td>
            </tr>
          ) : (
            rows.map((row, i) => {
              const k = rowKey(row, i);
              const sel = selectedKey != null && k === selectedKey;
              return (
                <tr
                  key={k}
                  className={sel ? "ui-table__row ui-table__row--sel" : "ui-table__row"}
                  onClick={() => onRowClick && onRowClick(row, k)}
                  onKeyDown={
                    onRowClick
                      ? (e) => {
                          if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault();
                            onRowClick(row, k);
                          }
                        }
                      : undefined
                  }
                  tabIndex={onRowClick ? 0 : undefined}
                  style={{ cursor: onRowClick ? "pointer" : "default" }}
                >
                  {columns.map((c) => (
                    <td key={c.key}>{c.render ? c.render(row) : row[c.key]}</td>
                  ))}
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}
