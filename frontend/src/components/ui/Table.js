import React from "react";

export function Table({ columns, rows, onRowClick, selectedKey, rowKey = (r, i) => i }) {
  return (
    <div className="ui-table-wrap">
      <table className="ui-table">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="ui-table__empty">
                No rows
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
