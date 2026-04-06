import React from "react";

export function Panel({ title, subtitle, children, className = "" }) {
  return (
    <section className={`ui-panel ${className}`.trim()}>
      {(title || subtitle) && (
        <header className="ui-panel__head">
          {title && <h2 className="ui-panel__title">{title}</h2>}
          {subtitle && <p className="ui-panel__sub">{subtitle}</p>}
        </header>
      )}
      <div className="ui-panel__body">{children}</div>
    </section>
  );
}
