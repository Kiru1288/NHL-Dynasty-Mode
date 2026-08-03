import React from "react";

export function Panel({
  title,
  subtitle,
  children,
  className = "",
  register = "ops",
  flat = false,
  code,
  status,
}) {
  const cls = [
    "ui-panel",
    flat ? "ui-panel--flat" : "",
    register === "office" ? "ui-panel--office" : "",
    register === "shell" ? "ui-panel--shell" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <section className={cls} data-register={register}>
      {(title || subtitle || code || status) && (
        <header className="ui-panel__head">
          {(code || status) && (
            <div className="ui-panel__scorebug">
              {code && <span className="fcn-scorebug__code">{code}</span>}
              {status && <span className="fcn-stamp">{status}</span>}
            </div>
          )}
          {title && <h2 className="ui-panel__title">{title}</h2>}
          {subtitle && <p className="ui-panel__sub">{subtitle}</p>}
        </header>
      )}
      <div className="ui-panel__body">{children}</div>
    </section>
  );
}
