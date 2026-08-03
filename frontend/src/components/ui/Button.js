import React from "react";

export function Button({
  children,
  onClick,
  disabled,
  variant = "primary",
  type = "button",
  className = "",
  register,
  loading = false,
  iconOnly = false,
  "aria-label": ariaLabel,
}) {
  const registerAttr = register ? { "data-register": register } : {};
  const cls = [
    "ui-btn",
    `ui-btn--${variant}`,
    iconOnly ? "ui-btn--icon" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <button
      type={type}
      className={cls}
      onClick={onClick}
      disabled={disabled || loading}
      aria-label={ariaLabel}
      aria-busy={loading || undefined}
      data-loading={loading ? "true" : undefined}
      {...registerAttr}
    >
      {children}
    </button>
  );
}
