import React, { useMemo } from "react";
import { pickFranchiseData } from "../shared/eventHelpers";
import {
  changeTone,
  formatCapChange,
  formatCapMoney,
  formatCapPercent,
  pickCapReport,
  seasonCapLabel,
  statusTone,
} from "./capReportHelpers";
import "./CapReportBoard.css";

function firstDefined(...values) {
  return values.find(
    (value) => value !== undefined && value !== null && value !== ""
  );
}

function seasonLabel(franchiseState) {
  const year =
    franchiseState?.season_year ||
    franchiseState?.seasonYear;

  return year ? `${year}–${Number(year) + 1}` : "";
}

function toMillions(value) {
  if (value === undefined || value === null || value === "") {
    return null;
  }

  if (typeof value === "number") {
    if (!Number.isFinite(value)) return null;

    return Math.abs(value) >= 100000
      ? value / 1000000
      : value;
  }

  const text = String(value).trim();

  if (!text) return null;

  const parsed = Number(
    text.replace(/[$,%\s,]/g, "")
  );

  if (!Number.isFinite(parsed)) return null;

  if (/b/i.test(text)) return parsed * 1000;
  if (/m/i.test(text)) return parsed;

  return Math.abs(parsed) >= 100000
    ? parsed / 1000000
    : parsed;
}

function money(value) {
  const normalized = toMillions(value);

  return normalized === null
    ? "—"
    : formatCapMoney(normalized);
}

function clamp(value, minimum, maximum) {
  return Math.min(
    Math.max(value, minimum),
    maximum
  );
}

function normalizeReason(value) {
  const text = String(value || "").trim();

  if (!text) {
    return "League revenues produced a routine salary-cap adjustment.";
  }

  return text.replace(/\s+/g, " ");
}

function ReportMetric({
  label,
  value,
  tone = "",
  detail = "",
}) {
  return (
    <article
      className={[
        "cap-report-metric",
        tone ? `is-${tone}` : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <span>{label}</span>
      <strong>{value}</strong>
      {detail ? <small>{detail}</small> : null}
    </article>
  );
}

export default function CapReportBoard({
  franchiseState = {},
  eventData = {},
  onContinue,
  onBack,
}) {
  const raw =
    pickFranchiseData(
      franchiseState,
      eventData,
      [
        "salary_cap",
        "cap",
        "offseason.salary_cap",
      ]
    ) || {};

  const report = useMemo(
    () =>
      pickCapReport(
        { salary_cap: raw },
        franchiseState
      ),
    [franchiseState, raw]
  );

  const capPayload =
    raw?.cap_report ||
    raw?.capReport ||
    raw;

  const currentCap =
    toMillions(report.currentCap) || 0;

  const payroll =
    toMillions(report.userTeam.payroll) || 0;

  const capSpace =
    toMillions(report.userTeam.capSpace);

  const projectedSpace =
    toMillions(report.userTeam.projectedSpace);

  const deadCap =
    toMillions(report.userTeam.deadCap);

  const retainedSalary =
    toMillions(report.userTeam.retainedSalary);

  const bonusOverages =
    toMillions(report.userTeam.bonusOverages);

  const utilization =
    currentCap > 0
      ? clamp((payroll / currentCap) * 100, 0, 100)
      : 0;

  const movementTone =
    changeTone(report.capChange);

  const teamTone =
    statusTone(report.userTeam.capStatus);

  const capSeason =
    seasonCapLabel(
      franchiseState,
      report.season
    );

  const currentSeason =
    seasonLabel(franchiseState);

  const movementReason = normalizeReason(
    firstDefined(
      capPayload?.movement_reason,
      capPayload?.movementReason,
      raw?.movement_reason,
      raw?.movementReason,
      report?.movementReason,
      report?.notes?.[0]
    )
  );

  const capSpaceDetail =
    capSpace !== null && capSpace < 0
      ? "Roster exceeds the league ceiling"
      : capSpace !== null && capSpace <= 1
        ? "Minimal operating flexibility"
        : capSpace !== null && capSpace <= 4
          ? "Limited in-season flexibility"
          : capSpace !== null && capSpace >= 10
            ? "Strong offseason flexibility"
            : "Usable operating flexibility";

  const metrics = [
    {
      label: "Payroll",
      value: money(report.userTeam.payroll),
      detail: "Current committed cap hit",
    },
    {
      label: "Dead Cap",
      value: money(deadCap),
      detail: "Non-roster obligations",
    },
    {
      label: "Retained",
      value: money(retainedSalary),
      detail: "Salary retained in trades",
    },
    {
      label: "Bonus Carry",
      value: money(bonusOverages),
      detail: "Previous bonus overage",
    },
    {
      label: "Projected Space",
      value: money(projectedSpace),
      detail: "Estimated deadline position",
      tone:
        projectedSpace !== null &&
        projectedSpace < 0
          ? "danger"
          : projectedSpace !== null &&
              projectedSpace >= 8
            ? "positive"
            : "",
    },
  ];

  return (
    <section className="cap-report-page">
      <header className="cap-report-topbar">
        <button
          type="button"
          className="cap-report-back-button"
          onClick={onBack}
        >
          <span aria-hidden="true">←</span>
          Back
        </button>

        <div className="cap-report-title">
          <span>Offseason</span>
          <h1>Cap Report</h1>
        </div>

        <div className="cap-report-season">
          <span>Season</span>
          <strong>
            {currentSeason || "Current"}
          </strong>
        </div>
      </header>

      <main className="cap-report-content">
        <section className="cap-report-primary-panel">
          <div className="cap-report-league-summary">
            <div className="cap-report-section-heading">
              <div>
                <span>League Salary Cap</span>
                <h2>
                  {capSeason ||
                    "Updated League Ceiling"}
                </h2>
              </div>

              <span
                className={[
                  "cap-report-movement-badge",
                  `is-${movementTone}`,
                ].join(" ")}
              >
                {report.movementLabel ||
                  "Cap Updated"}
              </span>
            </div>

            <div className="cap-report-cap-display">
              <strong>
                {money(report.currentCap)}
              </strong>

              <span
                className={[
                  "cap-report-change-badge",
                  `is-${movementTone}`,
                ].join(" ")}
              >
                {formatCapChange(
                  report.capChange
                )}
              </span>
            </div>

            <p className="cap-report-reason">
              {movementReason}
            </p>

            <div className="cap-report-change-strip">
              <div>
                <span>Previous</span>
                <strong>
                  {money(report.previousCap)}
                </strong>
              </div>

              <div>
                <span>Change</span>
                <strong>
                  {formatCapChange(
                    report.capChange
                  )}
                </strong>
              </div>

              <div>
                <span>Growth</span>
                <strong>
                  {formatCapPercent(
                    report.capChangePercent
                  )}
                </strong>
              </div>
            </div>
          </div>

          <aside className="cap-report-club-summary">
            <div className="cap-report-club-header">
              <div>
                <span>Your Club</span>
                <h2>
                  {report.userTeam.teamName ||
                    "Franchise"}
                </h2>
              </div>

              {report.userTeam.capStatus ? (
                <span
                  className={[
                    "cap-report-status",
                    `is-${teamTone}`,
                  ].join(" ")}
                >
                  {report.userTeam.capStatus}
                </span>
              ) : null}
            </div>

            <div className="cap-report-space-block">
              <span>Available Cap Space</span>

              <strong
                className={
                  capSpace !== null &&
                  capSpace < 0
                    ? "is-negative"
                    : ""
                }
              >
                {money(capSpace)}
              </strong>

              <p>{capSpaceDetail}</p>
            </div>

            <div className="cap-report-utilization">
              <div className="cap-report-utilization-header">
                <span>Payroll Used</span>
                <strong>
                  {Math.round(utilization)}%
                </strong>
              </div>

              <div
                className="cap-report-utilization-track"
                aria-label={`${Math.round(
                  utilization
                )}% of the salary cap used`}
              >
                <span
                  style={{
                    width: `${utilization}%`,
                  }}
                />
              </div>

              <div className="cap-report-utilization-labels">
                <span>
                  {money(payroll)} committed
                </span>

                <span>
                  {money(currentCap)} ceiling
                </span>
              </div>
            </div>

            <div className="cap-report-club-footer">
              <div>
                <span>Projected</span>
                <strong>
                  {money(projectedSpace)}
                </strong>
              </div>

              <div>
                <span>Dead Money</span>
                <strong>
                  {money(deadCap)}
                </strong>
              </div>
            </div>
          </aside>
        </section>

        <section className="cap-report-metrics-panel">
          <header className="cap-report-metrics-header">
            <div>
              <span>Club Position</span>
              <h2>Cap Commitments</h2>
            </div>

            <p>
              Current offseason snapshot
            </p>
          </header>

          <div className="cap-report-metrics-grid">
            {metrics.map((metric) => (
              <ReportMetric
                key={metric.label}
                label={metric.label}
                value={metric.value}
                detail={metric.detail}
                tone={metric.tone}
              />
            ))}
          </div>
        </section>
      </main>

      <footer className="cap-report-actions">
        <div className="cap-report-next-stage">
          <span>Next Stage</span>
          <strong>Player Development</strong>
        </div>

        <button
          type="button"
          className="cap-report-continue-button"
          onClick={onContinue}
        >
          View Development
          <span aria-hidden="true">→</span>
        </button>
      </footer>
    </section>
  );
}