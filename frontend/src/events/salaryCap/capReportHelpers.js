import { firstDefined } from "../shared/eventHelpers";

export function normalizeMoneyToMillions(value) {
  if (value === undefined || value === null || value === "") return null;
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  if (Math.abs(n) > 250) return n / 1_000_000;
  return n;
}

export function formatCapMoney(value) {
  const m = normalizeMoneyToMillions(value);
  if (m == null) return "—";
  const sign = m < 0 ? "-" : "";
  return `${sign}$${Math.abs(m).toFixed(1)}M`;
}

export function formatCapChange(value) {
  const m = normalizeMoneyToMillions(value);
  if (m == null) return "—";
  if (Math.abs(m) < 0.0001) return "$0.0M";
  const sign = m > 0 ? "+" : m < 0 ? "-" : "";
  return `${sign}$${Math.abs(m).toFixed(1)}M`;
}

export function formatCapPercent(value) {
  if (value === undefined || value === null || value === "") return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) < 0.05) return "0.0%";
  const sign = n > 0 ? "+" : "";
  return `${sign}${n.toFixed(1)}%`;
}

export function pickCapReport(raw = {}, franchiseState = {}) {
  const salaryCap = raw?.salary_cap || raw || {};
  const report =
    salaryCap.cap_report ||
    salaryCap.salary_cap_report ||
    franchiseState?.cap_report ||
    franchiseState?.salary_cap?.cap_report ||
    null;

  const previousCap = normalizeMoneyToMillions(
    firstDefined(
      report?.previous_cap,
      salaryCap.previous_cap,
      salaryCap.last_season_cap,
      salaryCap.previousSalaryCap,
      franchiseState?.previousSalaryCap
    )
  );

  const currentCap = normalizeMoneyToMillions(
    firstDefined(
      report?.current_cap,
      salaryCap.current_cap,
      salaryCap.new_season_cap,
      salaryCap.salary_cap,
      franchiseState?.team?.salary_cap,
      franchiseState?.salary_cap
    )
  );

  let capChange = normalizeMoneyToMillions(
    firstDefined(report?.cap_change, salaryCap.cap_change, salaryCap.change, salaryCap.capGrowth)
  );
  if (capChange == null && previousCap != null && currentCap != null) {
    capChange = currentCap - previousCap;
  }

  let capChangePercent = firstDefined(report?.cap_change_percent, salaryCap.cap_change_percent);
  if ((capChangePercent === undefined || capChangePercent === null) && previousCap > 0 && capChange != null) {
    capChangePercent = (capChange / previousCap) * 100;
  }

  const userRaw =
    report?.user_team ||
    salaryCap.user_team ||
    salaryCap.user_team_cap ||
    franchiseState?.team ||
    {};

  const payroll = normalizeMoneyToMillions(
    firstDefined(userRaw.payroll, userRaw.cap_hit, userRaw.totalCapHit, franchiseState?.team?.cap_hit)
  );
  let capSpace = normalizeMoneyToMillions(
    firstDefined(userRaw.cap_space, userRaw.capSpace, userRaw.usable_cap_space, franchiseState?.team?.cap_space)
  );
  if (capSpace == null && currentCap != null && payroll != null) {
    capSpace = currentCap - payroll;
  }

  const notes = Array.isArray(report?.notes)
    ? report.notes
    : Array.isArray(salaryCap.notes)
      ? salaryCap.notes
      : [];

  return {
    season: firstDefined(report?.season, salaryCap.season) || "",
    previousCap,
    currentCap,
    capChange,
    capChangePercent,
    movementType: firstDefined(report?.movement_type, salaryCap.movement_type) || "",
    movementLabel: firstDefined(report?.movement_label, salaryCap.movement_label) || "",
    movementReason: firstDefined(report?.movement_reason, salaryCap.movement_reason) || "",
    notes,
    userTeam: {
      teamId: firstDefined(userRaw.team_id, userRaw.teamId, franchiseState?.user_team_id) || "",
      teamName: firstDefined(userRaw.team_name, userRaw.teamName, franchiseState?.team?.name) || "Your Team",
      payroll,
      capSpace,
      deadCap: normalizeMoneyToMillions(firstDefined(userRaw.dead_cap, userRaw.deadCap)),
      retainedSalary: normalizeMoneyToMillions(firstDefined(userRaw.retained_salary, userRaw.retainedSalary)),
      bonusOverages: normalizeMoneyToMillions(firstDefined(userRaw.bonus_overages, userRaw.bonusOverage)),
      projectedSpace: normalizeMoneyToMillions(firstDefined(userRaw.projected_space, userRaw.projectedDeadlineSpace)),
      capStatus: firstDefined(userRaw.cap_status, userRaw.capStatus) || "",
    },
  };
}

export function seasonCapLabel(franchiseState, reportSeason) {
  if (reportSeason) {
    const raw = String(reportSeason);
    if (raw.includes("–")) return raw;
    const parts = raw.split("-");
    if (parts.length >= 2) {
      const start = parts[0];
      const endPart = parts[1];
      const end =
        endPart.length === 2 ? `${String(start).slice(0, Math.max(0, String(start).length - 2))}${endPart}` : endPart;
      return `${start}–${String(end).slice(-2)}`;
    }
    return raw;
  }
  const y = franchiseState?.season_year || franchiseState?.seasonYear;
  return y ? `${Number(y) + 1}–${Number(y) + 2}` : "";
}

export function changeTone(capChange) {
  if (capChange == null || Math.abs(capChange) < 0.0001) return "flat";
  return capChange > 0 ? "up" : "down";
}

export function statusTone(status) {
  const s = String(status || "").toLowerCase();
  if (s.includes("over")) return "danger";
  if (s.includes("tight") || s.includes("deadline")) return "warn";
  if (s.includes("flexible")) return "gold";
  return "healthy";
}
