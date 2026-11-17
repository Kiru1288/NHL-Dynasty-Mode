/** Shared flag helpers — same API surface as DraftClass / RosterScreen. */

const COUNTRY_NAME_TO_ISO = {
  CAN: "CA",
  CANADA: "CA",
  Canada: "CA",
  USA: "US",
  US: "US",
  "UNITED STATES": "US",
  "United States": "US",
  SWE: "SE",
  SWEDEN: "SE",
  Sweden: "SE",
  FIN: "FI",
  FINLAND: "FI",
  Finland: "FI",
  RUS: "RU",
  RUSSIA: "RU",
  Russia: "RU",
  CZE: "CZ",
  CZECHIA: "CZ",
  Czechia: "CZ",
  "Czech Republic": "CZ",
  SVK: "SK",
  SLOVAKIA: "SK",
  Slovakia: "SK",
  GER: "DE",
  GERMANY: "DE",
  Germany: "DE",
  SUI: "CH",
  SWITZERLAND: "CH",
  Switzerland: "CH",
  DEN: "DK",
  DENMARK: "DK",
  Denmark: "DK",
  LAT: "LV",
  LATVIA: "LV",
  Latvia: "LV",
  NOR: "NO",
  NORWAY: "NO",
  Norway: "NO",
  AUT: "AT",
  AUSTRIA: "AT",
  Austria: "AT",
  FRA: "FR",
  FRANCE: "FR",
  France: "FR",
  BLR: "BY",
  BELARUS: "BY",
  Belarus: "BY",
  KAZ: "KZ",
  KAZAKHSTAN: "KZ",
  Kazakhstan: "KZ",
  "UNITED STATES OF AMERICA": "US",
  "United States of America": "US",
  UK: "GB",
  GBR: "GB",
  "UNITED KINGDOM": "GB",
  "United Kingdom": "GB",
  ENGLAND: "GB",
  GREATBRITAIN: "GB",
  "GREAT BRITAIN": "GB",
  POL: "PL",
  POLAND: "PL",
  Poland: "PL",
  UKR: "UA",
  UKRAINE: "UA",
  Ukraine: "UA",
  JPN: "JP",
  JAPAN: "JP",
  Japan: "JP",
  KOR: "KR",
  "SOUTH KOREA": "KR",
  "South Korea": "KR",
  CHN: "CN",
  CHINA: "CN",
  China: "CN",
  AUS: "AU",
  AUSTRALIA: "AU",
  Australia: "AU",
  NZL: "NZ",
  "NEW ZEALAND": "NZ",
  "New Zealand": "NZ",
  BRA: "BR",
  BRAZIL: "BR",
  Brazil: "BR",
  ARG: "AR",
  ARGENTINA: "AR",
  Argentina: "AR",
  MEX: "MX",
  MEXICO: "MX",
  Mexico: "MX",
  NGA: "NG",
  NGR: "NG",
  NIGERIA: "NG",
  Nigeria: "NG",
  KEN: "KE",
  KENYA: "KE",
  Kenya: "KE",
  RSA: "ZA",
  "SOUTH AFRICA": "ZA",
  "South Africa": "ZA",
  IND: "IN",
  INDIA: "IN",
  India: "IN",
  PHI: "PH",
  PHL: "PH",
  PHILIPPINES: "PH",
  Philippines: "PH",
  SLO: "SI",
  SVN: "SI",
  SLOVENIA: "SI",
  Slovenia: "SI",
};

const WJC_THREE_TO_ISO = {
  CAN: "CA",
  USA: "US",
  SWE: "SE",
  FIN: "FI",
  CZE: "CZ",
  SVK: "SK",
  GER: "DE",
  SUI: "CH",
  DEN: "DK",
  LAT: "LV",
};

export function resolveCountryCode(raw) {
  const s = String(raw || "").trim();
  if (!s) return null;
  if (/^[A-Za-z]{2}$/.test(s)) return s.toUpperCase();
  const upper = s.toUpperCase();
  if (WJC_THREE_TO_ISO[upper]) return WJC_THREE_TO_ISO[upper];
  if (COUNTRY_NAME_TO_ISO[upper]) return COUNTRY_NAME_TO_ISO[upper];
  if (COUNTRY_NAME_TO_ISO[s]) return COUNTRY_NAME_TO_ISO[s];
  return null;
}

/** flagsapi.com serves these sizes only; anything else 500s and renders as a broken image. */
const FLAG_API_SIZES = [16, 24, 32, 48, 64];

/** Rounds up so a flag is never sourced smaller than the box it renders into. */
export function nearestFlagApiSize(size) {
  const n = Number(size);
  if (!Number.isFinite(n)) return 64;
  return FLAG_API_SIZES.find((candidate) => candidate >= n) || 64;
}

export function flagApiUrl(countryOrCode, size = 64, style = "flat") {
  const iso2 = resolveCountryCode(countryOrCode);
  if (!iso2) return null;
  return `https://flagsapi.com/${iso2}/${style}/${nearestFlagApiSize(size)}.png`;
}

export function wjcCodeToIso(wjcCode) {
  return resolveCountryCode(wjcCode);
}

/** Neutral Olympic flag for Olympic Athletes from Russia (OAR): white field, five rings. */
export const OAR_FLAG_URL =
  "data:image/svg+xml;utf8," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 90 60"><rect width="90" height="60" fill="#fff"/>' +
      '<g fill="none" stroke-width="3.2"><circle cx="27" cy="26" r="9" stroke="#0081c8"/><circle cx="45" cy="26" r="9" stroke="#000"/>' +
      '<circle cx="63" cy="26" r="9" stroke="#ee334e"/><circle cx="36" cy="35" r="9" stroke="#fcb131"/><circle cx="54" cy="35" r="9" stroke="#00a651"/></g></svg>'
  );

/** Convenience alias for WJC / international nation chips. Russian teams play as neutral athletes (OAR). */
export function wjcFlagUrl(countryOrCode, size = 48, style = "flat") {
  const raw = String(countryOrCode || "").trim().toUpperCase();
  if (raw === "OAR" || raw === "RUS" || resolveCountryCode(countryOrCode) === "RU") return OAR_FLAG_URL;
  return flagApiUrl(countryOrCode, size, style);
}