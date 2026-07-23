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

export function flagApiUrl(countryOrCode, size = 64, style = "flat") {
  const iso2 = resolveCountryCode(countryOrCode);
  if (!iso2) return null;
  return `https://flagsapi.com/${iso2}/${style}/${size}.png`;
}

export function wjcCodeToIso(wjcCode) {
  return resolveCountryCode(wjcCode);
}

/** Convenience alias for WJC nation chips (same flagsapi.com source as Draft Class). */
export function wjcFlagUrl(countryOrCode, size = 48, style = "flat") {
  return flagApiUrl(countryOrCode, size, style);
}