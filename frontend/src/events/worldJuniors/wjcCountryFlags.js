import { flagApiUrl, resolveCountryCode, wjcCodeToIso } from "../../utils/countryFlags";

export { resolveCountryCode, wjcCodeToIso };

export function wjcFlagUrl(countryOrCode, size = 48, style = "flat") {
  return flagApiUrl(countryOrCode, size, style);
}
