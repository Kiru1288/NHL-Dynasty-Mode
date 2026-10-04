import React from "react";
import { EvenStrengthLines } from "./editLines";

/** AHL affiliate lines: the exact NHL Edit Lines screen in AHL mode (no chemistry layer).
 *  AHL stats live in Stats Central (NHL / AHL switch). */
export default function AhlCenter(props) {
  return <EvenStrengthLines {...(props || {})} ahl />;
}
