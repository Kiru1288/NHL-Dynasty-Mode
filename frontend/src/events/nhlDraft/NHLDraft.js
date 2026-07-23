import React from "react";
import { DraftEventMenu } from "../offseasonEventMenus";

/** Permanent NHL Draft event — delegates to draft floor until full NHLDraft UI ships. */
export default function NHLDraft(props) {
  return <DraftEventMenu {...props} />;
}
