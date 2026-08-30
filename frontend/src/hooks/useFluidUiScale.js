import { useEffect } from "react";
import { applyFluidUiScale, UI_SCALE_CHANGE_EVENT } from "../utils/fluidUiScale";

export function useFluidUiScale() {
  useEffect(() => {
    applyFluidUiScale();
    let frame = 0;
    const onChange = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => applyFluidUiScale());
    };
    window.addEventListener("resize", onChange);
    window.addEventListener(UI_SCALE_CHANGE_EVENT, onChange);
    window.visualViewport?.addEventListener("resize", onChange);
    const probe = document.getElementById("ui-scale-probe");
    const observer = probe && typeof ResizeObserver !== "undefined" ? new ResizeObserver(onChange) : null;
    observer?.observe(probe);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", onChange);
      window.removeEventListener(UI_SCALE_CHANGE_EVENT, onChange);
      window.visualViewport?.removeEventListener("resize", onChange);
      observer?.disconnect();
    };
  }, []);
}
