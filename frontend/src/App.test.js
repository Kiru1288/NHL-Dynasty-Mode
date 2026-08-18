import { fireEvent, render } from "@testing-library/react";
import PlayerHeadshot from "./components/PlayerHeadshot";
import {
  getNhlHeadshotUrl,
  resolvePlayerHeadshot,
} from "./utils/playerHeadshots";

test("resolves NHL photography ahead of generated metadata", () => {
  const result = resolvePlayerHeadshot({
    player_id: "NHL_8478402",
    nhl_player_id: 8478402,
    nhl_headshot_url: "https://assets.nhle.com/mugs/nhl/20252026/EDM/8478402.png",
  });

  expect(result.source).toBe("nhl");
  expect(result.src).toContain("assets.nhle.com");
  expect(result.player.headshot_id).toBeGreaterThanOrEqual(1);
});

test("generated and old-save players do not accept arbitrary image hosts", () => {
  const oldSave = resolvePlayerHeadshot({
    player_id: "legacy-player",
    name: "Legacy Player",
    portrait_url: "https://example.invalid/player.png",
  });

  expect(getNhlHeadshotUrl(oldSave.player)).toBe("");
  expect(oldSave.source).toBe("generated");
  expect(oldSave.player.headshot_id).toBeGreaterThanOrEqual(1);
});

test("a broken NHL image falls back once to the generated portrait", () => {
  const { container } = render(
    <PlayerHeadshot
      player={{
        nhl_player_id: 1,
        nhl_headshot_url: "https://assets.nhle.com/missing.png",
        name: "Fallback Player",
      }}
    />
  );

  const image = container.querySelector(".ph-nhl-image");
  expect(image).not.toBeNull();
  fireEvent.error(image);
  expect(container.querySelector(".ph-nhl-image")).toBeNull();
  expect(container.querySelector(".ph-face")).not.toBeNull();
});
