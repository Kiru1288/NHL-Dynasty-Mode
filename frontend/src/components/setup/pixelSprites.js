import React from "react";

/* Hand-placed pixel sprites for the setup loading notepad. Pure SVG rects with
   crisp edges, no external images. */

const PENCIL_ROWS = [
  "....kkkkkkkkkkkkkkkkk",
  "..kkwwYYYYYYYYYYssppk",
  ".kgwwyyyyyyyyyyyssppk",
  "kggwwyyyyyyyyyyyssppk",
  ".kgwwooooooooooossppk",
  "..kkwwoooooooooossppk",
  "....kkkkkkkkkkkkkkkkk",
];

const PENCIL_COLORS = {
  k: "#1a1030",
  w: "#f2c38a",
  g: "#3a3a48",
  y: "#ffd23a",
  Y: "#fff08a",
  o: "#e89a1c",
  s: "#b8c4d6",
  p: "#ff7aa8",
};

function rowsToRects(rows, colors, keyPrefix) {
  const out = [];
  rows.forEach((row, y) => {
    let x = 0;
    while (x < row.length) {
      const ch = row[x];
      const color = colors[ch];
      if (!color) {
        x += 1;
        continue;
      }
      // Merge horizontal runs of the same colour into one rect.
      let run = 1;
      while (x + run < row.length && row[x + run] === ch) run += 1;
      out.push(<rect key={`${keyPrefix}${x}-${y}`} x={x} y={y} width={run} height={1} fill={color} />);
      x += run;
    }
  });
  return out;
}

const PENCIL_RECTS = rowsToRects(PENCIL_ROWS, PENCIL_COLORS, "p");

export function PixelPencil({ className = "" }) {
  return (
    <svg
      className={className}
      viewBox="0 0 21 7"
      shapeRendering="crispEdges"
      aria-hidden="true"
      focusable="false"
    >
      {PENCIL_RECTS}
    </svg>
  );
}

/* Zamboni facing right: conditioner + towel at the back (left), snow tank on top,
   cab with driver at the front (right). [x, y, w, h, colour] */
const INK = "#071233";
const ZAMBONI_PARTS = [
  // snow tank
  [6, 1, 18, 9, INK],
  [7, 2, 16, 7, "#eaf6ff"],
  [7, 2, 16, 1, "#ffffff"],
  [8, 4, 3, 1, "#9fd8ff"],
  [13, 4, 3, 1, "#9fd8ff"],
  [18, 4, 3, 1, "#9fd8ff"],
  // cab
  [23, 3, 9, 7, INK],
  [24, 4, 7, 5, "#9fd8ff"],
  [24, 4, 7, 1, "#d8f0ff"],
  [27, 6, 2, 2, "#ffcf9e"],
  [26, 5, 4, 1, "#ff3b3b"],
  [29, 5, 1, 1, "#ff3b3b"],
  // body
  [3, 9, 30, 8, INK],
  [4, 10, 28, 6, "#2f6fe0"],
  [4, 10, 28, 1, "#6aa0ff"],
  [4, 12, 28, 2, "#ffffff"],
  [4, 13, 28, 1, "#ffe23a"],
  [4, 15, 28, 1, "#1d4aa8"],
  // front bumper + headlight
  [32, 11, 2, 5, INK],
  [32, 11, 1, 2, "#ffe23a"],
  // conditioner + towel
  [0, 13, 5, 5, INK],
  [1, 14, 3, 3, "#b8c4d6"],
  [0, 18, 6, 1, "#ffffff"],
  [0, 18, 6, 1, "#eaf6ff"],
  // wheels
  [7, 15, 6, 5, INK],
  [9, 16, 2, 2, "#b8c4d6"],
  [22, 15, 6, 5, INK],
  [24, 16, 2, 2, "#b8c4d6"],
];

export function PixelZamboni({ className = "" }) {
  return (
    <svg
      className={className}
      viewBox="0 0 34 20"
      shapeRendering="crispEdges"
      aria-hidden="true"
      focusable="false"
    >
      {ZAMBONI_PARTS.map(([x, y, w, h, fill], i) => (
        <rect key={i} x={x} y={y} width={w} height={h} fill={fill} />
      ))}
    </svg>
  );
}
