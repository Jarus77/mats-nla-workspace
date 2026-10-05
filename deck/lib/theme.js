// Theme, palette and slide layouts. Change colours or fonts here; every slide follows.
// Colour roles are fixed for the whole deck:
//   accent1 = NLA (vermilion), accent2 = Oracle Lens (violet), accent3 = J-Lens (teal),
//   accent4 = text-only baselines (ochre), accent5 = muted ink, accent6 = hairlines.

const THEME = {
  name: "Residual Ink",
  headFontFace: "Cambria",
  bodyFontFace: "Arial",
  colors: {
    dk1: "15222C", // body text ink
    lt1: "FFFFFF",
    dk2: "0C1A24", // dark slide ground
    lt2: "EEF2F4", // light panel
    accent1: "E2553A", // NLA
    accent2: "7A5BC2", // Oracle Lens
    accent3: "0F9D8A", // J-Lens
    accent4: "C8921E", // text-only / re-solving baselines
    accent5: "5E6E78", // muted ink
    accent6: "C9D3D9", // hairlines, muted text on dark
    hlink: "0F9D8A",
    folHlink: "7A5BC2",
  },
};

// Hex values for the places that only accept hex (charts, shadows).
const HEX = {
  ink: THEME.colors.dk1,
  dark: THEME.colors.dk2,
  panel: THEME.colors.lt2,
  nla: THEME.colors.accent1,
  ol: THEME.colors.accent2,
  jl: THEME.colors.accent3,
  text: THEME.colors.accent4,
  muted: THEME.colors.accent5,
  line: THEME.colors.accent6,
  grid: "E3E8EB",
  // pale tints used for "published"/reference series in charts
  nlaPale: "F3B6A8",
  olPale: "CBBEE8",
  jlPale: "9FD9CF",
  textPale: "E9D3A1",
  ramp: ["2C3B46", "5F7987", "93A9B5", "C3D1D8", "E6ECEF"],
};

const W = 13.333;
const H = 7.5;
const M = 0.6; // side margin
const FOOTER = "MATS 12.0 · Neel Nanda stream · Do NLAs read what the model can't say?";

function defineLayouts(pres) {
  const C = pres.SchemeColor;
  const num = (color) => ({ x: W - M - 0.8, y: H - 0.42, w: 0.8, h: 0.3, color, fontSize: 10, fontFace: "Arial", align: "right" });

  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: THEME.colors.dk2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.55, w: 11.7, h: 2.3, fontFace: "Cambria", fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.05, w: 11.7, h: 1.1, fontFace: "Arial", fontSize: 18, color: C.accent6, valign: "top", align: "left", margin: 0 }, text: "" } },
    ],
  });

  pres.defineSlideMaster({
    title: "SECTION_DARK",
    background: { color: THEME.colors.dk2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.35, w: 11.7, h: 1.5, fontFace: "Cambria", fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.0, w: 11.0, h: 1.1, fontFace: "Arial", fontSize: 18, color: C.accent6, valign: "top", align: "left", margin: 0 }, text: "" } },
    ],
    slideNumber: num(THEME.colors.accent6),
  });

  pres.defineSlideMaster({
    title: "CLOSING_DARK",
    background: { color: THEME.colors.dk2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: M, y: 0.6, w: W - 2 * M, h: 0.95, fontFace: "Cambria", fontSize: 30, bold: true, color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
    ],
    slideNumber: num(THEME.colors.accent6),
  });

  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: THEME.colors.lt1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: M, y: 0.62, w: W - 2 * M, h: 0.98, fontFace: "Cambria", fontSize: 28, bold: true, color: C.text1, valign: "top", align: "left", margin: 0 }, text: "" } },
      { text: { text: FOOTER, options: { x: M, y: H - 0.42, w: 9, h: 0.3, fontSize: 10, fontFace: "Arial", color: THEME.colors.accent5, margin: 0 } } },
    ],
    slideNumber: num(THEME.colors.accent5),
  });
}

module.exports = { THEME, HEX, W, H, M, defineLayouts };
