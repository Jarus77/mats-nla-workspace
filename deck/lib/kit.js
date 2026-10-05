// Drawing primitives shared by every renderer. Positions are inches on a 13.333 x 7.5 canvas.
const { HEX, W, H, M } = require("./theme");

function makeKit(pres) {
  const C = pres.SchemeColor;
  const SHAPE = pres.ShapeType;

  // Reader identity: colour + display name. Used by every chart, chip and diagram.
  const READER = {
    nla: { name: "NLA", color: C.accent1, hex: HEX.nla },
    ol: { name: "Oracle Lens", color: C.accent2, hex: HEX.ol },
    jl: { name: "J-Lens", color: C.accent3, hex: HEX.jl },
    text: { name: "Text-only", color: C.accent4, hex: HEX.text },
    probe: { name: "Probe", color: C.text1, hex: HEX.ink },
  };

  /** Plain text box; margin 0 so text aligns with shapes at the same x. */
  function tx(slide, text, opts) {
    slide.addText(text, { isTextBox: true, margin: 0, color: C.text1, fontSize: 16, valign: "top", ...opts });
  }

  /** Section kicker above the title: three reader dots + caps label (the deck's motif). */
  function kicker(slide, label) {
    const y = 0.3;
    ["accent1", "accent2", "accent3"].forEach((k, i) => {
      slide.addShape(SHAPE.ellipse, { x: M + i * 0.17, y: y + 0.07, w: 0.12, h: 0.12, fill: { color: C[k] }, line: { color: C[k], width: 0 } });
    });
    tx(slide, label.toUpperCase(), { x: M + 0.6, y, w: 9, h: 0.26, fontSize: 11, bold: true, color: C.accent5, charSpacing: 1.5 });
  }

  /** Tinted rounded card with optional heading and body. */
  function card(slide, o) {
    const fillColor = o.fill || C.background2;
    slide.addShape(SHAPE.roundRect, {
      x: o.x, y: o.y, w: o.w, h: o.h, rectRadius: 0.08,
      fill: { color: fillColor, transparency: o.transparency || 0 },
      line: o.line ? { color: o.line, width: 1.25, dashType: o.dash || "solid" } : { color: fillColor, width: 0 },
    });
    const pad = o.pad == null ? 0.18 : o.pad;
    let y = o.y + pad;
    if (o.title) {
      const th = o.titleH || 0.36;
      tx(slide, o.title, { x: o.x + pad, y, w: o.w - 2 * pad, h: th, fontFace: "Cambria", bold: true, fontSize: o.titleSize || 17, color: o.titleColor || C.text1 });
      y += th + 0.05;
    }
    if (o.body) {
      tx(slide, o.body, { x: o.x + pad, y, w: o.w - 2 * pad, h: o.y + o.h - y - pad * 0.6, fontSize: o.bodySize || 14, color: o.bodyColor || C.text1, paraSpaceAfter: 4 });
    }
  }

  /** Small pill label. */
  function chip(slide, o) {
    const w = o.w || 0.12 * o.text.length + 0.35;
    slide.addShape(SHAPE.roundRect, { x: o.x, y: o.y, w, h: o.h || 0.3, rectRadius: 0.15, fill: { color: o.color }, line: { color: o.color, width: 0 } });
    tx(slide, o.text, { x: o.x, y: o.y, w, h: o.h || 0.3, fontSize: o.size || 11, bold: true, color: o.textColor || C.background1, align: "center", valign: "middle" });
    return w;
  }

  /** Numbered circle. */
  function badge(slide, x, y, label, color, d) {
    const s = d || 0.42;
    slide.addShape(SHAPE.ellipse, { x, y, w: s, h: s, fill: { color }, line: { color, width: 0 } });
    tx(slide, String(label), { x, y, w: s, h: s, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle" });
  }

  /** Big number + caption. */
  function stat(slide, o) {
    tx(slide, o.value, { x: o.x, y: o.y, w: o.w, h: o.vh || 0.8, fontFace: "Cambria", bold: true, fontSize: o.size || 40, color: o.color || C.text1, valign: "bottom" });
    tx(slide, o.label, { x: o.x, y: o.y + (o.vh || 0.8) + 0.06, w: o.w, h: o.lh || 0.7, fontSize: o.labelSize || 13, color: C.accent5 });
  }

  function arrow(slide, x1, y1, x2, y2, color, width) {
    const flipH = x2 < x1;
    const flipV = y2 < y1;
    slide.addShape(SHAPE.line, {
      x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.001), h: Math.max(Math.abs(y2 - y1), 0.001),
      flipH, flipV, line: { color: color || C.accent5, width: width || 1.5, endArrowType: "triangle" },
    });
  }

  function line(slide, x1, y1, x2, y2, color, width, dash) {
    slide.addShape(SHAPE.line, {
      x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.001), h: Math.max(Math.abs(y2 - y1), 0.001),
      flipH: x2 < x1, flipV: y2 < y1, line: { color: color || C.accent6, width: width || 1, dashType: dash || "solid" },
    });
  }

  /** Bulleted list from an array of strings or [{text, options}] runs. */
  function bullets(slide, items, o) {
    const runs = [];
    items.forEach((it, i) => {
      const last = i === items.length - 1;
      const base = { bullet: o.bullet === false ? false : { indent: 16 }, breakLine: !last, paraSpaceAfter: o.gap == null ? 8 : o.gap };
      if (typeof it === "string") runs.push({ text: it, options: base });
      else if (Array.isArray(it)) it.forEach((r, j) => runs.push({ text: r.text, options: { ...r.options, ...(j === 0 ? { bullet: base.bullet } : {}), breakLine: j === it.length - 1 ? !last : false, paraSpaceAfter: base.paraSpaceAfter } }));
      else runs.push({ text: it.text, options: { ...base, ...it.options } });
    });
    tx(slide, runs, { x: o.x, y: o.y, w: o.w, h: o.h, fontSize: o.size || 16, color: o.color || C.text1 });
  }

  /** The "residual stream" motif for dark slides: a line of layer nodes with L42 ringed. */
  function stream(slide, y, opts) {
    const x0 = (opts && opts.x0) || 0.8;
    const x1 = (opts && opts.x1) || W - 0.8;
    line(slide, x0, y, x1, y, C.accent5, 1.25);
    const n = 16;
    for (let i = 0; i < n; i++) {
      const x = x0 + ((x1 - x0) * i) / (n - 1);
      slide.addShape(SHAPE.ellipse, { x: x - 0.045, y: y - 0.045, w: 0.09, h: 0.09, fill: { color: C.accent5 }, line: { color: C.accent5, width: 0 } });
    }
    const xr = x0 + (x1 - x0) * (42 / 63);
    ["accent1", "accent2", "accent3"].forEach((k, i) => {
      slide.addShape(SHAPE.ellipse, { x: xr - 0.11 + (i - 1) * 0.3, y: y - 0.11, w: 0.22, h: 0.22, fill: { color: C[k] }, line: { color: C.text2, width: 1.5 } });
    });
    tx(slide, "layer 42", { x: xr - 0.6, y: y + 0.18, w: 1.2, h: 0.25, fontSize: 10, color: C.accent6, align: "center" });
  }

  /** Reader legend chips in a row; returns nothing. */
  function legend(slide, x, y, keys, size) {
    let cx = x;
    keys.forEach((k) => {
      const r = READER[k] || k;
      slide.addShape(SHAPE.ellipse, { x: cx, y: y + 0.06, w: 0.16, h: 0.16, fill: { color: r.color }, line: { color: r.color, width: 0 } });
      tx(slide, r.name, { x: cx + 0.22, y, w: 1.8, h: 0.28, fontSize: size || 12, color: C.text1 });
      cx += 0.22 + 0.1 * r.name.length + 0.45;
    });
  }

  /** "Result pending" stamp. */
  function pending(slide, x, y, text) {
    chip(slide, { x, y, text: text || "RESULT PENDING", color: C.accent4, w: 2.3, h: 0.36, size: 12 });
  }

  /** Common, quiet chart styling. Always returns a fresh object (pptxgenjs mutates options). */
  function chartOpts(extra) {
    return {
      catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted,
      catAxisLabelFontFace: "Arial", valAxisLabelFontFace: "Arial",
      catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
      catAxisLineShow: true, catAxisLineColor: HEX.line, valAxisLineShow: false,
      valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" },
      dataLabelFontFace: "Arial", dataLabelFontSize: 11, dataLabelColor: HEX.ink,
      legendFontFace: "Arial", legendFontSize: 12, legendColor: HEX.ink,
      titleFontFace: "Arial", titleFontSize: 13, titleColor: HEX.ink,
      valAxisTitleFontFace: "Arial", valAxisTitleFontSize: 11, valAxisTitleColor: HEX.muted,
      catAxisTitleFontFace: "Arial", catAxisTitleFontSize: 11, catAxisTitleColor: HEX.muted,
      ...extra,
    };
  }

  return { C, SHAPE, READER, tx, kicker, card, chip, badge, stat, arrow, line, bullets, stream, legend, pending, chartOpts, W, H, M, HEX };
}

module.exports = { makeKit };
