// Renderers for generic slide types: title, section, chart, stats, table, anchor, grid, glossary,
// timeline, answer, gantt, quote, closing. Each takes (ctx, slide, s) where s is the content spec.
const { renderChart, hexOf } = require("./charts");

function header(ctx, slide, s) {
  const { kit } = ctx;
  if (s.kicker) kit.kicker(slide, s.kicker);
  slide.addText(s.title, { placeholder: "title" });
  const flag = s.flag || (s.pilot ? "PILOT · excluded from main analysis" : null);
  if (flag) kit.chip(slide, { x: kit.W - kit.M - 3.45, y: 0.26, w: 3.45, h: 0.32, text: flag, color: s.flag ? kit.C.accent5 : kit.C.accent4, size: 11 });
}

function toneColor(kit, tone) {
  const C = kit.C;
  return { nonj: C.text1, inj: C.accent3, diffuse: C.accent5, fail: C.accent4 }[tone] || C.accent5;
}

function estLines(text, widthIn, pt) {
  const cpi = 144 / pt; // ~chars per inch for Arial at this size
  return Math.max(1, Math.ceil((text || "").length / (widthIn * cpi)));
}

function runsText(r) {
  return Array.isArray(r) ? r.map((x) => x.text).join("") : String(r);
}

/** Right-hand commentary panel for chart slides. */
function renderSide(ctx, slide, side, x, y, w, h) {
  const { kit } = ctx;
  const C = kit.C;
  let cy = y;
  (side.stats || []).forEach((st) => {
    const size = st.size || 32;
    const vh = size / 72 + 0.12;
    const lh = estLines(st.label, w, 13) * 0.23 + 0.05;
    kit.tx(slide, st.value, { x, y: cy, w, h: vh, fontFace: "Cambria", bold: true, fontSize: size, color: colorKey(kit, st.color), valign: "bottom" });
    kit.tx(slide, st.label, { x, y: cy + vh + 0.03, w, h: lh, fontSize: 13, color: C.accent5 });
    cy += vh + lh + 0.22;
  });
  if (side.steps) {
    side.steps.forEach((st) => {
      const bh = 0.3 + estLines(st.body, w - 0.1, 13) * 0.23;
      kit.tx(slide, st.head, { x, y: cy, w, h: 0.28, fontSize: 14, bold: true, color: C.text1 });
      kit.tx(slide, st.body, { x, y: cy + 0.28, w, h: bh - 0.28, fontSize: 13, color: C.text1 });
      cy += bh + 0.14;
    });
  }
  if (side.rows) {
    side.rows.forEach((r) => {
      const col = r.color ? colorKey(kit, r.color) : toneColor(kit, r.tone);
      slide.addShape(kit.SHAPE.ellipse, { x, y: cy + 0.07, w: 0.16, h: 0.16, fill: { color: col }, line: { color: col, width: 0 } });
      const lines = estLines(r.body, w - 0.3, 13);
      kit.tx(slide, [{ text: r.head + "  ", options: { bold: true } }, { text: r.body, options: { color: C.accent5 } }], { x: x + 0.3, y: cy, w: w - 0.3, h: 0.25 + lines * 0.22, fontSize: 13 });
      cy += 0.3 + lines * 0.22;
    });
  }
  if (side.bullets) {
    const lines = side.bullets.reduce((a, b) => a + estLines(runsText(b), w - 0.3, 14), 0);
    const bh = lines * 0.245 + side.bullets.length * 0.11 + 0.1;
    kit.bullets(slide, side.bullets, { x, y: cy, w, h: bh, size: 14, gap: 6 });
    cy += bh + 0.15;
  }
  if (side.note) {
    const nh = estLines(side.note, w - 0.3, 12.5) * 0.215 + 0.3;
    kit.card(slide, { x, y: cy, w, h: nh, fill: C.background2, body: side.note, bodySize: 12.5, pad: 0.14 });
    cy += nh;
  }
  return cy - y;
}

function colorKey(kit, key) {
  const C = kit.C;
  return { nla: C.accent1, ol: C.accent2, jl: C.accent3, text: C.accent4, ink: C.text1, probe: C.text1, muted: C.accent5 }[key] || C.text1;
}

const R = {};

R.title = (ctx, slide, s) => {
  const { kit, meta } = ctx;
  slide.addText(meta.title, { placeholder: "title" });
  slide.addText(meta.subtitle, { placeholder: "body" });
  if (meta.titleNote) kit.chip(slide, { x: 0.8, y: 1.0, w: 6.4, h: 0.34, text: meta.titleNote, color: kit.C.accent4, size: 11 });
  kit.tx(slide, `${meta.author}  ·  ${meta.affiliation}  ·  ${meta.date}`, { x: 0.8, y: 5.2, w: 11.7, h: 0.35, fontSize: 14, color: kit.C.accent6 });
  kit.stream(slide, 6.45);
};

R.section = (ctx, slide, s) => {
  const { kit } = ctx;
  kit.tx(slide, s.num, { x: 0.8, y: 1.3, w: 3, h: 1.0, fontFace: "Cambria", fontSize: 54, bold: true, color: kit.C.accent4 });
  slide.addText(s.title, { placeholder: "title" });
  slide.addText(s.sub, { placeholder: "body" });
  kit.stream(slide, 6.25);
};

R.chart = (ctx, slide, s) => {
  const { kit, pres } = ctx;
  header(ctx, slide, s);
  const top = 1.8;
  if (s.charts) {
    const gap = 0.4;
    const cw = (kit.W - 2 * kit.M - gap) / 2;
    s.charts.forEach((c, i) => renderChart(pres, kit, slide, c, { x: kit.M + i * (cw + gap), y: top, w: cw, h: 3.75 }));
    const sw = (kit.W - 2 * kit.M - 2 * 0.3) / 3;
    (s.strip || []).forEach((t, i) => kit.card(slide, { x: kit.M + i * (sw + 0.3), y: 5.75, w: sw, h: 1.1, body: t, bodySize: 12.5, pad: 0.14 }));
    return;
  }
  const cw = s.chartW || 7.2;
  renderChart(pres, kit, slide, s.chart, { x: kit.M - 0.1, y: top, w: cw, h: 5.0 });
  const sx = kit.M + cw + 0.35;
  if (s.side) renderSide(ctx, slide, s.side, sx, top + 0.05, kit.W - kit.M - sx, 5.0);
};

R.stats = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const n = s.stats.length;
  const gap = 0.35;
  const w = (kit.W - 2 * kit.M - gap * (n - 1)) / n;
  s.stats.forEach((st, i) => {
    const x = kit.M + i * (w + gap);
    kit.stat(slide, { x, y: 1.75, w, value: st.value, label: st.label, color: colorKey(kit, st.color), size: 38, vh: 0.75, lh: 0.75, labelSize: 13 });
  });
  const fw = (kit.W - 2 * kit.M - gap * (s.families.length - 1)) / s.families.length;
  s.families.forEach((f, i) => {
    kit.card(slide, { x: kit.M + i * (fw + gap), y: 3.55, w: fw, h: 1.6, title: f.head, titleSize: 15, body: f.body, bodySize: 13 });
  });
  kit.card(slide, { x: kit.M, y: 5.45, w: kit.W - 2 * kit.M, h: 1.25, fill: C.background1, line: C.accent4, body: s.caveat, bodySize: 14, pad: 0.2 });
};

R.table = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const fs = s.small ? 11.5 : 13;
  const head = s.columns.map((c) => ({ text: c, options: { bold: true, color: "FFFFFF", fill: { color: ctx.HEX.dark }, fontSize: fs - 0.5, valign: "middle" } }));
  const body = s.rows.map((row, ri) =>
    row.map((cell) => {
      const zebra = ri % 2 === 0 ? "FFFFFF" : ctx.HEX.panel;
      if (cell && typeof cell === "object" && cell.label) {
        return { text: cell.label, options: { bold: true, color: "FFFFFF", fill: { color: toneHex(ctx, cell.tone) }, align: "center", fontSize: fs - 1, valign: "middle" } };
      }
      if (cell && typeof cell === "object") return { text: cell.text, options: { bold: !!cell.bold, fill: { color: zebra }, fontSize: fs, valign: "middle", color: ctx.HEX.ink } };
      return { text: String(cell), options: { fill: { color: zebra }, fontSize: fs, valign: "middle", color: ctx.HEX.ink } };
    })
  );
  const w = s.colW.reduce((a, b) => a + b, 0);
  slide.addTable([head, ...body], { x: kit.M, y: 1.85, w, colW: s.colW, fontFace: "Arial", border: { type: "solid", pt: 0.5, color: ctx.HEX.line }, margin: [0.05, 0.08, 0.05, 0.08], rowH: s.small ? 0.36 : 0.44 });
  if (s.footnote) kit.tx(slide, s.footnote, { x: kit.M, y: 6.35, w: kit.W - 2 * kit.M, h: 0.5, fontSize: 12, color: C.accent5 });
};

function toneHex(ctx, tone) {
  const H = ctx.HEX;
  return { nonj: H.ink, inj: H.jl, diffuse: H.muted, fail: H.text }[tone] || H.muted;
}

const STATUS = {
  established: { label: "ESTABLISHED", key: "text1" },
  pilot: { label: "PILOT", key: "accent4" },
  tentative: { label: "TENTATIVE", key: "accent5" },
  pending: { label: "PENDING", key: "accent6" },
};

R.anchor = (ctx, slide, s) => {
  const { kit, knowledge } = ctx;
  const C = kit.C;
  kit.kicker(slide, "Running anchor · what we know so far");
  slide.addText(s.title, { placeholder: "title" });
  const items = knowledge.filter((k) => k.stage <= s.stage);
  const top = items.length > 8 ? 1.5 : 1.8;
  const avail = (s.next ? 5.85 : 6.9) - top;
  const rh = Math.min(0.62, avail / items.length);
  const fs = items.length > 8 ? 12.5 : items.length > 5 ? 13.5 : 15;
  if (s.next) kit.card(slide, { x: kit.M, y: 6.0, w: kit.W - 2 * kit.M, h: 0.7, fill: C.background1, line: C.accent4, body: s.next, bodySize: 16, pad: 0.18 });
  items.forEach((k, i) => {
    const y = top + i * rh;
    const st = STATUS[k.status];
    const isNew = k.stage === s.stage;
    kit.chip(slide, { x: kit.M, y: y + 0.06, w: 1.45, h: 0.28, text: st.label, color: C[st.key], size: 10 });
    kit.tx(slide, k.text, { x: kit.M + 1.7, y: y + 0.02, w: kit.W - 2 * kit.M - 1.7, h: rh - 0.04, fontSize: fs, bold: isNew && s.stage < 4, color: isNew || s.stage === 4 ? C.text1 : C.accent5 });
  });
};

R.grid = (ctx, slide, s) => {
  const { kit } = ctx;
  header(ctx, slide, s);
  const cols = 4;
  const gap = 0.3;
  const w = (kit.W - 2 * kit.M - gap * (cols - 1)) / cols;
  const h = 2.4;
  s.items.forEach((it, i) => {
    const r = Math.floor(i / cols);
    const c = i % cols;
    kit.card(slide, { x: kit.M + c * (w + gap), y: 1.85 + r * (h + gap), w, h, title: it.head, titleSize: 15, body: it.body, bodySize: 13 });
  });
};

R.glossary = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const half = Math.ceil(s.terms.length / 2);
  const cw = (kit.W - 2 * kit.M - 0.5) / 2;
  s.terms.forEach(([term, def], i) => {
    const c = i < half ? 0 : 1;
    const r = i % half;
    const x = kit.M + c * (cw + 0.5);
    const y = 1.8 + r * 0.62;
    kit.tx(slide, term, { x, y, w: 2.1, h: 0.55, fontSize: 13, bold: true, color: C.text1 });
    kit.tx(slide, def, { x: x + 2.15, y, w: cw - 2.15, h: 0.58, fontSize: 12, color: C.text1 });
  });
};

R.timeline = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const cw = (kit.W - 2 * kit.M - 0.6) / 2;
  s.days.forEach((d, c) => {
    const x = kit.M + c * (cw + 0.6);
    kit.chip(slide, { x, y: 1.8, w: 1.1, h: 0.32, text: d.date, color: C.text2, size: 12 });
    const rh = Math.min(0.42, 4.55 / Math.max(...s.days.map((dd) => dd.items.length)));
    kit.line(slide, x + 0.27, 2.2, x + 0.27, 2.2 + d.items.length * rh, C.accent6, 1.25);
    d.items.forEach(([id, text], i) => {
      const y = 2.25 + i * rh;
      kit.chip(slide, { x, y: y + 0.02, w: 0.55, h: 0.28, text: id, color: /Fable/.test(text) ? C.accent2 : C.accent5, size: 10 });
      kit.tx(slide, text, { x: x + 0.7, y, w: cw - 0.7, h: 0.36, fontSize: 13, color: C.text1, valign: "middle" });
    });
  });
  kit.tx(slide, "violet = a Fable critique round", { x: kit.M + 1.3, y: 1.83, w: 4, h: 0.28, fontSize: 11, color: C.accent5 });
};

R.checks = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const w = kit.W - 2 * kit.M;
  s.checks.forEach((c, i) => {
    const y = 1.8 + i * 0.86;
    if (c.done) kit.card(slide, { x: kit.M - 0.1, y: y - 0.04, w: w + 0.2, h: 0.84, fill: C.background2 });
    kit.badge(slide, kit.M, y + 0.15, c.done ? "✓" : c.tag, c.done ? C.accent3 : C.text1, 0.46);
    kit.tx(slide, c.head, { x: kit.M + 0.65, y, w: 3.0, h: 0.75, fontSize: 15, bold: true, valign: "middle", color: c.done ? C.accent3 : C.text1 });
    kit.tx(slide, c.body, { x: kit.M + 3.75, y, w: w - 3.75, h: 0.78, fontSize: 13.5, valign: "middle", color: c.done ? C.accent5 : C.text1 });
  });
  kit.card(slide, { x: kit.M, y: 6.15, w, h: 0.75, fill: C.background2, body: s.changes, bodySize: 12.5, pad: 0.14 });
};

module.exports = { R, header, renderSide, toneColor, estLines, colorKey };
