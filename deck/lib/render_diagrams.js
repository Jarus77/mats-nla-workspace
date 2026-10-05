// Native-shape diagrams for the opening, background and design slides.
const { header, estLines } = require("./render_basic");

const D = {};

D.question = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  kit.card(slide, { x: kit.M, y: 1.85, w: 5.0, h: 4.85, fill: C.background2 });
  kit.tx(slide, "“", { x: kit.M + 0.25, y: 1.85, w: 1, h: 1.0, fontFace: "Cambria", fontSize: 80, bold: true, color: C.accent1 });
  kit.tx(slide, s.quote, { x: kit.M + 0.35, y: 2.75, w: 4.35, h: 2.6, fontFace: "Cambria", italic: true, fontSize: 24, color: C.text1 });
  kit.tx(slide, s.quoteSource, { x: kit.M + 0.35, y: 5.6, w: 4.35, h: 0.8, fontSize: 12, color: C.accent5 });
  const x = kit.M + 5.45;
  const w = kit.W - kit.M - x;
  kit.tx(slide, "Our answer so far", { x, y: 1.85, w, h: 0.4, fontFace: "Cambria", bold: true, fontSize: 20, color: C.text1 });
  const cols = [C.accent4, C.accent1, C.accent3];
  s.answers.forEach((a, i) => {
    const y = 2.45 + i * 1.45;
    kit.badge(slide, x, y + 0.02, i + 1, cols[i]);
    kit.tx(slide, [{ text: a.lead + " ", options: { bold: true } }, { text: a.text }], { x: x + 0.6, y, w: w - 0.6, h: 1.35, fontSize: 16, color: C.text1 });
  });
};

D.why = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const gap = 0.55;
  const w = (kit.W - 2 * kit.M - gap * 2) / 3;
  const fills = [C.background2, C.background2, C.accent1];
  s.steps.forEach((st, i) => {
    const x = kit.M + i * (w + gap);
    kit.card(slide, { x, y: 2.0, w, h: 2.45, fill: fills[i], transparency: i === 2 ? 85 : 0, line: i === 2 ? C.accent1 : null });
    kit.badge(slide, x + 0.22, 2.2, i + 1, i === 2 ? C.accent1 : C.accent5);
    kit.tx(slide, st.head, { x: x + 0.78, y: 2.22, w: w - 1, h: 0.4, fontFace: "Cambria", bold: true, fontSize: 19 });
    kit.tx(slide, st.body, { x: x + 0.22, y: 2.85, w: w - 0.44, h: 1.5, fontSize: 15 });
    if (i < 2) kit.arrow(slide, x + w + 0.08, 3.22, x + w + gap - 0.08, 3.22, C.accent5, 2);
  });
  kit.card(slide, { x: kit.M, y: 4.95, w: kit.W - 2 * kit.M, h: 1.25, fill: C.background1, line: C.accent4 });
  kit.tx(slide, s.test, { x: kit.M + 0.3, y: 5.05, w: kit.W - 2 * kit.M - 0.6, h: 1.05, fontSize: 18, bold: true, color: C.text1, valign: "middle" });
};

D.residual = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const n = s.prompt.length;
  const tw = 0.7;
  const step = 0.75;
  const yTok = 6.15;
  const yTop = 2.0;
  s.prompt.forEach((t, i) => {
    const x = kit.M + i * step;
    const last = i === n - 1;
    kit.line(slide, x + tw / 2, yTop, x + tw / 2, yTok - 0.05, last ? C.text1 : C.accent6, last ? 2.25 : 1);
    slide.addShape(kit.SHAPE.roundRect, { x, y: yTok, w: tw, h: 0.4, rectRadius: 0.06, fill: { color: last ? C.text1 : C.background2 }, line: { color: last ? C.text1 : C.accent6, width: 0.75 } });
    kit.tx(slide, t, { x: x - 0.05, y: yTok, w: tw + 0.1, h: 0.4, fontSize: 11, align: "center", valign: "middle", color: last ? C.background1 : C.text1 });
  });
  const xl = kit.M + (n - 1) * step + tw / 2;
  for (let k = 0; k <= 8; k++) {
    const y = yTok - 0.05 - ((yTok - 0.05 - yTop) * k) / 8;
    slide.addShape(kit.SHAPE.ellipse, { x: xl - 0.06, y: y - 0.06, w: 0.12, h: 0.12, fill: { color: C.text1 }, line: { color: C.text1, width: 0 } });
  }
  const y42 = yTok - 0.05 - (yTok - 0.05 - yTop) * (42 / 63);
  slide.addShape(kit.SHAPE.ellipse, { x: xl - 0.2, y: y42 - 0.2, w: 0.4, h: 0.4, fill: { color: C.accent1 }, line: { color: C.background1, width: 2 } });
  kit.tx(slide, "layer 63", { x: xl + 0.22, y: yTop - 0.15, w: 1.0, h: 0.3, fontSize: 11, color: C.accent5 });
  kit.tx(slide, "layer 0", { x: xl + 0.22, y: yTok - 0.4, w: 1.0, h: 0.3, fontSize: 11, color: C.accent5 });
  slide.addShape(kit.SHAPE.roundRect, { x: xl - 3.55, y: y42 - 0.42, w: 3.2, h: 0.84, rectRadius: 0.08, fill: { color: C.background1 }, line: { color: C.accent1, width: 1.25 } });
  kit.tx(slide, [{ text: "Read here: ", options: { bold: true, color: C.accent1 } }, { text: "layer 42, last token → one vector of 5,120 numbers" }], { x: xl - 3.45, y: y42 - 0.38, w: 3.0, h: 0.76, fontSize: 13, valign: "middle" });
  kit.arrow(slide, xl - 0.35, y42, xl - 0.22, y42, C.accent1, 1.5);
  const x = 9.55;
  kit.bullets(slide, s.side, { x, y: 2.0, w: kit.W - kit.M - x, h: 4.5, size: 16, gap: 14 });
};

D.jspace = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  kit.card(slide, { x: kit.M, y: 1.9, w: 5.6, h: 4.85, fill: C.background2 });
  kit.tx(slide, "Residual stream · 5,120 dimensions", { x: kit.M + 0.25, y: 2.05, w: 5.1, h: 0.35, fontSize: 14, bold: true, color: C.accent5 });
  slide.addShape(kit.SHAPE.ellipse, { x: kit.M + 1.3, y: 2.75, w: 3.0, h: 2.2, fill: { color: C.accent3, transparency: 10 }, line: { color: C.accent3, width: 1 } });
  kit.tx(slide, s.inner, { x: kit.M + 1.3, y: 3.2, w: 3.0, h: 1.3, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle" });
  kit.tx(slide, s.innerSub, { x: kit.M + 0.4, y: 5.05, w: 4.8, h: 0.35, fontSize: 13, italic: true, color: C.accent3, align: "center" });
  kit.tx(slide, s.outerSub, { x: kit.M + 0.25, y: 6.15, w: 5.1, h: 0.45, fontSize: 13, italic: true, color: C.accent5 });
  const x = kit.M + 6.0;
  kit.bullets(slide, s.points, { x, y: 1.95, w: kit.W - kit.M - x, h: 4.8, size: 16, gap: 16 });
};

D.readers = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const gap = 0.35;
  const w = (kit.W - 2 * kit.M - gap * 2) / 3;
  const cx = kit.W / 2;
  kit.chip(slide, { x: cx - 2.1, y: 1.78, w: 4.2, h: 0.36, text: "one vector h · layer 42 · last token", color: C.text2, size: 12 });
  s.readers.forEach((r, i) => {
    const x = kit.M + i * (w + gap);
    const col = kit.READER[r.key].color;
    kit.arrow(slide, cx, 2.15, x + w / 2, 2.5, C.accent6, 1.25);
    kit.card(slide, { x, y: 2.55, w, h: 4.25, fill: C.background2 });
    slide.addShape(kit.SHAPE.ellipse, { x: x + 0.22, y: 2.75, w: 0.5, h: 0.5, fill: { color: col }, line: { color: col, width: 0 } });
    kit.tx(slide, r.name, { x: x + 0.85, y: 2.72, w: w - 1.0, h: 0.38, fontFace: "Cambria", bold: true, fontSize: 21, color: col });
    kit.tx(slide, r.tag.toUpperCase(), { x: x + 0.85, y: 3.1, w: w - 1.0, h: 0.25, fontSize: 10.5, bold: true, color: C.accent5, charSpacing: 1 });
    kit.bullets(slide, r.lines, { x: x + 0.22, y: 3.55, w: w - 0.44, h: 2.2, size: 14, gap: 6 });
    kit.card(slide, { x: x + 0.15, y: 5.85, w: w - 0.3, h: 0.8, fill: C.background1, body: r.role, bodySize: 12.5, pad: 0.12 });
  });
};

D.nonj = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const w = 6.6;
  s.criteria.forEach((c, i) => {
    const y = 1.9 + i * 1.2;
    const gated = i < 3;
    kit.badge(slide, kit.M, y + 0.02, gated ? `G${[1, 2, 4][i]}` : "—", gated ? C.text1 : C.accent6, 0.5);
    kit.tx(slide, c.head, { x: kit.M + 0.7, y, w: w - 0.7, h: 0.35, fontSize: 17, bold: true, color: gated ? C.text1 : C.accent5 });
    kit.tx(slide, c.body, { x: kit.M + 0.7, y: y + 0.36, w: w - 0.7, h: 0.75, fontSize: 14, color: gated ? C.text1 : C.accent5 });
  });
  const x = kit.M + w + 0.45;
  const ew = kit.W - kit.M - x;
  kit.card(slide, { x, y: 1.9, w: ew, h: 4.85, fill: C.background2, title: s.example.head, titleSize: 18, body: s.example.body, bodySize: 15 });
  slide.addShape(kit.SHAPE.roundRect, { x: x + 0.2, y: 4.55, w: ew - 0.4, h: 1.0, rectRadius: 0.06, fill: { color: C.background1 }, line: { color: C.accent6, width: 1 } });
  kit.tx(slide, s.example.sample, { x: x + 0.32, y: 4.6, w: ew - 0.6, h: 0.9, fontFace: "Courier New", fontSize: 12, color: C.text1, valign: "middle" });
  kit.tx(slide, "col = 37 here ↑ (the model must track it to know a break is near)", { x: x + 0.2, y: 5.7, w: ew - 0.4, h: 0.6, fontSize: 12, color: C.accent5 });
};

D.confound = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const box = (x, y, w, h, title, body, fill, line, transp) => {
    slide.addShape(kit.SHAPE.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: fill, transparency: transp || 0 }, line: { color: line || fill, width: 1.25 } });
    kit.tx(slide, title, { x: x + 0.15, y: y + 0.1, w: w - 0.3, h: 0.35, fontFace: "Cambria", bold: true, fontSize: 16 });
    if (body) kit.tx(slide, body, { x: x + 0.15, y: y + 0.47, w: w - 0.3, h: h - 0.55, fontSize: 13 });
  };
  box(kit.M, 2.75, 2.2, 1.35, "Prompt text", "\"…famous for the Eiffel Tower is\"", C.background2);
  kit.arrow(slide, kit.M + 2.25, 3.42, 3.2, 3.42, C.accent5, 2);
  box(3.25, 2.55, 2.75, 1.75, "Activation, layer 42", "encodes the prompt and the features the model computed", C.background2, C.text1);
  box(6.75, 1.85, 3.35, 1.3, s.readPath.head, s.readPath.body, C.accent1, C.accent1, 85);
  box(6.75, 3.65, 3.35, 1.5, s.solvePath.head, s.solvePath.body, C.accent4, C.accent4, 82);
  kit.arrow(slide, 6.05, 3.2, 6.7, 2.55, C.accent1, 2);
  kit.arrow(slide, 6.05, 3.65, 6.7, 4.35, C.accent4, 2);
  box(10.8, 2.75, 1.93, 1.35, "Same answer", "\"France\"", C.background2);
  kit.arrow(slide, 10.15, 2.55, 10.75, 3.2, C.accent5, 1.5);
  kit.arrow(slide, 10.15, 4.35, 10.75, 3.65, C.accent5, 1.5);
  kit.card(slide, { x: kit.M, y: 5.45, w: kit.W - 2 * kit.M, h: 0.95, fill: C.background2 });
  kit.tx(slide, [{ text: s.natural + " ", options: {} }, { text: s.fix, options: { bold: true, color: C.accent1 } }], { x: kit.M + 0.2, y: 5.5, w: kit.W - 2 * kit.M - 0.4, h: 0.85, fontSize: 15, valign: "middle" });
  kit.tx(slide, s.aside, { x: kit.M, y: 6.5, w: kit.W - 2 * kit.M, h: 0.3, fontSize: 12, color: C.accent5 });
};

D.prior = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const gap = 0.3;
  const w = (kit.W - 2 * kit.M - gap * 3) / 4;
  s.cards.forEach((c, i) => {
    const x = kit.M + i * (w + gap);
    kit.card(slide, { x, y: 1.85, w, h: 3.95, fill: C.background2 });
    kit.tx(slide, c.head, { x: x + 0.16, y: 1.98, w: w - 0.32, h: 0.36, fontFace: "Cambria", bold: true, fontSize: 16 });
    kit.tx(slide, c.src, { x: x + 0.16, y: 2.34, w: w - 0.32, h: 0.3, fontSize: 10.5, color: C.accent5 });
    kit.tx(slide, c.body, { x: x + 0.16, y: 2.72, w: w - 0.32, h: 3.0, fontSize: 12.5 });
  });
  kit.card(slide, { x: kit.M, y: 6.0, w: kit.W - 2 * kit.M, h: 0.85, fill: C.background1, line: C.accent1 });
  kit.tx(slide, s.open, { x: kit.M + 0.2, y: 6.04, w: kit.W - 2 * kit.M - 0.4, h: 0.77, fontSize: 14, bold: true, valign: "middle" });
};

D.twoByTwo = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const x0 = kit.M + 2.55;
  const cw = (kit.W - kit.M - x0 - 0.2) / 2;
  const y0 = 2.85;
  const ch = 1.6;
  s.colHeads.forEach((h, j) => {
    const x = x0 + j * (cw + 0.2);
    kit.tx(slide, h, { x, y: 1.85, w: cw, h: 0.4, fontSize: 17, bold: true, align: "center", color: j === 0 ? C.accent1 : C.accent4 });
    kit.tx(slide, s.colSubs[j], { x, y: 2.25, w: cw, h: 0.5, fontSize: 12, align: "center", color: C.accent5 });
  });
  s.rowHeads.forEach((h, i) => {
    const y = y0 + i * (ch + 0.2);
    kit.tx(slide, h, { x: kit.M, y, w: 2.35, h: ch, fontSize: 16, bold: true, valign: "middle", color: i === 0 ? C.accent3 : C.text1 });
  });
  s.cells.forEach((c, k) => {
    const i = Math.floor(k / 2);
    const j = k % 2;
    const x = x0 + j * (cw + 0.2);
    const y = y0 + i * (ch + 0.2);
    kit.card(slide, { x, y, w: cw, h: ch, fill: c.highlight ? C.accent1 : C.background2, transparency: c.highlight ? 82 : 0, line: c.highlight ? C.accent1 : null, title: c.head, titleSize: 17, body: c.body, bodySize: 14 });
  });
  kit.tx(slide, s.footnote, { x: kit.M, y: 6.45, w: kit.W - 2 * kit.M, h: 0.4, fontSize: 12, color: C.accent5 });
};

D.tools = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const gap = 0.35;
  const w = (kit.W - 2 * kit.M - gap * 2) / 3;
  const cols = [C.accent1, C.text1, C.accent4];
  s.tools.forEach((t, i) => {
    const x = kit.M + i * (w + gap);
    kit.card(slide, { x, y: 1.85, w, h: 3.75, fill: C.background2 });
    kit.badge(slide, x + 0.2, 2.02, t.tag, cols[i], 0.55);
    kit.tx(slide, t.head, { x: x + 0.9, y: 2.0, w: w - 1.05, h: 0.65, fontFace: "Cambria", bold: true, fontSize: 16, valign: "middle" });
    kit.tx(slide, t.body, { x: x + 0.2, y: 2.8, w: w - 0.4, h: 2.1, fontSize: 14 });
    kit.tx(slide, t.cell, { x: x + 0.2, y: 5.05, w: w - 0.4, h: 0.4, fontSize: 12, bold: true, color: C.accent5 });
  });
  kit.card(slide, { x: kit.M, y: 5.85, w: kit.W - 2 * kit.M, h: 0.95, fill: C.background1, line: C.accent6, body: s.currency, bodySize: 14, pad: 0.2 });
};

D.followFeature = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const gap = 0.38;
  const n = s.steps.length;
  const w = (kit.W - 2 * kit.M - gap * (n - 1)) / n;
  s.steps.forEach((st, i) => {
    const x = kit.M + i * (w + gap);
    kit.card(slide, { x, y: 1.85, w, h: 1.95, fill: i === 0 ? C.text2 : C.background2 });
    kit.tx(slide, st.head, { x: x + 0.15, y: 1.95, w: w - 0.3, h: 0.35, fontSize: 15, bold: true, color: i === 0 ? C.background1 : C.text1 });
    kit.tx(slide, st.body, { x: x + 0.15, y: 2.38, w: w - 0.3, h: 1.35, fontSize: 13, color: i === 0 ? C.accent6 : C.text1, fontFace: i < 2 ? "Courier New" : "Arial" });
    if (i < n - 1) kit.arrow(slide, x + w + 0.05, 2.82, x + w + gap - 0.05, 2.82, C.accent5, 2);
  });
  kit.tx(slide, "GUARDS", { x: kit.M, y: 4.05, w: 3, h: 0.3, fontSize: 11, bold: true, color: C.accent5, charSpacing: 1.5 });
  const gw = (kit.W - 2 * kit.M - 0.35 * 2) / 3;
  const gcol = [C.accent3, C.accent4, C.text1];
  s.guards.forEach((g, i) => {
    const x = kit.M + i * (gw + 0.35);
    kit.card(slide, { x, y: 4.4, w: gw, h: 2.4, fill: C.background1, line: gcol[i], title: g.head, titleSize: 16, titleColor: gcol[i], body: g.body, bodySize: 13.5 });
  });
};

D.loop = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const cx = 3.2;
  const cy = 4.3;
  const r = 2.0;
  const n = s.loop.length;
  const pts = s.loop.map((_, i) => {
    const a = -Math.PI / 2 + (2 * Math.PI * i) / n;
    return [cx + r * 0.86 * Math.cos(a), cy + r * 0.95 * Math.sin(a)];
  });
  pts.forEach(([x, y], i) => {
    const [x2, y2] = pts[(i + 1) % n];
    const dx = x2 - x;
    const dy = y2 - y;
    const L = Math.hypot(dx, dy);
    const ux = dx / L;
    const uy = dy / L;
    const off = (vx, vy) => Math.min(0.95 / Math.max(Math.abs(vx), 1e-6), 0.34 / Math.max(Math.abs(vy), 1e-6)) + 0.06;
    const o = off(ux, uy);
    kit.arrow(slide, x + ux * o, y + uy * o, x2 - ux * o, y2 - uy * o, C.accent5, 1.75);
  });
  pts.forEach(([x, y], i) => {
    const fable = i >= 3;
    slide.addShape(kit.SHAPE.roundRect, { x: x - 0.9, y: y - 0.28, w: 1.8, h: 0.56, rectRadius: 0.1, fill: { color: fable ? C.accent2 : C.background2 }, line: { color: fable ? C.accent2 : C.accent6, width: 1 } });
    kit.tx(slide, s.loop[i], { x: x - 0.85, y: y - 0.28, w: 1.7, h: 0.56, fontSize: 12, bold: true, align: "center", valign: "middle", color: fable ? C.background1 : C.text1 });
  });
  const x = 6.0;
  const w = kit.W - kit.M - x;
  s.changes.forEach((c, i) => {
    const y = 1.85 + i * 1.1;
    kit.chip(slide, { x, y: y + 0.03, w: 1.15, h: 0.3, text: c.round, color: C.accent2, size: 11 });
    kit.tx(slide, c.what, { x: x + 1.3, y, w: w - 1.3, h: 0.52, fontSize: 13.5 });
    kit.tx(slide, c.fix, { x: x + 1.3, y: y + 0.54, w: w - 1.3, h: 0.3, fontSize: 13.5, bold: true, color: C.accent1 });
  });
  kit.tx(slide, s.pushback, { x, y: 6.35, w, h: 0.55, fontSize: 12, italic: true, color: C.accent5 });
};

module.exports = { D, estLines };
