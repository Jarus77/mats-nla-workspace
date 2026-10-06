// Renderers for the closing story slides: answer, quote, gantt, closing.
const { header, estLines } = require("./render_basic");

const S = {};

S.answer = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const lw = 5.3;
  kit.card(slide, { x: kit.M, y: 1.85, w: lw, h: 4.95, fill: C.text2 });
  kit.tx(slide, s.question, { x: kit.M + 0.3, y: 2.05, w: lw - 0.6, h: 0.7, fontSize: 15, italic: true, color: C.accent6 });
  kit.tx(slide, s.answer, { x: kit.M + 0.3, y: 2.75, w: lw - 0.6, h: 0.9, fontFace: "Cambria", bold: true, fontSize: 44, color: C.background1 });
  kit.tx(slide, s.answerBody, { x: kit.M + 0.3, y: 3.75, w: lw - 0.6, h: 1.9, fontSize: 16, color: C.background1 });
  kit.tx(slide, s.confidence, { x: kit.M + 0.3, y: 5.85, w: lw - 0.6, h: 0.8, fontSize: 13, bold: true, color: C.accent4 });
  const x = kit.M + lw + 0.45;
  const w = kit.W - kit.M - x;
  kit.tx(slide, "Caveats we state with it", { x, y: 1.85, w, h: 0.4, fontFace: "Cambria", bold: true, fontSize: 19 });
  let y = 2.4;
  s.caveats.forEach((c, i) => {
    const h = estLines(c, w - 0.6, 14) * 0.25 + 0.2;
    kit.badge(slide, x, y, i + 1, C.accent5, 0.36);
    kit.tx(slide, c, { x: x + 0.55, y: y - 0.02, w: w - 0.55, h, fontSize: 14 });
    y += h + 0.12;
  });
};

S.quote = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const lw = 7.0;
  kit.tx(slide, "PROMPT (UNCHANGED)", { x: kit.M, y: 1.85, w: lw, h: 0.28, fontSize: 11, bold: true, color: C.accent5, charSpacing: 1.5 });
  kit.card(slide, { x: kit.M, y: 2.15, w: lw, h: 0.75, fill: C.background2 });
  kit.tx(slide, s.prompt, { x: kit.M + 0.2, y: 2.2, w: lw - 0.4, h: 0.65, fontFace: "Courier New", fontSize: 14, valign: "middle" });
  kit.tx(slide, s.planted, { x: kit.M, y: 3.0, w: lw, h: 0.55, fontSize: 14, bold: true, color: C.accent5 });
  kit.arrow(slide, kit.M + lw / 2, 3.55, kit.M + lw / 2, 3.85, C.accent1, 2);
  kit.card(slide, { x: kit.M, y: 3.9, w: lw, h: 1.25, fill: C.accent1, transparency: 85, line: C.accent1 });
  kit.tx(slide, s.output, { x: kit.M + 0.25, y: 3.95, w: lw - 0.5, h: 1.15, fontFace: "Cambria", italic: true, fontSize: 24, valign: "middle" });
  kit.tx(slide, s.outputSource, { x: kit.M, y: 5.3, w: lw, h: 0.9, fontSize: 12.5, color: C.accent5 });
  const x = kit.M + lw + 0.4;
  const w = kit.W - kit.M - x;
  s.lessons.forEach((l, i) => {
    kit.card(slide, { x, y: 1.85 + i * 2.5, w, h: 2.3, fill: C.background2, title: l.head, titleSize: 16, body: l.body, bodySize: 13.5 });
  });
};

S.gantt = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const lx = kit.M;
  const lw = 5.6;
  const gx = lx + lw + 0.1;
  const dw = (kit.W - kit.M - gx) / s.days.length;
  const y0 = 1.85;
  const rh = Math.min(0.39, 3.75 / s.rows.length);
  kit.tx(slide, "Oct", { x: gx, y: y0 - 0.02, w: 1, h: 0.25, fontSize: 10, color: C.accent5 });
  kit.tx(slide, "Nov", { x: gx + dw * 13, y: y0 - 0.02, w: 1, h: 0.25, fontSize: 10, color: C.accent5 });
  s.days.forEach((d, i) => {
    kit.tx(slide, d, { x: gx + i * dw, y: y0 + 0.2, w: dw, h: 0.28, fontSize: 11, bold: true, align: "center", color: C.text1 });
    if (i % 2 === 0) slide.addShape(kit.SHAPE.rect, { x: gx + i * dw, y: y0 + 0.5, w: dw, h: s.rows.length * rh, fill: { color: C.background2 }, line: { color: C.background2, width: 0 } });
  });
  const colors = { nla: C.accent1, ol: C.accent2, jl: C.accent3, text: C.accent4, muted: C.accent5, line: C.accent6 };
  s.rows.forEach((r, i) => {
    const y = y0 + 0.55 + i * rh;
    kit.tx(slide, r.label, { x: lx, y, w: lw, h: 0.32, fontSize: 12, valign: "middle" });
    const bx = gx + (r.from - 1) * dw + 0.04;
    const bw = (r.to - r.from + 1) * dw - 0.08;
    slide.addShape(kit.SHAPE.roundRect, { x: bx, y: y + 0.04, w: bw, h: 0.24, rectRadius: 0.06, fill: { color: colors[r.color] }, line: { color: colors[r.color], width: 0 } });
    if (r.check) {
      const kx = gx + r.to * dw - 0.02;
      slide.addShape(kit.SHAPE.diamond, { x: kx - 0.14, y: y + 0.02, w: 0.28, h: 0.28, fill: { color: C.text1 }, line: { color: C.background1, width: 1 } });
      kit.tx(slide, r.check, { x: kx + 0.16, y, w: 0.5, h: 0.32, fontSize: 10, bold: true, valign: "middle" });
    }
  });
  const ky = y0 + 0.65 + s.rows.length * rh;
  const kw = (kit.W - 2 * kit.M - 0.3 * 2) / 3;
  s.kills.forEach((k, i) => {
    const x = kit.M + i * (kw + 0.3);
    slide.addShape(kit.SHAPE.diamond, { x, y: ky + 0.06, w: 0.3, h: 0.3, fill: { color: C.text1 }, line: { color: C.text1, width: 0 } });
    kit.tx(slide, [{ text: k.id + "  ", options: { bold: true } }, { text: k.text }], { x: x + 0.42, y: ky, w: kw - 0.42, h: 0.5, fontSize: 12.5 });
  });
};

S.closing = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  slide.addText(s.title, { placeholder: "title" });
  const lw = 7.3;
  kit.tx(slide, "CONTRIBUTIONS", { x: kit.M, y: 1.75, w: lw, h: 0.3, fontSize: 12, bold: true, color: C.accent4, charSpacing: 1.5 });
  s.contributions.forEach((c, i) => {
    const y = 2.15 + i * 1.08;
    kit.badge(slide, kit.M, y, i + 1, [C.accent4, C.accent3, C.accent1, C.accent2][i], 0.4);
    kit.tx(slide, c, { x: kit.M + 0.6, y: y - 0.04, w: lw - 0.6, h: 1.0, fontSize: 14.5, color: C.background1 });
  });
  const x = kit.M + lw + 0.5;
  const w = kit.W - kit.M - x;
  kit.tx(slide, "WHAT WOULD CHANGE OUR MINDS", { x, y: 1.75, w, h: 0.3, fontSize: 12, bold: true, color: C.accent4, charSpacing: 1.5 });
  s.open.forEach((o, i) => {
    kit.card(slide, { x, y: 2.15 + i * 1.2, w, h: 1.02, fill: C.background1, transparency: 90, line: C.accent5, body: o, bodySize: 14, bodyColor: C.background1, pad: 0.16 });
  });
  kit.stream(slide, 6.6);
};

S.branches = (ctx, slide, s) => {
  const { kit } = ctx;
  const C = kit.C;
  header(ctx, slide, s);
  const gap = 0.4;
  const w = (kit.W - 2 * kit.M - gap) / 2;
  s.branches.forEach((b, i) => {
    const x = kit.M + i * (w + gap);
    const dark = i === 0;
    kit.card(slide, { x, y: 1.8, w, h: 2.75, fill: dark ? C.text2 : C.background2 });
    kit.tx(slide, b.head.toUpperCase(), { x: x + 0.25, y: 1.95, w: w - 0.5, h: 0.3, fontSize: 12, bold: true, color: C.accent4, charSpacing: 1.2 });
    kit.tx(slide, b.answer, { x: x + 0.25, y: 2.3, w: w - 0.5, h: 1.2, fontFace: "Cambria", bold: true, fontSize: 21, color: dark ? C.background1 : C.text1 });
    kit.tx(slide, b.body, { x: x + 0.25, y: 3.55, w: w - 0.5, h: 0.9, fontSize: 13.5, color: dark ? C.accent6 : C.accent5 });
  });
  kit.tx(slide, "STANDS ON EITHER BRANCH", { x: kit.M, y: 4.75, w: 6, h: 0.3, fontSize: 12, bold: true, color: C.accent3, charSpacing: 1.2 });
  const sw = (kit.W - 2 * kit.M - 0.3 * 2) / 3;
  s.stands.forEach((t, i) => {
    kit.card(slide, { x: kit.M + i * (sw + 0.3), y: 5.1, w: sw, h: 1.15, fill: C.background1, line: C.accent3, body: t, bodySize: 13, pad: 0.15 });
  });
  kit.tx(slide, s.caveats, { x: kit.M, y: 6.4, w: kit.W - 2 * kit.M, h: 0.35, fontSize: 12, color: C.accent5 });
};

module.exports = { S };
