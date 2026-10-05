// Native chart rendering from the declarative specs in content/*.js.
// spec: { kind: "bar"|"stacked"|"line", dir: "col"|"bar", categories, series:[{name, values, color, dash}],
//         pointColors (one-series bars), lines:[{name, value, color}] (horizontal reference lines),
//         min, max, fmt, labels, legend ("b"|false), title, valTitle, catTitle, labelColor }
const { HEX } = require("./theme");

function hexOf(key) {
  if (!key) return HEX.muted;
  const map = { nla: HEX.nla, ol: HEX.ol, jl: HEX.jl, text: HEX.text, probe: HEX.ink, ink: HEX.ink, muted: HEX.muted, line: HEX.line, nlaPale: HEX.nlaPale, olPale: HEX.olPale, jlPale: HEX.jlPale, textPale: HEX.textPale };
  return map[key] || key;
}

function renderChart(pres, kit, slide, spec, box) {
  const horizontal = spec.dir === "bar";
  const isLine = spec.kind === "line";
  const stacked = spec.kind === "stacked";
  const showLegend = !!spec.legend;

  const base = kit.chartOpts({
    x: box.x, y: box.y, w: box.w, h: box.h,
    showLegend, legendPos: spec.legend || "b",
    showTitle: !!spec.title, title: spec.title || "",
    valAxisMinVal: spec.min, valAxisMaxVal: spec.max,
    valAxisLabelFormatCode: spec.fmt || "0.00",
    showValAxisTitle: !!spec.valTitle, valAxisTitle: spec.valTitle || "",
    showCatAxisTitle: !!spec.catTitle, catAxisTitle: spec.catTitle || "",
  });
  if (spec.step) base.valAxisMajorUnit = spec.step;
  if (horizontal) {
    base.catAxisOrientation = "maxMin";
    base.valAxisHidden = true;
    base.valGridLine = { style: "none" };
  }

  const solid = spec.series.filter((s) => !s.dash);
  const dashed = spec.series.filter((s) => s.dash);
  const dataOf = (arr) => arr.map((s) => ({ name: s.name, labels: spec.categories, values: s.values }));

  if (isLine) {
    const groups = [
      { type: pres.charts.LINE, data: dataOf(solid), options: { chartColors: solid.map((s) => hexOf(s.color)), lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 6 } },
    ];
    if (dashed.length) {
      groups.push({ type: pres.charts.LINE, data: dataOf(dashed), options: { chartColors: dashed.map((s) => hexOf(s.color)), lineSize: 2, lineDataSymbol: "none", lineDash: "dash" } });
    }
    slide.addChart(groups, base);
    return;
  }

  const barOpts = {
    barDir: horizontal ? "bar" : "col",
    barGrouping: stacked ? "stacked" : "clustered",
    barGapWidthPct: spec.gap || (spec.series.length > 2 ? 60 : 80),
    chartColors: spec.pointColors ? spec.pointColors.map(hexOf) : spec.series.map((s) => hexOf(s.color)),
    showValue: spec.labels !== false,
    dataLabelPosition: stacked ? "ctr" : "outEnd",
    dataLabelFormatCode: spec.fmt || "0.00",
    dataLabelColor: spec.labelColor || HEX.ink,
    dataLabelFontSize: spec.series.length > 2 ? 10 : 11,
  };
  if (stacked) barOpts.barOverlapPct = 100;

  if (spec.lines && spec.lines.length) {
    const lineData = spec.lines.map((l) => ({ name: l.name, labels: spec.categories, values: spec.categories.map(() => l.value) }));
    const groups = [
      { type: pres.charts.BAR, data: dataOf(spec.series), options: barOpts },
      { type: pres.charts.LINE, data: lineData, options: { chartColors: spec.lines.map((l) => hexOf(l.color)), lineSize: 1.5, lineDataSymbol: "none", lineDash: "dash", showValue: false } },
    ];
    slide.addChart(groups, { ...base, ...{ showValue: false } });
    return;
  }

  slide.addChart(pres.charts.BAR, dataOf(spec.series), { ...base, ...barOpts });
}

module.exports = { renderChart, hexOf };
