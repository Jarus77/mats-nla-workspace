#!/usr/bin/env node
// Builds deck/mats_nla_workspace.pptx from deck/content/*.js.
// Usage: node build.js            (from the deck/ directory)
// Edit numbers and text in content/*.js; layout code lives in lib/.
const path = require("path");
const pptxgen = require("pptxgenjs");
const { THEME, HEX, defineLayouts } = require("./lib/theme");
const { makeKit } = require("./lib/kit");
const { R } = require("./lib/render_basic");
const { D } = require("./lib/render_diagrams");
const { S } = require("./lib/render_story");
const content = require("./content");

const SKILL_THEME_SCRIPT = process.env.APPLY_THEME_JS || path.join(__dirname, "lib", "apply_theme.js");

const RENDERERS = { ...R, ...D, ...S };
const LAYOUT_OF = { title: "TITLE_DARK", section: "SECTION_DARK", closing: "CLOSING_DARK" };

function validateSlide(s, i) {
  if (!s || typeof s !== "object") throw new Error(`slide ${i + 1}: not an object`);
  if (!RENDERERS[s.type]) throw new Error(`slide ${i + 1}: unknown type "${s.type}"`);
  if (!s.section) throw new Error(`slide ${i + 1} (${s.type}): missing section`);
  if (!s.notes || s.notes.length < 40) throw new Error(`slide ${i + 1} (${s.type}): speaker notes missing or too short`);
  if (s.type !== "title" && s.type !== "section" && !s.title) throw new Error(`slide ${i + 1} (${s.type}): missing title`);
}

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = content.meta.title;
  pres.author = content.meta.author;
  pres.subject = "MATS 12.0 research update";
  defineLayouts(pres);

  const kit = makeKit(pres);
  const ctx = { pres, kit, HEX, meta: content.meta, knowledge: content.knowledge };

  let currentSection = null;
  content.slides.forEach((s, i) => {
    validateSlide(s, i);
    if (s.section !== currentSection) {
      pres.addSection({ title: s.section });
      currentSection = s.section;
    }
    const slide = pres.addSlide({ masterName: LAYOUT_OF[s.type] || "CONTENT", sectionTitle: s.section });
    RENDERERS[s.type](ctx, slide, s);
    slide.addNotes(s.notes);
  });

  const out = path.join(__dirname, `${content.meta.file}.pptx`);
  await pres.writeFile({ fileName: out });
  const { applyTheme } = require(SKILL_THEME_SCRIPT);
  await applyTheme(out, THEME);
  process.stdout.write(`wrote ${out} (${content.slides.length} slides)\n`);
}

main().catch((err) => {
  process.stderr.write(`build failed: ${err.stack || err}\n`);
  process.exit(1);
});
