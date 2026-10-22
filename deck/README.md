# MATS 12.0 research deck: "Do NLAs read what the model can't say, or re-solve what it read?"

Outputs: `mats_nla_workspace.pptx` (62 slides: 55 main + 7 appendix, speaker notes on every slide) and `mats_nla_workspace.pdf`.

## Rebuild (takes a few seconds)

```bash
cd deck
npm install          # first time only (pptxgenjs + jszip into deck/node_modules)
node build.js        # writes mats_nla_workspace.pptx and applies the theme colours
```

PDF (needs LibreOffice; `brew install --cask libreoffice`):

```bash
soffice --headless --convert-to pdf mats_nla_workspace.pptx
```

Check the file (optional; uses the pptx skill's validator):

```bash
python3 ~/.claude/skills/synced/*/pptx/scripts/office/validate.py mats_nla_workspace.pptx
```

## Changing a number or a sentence

All text, numbers, chart data and speaker notes live in `content/`. The layout code is in `lib/`.

| File | Contents |
|---|---|
| `content/00_meta.js` | title, author, date, and the "what we know so far" ledger (`stage` = which anchor slide an item first appears on; `status` = established / pilot / tentative / pending) |
| `content/01_intro.js` | title, question + answer up front, why it matters, background, prior work |
| `content/02_design.js` | 2×2 frame, tools, the FollowFeature measurement, the Fable loop |
| `content/03_gates.js` | K0, WorkspaceBench replication, best-of-11 protocol, anchor 1 |
| `content/05_partA.js` | Part A: survival test, calibrated gate, gate table, ~12-feature survey, anchor 2 |
| `content/06_partB.js` | Part B: why col, method lessons, col result, Figure 1 (K3), equal norm, content type, power, natural arm, anchor 3 |
| `content/06b_partC.js` | Part C: tense tail, planted intermediates (in-J calibration), the NLA rewrite |
| `content/04_wsbench.js` | Part D: text-only baselines, same-template re-solver comparison (D22), claim precision, prompt-only floors |
| `content/07_close.js` | ledger, answer ("neither"), open checks, limitations, sprint plan, closing |
| `content/08_appendix.js` | decision log, pilot re-analysis, outcome map, costs, glossary, settings |
| `content/index.js` | slide order (reorder sections here) |

Conventions:
- `title` is the claim (claim-as-title). `notes` is the speaker narration; the build refuses a slide without notes.
- Chart data: `chart: { kind: "bar" | "stacked" | "line", categories, series: [{ name, values, color }], lines: [{ name, value, color }] }`. Colour keys: `nla`, `ol`, `jl`, `text`, `probe`, `muted`, `line`, `nlaPale`, `olPale`, `jlPale`, `textPale`, or a 6-digit hex.
- `pilot: true` adds the "PILOT · excluded from main analysis" chip; `flag: "..."` adds a grey chip (used for "CONSISTENCY CHECK PENDING" on Part D).

Design system (in `lib/theme.js`): Cambria headings, Arial body. NLA = vermilion, Oracle Lens = violet, J-Lens = teal, text-only baselines = ochre in every chart and diagram. Dark title/section slides with the "residual stream" motif (layer nodes, three reader dots at layer 42); light content slides with a three-dot kicker. Style principles and sources: `STYLE_NOTES.md`.

## Import into Google Slides

1. Upload `mats_nla_workspace.pptx` to Google Drive.
2. Right-click → Open with → Google Slides (or File → Save as Google Slides from the preview).
3. Speaker notes come across as Google Slides speaker notes.
4. Expected differences: native charts arrive as static chart images in Google Slides (edit the numbers in `content/` and re-import, or re-create a chart in Sheets if you need it live). Fonts are Google-Slides-safe (Cambria, Arial, Courier New), so no substitution should occur. Check the two-column chart slides once after import for any line-wrap differences.

## Files

- `build.js`: entry point. `lib/theme.js` (palette, fonts, layouts), `lib/kit.js` (shapes, chips, motif), `lib/charts.js` (native charts), `lib/render_basic.js`, `lib/render_diagrams.js`, `lib/render_story.js` (one renderer per slide `type`), `lib/apply_theme.js` (copied from the pptx skill; writes theme colours into the file).
- `STYLE_NOTES.md`: research-talk principles applied in the deck, with sources.
