# 05: (C) planted intermediates: yield check (K4) plus a first reader pass

**Verdict: K4 passes.** At planting layer 38 with the centred swap, 431 of 614 candidates meet all the keep criteria under the per-item minimal-dose policy (alpha ≤ 4). 22 of the 431 are WorkspaceBench bank items. At one fixed dose, alpha = 3, 348 items are kept. L34 keeps 212 and L42 keeps 542. L24 and L28 do not pass (18 and 39).

**Caveat that matters for the sprint.** The doses that flip the answer are large: ‖Δ‖/‖h‖ has a median of 0.49 (q10–q90: 0.33–0.71) at L38. That is 1.5–3× the X-to-Y class-mean distance, or about 10 within-class SD along the swap direction. At the natural class-mean swap (alpha = 1) only 9–14 items flip at L38. So K4 passes only if the prereg accepts "beyond the natural class-mean distance, but with the model not broken": KL on neutral prompts ≤ 0.1 (median 0.015 at alpha 2), the relation still decodable, and the prompt still self-retrieved at L42. [DECIDE: lead/Fable]

Date 2026-10-06. Qwen3.6-27B bf16. Layer L means block-L output = `hidden_states[L+1]`. Readers read at L42. Compute: roughly 1.3 H100-h plus 0.3 H200-h.

## Items
- **Constructed bank** (`plant_items.py`): 31 countries × 5 clues × up to 4 relations (capital, official language, currency, continent). Relations that are ambiguous for a country are skipped. Example: "Fact: The capital of the country famous for the Eiffel Tower is". 605 prompts in total.
- **Bank items:** the 30 WorkspaceBench `multihop` items whose bridge is one of these countries and whose relation is one of the four.
- **Gate** (model's own answer): some alias of the correct answer, including case variants, must be greedy-consistent (every alias token is the argmax). 585/605 constructed and 29/30 bank items pass. The real failures are Rio for Brazil, Munich, Barcelona, Istanbul and similar.
- **Partners:** one partner bridge Y per gated item, chosen with a seed from the item id **before any planting** (never by outcome). Y must have the same relation, ≥ 3 gated prompts in that relation (so the model knows Y's answer), and answer first tokens disjoint from X's. That gives 614 candidates (585 constructed + 29 bank).

## Planting
- **Direction.** At layer L, d = μ_Y − μ_X, where μ is the class mean over gated constructed prompts after centring by relation. **μ_X leaves out the item's own clue.** Bank items use all of X's constructed prompts.
- **Positions:** from the read token to the end of the prompt (the last token, or the last two for the one bank prompt that ends in a space). The prompt text is unchanged.
- **Two constructions:**
  - **clamp (primary, per D11):** the centred swap h' = h + (t − u·h)u, with u = d̂ and t = c_X + alpha(c_Y − c_X), where c is the class-mean projection. The item's own u-projection is removed and replaced, which avoids unequal dilution. Alpha = 1 puts it exactly on the Y class mean.
  - **add:** norm-preserving h' = (h + alpha·d)‖h‖/‖h + alpha·d‖.
- **Alpha grid:** 0, 0.5, 1, 1.5, 2, 3, 4, 6, 8. Layers 24, 28, 34, 38 and 42 (42 is the readers' own layer, kept as a reference).
- **Keep criteria**, per (item, layer, construction, alpha):
  - **(i) Flip:** some Y alias is greedy-consistent AND no X alias is.
  - **(ii-a) Not broken:** the same edit applied at the last token of 6 unrelated prompts gives a mean next-token KL ≤ 0.1.
  - **(ii-b) Prompt still decodable at L42:**
    - after removing the span of the 31 country class means, the planted L42 read vector is closer (cosine) to its own unplanted vector than to any of the 614 other unplanted items;
    - the nearest relation mean is still the item's relation.
  - **(iii) Bridge probe at L42 reads Y:** the X-vs-Y class-mean midpoint rule, using leave-own-clue-out means at L42.
- **Per-item minimal-alpha policy:** the smallest alpha in (0, 4] that meets all criteria. Choosing alpha this way is selection on the *model's* behaviour, never on a reader's. Fixed-alpha yields are given below too.

## Yield (keep counts out of 614; bank items in brackets)

| layer | construction | alpha 1 | 1.5 | 2 | 3 | 4 | min-alpha ≤ 4 | ‖Δ‖/‖h‖ median at alpha 2 | neutral KL median at alpha 2 |
|---|---|---|---|---|---|---|---|---|---|
| 24 | clamp | 0 | 0 | 0 | 3 | 17 | 18 (0) | 0.31 | 0.004 |
| 28 | clamp | 0 | 0 | 1 | 10 | 39 | 39 (3) | 0.26 | 0.004 |
| 34 | clamp | 0 | 6 | 36 | 141 | 181 | 212 (13) | 0.27 | 0.008 |
| **38** | **clamp** | 9 | 109 | 236 | **348** | 185 | **431 (22)** | 0.42 | 0.015 |
| 42 | clamp | 137 | 384 | 480 | 487 | 234 | 542 (25) | 0.50 | 0.020 |
| 28 | add | 0 | 0 | 1 | 7 | 21 | 22 (1) | 0.31 | 0.007 |
| 34 | add | 0 | 6 | 41 | 143 | 152 | 197 (12) | 0.31 | 0.014 |
| 38 | add | 14 | 127 | 257 | 250 | 107 | 350 (18) | 0.46 | 0.024 |
| 42 | add | 176 | 425 | 481 | 462 | 333 | 519 (25) | 0.53 | 0.030 |

How each criterion behaves at L38 clamp (full tables in `results/yield.json`):
- **Flips peak at alpha 4** (451/614) and then fall as the model breaks: at alpha 8, KL-ok drops to 5/614.
- **(iii) saturates at alpha 1.5.**
- **(ii-b) is the binding constraint at high alpha.** The bridge-removed self-retrieval falls from 605 to 456 between alpha 1.5 and 4.
- **A stricter variant is far too strict.** It asks the raw planted vector (bridge content included) to stay nearest its own unplanted vector rather than any Y-bridge prompt. Only 7–27 items pass it. With a 10-SD bridge swap, the raw nearest neighbour is naturally a Y prompt, so this variant mixes up "the prompt is decodable" with "the bridge did not move". The primary rule removes the bridge span first.
- **Alpha = 0 keeps 0 items, and probe-reads-Y is 0/614.** This sanity check passes.
- **Layer trend.** Below L34, planting at the read token barely flips the answer: the bridge is presumably re-derived from the clue tokens, which are unedited. From L34 on, the planted value is the one the model uses.
- **Dose against the natural spread.** Along d̂, the X and Y class means are about 4–5 within-class SD apart. The kept doses have a median of about 10 SD, i.e. 1.5–3× the class-mean distance. The pilot's escalation (34/50 at a 16× cap of a token-readout direction) was much more extreme and was selected on a reader. Ours is selected on the model and stays coherent, but it is not "inside the natural spread" as §4.5 says. Fixed-dose subsets (alpha ≤ 1.5: 109 items at L38 clamp) are available if the prereg insists.

## Reader pass (feasibility; L38 clamp, minimal-alpha policy, 200 of 431 kept items, random subsample, seed 0)

Setup:
- LO = log p(Y) − log p(X) in native format, averaged over 3 templates:
  - OL: `- {c}`, `- {c} (country)`, `- The country of {c}`;
  - NLA: `{c}`, `The text is about {c}`, `A factual question involving {c}`, after `<explanation>\n`;
  - J-Lens: max cosine score over the name/adjective token variants.
- Null bands: K = 20 isotropic plus K = 20 J-complement (top-r* = 192 removed) random directions. Each is planted at L38 with the item's own ‖Δ‖ and a random sign, and propagated to L42. Results are reported as z, never subtracted.
- The swap direction u has J-space enrichment **6.5** at r* (strongly J-enriched). The bridge is in-J content.

| reader | P(reads Y), unplanted | **P(reads Y), planted** | P(text-implied X), planted | mean LO shift | paired win | z (shift), iso / J-compl. band | null P(Y) band (iso) |
|---|---|---|---|---|---|---|---|
| OL@42 | 0.01 | **1.00** | 0.00 | +20.9 | 1.00 | +166 / +189 | [0.00, 0.02] |
| NLA@42 | 0.03 | **0.98** | 0.01 | +14.2 | 1.00 | +112 / +84 | [0.01, 0.04] |
| J-Lens@42 | 0.01 | **1.00** | 0.00 | +17.4 | 1.00 | +168 / +164 | [0.01, 0.01] |

- Per template, P(Y) is 0.99–1.00 for OL and 0.89 / 0.99 / 0.985 for the NLA (the bare `{c}` template is weakest). It is 1.00 at every alpha level, including the 5 items kept at alpha = 1.
- **Free-form NLA** (12 items, k = 1): the planted explanation names Y and not X in 12/12. The unplanted one names X in 12/12. **The NLA also rewrites the clue to fit Y.** "Eiffel Tower" becomes "…the country whose CEO designed adopted Volvo… Sweden", or "…maple leaves (Canada)". So under planting, its explanation does **not** keep the original subject. Prereg keep criterion (ii)'s TextNLA-on-NLA-explanation subject check would fail here, even though the subject is still decodable from the vector by our geometric check.

**Reading of this pass:** on this in-J bridge, all three readers report the planted value, not the text-implied one. That is the "reading" pattern. It is the expected calibration result, because the swap is J-enriched and the J-Lens reads it too. It does **not** yet tell non-J reading from re-solving. That needs the split arm (§4.5 optional: plant only the J-removed component of d) and the leak check.

## Caveats / open issues
1. **Dose.** Answer flips need about 1.5–3× the class-mean distance. Decide whether that is acceptable for K4, or restrict to alpha ≤ 1.5 (109 items at L38 clamp, which still passes ≥ 60).
2. **Selection by (iii).** Kept items have a planted Y bridge at L42 by construction. That is the point of the control, but with a J-enriched direction, J-Lens and OL reading Y is close to guaranteed.
3. **Answer-vs-bridge confound.** At L42 the model already encodes Y's *answer* (e.g. "Stockholm"). A reader could name Y by back-inferring from the answer, not by reading the bridge. A next step is to score answer-token candidates too, or plant earlier (L34) and read before the answer forms.
4. **One partner per item, 31 countries:** partners repeat, and directions share class means.
5. **Not done:**
   - TextNLA-on-explanation subject recovery (§4.5 ii);
   - FVE guard on planted vectors;
   - OL@40/44 sensitivity;
   - the split (J vs non-J component) plant.

   The split arm is the natural next run. Plumbing is ready: `JS.j_removed_direction(basis, d, r)`, with `U` passed in place of `d̂` in `stage_sweep`.

## Files
- `plant_items.py`: country table, clues, templates, bank mapping, neutral prompts.
- `plant_lib.py`: batched forward with per-row edits (additive norm-preserving, or centred clamp), greedy-consistency scoring, capture, class-mean directions, seeded partner assignment.
- `plant_modal.py`: app `mats-05-planted`. Stages `gate`, `sweep` (.map over layer × construction, 10 H100 containers), `readers` (H100: OL + J-Lens + null plants), `nla` (H200). `env.py` holds the image (03 pins).
- `analyze.py`: keep criteria, giving `results/yield.json` and `results/keep.json`. `analyze_readers.py` gives `results/readers.json`.
- Volume `/results/05_planted_intermediates/`:
  - `gate.pt`;
  - `sweep_{add,clamp}_L{24,28,34,38,42}.pt` (including the planted L42 vectors for every item × alpha);
  - `readers_base.pt`;
  - `readers_nla.pt`.

  These are mirrored in `results/` (`*.pt` is git-ignored).
