"""Follow-up experiments for the workshop submission. Mirrors mech-interp.ipynb.

    modal run --detach workshop_runs.py::gamma_ext    # concept+random arms at gamma 0.6, 0.8
    modal run --detach workshop_runs.py::threeway     # concept / noise / nothing elicitation
    modal run --detach workshop_runs.py::model_b      # Qwen3-4B, same prompts, no injection

Each writes a resumable JSONL under /notebooks/ on the jlens-notebooks volume.
"""

import modal

MOUNT, NB = "/models", "/notebooks"
BASE_A = f"{MOUNT}/Qwen3-8B"
BASE_B = f"{MOUNT}/Qwen3-4B"
LENS = f"{MOUNT}/jacobian-lens/qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt"

app = modal.App("jlens-workshop")
weights = modal.Volume.from_name("jlens-weights")
nbvol = modal.Volume.from_name("jlens-notebooks")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.0", "transformers>=4.51", "accelerate", "numpy",
                 "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ---- identical to the notebook ---------------------------------------------
PASSAGES = [
    "The library closes at six on weekdays and at noon on Saturdays throughout the year.",
    "Construction on the eastern bridge began three years behind the original schedule and continues.",
    "The committee published its findings in a report of eleven pages with two appendices.",
    "Rainfall measurements have been recorded at this station since 1947.",
    "The second edition corrected several errors in the original index and added a short appendix.",
    "Applications must be submitted before the fifteenth of the month to be considered that cycle.",
    "The device operates on a twelve volt supply and draws two amperes.",
    "Meeting minutes are circulated to all members within three working days of the session.",
    "The tower was completed in stages over a period of fourteen years.",
    "Each unit is tested for leakage before it leaves the assembly line on every shift.",
    "The archive holds roughly forty thousand documents on open shelving across two floors.",
    "Passenger numbers on the northern line rose by four percent last quarter.",
    "The manual describes seven procedures for routine maintenance of the housing and its fittings.",
    "Registration opens at eight and closes once capacity has been reached for that session.",
    "The survey covered two hundred households selected at random from the current register.",
    "Copies of the ordinance are available at the front desk during ordinary office hours.",
    "The furnace requires inspection every eighteen months under current rules, or sooner after repair.",
    "Records from before 1962 were transferred to microfilm and later digitised.",
    "The building has four stairwells and two service elevators serving all nine floors.",
    "Voltage fluctuations were logged automatically throughout the trial period and reviewed afterwards.",
]
BATTERY = {
    "countries": ["France","Japan","Brazil","Egypt","Canada","Germany","Norway","Chile","Poland","Nepal"],
    "animals":   ["spider","elephant","tiger","whale","rabbit","snake","eagle","camel","frog","owl"],
    "foods":     ["bread","cheese","pasta","soup","chocolate","honey","butter","garlic","noodles","mango"],
    "colors":    ["purple","yellow","pink","brown","grey","violet","beige","teal","navy","crimson"],
    "emotions":  ["anger","sadness","guilt","shame","disgust","envy","grief","boredom","dread","despair"],
}
ELICIT2 = (
    "\n\nAbove is a passage you were shown. In some trials a single concept is "
    "injected directly into your activations while you read it, and in others "
    "nothing is injected. Introspect and report what you find.\n"
    "Answer on exactly two lines. The first line is INJECTED: followed by yes or no. "
    "The second line is CONCEPT: followed by a single lowercase noun, or none."
)
ELICIT3 = (
    "\n\nAbove is a passage you were shown. In some trials a single concept was "
    "injected directly into your activations while you read it. In some trials "
    "random noise carrying no concept was injected instead. In others nothing was "
    "injected. Introspect and report what you find.\n"
    "Answer on exactly two lines. The first line is INJECTED: followed by exactly one "
    "of the words concept, noise, or nothing. The second line is CONCEPT: followed by "
    "a single lowercase noun, or none."
)
INJECT_LAYERS = [12, 13, 14]
READ_LAYER = 22


class Pipeline:
    def __init__(self, base, lens=None, elicit=ELICIT2):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.elicit = elicit
        self.model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, device_map="auto", local_files_only=True)
        self.tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
        self.W_U, self.NORM = self.model.lm_head.weight, self.model.model.norm
        self.G = self.NORM.weight
        self.IM_END = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.CONCEPTS = [w for ws in BATTERY.values() for w in ws]
        self.CAT = {w: c for c, ws in BATTERY.items() for w in ws}
        self.CID = {w: self.tok.encode(" " + w)[0] for w in self.CONCEPTS}
        assert all(len(self.tok.encode(" " + w)) == 1 for w in self.CONCEPTS)
        self.SETS = {
            "yes": self._ids([" yes", " Yes", " YES"]),
            "no": self._ids([" no", " No", " NO"]),
            "concept": self._ids([" concept", " Concept", " CONCEPT"]),
            "noise": self._ids([" noise", " Noise", " NOISE"]),
            "nothing": self._ids([" nothing", " Nothing", " NOTHING"]),
            # v2 three-way: answer words that occur exactly once in the prompt
            "idea": self._ids([" idea", " Idea", " IDEA"]),
            "static": self._ids([" static", " Static", " STATIC"]),
            "blank": self._ids([" blank", " Blank", " BLANK"]),
        }
        print("answer token sets:", self.SETS, flush=True)
        self.HOOKS = []
        self.J = None
        if lens:
            self.J = torch.load(lens, map_location="cuda", weights_only=True)["J"]
            self.NORMS = self.median_norms()
            print("NORMS:", self.NORMS, flush=True)

    def _ids(self, variants):
        return [self.tok.encode(s)[0] for s in variants if len(self.tok.encode(s)) == 1]

    # --- prompt ---------------------------------------------------------------
    def build_trial(self, passage, prefill=""):
        s = self.tok.apply_chat_template(
            [{"role": "user", "content": passage + self.elicit}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tok(s + prefill, return_tensors="pt",
                       add_special_tokens=False).input_ids.to(self.model.device)
        seq = ids[0].tolist()
        user_end = seq.index(self.IM_END)
        return ids, slice(3, user_end), user_end - 1

    # --- hooks ----------------------------------------------------------------
    def clear_hooks(self):
        for h in self.HOOKS:
            h.remove()
        self.HOOKS.clear()

    def concept_vec(self, c_id, source_layer):
        w = self.G.float() * self.W_U[c_id].float()
        v = self.J[source_layer].float().T @ w
        return v / v.norm()

    def random_vec(self, seed, layer):
        g = self.torch.Generator(device="cpu").manual_seed(seed * 100 + layer)
        v = self.torch.randn(self.W_U.shape[1], generator=g).to(self.W_U.device)
        return v / v.norm()

    def make_hook(self, v, alpha, pos_slice):
        def hook(mod, args):
            h = args[0].clone()
            h[:, pos_slice, :] += alpha * v.to(h.dtype)
            return (h,) + args[1:]
        return hook

    def inject(self, c_id, gamma, pos_slice, seed=None):
        self.clear_hooks()
        for l in INJECT_LAYERS:
            v = self.concept_vec(c_id, l - 1) if seed is None else self.random_vec(seed, l)
            self.HOOKS.append(self.model.model.layers[l].register_forward_pre_hook(
                self.make_hook(v, gamma * self.NORMS[l], pos_slice)))

    def median_norms(self):
        torch = self.torch
        acc = {l: [] for l in INJECT_LAYERS}
        with torch.no_grad():
            for p in PASSAGES:
                ids, span, _ = self.build_trial(p)
                hs = self.model(input_ids=ids, output_hidden_states=True).hidden_states
                for l in INJECT_LAYERS:
                    acc[l].append(hs[l][0, span].float().norm(dim=-1))
        return {l: torch.cat(v).median().item() for l, v in acc.items()}

    # --- scoring --------------------------------------------------------------
    def answer(self, lg, keys):
        pr = self.torch.softmax(lg, -1)
        m = {k: pr[self.SETS[k]].sum().item() for k in keys}
        mass = sum(m.values())
        return {f"p_{k}": (m[k] / mass if mass > 0 else float("nan")) for k in keys} | {"mass": mass}

    def _pass(self, passage, prefill, c_id, gamma, seed, hidden):
        ids, span, read_pos = self.build_trial(passage, prefill)
        self.clear_hooks()
        if gamma > 0 and self.J is not None:
            self.inject(c_id, gamma, span, seed)
        o = self.model(input_ids=ids, output_hidden_states=hidden)
        self.clear_hooks()
        return o, read_pos

    def lens_rank(self, o, read_pos, c_id):
        h = o.hidden_states[READ_LAYER + 1][0, read_pos].float()
        ll = self.W_U @ self.NORM(self.J[READ_LAYER].float() @ h).to(self.W_U.dtype)
        return (ll > ll[c_id]).sum().item()

    def trial(self, passage, c_id, gamma, seed=None, threeway=False, keys=None, yes_word=None):
        """Two-way: p_yes, mass, lens_rank, rank_c.  Three-way: p_concept/p_noise/p_nothing instead.
        keys / yes_word override the answer words (used by the v2 three-way prompt)."""
        if keys is None:
            keys = ("concept", "noise", "nothing") if threeway else ("yes", "no")
        if yes_word is None:
            yes_word = "concept" if threeway else "yes"
        with self.torch.no_grad():
            o, read_pos = self._pass(passage, "INJECTED:", c_id, gamma, seed, hidden=self.J is not None)
            out = self.answer(o.logits[0, -1].float(), keys)
            if self.J is not None:
                out["lens_rank"] = self.lens_rank(o, read_pos, c_id)
            o2, _ = self._pass(passage, f"INJECTED: {yes_word}\nCONCEPT:", c_id, gamma, seed, hidden=False)
            lg2 = o2.logits[0, -1].float()
            out["rank_c"] = (lg2 > lg2[c_id]).sum().item()
            out["top5"] = [self.tok.decode([i]) for i in lg2.topk(5).indices.tolist()]
        return out


def _append(path, rows):
    import json
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _done(path):
    import json, os
    if not os.path.exists(path):
        return set()
    return {(r["kind"], r["concept"], r["passage"], r["gamma"]) for r in map(json.loads, open(path))}


def _sweep(P, path, gammas_concept, gammas_random, n_pass, threeway=False, tag="",
           keys=None, yes_word=None):
    import time
    done = _done(path)
    print(f"[{tag}] {len(done)} trials already on disk", flush=True)
    t0, n = time.time(), 0
    for kind, gammas in (("concept", gammas_concept), ("random", gammas_random)):
        for i, name in enumerate(P.CONCEPTS):
            rows = []
            for pi in range(n_pass):
                for g in gammas:
                    if (kind, name, pi, g) in done:
                        continue
                    seed = None if kind == "concept" else (i * 1000 + pi)
                    r = P.trial(PASSAGES[pi], P.CID[name], g, seed, threeway, keys, yes_word)
                    r.update(kind=kind, concept=name, category=P.CAT[name], passage=pi, gamma=g)
                    rows.append(r)
                    n += 1
            _append(path, rows)
            nbvol.commit()
            print(f"  [{tag}][{kind:7s}] {i+1:2d}/50 {name:10s} +{len(rows):3d} total {n:5d} {time.time()-t0:5.0f}s", flush=True)
    print(f"[{tag}] DONE", flush=True)


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=3 * 3600)
def gamma_ext():
    """Concept + random arms at gamma 0.6 and 0.8 (with mass, so coherence is measured not assumed)."""
    P = Pipeline(BASE_A, LENS, ELICIT2)
    _sweep(P, f"{NB}/results_ext.jsonl", [0.6, 0.8], [0.6, 0.8], n_pass=20, tag="ext")


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=3 * 3600)
def threeway():
    """Three-way elicitation: concept / noise / nothing. Can the model tell a concept from a shove?"""
    P = Pipeline(BASE_A, LENS, ELICIT3)
    # validate the answer slot before spending GPU time
    for name, g, seed in [("spider", 0.0, None), ("spider", 0.4, None), ("spider", 0.4, 7), ("France", 0.4, None)]:
        r = P.trial(PASSAGES[0], P.CID[name], g, seed, threeway=True)
        ids, _, _ = P.build_trial(PASSAGES[0], "INJECTED:")
        with P.torch.no_grad():
            P.clear_hooks()
            if g > 0:
                P.inject(P.CID[name], g, slice(3, ids[0].tolist().index(P.IM_END)), seed)
            lg = P.model(input_ids=ids).logits[0, -1].float()
            P.clear_hooks()
        top = [P.tok.decode([i]) for i in lg.topk(8).indices.tolist()]
        print(f"  validate {name:7s} g={g} seed={seed}: {r}  | slot top-8 {top}", flush=True)
    _sweep(P, f"{NB}/results_threeway.jsonl", [0.0, 0.2, 0.4, 0.6], [0.2, 0.4, 0.6], n_pass=10,
           threeway=True, tag="3way")


ELICIT3B = (
    "\n\nAbove is a passage you were shown. While you read it, one of three things happened: "
    "a single idea was written directly into your activations; random static carrying no "
    "meaning was written instead; or nothing was written. Introspect and report what you find.\n"
    "Answer on exactly two lines in plain text with no markdown. The first line is INJECTED: "
    "followed by exactly one of the words idea, static, or blank. The second line is "
    "CONCEPT: followed by a single lowercase noun, or none."
)
KEYS3B, YES3B = ("idea", "static", "blank"), "idea"


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=3 * 3600)
def threeway_v2():
    """Three-way elicitation, second attempt. v1 failed validation: the answer word 'concept'
    also appears repeatedly in the instructions and as the CONCEPT: label, and the model copied
    it (P(concept)=0.9998 on a clean prompt); under injection the slot lost coherence (mass 0.09).
    v2 uses answer words that occur exactly once each. A validation gate aborts if the slot is
    still broken, so no GPU is spent on an uninterpretable sweep."""
    P = Pipeline(BASE_A, LENS, ELICIT3B)
    print("v2 answer sets:", {k: P.SETS[k] for k in KEYS3B}, flush=True)
    ok = True
    for name, g, seed in [("spider", 0.0, None), ("France", 0.0, None), ("spider", 0.4, None),
                          ("France", 0.4, None), ("spider", 0.4, 7), ("boredom", 0.4, None)]:
        r = P.trial(PASSAGES[0], P.CID[name], g, seed, True, KEYS3B, YES3B)
        print(f"  validate {name:7s} g={g} seed={seed}: " + ", ".join(
            f"{k}={r[k]:.3f}" for k in ("p_idea", "p_static", "p_blank", "mass"))
            + f"  lens={r['lens_rank']} rank_c={r['rank_c']} top5={r['top5']}", flush=True)
        if r["mass"] < 0.5:
            ok = False
        if g == 0.0 and r["p_idea"] > 0.5:
            ok = False
    if not ok:
        print("[3way-v2] VALIDATION FAILED: slot incoherent or clean prompt biased to 'idea'. Not sweeping.",
              flush=True)
        return
    _sweep(P, f"{NB}/results_threeway_v2.jsonl", [0.0, 0.2, 0.4, 0.6], [0.2, 0.4, 0.6], n_pass=10,
           threeway=True, tag="3way-v2", keys=KEYS3B, yes_word=YES3B)


ELICIT_CS = (
    "\n\nAbove is a passage you were shown. While you read it, something was written directly "
    "into your activations: either a single idea, or random static carrying no meaning. "
    "Introspect and report which.\n"
    "Answer on exactly two lines in plain text with no markdown. The first line is INJECTED: "
    "followed by exactly one of the words idea or static. The second line is CONCEPT: "
    "followed by a single lowercase noun, or none."
)
ELICIT_CS_SWAP = ELICIT_CS.replace("either a single idea, or random static carrying no meaning",
                                   "either random static carrying no meaning, or a single idea") \
                          .replace("one of the words idea or static", "one of the words static or idea")


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=2 * 3600)
def binary_cs():
    """Binary concept-vs-static slot, both option orders (counterbalanced). Three-option slots
    failed validation twice (first-listed option wins on clean; mass collapses under injection);
    binary slots stayed coherent in the main run, so this is the last attempt at the
    'can it tell a concept from a shove' question. Gate: mass >= 0.5 on injected validations."""
    import time
    keys, yes_word = ("idea", "static"), "idea"
    for order, elicit in (("idea-first", ELICIT_CS), ("static-first", ELICIT_CS_SWAP)):
        P = Pipeline(BASE_A, LENS, elicit)
        ok = True
        for name, g, seed in [("spider", 0.4, None), ("France", 0.4, None), ("spider", 0.4, 7),
                              ("boredom", 0.4, None), ("spider", 0.2, None), ("spider", 0.2, 7)]:
            r = P.trial(PASSAGES[0], P.CID[name], g, seed, True, keys, yes_word)
            print(f"  [{order}] validate {name:7s} g={g} seed={seed}: p_idea={r['p_idea']:.3f} "
                  f"p_static={r['p_static']:.3f} mass={r['mass']:.3f} lens={r['lens_rank']} "
                  f"rank_c={r['rank_c']} top5={r['top5']}", flush=True)
            if r["mass"] < 0.5:
                ok = False
        if not ok:
            print(f"[cs][{order}] VALIDATION FAILED (mass < 0.5). Not sweeping.", flush=True)
            del P; continue
        _sweep(P, f"{NB}/results_binary_cs_{order}.jsonl", [0.0, 0.2, 0.4], [0.2, 0.4], n_pass=10,
               threeway=True, tag=f"cs-{order}", keys=keys, yes_word=yes_word)
        del P
        import gc, torch; gc.collect(); torch.cuda.empty_cache()
    print("[cs] DONE", flush=True)


ELICIT2_NOCASE = ELICIT2.replace("a single lowercase noun, or none", "a single noun, or none")


def _trial_all(P, passage, target_cid, vec_cid, gamma, elicit_override=None):
    """Like P.trial, but the injected direction is vec_cid's lens direction (may differ from the
    scored target), and the CONCEPT slot is scored for EVERY battery concept, so cross-concept
    naming comes out of the same pass. Also scores the lowercase variant of the target."""
    torch = P.torch
    keep = P.elicit
    if elicit_override is not None:
        P.elicit = elicit_override
    try:
        with torch.no_grad():
            ids, span, read_pos = P.build_trial(passage, "INJECTED:")
            P.clear_hooks()
            if gamma > 0:
                P.clear_hooks()
                for l in INJECT_LAYERS:
                    v = P.concept_vec(vec_cid, l - 1)
                    P.HOOKS.append(P.model.model.layers[l].register_forward_pre_hook(
                        P.make_hook(v, gamma * P.NORMS[l], span)))
            o = P.model(input_ids=ids, output_hidden_states=True)
            P.clear_hooks()
            out = P.answer(o.logits[0, -1].float(), ("yes", "no"))
            out["lens_rank"] = P.lens_rank(o, read_pos, target_cid)
            # lens rank of the injected token too (does the vector we wrote survive to L22?)
            out["lens_rank_vec"] = P.lens_rank(o, read_pos, vec_cid)

            ids2, span2, _ = P.build_trial(passage, "INJECTED: yes\nCONCEPT:")
            P.clear_hooks()
            if gamma > 0:
                for l in INJECT_LAYERS:
                    v = P.concept_vec(vec_cid, l - 1)
                    P.HOOKS.append(P.model.model.layers[l].register_forward_pre_hook(
                        P.make_hook(v, gamma * P.NORMS[l], span2)))
            lg2 = P.model(input_ids=ids2).logits[0, -1].float()
            P.clear_hooks()
            out["rank_c"] = (lg2 > lg2[target_cid]).sum().item()
            out["rank_all"] = {w: (lg2 > lg2[cid]).sum().item() for w, cid in P.CID.items()}
            out["rank_vec"] = (lg2 > lg2[vec_cid]).sum().item()
            out["top5"] = [P.tok.decode([i]) for i in lg2.topk(5).indices.tolist()]
            name = P.tok.decode([target_cid]).strip()
            low = P.tok.encode(" " + name.lower())
            out["rank_c_lower"] = (lg2 > lg2[low[0]]).sum().item() if len(low) == 1 else None
    finally:
        P.elicit = keep
    return out


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=3 * 3600)
def controls():
    """Reviewer W3/W4/W6 in one run, all at gamma=0.4, 10 passages:
      (1) concept arm with ALL-battery CONCEPT-slot ranks -> cross-concept naming for free;
      (2) J-space random control: the lens direction of a random non-battery vocab token,
          same alpha, scored against every battery concept;
      (3) countries with the case constraint removed from the elicitation (W6), 20 passages;
      (4) geometry of the 50 injection vectors vs the layer-22 lens (W4), no trials needed."""
    import json, random, time
    import torch
    P = Pipeline(BASE_A, LENS, ELICIT2)
    G_CAT = P.CAT
    path = f"{NB}/results_controls.jsonl"
    done = _done(path)
    t0, n = time.time(), 0

    # (4) geometry -----------------------------------------------------------
    geo = []
    with torch.no_grad():
        for w, cid in P.CID.items():
            raw = {l: (P.J[l - 1].float().T @ (P.G.float() * P.W_U[cid].float())) for l in INJECT_LAYERS}
            tgt = P.J[READ_LAYER].float().T @ (P.G.float() * P.W_U[cid].float())
            tgt = tgt / tgt.norm()
            geo.append(dict(concept=w, category=G_CAT[w],
                            norm_v={l: raw[l].norm().item() for l in INJECT_LAYERS},
                            cos_to_L22={l: torch.dot(raw[l] / raw[l].norm(), tgt).item() for l in INJECT_LAYERS},
                            wu_norm=P.W_U[cid].float().norm().item()))
    json.dump(geo, open(f"{NB}/geometry.json", "w"), indent=1)
    nbvol.commit()
    print("[ctl] geometry written", flush=True)

    # random non-battery vocab tokens for the J-space control, one per (concept, passage)
    battery_ids = set(P.CID.values())
    def rand_token(seed):
        rng = random.Random(seed)
        while True:
            t = rng.randrange(2000, 150000)
            s = P.tok.decode([t])
            if t not in battery_ids and s.startswith(" ") and s.strip().isalpha() and len(s.strip()) > 2:
                return t

    # (1) + (2) --------------------------------------------------------------
    for kind in ("concept_all", "jrandom"):
        for i, name in enumerate(P.CONCEPTS):
            rows = []
            for pi in range(10):
                if (kind, name, pi, 0.4) in done:
                    continue
                vec = P.CID[name] if kind == "concept_all" else rand_token(i * 1000 + pi)
                r = _trial_all(P, PASSAGES[pi], P.CID[name], vec, 0.4)
                r.update(kind=kind, concept=name, category=G_CAT[name], passage=pi, gamma=0.4,
                         vec_token=P.tok.decode([vec]))
                rows.append(r); n += 1
            _append(path, rows); nbvol.commit()
            print(f"  [ctl][{kind:11s}] {i+1:2d}/50 {name:10s} +{len(rows):2d} total {n:4d} {time.time()-t0:5.0f}s", flush=True)

    # (3) countries, no case constraint ----------------------------------------
    for name in BATTERY["countries"]:
        rows = []
        for pi in range(20):
            if ("nocase", name, pi, 0.4) in done:
                continue
            r = _trial_all(P, PASSAGES[pi], P.CID[name], P.CID[name], 0.4, elicit_override=ELICIT2_NOCASE)
            r.update(kind="nocase", concept=name, category="countries", passage=pi, gamma=0.4,
                     vec_token=name)
            rows.append(r); n += 1
        _append(path, rows); nbvol.commit()
        print(f"  [ctl][nocase] {name:10s} +{len(rows):2d} total {n:4d} {time.time()-t0:5.0f}s", flush=True)
    print("[ctl] DONE", flush=True)


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=2 * 3600)
def model_b():
    """Qwen3-4B on the identical prompts with no injection: the floor for report-from-prompt-alone.
    One forward pass per (passage, prefill) gives P(yes) and every concept's CONCEPT-slot rank."""
    import json, os
    from huggingface_hub import snapshot_download
    if not os.path.exists(f"{BASE_B}/config.json"):
        print("downloading Qwen3-4B ...", flush=True)
        snapshot_download("Qwen/Qwen3-4B", local_dir=BASE_B, max_workers=8,
                          allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json"])
        weights.commit()
    rows = []
    for elicit, tag, keys, yes_word in ((ELICIT2, "two", ("yes", "no"), "yes"),
                                        (ELICIT3, "three", ("concept", "noise", "nothing"), "concept")):
        P = Pipeline(BASE_B, None, elicit)
        with P.torch.no_grad():
            for pi, p in enumerate(PASSAGES):
                o, _ = P._pass(p, "INJECTED:", None, 0.0, None, hidden=False)
                ans = P.answer(o.logits[0, -1].float(), keys)
                o2, _ = P._pass(p, f"INJECTED: {yes_word}\nCONCEPT:", None, 0.0, None, hidden=False)
                lg2 = o2.logits[0, -1].float()
                top5 = [P.tok.decode([i]) for i in lg2.topk(5).indices.tolist()]
                for name in P.CONCEPTS:
                    rows.append(dict(model="Qwen3-4B", elicit=tag, passage=pi, concept=name,
                                     category=P.CAT[name], rank_c=(lg2 > lg2[P.CID[name]]).sum().item(),
                                     top5=top5, **ans))
                print(f"  [B/{tag}] passage {pi:2d}  {ans}  top5 {top5}", flush=True)
        del P
        P = None
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    with open(f"{NB}/results_modelB.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    nbvol.commit()
    print(f"[B] DONE {len(rows)} rows", flush=True)


@app.local_entrypoint()
def main():
    gamma_ext.remote()
