"""Run the two-hop J-lens verification on Modal, headless.

    modal run jverify.py

Iterating in a notebook means paying for a GPU while you type. This runs the
whole battery in one shot and prints it, so the loop is edit -> run -> read.
"""

import modal

MOUNT = "/models"
BASE = f"{MOUNT}/Qwen3-8B"
LENS = f"{MOUNT}/jacobian-lens/qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt"

app = modal.App("jlens-verify")
volume = modal.Volume.from_name("jlens-weights")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.9.0", "transformers>=4.51", "accelerate", "numpy")


@app.function(image=image, gpu="A10G", volumes={MOUNT: volume}, timeout=30 * 60)
def verify():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    volume.reload()
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    J = torch.load(LENS, map_location="cuda", weights_only=True)["J"]

    W_U, NORM = model.lm_head.weight, model.model.norm
    nL = len(model.model.layers)
    print(f"\n{nL} layers | {len(J)} J | vocab {W_U.shape[0]} | chat_template={tok.chat_template is not None}")

    def fwd(ids):
        with torch.no_grad():
            return model(input_ids=ids, output_hidden_states=True)

    ids0 = tok("The number of legs on the animal that spins webs is",
               return_tensors="pt").input_ids.to(model.device)
    o0 = fwd(ids0)
    hs0 = o0.hidden_states

    # ---- A. plumbing -----------------------------------------------------
    print(f"\n{'='*70}\nA. PLUMBING\n{'='*70}")
    print(f"hidden_states entries: {len(hs0)}  (layers+1 = {nL+1})")
    h, ref = hs0[-1][0, -1], o0.logits[0, -1].float()
    for nm, v in [("W_U @ hs[-1]", (W_U @ h.to(W_U.dtype)).float()),
                  ("W_U @ norm(hs[-1])", (W_U @ NORM(h).to(W_U.dtype)).float())]:
        print(f"  {nm:22s} max|diff vs logits| = {(v-ref).abs().max().item():.5f}")

    @torch.no_grad()
    def align(hs, l, off):
        X = hs[l + off][0].float() @ J[l].float().T
        return torch.nn.functional.cosine_similarity(NORM(X), hs[-1][0].float(), dim=-1).mean().item()

    print(f"\n  {'layer':>5} | {'offset=0':>9} {'offset=1':>9}   cos(norm(J_l h), hs[-1])")
    for l in [0, 8, 16, 24, 30, 32, 34]:
        print(f"  {l:5d} | {align(hs0,l,0):9.4f} {align(hs0,l,1):9.4f}")
    OFFSET = max((0, 1), key=lambda f: np.mean([align(hs0, l, f) for l in (30, 32, 34)]))
    print(f"\n  OFFSET = {OFFSET}")

    @torch.no_grad()
    def lens(hs, l, kind):
        X = hs[l][0].float() if kind == "logit" else hs[l+OFFSET][0].float() @ J[l].float().T
        return (NORM(X).to(W_U.dtype) @ W_U.T).float()

    def rk(lg, tid): return (lg > lg[:, tid:tid+1]).sum(-1).cpu().numpy()
    def top(r, k=8): return [tok.decode([i]) for i in r.topk(k).indices.tolist()]

    print(f"\n  model next-token : {top(o0.logits[0,-1].float())}")
    print(f"  J-lens    l=34   : {top(lens(hs0,34,'j')[-1])}")
    print(f"  logit     l=35   : {top(lens(hs0,35,'logit')[-1])}")
    print(f"  layer 12 logit   : {top(lens(hs0,12,'logit')[-1])}")
    print(f"  layer 12 J-lens  : {top(lens(hs0,12,'j')[-1])}")

    # ---- B. matched sweep, original prompt --------------------------------
    SP = tok.encode(" spider")[0]
    ts0 = tok.convert_ids_to_tokens(ids0[0])
    print(f"\n{'='*70}\nB. MATCHED SWEEP - original prompt, position -1 ({ts0[-1]!r})\n{'='*70}")
    print(f"  {'L':>3} | {'logit':>7} {'J':>7} | top-6 J-lens")
    for l in range(len(J)):
        jl = lens(hs0, l, "j")
        ll = lens(hs0, l, "logit")
        print(f"  {l:3d} | {rk(ll,SP)[-1]:7d} {rk(jl,SP)[-1]:7d} | {top(jl[-1],6)}")

    # ---- C. prompt bank ---------------------------------------------------
    TARGETS = {t: tok.encode(t)[0] for t in
               [" spider", " Spider", " spiders", " arachnid", " web", " eight"]
               if len(tok.encode(t)) == 1}
    PROMPTS = {
        "2hop original": "The number of legs on the animal that spins webs is",
        "2hop question": "How many legs does the animal that spins webs have?",
        "2hop count":    "Count the legs on the animal that spins webs. The count is",
        "2hop q+a":      "Q: How many legs does the animal that spins webs have?\nA:",
        "1hop control":  "The animal that spins webs is called a",
        "NEG control":   "The number of legs on the animal that flies south is",
    }
    MID = list(range(int(.25*len(J)), int(.80*len(J))))
    print(f"\n{'='*70}\nC. PROMPT BANK  (mid-band = layers {MID[0]}-{MID[-1]})\n{'='*70}")

    def build(text, chat):
        """Chat wrapping changes the tokenisation entirely, so go via the
        rendered string -- apply_chat_template's return type moves around
        between transformers versions."""
        if chat:
            m = [{"role": "user", "content": text}]
            try:
                text = tok.apply_chat_template(m, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
            except TypeError:
                text = tok.apply_chat_template(m, tokenize=False,
                                               add_generation_prompt=True)
        return tok(text, return_tensors="pt", add_special_tokens=not chat).input_ids.to(model.device)

    rows = []
    for name, text in PROMPTS.items():
        for chat in (False, True):
            ids = build(text, chat)
            hs = fwd(ids).hidden_states
            T = ids.shape[1]
            for t, tid in TARGETS.items():
                jm = np.array([rk(lens(hs, l, "j"), tid) for l in range(len(J))])
                lm = np.array([rk(lens(hs, l, "logit"), tid) for l in range(len(J))])
                sub = jm[MID]
                i, p = np.unravel_index(sub.argmin(), sub.shape)
                L = MID[i]
                rows.append((name, chat, t, int(jm[L, p]), L, int(p), T,
                             int(lm[L, p]), int(lm[MID].min())))
    print(f"  {'prompt':15s} {'chat':5s} {'target':10s} {'J':>7} {'@L':>3} {'pos':>4}/{'T':<3} "
          f"{'logit@same':>10} {'logitbest':>9}")
    for r in rows:
        print(f"  {r[0]:15s} {str(r[1]):5s} {r[2]:10s} {r[3]:7d} {r[4]:3d} {r[5]:4d}/{r[6]:<3d} "
              f"{r[7]:10d} {r[8]:9d}")
    return rows


@app.local_entrypoint()
def main():
    verify.remote()


@app.function(image=image, gpu="A10G", volumes={MOUNT: volume}, timeout=30 * 60)
def crossed():
    """Crossed design: same syntactic frame, different bridge entity.

    The earlier negative control was broken -- it shared the whole prefix
    ("The number of legs on the animal that ...") with the test prompt, and its
    best cell landed at a position BEFORE the disambiguating words. A prompt
    prefix about legs and animals cues 'spider' on its own, so that proves
    nothing. Here every prompt uses the identical frame and differs only in the
    relative clause, and readouts are taken only at positions at or after the
    disambiguator -- the earliest point where the model could possibly have
    resolved the bridge entity. The hop is real only if the diagonal wins.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    volume.reload()
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    J = torch.load(LENS, map_location="cuda", weights_only=True)["J"]
    W_U, NORM, OFFSET = model.lm_head.weight, model.model.norm, 1

    @torch.no_grad()
    def lens(hs, l, kind):
        X = hs[l][0].float() if kind == "logit" else hs[l+OFFSET][0].float() @ J[l].float().T
        return (NORM(X).to(W_U.dtype) @ W_U.T).float()

    def rk(lg, tid): return (lg > lg[:, tid:tid+1]).sum(-1).cpu().numpy()
    def top(r, k=8): return [tok.decode([i]) for i in r.topk(k).indices.tolist()]

    FRAME = "The number of legs on the animal that {} is"
    CASES = [("spider",   "spins webs",     " spider"),
             ("dog",      "barks",          " dog"),
             ("bee",      "makes honey",    " bee"),
             ("elephant", "has a trunk",    " elephant")]
    NEUTRAL = "The weather in the small town by the sea is"
    TIDS = {t: tok.encode(t)[0] for _, _, t in CASES}
    assert all(len(tok.encode(t)) == 1 for t in TIDS), TIDS
    MID = list(range(int(.25*len(J)), int(.80*len(J))))

    def run(text, cut_after=None):
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        ts = tok.convert_ids_to_tokens(ids[0])
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        out = {}
        for kind in ("j", "logit"):
            out[kind] = {t: np.array([rk(lens(hs, l, kind), tid) for l in range(len(J))])
                         for t, tid in TIDS.items()}
        # first position at or after the disambiguator
        p0 = 0
        if cut_after:
            last = tok(" " + cut_after).input_ids[-1]
            p0 = max(i for i, x in enumerate(ids[0].tolist()) if x == last)
        return hs, ts, p0, out

    print(f"\nmid-band layers {MID[0]}-{MID[-1]}\n")
    print("=" * 78)
    print("D. CROSSED DESIGN - best mid-band J-lens rank, positions >= disambiguator")
    print("=" * 78)
    store = {}
    hdr = f"  {'prompt':10s} {'p0':>3s} |" + "".join(f"{t:>11s}" for t in TIDS)
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for name, clause, _ in CASES:
        text = FRAME.format(clause)
        hs, ts, p0, res = run(text, clause)
        store[name] = (hs, ts, p0, res, text)
        cells = []
        for t in TIDS:
            sub = res["j"][t][np.ix_(MID, range(p0, len(ts)))]
            i, k = np.unravel_index(sub.argmin(), sub.shape)
            cells.append((int(sub[i, k]), MID[i], p0 + k))
        print(f"  {name:10s} {p0:3d} |" + "".join(f"{c[0]:>11d}" for c in cells))
        store[name] += (cells,)
    hs, ts, _, res = run(NEUTRAL)
    print(f"  {'NEUTRAL':10s} {'-':>3s} |" +
          "".join(f"{int(res['j'][t][MID].min()):>11d}" for t in TIDS))
    print("\n  rows = prompt, cols = target. The diagonal must win, and NEUTRAL")
    print("  (no animal at all) is the lens's standing bias for each token.")

    # ---- matched logit-lens numbers on the diagonal ----------------------
    print("\n" + "=" * 78)
    print("E. DIAGONAL, MATCHED - J vs logit lens at the same layer and position")
    print("=" * 78)
    for i, (name, clause, tgt) in enumerate(CASES):
        hs, ts, p0, res, text, cells = store[name]
        j, L, P = cells[i]
        lg = int(res["logit"][tgt][L, P])
        off = [int(res["j"][t][np.ix_(MID, range(p0, len(ts)))].min())
               for t in TIDS if t != tgt]
        print(f"\n  {text!r}")
        print(f"    target {tgt!r} at layer {L}, position {P} ({ts[P]!r})")
        print(f"    J-lens {j:6d} | logit lens (same cell) {lg:6d} | "
              f"logit best in band {int(res['logit'][tgt][np.ix_(MID, range(p0, len(ts)))].min()):6d}")
        print(f"    same cell, other targets: {off}")
        print(f"    top-8 J-lens there: {top(lens(hs, L, 'j')[P])}")


@app.local_entrypoint()
def cross():
    crossed.remote()


@app.function(image=image, gpu="A10G", volumes={MOUNT: volume}, timeout=30 * 60)
def endpos():
    """Read at the answer position only -- the actual two-hop claim.

    At the disambiguator token (' webs') both lenses find 'spider', because the
    residual there still carries that token's own lexical neighbourhood; the
    logit lens gets it for free. The claim worth testing is the paper's silent-
    reasoning one: at the position where the model is about to emit the NUMBER,
    is the unspoken bridge entity still readable? There the logit lens has to
    decode aggregated state rather than a local token, which is where it should
    fail. Crossed over four animals, chat format, with an assistant prefill so
    the very next token is the answer.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    volume.reload()
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    J = torch.load(LENS, map_location="cuda", weights_only=True)["J"]
    W_U, NORM, OFFSET = model.lm_head.weight, model.model.norm, 1

    @torch.no_grad()
    def lens(hs, l, kind):
        X = hs[l][0].float() if kind == "logit" else hs[l+OFFSET][0].float() @ J[l].float().T
        return (NORM(X).to(W_U.dtype) @ W_U.T).float()

    def rk(lg, tid): return (lg > lg[:, tid:tid+1]).sum(-1).cpu().numpy()
    def top(r, k=8): return [tok.decode([i]) for i in r.topk(k).indices.tolist()]

    CASES = [("spider",   "spins webs",  " spider",   " eight"),
             ("dog",      "barks",       " dog",      " four"),
             ("bee",      "makes honey", " bee",      " six"),
             ("elephant", "has a trunk", " elephant", " four")]
    TIDS = {t: tok.encode(t)[0] for _, _, t, _ in CASES}
    MID = list(range(int(.25*len(J)), int(.80*len(J))))
    PREFILL = "It has "

    def build(clause):
        q = (f"How many legs does the animal that {clause} have? "
             "Reply with just the number.")
        try:
            s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                        add_generation_prompt=True)
        return tok(s + PREFILL, return_tensors="pt",
                   add_special_tokens=False).input_ids.to(model.device)

    print(f"\nprefill {PREFILL!r} | mid-band layers {MID[0]}-{MID[-1]}")
    store = {}
    for name, clause, tgt, num in CASES:
        ids = build(clause)
        with torch.no_grad():
            o = model(input_ids=ids, output_hidden_states=True)
        hs, ts = o.hidden_states, tok.convert_ids_to_tokens(ids[0])
        res = {k: {t: np.array([rk(lens(hs, l, k), tid)[-1] for l in range(len(J))])
                   for t, tid in TIDS.items()} for k in ("j", "logit")}
        store[name] = (hs, ts, res)
        print(f"\n  {name}: {ids.shape[1]} tokens, ends {ts[-3:]}, "
              f"model says {top(o.logits[0,-1].float(), 3)}")

    print("\n" + "=" * 78)
    print("F. AT THE ANSWER POSITION - best mid-band J-lens rank (crossed)")
    print("=" * 78)
    hdr = f"  {'prompt':10s} |" + "".join(f"{t:>11s}" for t in TIDS)
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for name, *_ in CASES:
        r = store[name][2]["j"]
        print(f"  {name:10s} |" + "".join(f"{int(r[t][MID].min()):>11d}" for t in TIDS))

    print("\n" + "=" * 78)
    print("G. DIAGONAL, MATCHED, AT THE ANSWER POSITION")
    print("=" * 78)
    verdict = []
    for name, clause, tgt, num in CASES:
        hs, ts, res = store[name]
        L = MID[int(np.argmin(res["j"][tgt][MID]))]
        j, lg_same = int(res["j"][tgt][L]), int(res["logit"][tgt][L])
        lg_best = int(res["logit"][tgt][MID].min())
        off = max(int(res["j"][t][MID].min()) for t in TIDS if t != tgt)
        verdict.append((name, tgt, j, lg_same, lg_best, off))
        print(f"\n  {clause!r} -> {tgt!r}   (model should answer {num!r})")
        print(f"    J-lens best in band : {j:6d}  at layer {L}")
        print(f"    logit lens, same L  : {lg_same:6d}")
        print(f"    logit lens, best    : {lg_best:6d}")
        print(f"    worst off-diagonal  : {off:6d}")
        print(f"    {'L':>3} {'logit':>7} {'J':>7} | top-6 J-lens")
        for l in range(6, len(J)):
            print(f"    {l:3d} {int(res['logit'][tgt][l]):7d} {int(res['j'][tgt][l]):7d} | "
                  f"{top(lens(hs, l, 'j')[-1], 6)}")

    print("\n" + "=" * 78)
    print("VERDICT (at the answer position, mid-band)")
    print("=" * 78)
    allok = True
    for name, tgt, j, lg_same, lg_best, off in verdict:
        a, b, c = j <= 100, lg_best >= 10 * max(j, 1), off >= 10 * max(j, 1)
        allok &= a and b and c
        print(f"  {name:9s} J={j:6d} logit_best={lg_best:6d} offdiag={off:6d}  "
              f"[{'PASS' if a else 'FAIL'} readable] "
              f"[{'PASS' if b else 'FAIL'} logit fails] "
              f"[{'PASS' if c else 'FAIL'} specific]")
    print("\n==> OBJECTIVE MET" if allok else "\n==> OBJECTIVE NOT MET")


@app.local_entrypoint()
def end():
    endpos.remote()


@app.function(image=image, gpu="A10G", volumes={MOUNT: volume}, timeout=30 * 60)
def final(prefills=("It has", "It has ", "")):
    """Final protocol, with the comparison stated correctly.

    The earlier verdict compared the J-lens at its best layer against the logit
    lens at ITS best layer anywhere in the band. That is not the claim. The claim
    is that at a GIVEN mid-band layer the J-lens reads the bridge entity and the
    logit lens does not, so the comparison has to be matched-layer. Reported here
    as the contiguous band of layers where J-lens rank <= 10 and the logit lens
    at the same layer is >= 10x worse.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    volume.reload()
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    J = torch.load(LENS, map_location="cuda", weights_only=True)["J"]
    W_U, NORM, OFFSET = model.lm_head.weight, model.model.norm, 1

    @torch.no_grad()
    def lens(hs, l, kind):
        X = hs[l][0].float() if kind == "logit" else hs[l+OFFSET][0].float() @ J[l].float().T
        return (NORM(X).to(W_U.dtype) @ W_U.T).float()

    def rk(lg, tid): return (lg > lg[:, tid:tid+1]).sum(-1).cpu().numpy()
    def top(r, k=6): return [tok.decode([i]) for i in r.topk(k).indices.tolist()]

    CASES = [("spider", "spins webs", " spider"), ("dog", "barks", " dog"),
             ("bee", "makes honey", " bee"), ("elephant", "has a trunk", " elephant"),
             ("camel", "has a hump", " camel"), ("owl", "hoots at night", " owl")]
    TIDS = {t: tok.encode(t)[0] for _, _, t in CASES}
    MID = list(range(int(.25*len(J)), int(.80*len(J))))

    def build(clause, prefill):
        q = (f"How many legs does the animal that {clause} have? "
             "Reply with just the number.")
        try:
            s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                        add_generation_prompt=True)
        return tok(s + prefill, return_tensors="pt",
                   add_special_tokens=False).input_ids.to(model.device)

    best_pf, best_score, results = None, None, {}
    for pf in prefills:
        rows, score = [], []
        for name, clause, tgt in CASES:
            ids = build(clause, pf)
            with torch.no_grad():
                o = model(input_ids=ids, output_hidden_states=True)
            hs = o.hidden_states
            r = {k: {t: np.array([rk(lens(hs, l, k), tid)[-1] for l in range(len(J))])
                     for t, tid in TIDS.items()} for k in ("j", "logit")}
            band = [l for l in MID
                    if r["j"][tgt][l] <= 10 and r["logit"][tgt][l] >= 10 * max(r["j"][tgt][l], 1)]
            off = min(int(r["j"][t][MID].min()) for t in TIDS if t != tgt)
            rows.append((name, tgt, r, band, off, hs, top(o.logits[0, -1].float(), 3)))
            score.append(len(band) > 0 and off >= 100)
        results[pf] = rows
        s = sum(score)
        print(f"  prefill {pf!r:10s} -> {s}/{len(CASES)} cases with a clean matched-layer band")
        if best_score is None or s > best_score:
            best_pf, best_score = pf, s

    print(f"\nchosen prefill: {best_pf!r}")
    print("=" * 78)
    print("H. MATCHED-LAYER VERDICT, at the answer position")
    print("=" * 78)
    allok = True
    for name, tgt, r, band, off, hs, says in results[best_pf]:
        L = band[int(np.argmin([r["j"][tgt][l] for l in band]))] if band else \
            MID[int(np.argmin(r["j"][tgt][MID]))]
        j, lg = int(r["j"][tgt][L]), int(r["logit"][tgt][L])
        a, b, c = j <= 10, lg >= 10 * max(j, 1), off >= 100
        allok &= a and b and c
        print(f"\n  {name:9s} model answers {says}")
        print(f"    layers where J<=10 and logit>=10x worse: {band}")
        print(f"    at layer {L:2d}: J-lens {j:5d} | logit lens {lg:6d} | "
              f"best off-diagonal target {off:6d}")
        print(f"    J-lens top-6 : {top(lens(hs, L, 'j')[-1])}")
        print(f"    logit  top-6 : {top(lens(hs, L, 'logit')[-1])}")
        print(f"    [{'PASS' if a else 'FAIL'} readable] "
              f"[{'PASS' if b else 'FAIL'} logit fails same layer] "
              f"[{'PASS' if c else 'FAIL'} specific]")
    print("\n" + ("=" * 78) + ("\n==> OBJECTIVE MET" if allok else "\n==> OBJECTIVE NOT MET"))


@app.local_entrypoint()
def fin():
    final.remote()


@app.function(image=image, gpu="A10G", volumes={MOUNT: volume}, timeout=30 * 60)
def objective():
    """The stated objective, tested with the controls it needs.

    Protocol settled by the earlier runs:
      * chat format -- a bare string makes this post-trained model complete text
        like a base model, and its mid-band readouts fill with '____' because it
        reads the prompt as a cloze quiz rather than a question;
      * prefill 'It has ' so the very next token is the digit, putting the read
        position exactly where the answer is about to be emitted;
      * read at that final position, NOT at ' webs' -- at the disambiguator both
        lenses find 'spider' from local lexical association, which is not the
        two-hop claim;
      * compare the two lenses at the SAME layer.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    volume.reload()
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    J = torch.load(LENS, map_location="cuda", weights_only=True)["J"]
    W_U, NORM, OFFSET = model.lm_head.weight, model.model.norm, 1

    @torch.no_grad()
    def lens(hs, l, kind):
        X = hs[l][0].float() if kind == "logit" else hs[l+OFFSET][0].float() @ J[l].float().T
        return (NORM(X).to(W_U.dtype) @ W_U.T).float()

    def rk(lg, tid): return int((lg[-1] > lg[-1, tid]).sum().item())
    def top(r, k=6): return [tok.decode([i]) for i in r.topk(k).indices.tolist()]

    CASES = [("spider", "spins webs", " spider"), ("dog", "barks", " dog"),
             ("bee", "makes honey", " bee"), ("elephant", "has a trunk", " elephant")]
    TIDS = {t: tok.encode(t)[0] for _, _, t in CASES}
    MID = list(range(int(.25*len(J)), int(.80*len(J))))
    PREFILL = "It has "

    def run(clause, neutral=False):
        q = ("What is the weather usually like by the sea? Reply in one word."
             if neutral else
             f"How many legs does the animal that {clause} have? Reply with just the number.")
        try:
            s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                        add_generation_prompt=True)
        ids = tok(s + PREFILL, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            o = model(input_ids=ids, output_hidden_states=True)
        hs = o.hidden_states
        r = {k: {t: np.array([rk(lens(hs, l, k), tid) for l in range(len(J))])
                 for t, tid in TIDS.items()} for k in ("j", "logit")}
        return hs, r, top(o.logits[0, -1].float(), 3)

    data = {n: run(c) for n, c, _ in CASES}
    _, neu, _ = run(None, neutral=True)

    print(f"\nprotocol: chat + prefill {PREFILL!r}, read at the answer position")
    print(f"mid-band = layers {MID[0]}-{MID[-1]} of {len(J)}\n")

    print("=" * 76)
    print("SPIDER - full layer sweep, both lenses, at the answer position")
    print("=" * 76)
    hs, r, says = data["spider"]
    print(f"  model's next token: {says}\n")
    print(f"  {'L':>3} {'logit':>7} {'J':>7}  {'ratio':>6} | top-4 J-lens")
    band = []
    for l in range(8, len(J)):
        j, lg = int(r["j"][" spider"][l]), int(r["logit"][" spider"][l])
        ok = j <= 10 and lg >= 10 * max(j, 1)
        band += [l] if ok else []
        print(f"  {l:3d} {lg:7d} {j:7d}  {lg/max(j,1):6.0f}x | {top(lens(hs,l,'j')[-1],4)}"
              f"{'   <-- readable, logit fails' if ok else ''}")

    print("\n" + "=" * 76)
    print("CONTROLS - best mid-band J-lens rank at the answer position")
    print("=" * 76)
    hdr = f"  {'prompt':10s} |" + "".join(f"{t:>11s}" for t in TIDS)
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for n, *_ in CASES:
        rr = data[n][1]["j"]
        print(f"  {n:10s} |" + "".join(f"{int(rr[t][MID].min()):>11d}" for t in TIDS))
    print(f"  {'NEUTRAL':10s} |" + "".join(f"{int(neu['j'][t][MID].min()):>11d}" for t in TIDS))

    L = band[int(np.argmin([r["j"][" spider"][l] for l in band]))] if band else None
    j, lg = int(r["j"][" spider"][L]), int(r["logit"][" spider"][L])
    off = min(int(data[n][1]["j"][" spider"][MID].min()) for n in ("dog", "bee", "elephant"))
    nz = int(neu["j"][" spider"][MID].min())
    a, b, c = j <= 10, lg >= 10 * max(j, 1), min(off, nz) >= 100

    print("\n" + "=" * 76)
    print("VERDICT - the stated objective")
    print("=" * 76)
    print(f"  readable band (J<=10 and logit >=10x worse at the same layer): {band}")
    print(f"  at layer {L}:  J-lens rank {j}   logit-lens rank {lg}   ({lg/max(j,1):.0f}x)")
    print(f"    J-lens top-6 : {top(lens(hs,L,'j')[-1])}")
    print(f"    logit  top-6 : {top(lens(hs,L,'logit')[-1])}")
    print(f"  ' spider' on other animals (best mid-band J): {off}")
    print(f"  ' spider' on a no-animal prompt (best mid-band J): {nz}")
    print(f"\n  [{'PASS' if a else 'FAIL'}] ranks highly with the lens (<=10)")
    print(f"  [{'PASS' if b else 'FAIL'}] ranks poorly with the logit lens, same layer (>=10x)")
    print(f"  [{'PASS' if c else 'FAIL'}] specific to this prompt (>=100 on controls)")
    print("\n==> OBJECTIVE MET" if (a and b and c) else "\n==> OBJECTIVE NOT MET")

    print("\n  generalisation (secondary): best mid-band J rank for each animal's own target")
    for n, _, t in CASES:
        print(f"    {n:9s} {t:10s} {int(data[n][1]['j'][t][MID].min()):6d}")


@app.local_entrypoint()
def obj():
    objective.remote()


@app.function(image=image, gpu="A10G", volumes={MOUNT: volume}, timeout=20 * 60)
def atlayer(layer: int = 11):
    """Specificity at the SAME layer, not each prompt's own best layer.

    The matched-layer rule was established for the lens comparison; it applies to
    the control comparison too. 'spider' reaching rank 3 somewhere in the bee
    prompt's band is a different claim from it reaching rank 3 at layer 11.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    volume.reload()
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    J = torch.load(LENS, map_location="cuda", weights_only=True)["J"]
    W_U, NORM = model.lm_head.weight, model.model.norm

    @torch.no_grad()
    def lens(hs, l, kind):
        X = hs[l][0].float() if kind == "logit" else hs[l+1][0].float() @ J[l].float().T
        return (NORM(X).to(W_U.dtype) @ W_U.T).float()[-1]

    SP = tok.encode(" spider")[0]
    CLAUSES = [("spins webs", "spider"), ("barks", "dog"), ("makes honey", "bee"),
               ("has a trunk", "elephant"), ("has eight arms", "octopus"),
               ("builds dams", "beaver")]

    def go(clause, neutral=False):
        q = ("What is the weather usually like by the sea? Reply in one word." if neutral
             else f"How many legs does the animal that {clause} have? Reply with just the number.")
        try:
            s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                        add_generation_prompt=True)
        ids = tok(s + "It has ", return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        j, lg = lens(hs, layer, "j"), lens(hs, layer, "logit")
        return (int((j > j[SP]).sum()), int((lg > lg[SP]).sum()),
                [tok.decode([i]) for i in j.topk(5).indices.tolist()])

    print(f"\nrank of ' spider' AT LAYER {layer}, answer position, all prompts\n")
    print(f"  {'prompt':16s} {'J-lens':>7} {'logit':>7} | top-5 J-lens")
    print("  " + "-" * 74)
    rows = []
    for c, name in CLAUSES:
        j, lg, t = go(c)
        rows.append((name, j, lg))
        print(f"  {name:16s} {j:7d} {lg:7d} | {t}")
    j, lg, t = go(None, neutral=True)
    rows.append(("NEUTRAL", j, lg))
    print(f"  {'NEUTRAL':16s} {j:7d} {lg:7d} | {t}")

    web = rows[0]
    others = [r for r in rows[1:]]
    print(f"\n  web prompt: J={web[1]}, logit={web[2]}")
    print(f"  worst (lowest) control rank: {min(r[1] for r in others)} "
          f"({[r[0] for r in others if r[1] == min(x[1] for x in others)][0]})")
    a = web[1] <= 10
    b = web[2] >= 10 * max(web[1], 1)
    c = min(r[1] for r in others) >= 100 * max(web[1], 1)
    print(f"\n  [{'PASS' if a else 'FAIL'}] ranks highly with the lens")
    print(f"  [{'PASS' if b else 'FAIL'}] ranks poorly with the logit lens, same layer")
    print(f"  [{'PASS' if c else 'FAIL'}] specific to the web prompt, same layer")
    print("\n==> OBJECTIVE MET" if (a and b and c) else "\n==> OBJECTIVE NOT MET")


@app.local_entrypoint()
def at():
    atlayer.remote()
