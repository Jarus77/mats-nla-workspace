"""Task 3 checks + FVE-as-OOD-detector stats (H200, AV iter_000400 + AR).

1. Natural FVE distribution per example (from fve_gate.json explanations, re-scored).
2. TextNLA explanations (textnla.json) through the AR: text-only re-description FVE.
3. Teacher-forced scoring: M[a, w] = log p(candidate(w) | act_a) for all 64x64 (act, item)
   pairs, three candidate types: bare topic word, topic word in a sentence frame, and the
   source's final word in the AV's native 'Final token "..."' frame. AUROCs:
     raw      : per activation, own candidate vs other items' candidates (string prior confound)
     word_ctl : per candidate string, its own activation vs other activations
     paired   : paired log-odds, P(lo(act_i; c_i vs c_j) > lo(act_j; c_i vs c_j))
"""
import json
import re
from collections import Counter
from pathlib import Path

import modal
from env import OUT, VOLUMES, app_image, load_examples

app = modal.App("mats-00-score-gate")
STOP = set("""about above after again against their there these those which while would could
should other being where every under first since until within without using between before
because through during around among another however still also into from with this that have
what when your more most some such than then them they very will just only over were been
here make like many much each both said says""".split())


def topic_word(text, taken):
    tail = text[-600:]
    words = [w for w in re.findall(r"[A-Za-z]{5,}", tail) if w.lower() not in STOP]
    cnt = Counter(w.lower() for w in words)
    last = {w.lower(): i for i, w in enumerate(words)}
    for w in sorted(cnt, key=lambda w: (-cnt[w], -last[w])):
        if w not in taken:
            return w
    return words[-1].lower() if words else "text"


def last_word(text):
    chunks = text.split()
    for c in reversed(chunks):
        c2 = c.strip(".,;:!?\"'()[]“”’")
        if c2:
            return c2
    return chunks[-1] if chunks else ""


def auroc(pos, neg):
    import numpy as np
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def metrics(M, idx):
    """AUROCs over the sub-block idx x idx of score matrix M[act, cand]."""
    import numpy as np
    M = np.asarray(M)[np.ix_(idx, idx)]
    n = len(idx)
    off = ~np.eye(n, dtype=bool)
    raw = auroc(np.diag(M), M[off])  # pooled; per-act version below
    raw_per = float(np.mean([(M[i, i] > M[i, off[i]]).mean() for i in range(n)]))
    word = float(np.mean([(M[j, j] > M[off[:, j], j]).mean() for j in range(n)]))
    dd = np.array([M[i, i] - M[i, j] - M[j, i] + M[j, j] for i in range(n) for j in range(n) if i < j])
    return {"raw_auroc_pooled": raw, "raw_forced_choice_acc": raw_per,
            "word_controlled_acc": word, "paired_logodds_auroc": float((dd > 0).mean() + 0.5 * (dd == 0).mean()),
            "n": n}


@app.function(image=app_image, volumes=VOLUMES, gpu="H200", timeout=3 * 3600)
def run(parts: str = "natural,textnla,scoring") -> dict:
    import torch
    from mats.readers.nla import load_nla, predict_mean_baseline

    acts, texts, _ = load_examples()
    nla = load_nla(adapter="av_rl_adapters/iter_000400")
    base = predict_mean_baseline(acts)
    res = {"baseline": base}

    # 1. natural FVE distribution
    fg = json.loads(Path(f"{OUT}/fve_gate.json").read_text())
    for name in [] if "natural" not in parts else ("iter400_nothink", "iter400_nothink_wsbench_sampling"):
        r = nla.fve_on(acts, baseline=base, explanations=fg["explanations"][name])
        ps = r["per_sample"]
        res[f"natural_{name}"] = {
            "mean": float(torch.nanmean(ps)), "sd_samples": float(ps[~ps.isnan()].std()),
            "sd_examples": float(r["per_example"].std()),
            "within_example_sd_mean": float(torch.nanmean(ps.std(dim=1))),
            "p05_samples": float(ps[~ps.isnan()].quantile(0.05)),
            "min_example": float(r["per_example"].min()),
            "per_example_mean": [round(float(x), 4) for x in r["per_example"]],
            "per_example_sd": [round(float(x), 4) for x in ps.std(dim=1)],
        }
        print(name, {k: v for k, v in res[f"natural_{name}"].items() if not k.startswith("per_")},
              flush=True)

    # 2. TextNLA
    tp = Path(f"{OUT}/textnla.json")
    if tp.exists() and "textnla" in parts:
        tx = json.loads(tp.read_text())["explanations"]
        r = nla.fve_on(acts, baseline=base, explanations=tx)
        perm = fg["perm"]
        rs = nla.fve_on(acts, baseline=base, explanations=[tx[p] for p in perm])
        res["textnla"] = {"mean": float(torch.nanmean(r["per_sample"])),
                          "sd_examples": float(r["per_example"].std()),
                          "per_example_mean": [round(float(x), 4) for x in r["per_example"]],
                          "shuffled_mean": float(torch.nanmean(rs["per_sample"]))}
        print("textnla", {k: v for k, v in res["textnla"].items() if k != "per_example_mean"},
              flush=True)

    # 3. teacher-forced scoring
    taken, topics = set(), []
    for t in texts:
        w = topic_word(t, taken)
        taken.add(w)
        topics.append(w)
    lasts = [last_word(t) for t in texts]
    cand_types = {
        "bare_topic_word": topics,
        "frame_topic": [f"The text discusses {w}." for w in topics],
        "frame_final_token": [f'Final token "{w}"' for w in lasts],
    }
    res["candidates"] = cand_types
    res["scoring"] = {}
    n = acts.shape[0]
    for name, cands in (cand_types.items() if "scoring" in parts else []):
        M = torch.stack([nla.logprob_of_strings(acts[a], cands, batch=64) for a in range(n)])
        res["scoring"][name] = {"all64": metrics(M.numpy(), list(range(n))),
                                "first10": metrics(M.numpy(), list(range(10))),
                                "M": M.tolist()}
        print(name, res["scoring"][name]["all64"], res["scoring"][name]["first10"], flush=True)
    # API check: paired_logodds equals the matrix difference
    if "scoring" in parts:
        c = cand_types["frame_final_token"]
        lo = nla.paired_logodds(acts[0], c[0], c[1])
        Mf = res["scoring"]["frame_final_token"]["M"]
        res["paired_logodds_api_check"] = {"api": lo, "matrix": Mf[0][0] - Mf[0][1]}
        print("api check", res["paired_logodds_api_check"], flush=True)
    if "lpcheck" in parts:  # regression check of logprob_of_strings against the saved matrix
        prev = json.loads(Path(f"{OUT}/score_gate_natural_scoring.json").read_text())
        c = prev["candidates"]["frame_final_token"]
        new = nla.logprob_of_strings(acts[0], c, batch=16)
        old = torch.tensor(prev["scoring"]["frame_final_token"]["M"][0])
        res["lpcheck"] = {"max_abs_diff": float((new - old).abs().max()),
                          "argmax_same": int(new.argmax()) == int(old.argmax())}
        print("lpcheck", res["lpcheck"], flush=True)
    tag = "" if parts == "natural,textnla,scoring" else "_" + parts.replace(",", "_")
    Path(f"{OUT}/score_gate{tag}.json").write_text(json.dumps(res, indent=1))
    VOLUMES["/results"].commit()
    return res


@app.local_entrypoint()
def main(parts: str = "natural,textnla,scoring"):
    res = run.remote(parts)
    tag = "" if parts == "natural,textnla,scoring" else "_" + parts.replace(",", "_")
    (Path(__file__).parent / "results" / f"score_gate{tag}.json").write_text(json.dumps(res, indent=1))
