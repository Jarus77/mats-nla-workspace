"""Step 4 (iv): causal use. Steer along the feature's steering direction at hs[43] (output of block 42,
J-lens L42) at every position of the context, and measure a feature-specific behavioural metric
vs equal-norm random directions. Doses = natural projection |dev| at p90, p99, 2x p99 (from s3).

Metrics (all teacher-forced, at "eval positions" of held-out items):
  col      log-odds that the next token contains a newline, at true line ends and 1-2 tokens before
           (+direction = more characters on the line -> expect earlier breaks / higher P(newline))
  next_pos log P(adjective-lexicon) - log P(noun-lexicon) at positions where the next word is ADJ/NOUN
  tense    log P(past-verb lexicon) - log P(present-verb lexicon) at verb slots
  depth    expected number of ']' in the next token, at positions where the next token closes brackets
  topic    log P(sports lexicon) - log P(business lexicon) at all positions
  nexttok  log P(' the') at word positions (direction = next token ' the' vs rest)
Also reported: steering norm / median residual norm, mean next-token KL(unsteered || steered) at
eval positions, and the J-orthogonalised direction (direction minus its top-25 J-lens component).

    modal run s4_causal.py [--features col,tense] [--hs 43]
"""

import modal

from nonj_common import OUT, VOLUMES, nonj_image

app = modal.App("nonj-s4-causal")

PAST = ["was", "were", "had", "did", "said", "became", "made", "went", "took", "came", "gave", "found",
        "began", "remained", "included", "served", "played", "led", "won", "received", "used", "called",
        "left", "saw", "held", "wrote", "moved", "died", "lived", "stated", "appeared", "showed", "seemed"]
PRES = ["is", "are", "has", "does", "says", "becomes", "makes", "goes", "takes", "comes", "gives", "finds",
        "begins", "remains", "includes", "serves", "plays", "leads", "wins", "receives", "uses", "calls",
        "leaves", "sees", "holds", "writes", "moves", "dies", "lives", "states", "appears", "shows", "seems"]
SPORTS = ["game", "team", "season", "win", "coach", "league", "players", "match", "victory", "Cup", "champion"]
BUSINESS = ["company", "shares", "market", "profit", "sales", "billion", "deal", "stock", "investors", "bank", "prices"]


def _single_ids(tok, words: list[str]) -> list[int]:  # noqa: ANN001
    out = []
    for w in words:
        ids = tok.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            out.append(ids[0])
    return sorted(set(out))


@app.function(image=nonj_image, volumes=VOLUMES, gpu="H100", timeout=6 * 3600, memory=131072)
def run_causal(features: list[str], hs: int = 43, n_items: int = 120, n_rand: int = 4) -> dict:
    import collections
    import pickle
    import time

    import numpy as np
    import torch

    from features import POS_CLASSES
    from nonj_lib import decoder_blocks, load_model, save_json

    VOLUMES["/results"].reload()
    t0 = time.time()
    tok, model = load_model()
    blocks = decoder_blocks(model)
    V = model.lm_head.weight.shape[0]
    vocab = [tok.decode([i]) for i in range(V)]
    nl_ids = torch.tensor([i for i, s in enumerate(vocab) if "\n" in s], device="cuda")
    close_cnt = torch.tensor([s.count("]") for s in vocab], device="cuda", dtype=torch.float32)
    state: dict = {"vec": None}

    def hook(_m, _i, out):  # noqa: ANN001, ANN202
        if state["vec"] is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + state["vec"].to(out[0].dtype),) + tuple(out[1:])
        return out + state["vec"].to(out.dtype)

    handle = blocks[hs - 1].register_forward_hook(hook)
    all_res = {}
    for fname in features:
        P = torch.load(f"{OUT}/probes/{fname}_hs{hs}.pt", weights_only=False)
        u = P["steer_dir_unit"].float().cuda()
        u_o = P.get("steer_dir_unit_jorth")
        u_o = u_o.float().cuda() if u_o is not None else None
        dose = P["dose"]
        dataset = {"col": "wrap", "next_pos": "prose", "tense": "prose", "depth": "brackets",
                   "topic": "topic", "nexttok": "prose", "pred_pos": "prose", "lang": "lang",
                   "col_report": "wrap_report"}[fname]
        jrem = {int(k.split("_r")[-1]): v.float().cuda() for k, v in P.items() if k.startswith("steer_dir_unit_jrem_r")}
        pred_lab = None
        if fname == "pred_pos":
            idx_ = np.load(f"{OUT}/acts/prose/index.npz")
            cm = np.load(f"{OUT}/acts/prose/pred.npz")["class_mass"]
            tot, arg = cm.sum(1), cm.argmax(1)
            share = cm.max(1) / np.maximum(tot, 1e-9)
            ok = (tot >= 0.3) & (share >= 0.6) & np.isin(arg, [POS_CLASSES.index("ADJ"), POS_CLASSES.index("NOUN")])
            pred_lab = collections.defaultdict(list)
            for r in np.nonzero(ok)[0]:
                pred_lab[int(idx_["item"][r])].append(int(idx_["pos"][r]))
        items = pickle.load(open(f"{OUT}/data/{dataset}.pkl", "rb"))
        rng = np.random.default_rng(11)
        # held-out-ish: take items from the end of the list (probes used grouped random splits)
        cand = list(range(len(items)))
        rng.shuffle(cand)
        sel = cand[:n_items] if fname != "topic" else cand[: n_items * 2]

        # lexicons / eval positions
        if fname in ("next_pos", "pred_pos"):
            cnt = collections.defaultdict(collections.Counter)
            for it in items:
                for j, lab in enumerate(it["next_pos"]):
                    if lab and j + 1 < len(it["ids"]):
                        cnt[it["ids"][j + 1]][lab] += 1
            adj = [t for t, c in cnt.items() if sum(c.values()) >= 3 and c["ADJ"] / sum(c.values()) >= 0.8]
            noun = [t for t, c in cnt.items() if sum(c.values()) >= 3 and c["NOUN"] / sum(c.values()) >= 0.8]
            setA, setB = torch.tensor(adj, device="cuda"), torch.tensor(noun, device="cuda")
        elif fname == "tense":
            setA, setB = torch.tensor(_single_ids(tok, PAST), device="cuda"), torch.tensor(_single_ids(tok, PRES), device="cuda")
        elif fname == "topic":
            setA, setB = torch.tensor(_single_ids(tok, SPORTS), device="cuda"), torch.tensor(_single_ids(tok, BUSINESS), device="cuda")
        elif fname == "lang":
            cyr = [i for i, s_ in enumerate(vocab) if any("\u0400" <= ch <= "\u04ff" for ch in s_)]
            lat = [i for i, s_ in enumerate(vocab) if len(s_.strip()) >= 2 and all(("a" <= ch.lower() <= "z") or ch == " " for ch in s_)]
            setA, setB = torch.tensor(cyr, device="cuda"), torch.tensor(lat, device="cuda")
        elif fname == "col_report":
            digit_ids = torch.tensor([tok.encode(str(k), add_special_tokens=False)[0] for k in range(10)], device="cuda")
        the_id = tok.encode(" the", add_special_tokens=False)[0]
        setAB_all = None
        if fname in ("tense",):
            setAB_all = set(setA.tolist()) | set(setB.tolist())

        def eval_positions(it: dict) -> list[int]:
            ids = it["ids"]
            n = len(ids)
            if fname == "col":
                out = []
                for j in range(n - 1):
                    if it["next_nl"][j]:
                        out += [k for k in (j - 2, j - 1, j) if k >= 4 and not it["has_nl"][k]]
                return sorted(set(out))
            if fname == "next_pos":
                return [j for j in range(4, n - 1) if it["next_pos"][j] in ("ADJ", "NOUN")]
            if fname == "pred_pos":
                return sorted(pred_lab.get(cur_item[0], []))
            if fname == "lang":
                return list(range(4, n))
            if fname == "col_report":
                return [n - 1]
            if fname == "tense":
                return [j for j in range(4, n - 1) if ids[j + 1] in setAB_all]
            if fname == "depth":
                return [j for j in range(4, n - 1) if "]" in tok.decode([ids[j + 1]])]
            if fname == "topic":
                return list(range(8, n))
            if fname == "nexttok":
                return [j for j in range(4, n - 1) if tok.decode([ids[j + 1]]).startswith(" ")]
            return []

        def metric(logp: torch.Tensor) -> torch.Tensor:  # logp [P, V] -> [P]
            if fname == "col":
                lp = torch.logsumexp(logp[:, nl_ids], 1)
                return lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-6))
            if fname == "col_report":  # expected first digit of the reported count
                pd = logp[:, digit_ids].exp()
                return (pd / pd.sum(1, keepdim=True) * torch.arange(10, device="cuda")).sum(1)
            if fname in ("next_pos", "pred_pos", "tense", "topic", "lang"):
                return torch.logsumexp(logp[:, setA], 1) - torch.logsumexp(logp[:, setB], 1)
            if fname == "depth":
                return (logp.exp() * close_cnt).sum(1) / (logp.exp() * (close_cnt > 0)).sum(1).clamp_min(1e-9)
            if fname == "nexttok":
                return logp[:, the_id]
            raise ValueError(fname)

        g = torch.Generator("cuda").manual_seed(3)
        rdirs = torch.randn(n_rand, u.shape[0], device="cuda", generator=g)
        rdirs = rdirs / rdirs.norm(dim=1, keepdim=True)
        conds = [("none", 0.0, None)]
        for dn, dv in dose["doses"].items():
            for sgn in (+1, -1):
                conds.append((f"probe{'+' if sgn > 0 else '-'}{dn}", sgn * dv, u))
                if u_o is not None and dn != "p90":
                    conds.append((f"jorth{'+' if sgn > 0 else '-'}{dn}", sgn * dv, u_o))
                for rr, ur in jrem.items():
                    conds.append((f"jrem{rr}{'+' if sgn > 0 else '-'}{dn}", sgn * dv, ur))
                for ri in range(n_rand):
                    conds.append((f"rand{ri}{'+' if sgn > 0 else '-'}{dn}", sgn * dv, rdirs[ri]))
        per_item: dict = {c[0]: [] for c in conds}
        kl_item: dict = {c[0]: [] for c in conds}
        norm_ratio = []
        cur_item = [None]
        if fname == "pred_pos":
            sel = [i for i in sel if pred_lab.get(i)] + [i for i in cand[n_items:] if pred_lab.get(i)][: max(0, n_items - len([i for i in sel if pred_lab.get(i)]))]
        for ii, i in enumerate(sel):
            it = items[i]
            cur_item[0] = i
            pos = eval_positions(it)
            if not pos:
                continue
            ids = torch.tensor(it["ids"], device="cuda")[None]
            base_lp = None
            for name, alpha, d in conds:
                state["vec"] = None if d is None else alpha * d
                with torch.no_grad():
                    out = model(input_ids=ids, output_hidden_states=(name == "none"))
                lp = torch.log_softmax(out.logits[0, pos].float(), -1)
                if name == "none":
                    base_lp = lp
                    norm_ratio.append(out.hidden_states[hs][0, pos].float().norm(dim=1).median().item())
                per_item[name].append(metric(lp).mean().item())
                kl_item[name].append((base_lp.exp() * (base_lp - lp)).sum(1).mean().item())
            state["vec"] = None
            if ii % 20 == 0:
                print(f"[{fname}] item {ii}/{len(sel)} t={time.time()-t0:.0f}s", flush=True)

        base = np.array(per_item["none"])
        med_norm = float(np.median(norm_ratio))
        summ: dict = {"n_items": len(base), "baseline_metric_mean": float(base.mean()),
                      "median_resid_norm_eval": med_norm, "doses": dose["doses"], "conditions": {}}
        rng2 = np.random.default_rng(0)
        for name, alpha, d in conds:
            if name == "none":
                continue
            delta = np.array(per_item[name]) - base
            boots = [delta[rng2.integers(0, len(delta), len(delta))].mean() for _ in range(1000)]
            summ["conditions"][name] = {"alpha": float(alpha), "steer_over_resid_norm": abs(alpha) / med_norm,
                                        "delta_mean": float(delta.mean()), "delta_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                        "kl_mean": float(np.mean(kl_item[name]))}
        # effect sizes: (delta(+d) - delta(-d))/2 for probe vs random directions
        eff = {}
        for dn in dose["doses"]:
            pe = (np.array(per_item[f"probe+{dn}"]) - np.array(per_item[f"probe-{dn}"])) / 2
            re = np.stack([(np.array(per_item[f"rand{r}+{dn}"]) - np.array(per_item[f"rand{r}-{dn}"])) / 2 for r in range(n_rand)])
            e = {"probe_signed_effect": float(pe.mean()),
                 "probe_effect_ci95": [float(np.percentile([pe[rng2.integers(0, len(pe), len(pe))].mean() for _ in range(1000)], q)) for q in (2.5, 97.5)],
                 "random_abs_effects": [float(abs(x)) for x in re.mean(1)],
                 "random_abs_effect_max": float(np.abs(re.mean(1)).max()),
                 "probe_kl_plus": summ["conditions"][f"probe+{dn}"]["kl_mean"],
                 "random_kl_plus_mean": float(np.mean([summ["conditions"][f"rand{r}+{dn}"]["kl_mean"] for r in range(n_rand)]))}
            for rr in jrem:
                je = (np.array(per_item[f"jrem{rr}+{dn}"]) - np.array(per_item[f"jrem{rr}-{dn}"])) / 2
                e[f"jrem{rr}_signed_effect"] = float(je.mean())
                e[f"jrem{rr}_effect_ci95"] = [float(np.percentile([je[rng2.integers(0, len(je), len(je))].mean() for _ in range(1000)], q)) for q in (2.5, 97.5)]
                e[f"jrem{rr}_over_random_max"] = abs(e[f"jrem{rr}_signed_effect"]) / max(float(np.abs(re.mean(1)).max()), 1e-9)
                e[f"jrem{rr}_kl_plus"] = summ["conditions"][f"jrem{rr}+{dn}"]["kl_mean"]
            if f"jorth+{dn}" in per_item:
                je = (np.array(per_item[f"jorth+{dn}"]) - np.array(per_item[f"jorth-{dn}"])) / 2
                e["jorth_signed_effect"] = float(je.mean())
                e["jorth_effect_ci95"] = [float(np.percentile([je[rng2.integers(0, len(je), len(je))].mean() for _ in range(1000)], q)) for q in (2.5, 97.5)]
            e["probe_over_random_max"] = abs(e["probe_signed_effect"]) / max(e["random_abs_effect_max"], 1e-9)
            eff[dn] = e
        summ["effects"] = eff
        all_res[fname] = summ
        print(fname, {dn: (round(v["probe_signed_effect"], 3), round(v["random_abs_effect_max"], 3)) for dn, v in eff.items()}, flush=True)
        save_json(summ, f"{OUT}/causal/{fname}_hs{hs}.json")
        VOLUMES["/results"].commit()
    handle.remove()
    return all_res


@app.local_entrypoint()
def main(features: str = "col,pred_pos,tense,depth,lang,col_report", hs: int = 43, n_items: int = 120) -> None:
    fl = features.split(",")
    groups = [[f] for f in fl]
    for r in run_causal.starmap([(g, hs, n_items) for g in groups], return_exceptions=True):
        print(r)
