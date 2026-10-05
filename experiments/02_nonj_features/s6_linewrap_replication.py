"""Step 6: replicate the J-space paper's line-width experiment (Fig. 21) on Qwen3.6-27B.

Same wrapped passage, three chat questions (thinking disabled):
  linewrap  "Continue this passage for several more lines, preserving the existing line wrapping."
  report    "How many characters (including spaces and punctuation) are in the first line ...?"
  flexible  "... spell that number out as an English word. What is its first letter?"
J-lens (neuronpedia n1000, standard readout) at every passage token, J-layers 26..58 step 4.
Count of passage positions where a number token (digit or number word) is in the top-3 / top-10.

    modal run s6_linewrap_replication.py
"""

import modal

from nonj_common import OUT, VOLUMES, nonj_image

app = modal.App("nonj-s6-linewrap")

QS = {
    "linewrap": "Continue this passage for several more lines, preserving the existing line-wrapping pattern exactly.",
    "report": "How many characters (including spaces and punctuation) are in the first line of this passage? Answer with just the number.",
    "flexible": "Count the characters (including spaces and punctuation) in the first line of this passage and spell that number out as an English word. What is the first letter of that word? Answer with just the letter.",
}
JL = list(range(26, 59, 4))
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
             "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}


@app.function(image=nonj_image, volumes=VOLUMES, gpu="H100", timeout=3 * 3600, memory=196608)
def replicate(n: int = 24) -> dict:
    import glob
    import pickle
    import textwrap

    import numpy as np
    import torch

    from nonj_lib import decoder_blocks, final_norm_module, load_jlens_layers, load_model, save_json

    VOLUMES["/results"].reload()
    tok, model = load_model()
    blocks = decoder_blocks(model)
    fnorm = final_norm_module(model)
    W_U = model.lm_head.weight
    lens_path = glob.glob(f"{OUT}/lenses/np/**/*.pt", recursive=True)[0]
    Js = {L: J.to(torch.bfloat16).cuda() for L, J in load_jlens_layers(lens_path, JL).items()}
    words = set(NUM_WORDS) | set(TENS.values()) | {"hundred"} | {str(i) for i in range(10)}
    num_ids = set()
    for w in words:
        for s in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                num_ids.add(ids[0])
    num_ids_t = torch.tensor(sorted(num_ids), device="cuda")
    items = pickle.load(open(f"{OUT}/data/wrap.pkl", "rb"))
    rng = np.random.default_rng(4)
    # passages without any digit, so number tokens in the lens cannot come from copying the text
    clean = [k for k, it in enumerate(items) if not any(ch.isdigit() for ch in "\n".join(it["text"].split("\n")[:6]))]
    sel = rng.choice(clean, n, replace=False)
    cap: dict = {}
    hooks = [blocks[L].register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach()))
             for L in JL]
    rows = []
    for i in sel:
        it = items[int(i)]
        lines = it["text"].split("\n")[:6]
        W = it["width"]
        passage = "\n".join(lines)
        first_len = len(lines[0])
        row = {"item": int(i), "width": W, "first_line_len": first_len}
        for cond, q in QS.items():
            msg = f"{q}\n\n{passage}"
            text = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
            enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")[0].tolist()
            p0 = text.index(passage)
            ppos = [j for j, (s, e) in enumerate(offs) if s >= p0 and e <= p0 + len(passage) and e > s]
            enc = enc.to("cuda")
            with torch.no_grad():
                model(**enc)
                hits3, hits10 = np.zeros(len(ppos), bool), np.zeros(len(ppos), bool)
                per_layer = {}
                for L in JL:
                    H = cap[L][0, ppos]
                    sc = fnorm(H @ Js[L].T) @ W_U.T
                    top = sc.topk(10, dim=1).indices
                    h3 = torch.isin(top[:, :3], num_ids_t).any(1).cpu().numpy()
                    h10 = torch.isin(top, num_ids_t).any(1).cpu().numpy()
                    per_layer[L] = int(h3.sum())
                    hits3 |= h3
                    hits10 |= h10
                gen = model.generate(**enc, max_new_tokens=120 if cond == "linewrap" else 6, do_sample=False)
            out = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            row[cond] = {"positions_num_top3": int(hits3.sum()), "positions_num_top10": int(hits10.sum()),
                         "n_positions": len(ppos), "per_layer_top3": per_layer, "answer": out}
            if cond == "linewrap":
                gl = [len(x) for x in out.strip("\n").split("\n")[:-1] if x.strip()]
                row[cond]["generated_line_lengths"] = gl
                row[cond]["mean_abs_dev_from_width"] = float(np.mean([abs(g - W) for g in gl])) if gl else None
        rows.append(row)
        print({k: (v["positions_num_top3"] if isinstance(v, dict) else v) for k, v in row.items()},
              row["report"]["answer"], row["flexible"]["answer"], flush=True)
    for h in hooks:
        h.remove()
    summ = {}
    for cond in QS:
        a3 = np.array([r[cond]["positions_num_top3"] for r in rows])
        a10 = np.array([r[cond]["positions_num_top10"] for r in rows])
        summ[cond] = {"mean_positions_num_top3": float(a3.mean()), "frac_passages_any_top3": float((a3 > 0).mean()),
                      "mean_positions_num_top10": float(a10.mean()),
                      "mean_n_positions": float(np.mean([r[cond]["n_positions"] for r in rows]))}
    rep = [r["report"]["answer"].strip() for r in rows]
    summ["report_exact"] = float(np.mean([a.isdigit() and int(a) == r["first_line_len"] for a, r in zip(rep, rows)]))
    summ["report_within2"] = float(np.mean([a.isdigit() and abs(int(a) - r["first_line_len"]) <= 2 for a, r in zip(rep, rows)]))
    devs = [r["linewrap"]["mean_abs_dev_from_width"] for r in rows if r["linewrap"]["mean_abs_dev_from_width"] is not None]
    summ["linewrap_mean_abs_dev_from_width"] = float(np.mean(devs)) if devs else None
    save_json({"summary": summ, "rows": rows}, f"{OUT}/linewrap_replication.json")
    VOLUMES["/results"].commit()
    print(summ, flush=True)
    return summ


@app.local_entrypoint()
def main(n: int = 24) -> None:
    print(replicate.remote(n))
