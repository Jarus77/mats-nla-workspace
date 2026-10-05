"""Step 1b: in-workspace calibration datasets (lead's round-2 gates).

  lang         IJ2 language identity: papluca/language-identification texts (ar, el, pl, ru, tr, vi;
               raw text) + WorkspaceBench multilingual_mt prompts of the same languages (plain render).
               Label at every position >= 4.
  wrap_report  IJ1 line/character count under EXPLICIT report: chat prompt "How many characters ...
               in the first line of this passage?" + 6 wrapped lines (digit-free passages); label =
               characters since last newline at passage tokens (same definition as `col`).

    modal run s1b_data.py
"""

from pathlib import Path

import modal

from nonj_common import OUT, VOLUMES, net_image

app = modal.App("nonj-s1b-data")
_P = Path(__file__).resolve().parents
_MT = (_P[2] / "third_party/workspace-bench/evals/multilingual_mt/items.json") if len(_P) > 2 else Path("/x")
image = net_image.add_local_file(str(_MT), "/root/multilingual_mt_items.json") if _MT.exists() else net_image

LANGS = {"ar": "Arabic", "el": "Greek", "pl": "Polish", "ru": "Russian", "tr": "Turkish", "vi": "Vietnamese"}
REPORT_Q = "How many characters (including spaces and punctuation) are in the first line of this passage? Answer with just the number."


@app.function(image=image, volumes=VOLUMES, timeout=3600, memory=32768, cpu=8)
def build(per_lang: int = 450, n_report: int = 500) -> dict:
    import json
    import pickle
    import random

    from datasets import load_dataset
    from transformers import AutoTokenizer

    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    ds = load_dataset("papluca/language-identification", split="train")
    rng = random.Random(5)
    items = []
    by = {k: [] for k in LANGS}
    for ex in ds:
        if ex["labels"] in by and len(by[ex["labels"]]) < per_lang * 3:
            by[ex["labels"]].append(ex["text"])
    for lg, texts in by.items():
        rng.shuffle(texts)
        texts = [t for t in texts if len(t) > 60][:per_lang]
        for k, t in enumerate(texts):
            enc = tok(t, return_offsets_mapping=True, add_special_tokens=False)
            items.append({"id": f"lang_{lg}_{k}", "text": t, "ids": enc["input_ids"][:128],
                          "offsets": enc["offset_mapping"][:128], "lang": LANGS[lg], "source": "papluca"})
    mt = json.load(open("/root/multilingual_mt_items.json"))["items"]
    for it in mt:
        if it.get("lang") in LANGS:
            enc = tok(it["prompt"], return_offsets_mapping=True, add_special_tokens=False)
            items.append({"id": f"mt_{it['name']}", "text": it["prompt"], "ids": enc["input_ids"],
                          "offsets": enc["offset_mapping"], "lang": LANGS[it["lang"]], "source": "multilingual_mt"})
    with open(f"{OUT}/data/lang.pkl", "wb") as f:
        pickle.dump(items, f)
    info = {"lang_items": len(items), "lang_tokens": sum(len(i["ids"]) for i in items)}

    wrap = pickle.load(open(f"{OUT}/data/wrap.pkl", "rb"))
    clean = [w for w in wrap if not any(ch.isdigit() for ch in "\n".join(w["text"].split("\n")[:6]))]
    rep = []
    for k, w in enumerate(clean[:n_report]):
        passage = "\n".join(w["text"].split("\n")[:6])
        text = tok.apply_chat_template([{"role": "user", "content": f"{REPORT_Q}\n\n{passage}"}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        p0 = text.index(passage)
        col, has_nl = [], []
        for s, e in enc["offset_mapping"]:
            if s >= p0 and e <= p0 + len(passage) and e > s:
                seg = text[p0:e]
                col.append(e - p0 - (seg.rfind("\n") + 1))
                has_nl.append("\n" in text[s:e])
            else:
                col.append(None)
                has_nl.append(True)
        rep.append({"id": f"rep{k}", "text": text, "ids": enc["input_ids"], "offsets": enc["offset_mapping"],
                    "width": w["width"], "col": col, "has_nl": has_nl, "first_line_len": len(passage.split("\n")[0])})
    with open(f"{OUT}/data/wrap_report.pkl", "wb") as f:
        pickle.dump(rep, f)
    info.update(report_items=len(rep), report_tokens=sum(len(i["ids"]) for i in rep))
    VOLUMES["/results"].commit()
    print(info)
    return info


@app.local_entrypoint()
def main() -> None:
    print(build.remote())
