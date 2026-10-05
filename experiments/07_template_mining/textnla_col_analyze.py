"""Fable round 5 (local): Gemini 3.8 Flash TextNLA (JSON P(A), both option orders, max_tokens 600)
on the SAME col selection items and frozen candidate pairs, then the comparison against NLA@42 /
OL@42 per template, with item-level overlap. Reads templates_frozen.json; never writes it.

    python textnla_col_analyze.py [--skip-gemini]   -> results/textnla_col_summary.json + table
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

from mats.metrics import auroc_ci, auroc_labels  # noqa: E402
from mats.textnla import TextNLAGemini  # noqa: E402

RES = HERE / "results"
FEATS = ("nj1_col", "nj1_col_break")  # Gemini budget is spent on nj1_col first
CAP_USD = 2.5            # usage DELTA on the (shared) OpenRouter key for this rerun
KEY_BASELINE_USD = 8.8079175  # key usage before the rerun (an earlier attempt crashed unsaved)
QUESTION = ("A language model has just read the text above, up to its very last character. Which "
            "description better fits the model's internal state at that last token?")
GEMINI_ROLES = ("selected", "fixed_label", "fixed_numeric")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def residual_auroc(lo: np.ndarray, text_lo: np.ndarray, y: np.ndarray) -> float:
    """AUROC of the reader LO after regressing out the TextNLA LO (OLS with intercept)."""
    X = np.c_[np.ones_like(text_lo), text_lo]
    beta, *_ = np.linalg.lstsq(X, lo, rcond=None)
    return auroc_labels(lo - X @ beta, y)


class RobustGemini(TextNLAGemini):
    """TextNLAGemini whose _call also survives connection resets (http.client.RemoteDisconnected
    is not a URLError, which crashed the first attempt). src/mats stays untouched."""

    def _call(self, prompt: str):  # noqa: ANN201
        import time

        for attempt in range(4):
            try:
                return super()._call(prompt)
            except (ConnectionError, OSError) as e:  # noqa: PERF203
                print(f"  conn retry {attempt}: {type(e).__name__}", flush=True)
                time.sleep(2 ** attempt)
        self.n_failed += 1
        return None


def key_usage() -> float:
    import urllib.request

    from mats.textnla import _api_key

    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": f"Bearer {_api_key()}"})
    return float(json.load(urllib.request.urlopen(req, timeout=30))["data"]["usage"])


def run_gemini(feature: str, local: dict, g: TextNLAGemini) -> dict:
    items = {}
    for name in ("gen_ol_" + feature,):
        for it in json.loads((RES / f"{name}.json").read_text())["items"]:
            items[it["id"]] = it
    texts = [items[i]["text"] for i in local["ids"]]
    vals = list(local["templates"][0]["candidates"])
    fz = json.loads((HERE / "templates_frozen.json").read_text())
    order = fz["meta"]["values"][feature]
    ck = RES / f"textnla_gemini_{feature}.partial.json"
    out = json.loads(ck.read_text()) if ck.exists() else {}
    # budget order: selected templates of both readers first (NLA, then OL), then fixed labels
    todo = sorted([t for t in local["templates"] if t["role"] in GEMINI_ROLES and t["id"] not in out],
                  key=lambda t: (t["role"] != "selected", t["reader"] != "nla"))
    for t in todo:
        delta = key_usage() - KEY_BASELINE_USD
        if delta >= CAP_USD:
            print(f"Gemini cap reached (key delta ${delta:.2f}); stopping")
            break
        a, b = t["candidates"][order[0]], t["candidates"][order[1]]
        out[t["id"]] = g.paired_logodds(texts, a, b, QUESTION).tolist()
        ck.write_text(json.dumps(out))
        print(f"  gemini {feature} {t['id']}: run spend ${g.spend_usd:.3f}, key delta ${delta:.2f}", flush=True)
    del vals
    return out


def main() -> None:
    skip = "--skip-gemini" in sys.argv
    g = RobustGemini(max_workers=8)
    summary = {}
    for f in FEATS:
        local = json.loads((RES / f"textnla_local_{f}.json").read_text())
        gpath = RES / f"textnla_gemini_{f}.json"
        if gpath.exists():
            gem = json.loads(gpath.read_text())
        elif not skip:
            gem = run_gemini(f, local, g)
            n_want = sum(t["role"] in GEMINI_ROLES for t in local["templates"])
            if len(gem) == n_want:
                gpath.write_text(json.dumps(gem))
        else:  # --skip-gemini: use whatever the budget-capped run checkpointed
            pp = RES / f"textnla_gemini_{f}.partial.json"
            gem = json.loads(pp.read_text()) if pp.exists() else {}
        s_nla = json.loads((RES / f"score_nla_{f}.json").read_text())["scores"]["nla@42"]
        s_ol = json.loads((RES / f"score_ol_{f}.json").read_text())["scores"]["ol@42"]
        gen = {it["id"]: it for it in json.loads((RES / f"gen_ol_{f}.json").read_text())["items"]}
        ids = local["ids"]
        assert s_nla["ids"] == ids or sorted(s_nla["ids"]) == sorted(ids)
        y = np.array(local["y"])
        clusters = [gen[i]["cluster"] for i in ids]
        rows = []
        for t in local["templates"]:
            sc = s_nla if t["reader"] == "nla" else s_ol
            pos = {i: k for k, i in enumerate(sc["ids"])}
            rd = np.array([sc["lo"][t["id"]][pos[i]] for i in ids])
            tl = np.array(local["lo"][t["id"]])
            row = {"template": t["id"], "reader": t["reader"], "role": t["role"],
                   "cand_a": t["candidates"][list(t["candidates"])[0]],
                   "reader_auroc": auroc_ci(rd, y, clusters, n_boot=1000).__dict__,
                   "textnla_local_auroc": auroc_ci(tl, y, clusters, n_boot=1000).__dict__,
                   "spearman_reader_vs_textlocal": spearman(rd, tl),
                   "reader_auroc_after_regressing_text": residual_auroc(rd, tl, y),
                   "sign_agreement": float(np.mean(np.sign(rd - np.median(rd)) == np.sign(tl - np.median(tl))))}
            if t["id"] in gem:
                gl = np.array(gem[t["id"]], float)
                ok = np.isfinite(gl)
                row["textnla_gemini_auroc"] = auroc_ci(gl[ok], y[ok], np.array(clusters)[ok], n_boot=1000).__dict__
                row["gemini_n_valid"] = int(ok.sum())
                row["spearman_reader_vs_gemini"] = spearman(rd[ok], gl[ok])
            rows.append(row)
        # template means (selected only) per reader
        means = {}
        for rdname, sc in (("nla", s_nla), ("ol", s_ol)):
            sel = [t for t in local["templates"] if t["reader"] == rdname and t["role"] == "selected"]
            pos = {i: k for k, i in enumerate(sc["ids"])}
            rd = np.mean([[sc["lo"][t["id"]][pos[i]] for i in ids] for t in sel], 0)
            tl = np.mean([local["lo"][t["id"]] for t in sel], 0)
            m = {"reader": auroc_ci(rd, y, clusters, n_boot=1000).__dict__,
                 "textnla_local": auroc_ci(tl, y, clusters, n_boot=1000).__dict__,
                 "spearman": spearman(rd, tl), "reader_after_text": residual_auroc(rd, tl, y)}
            gsel = [gem[t["id"]] for t in sel if t["id"] in gem]
            if gsel:
                gl = np.nanmean(np.array(gsel, float), 0)
                ok = np.isfinite(gl)
                m["textnla_gemini"] = auroc_ci(gl[ok], y[ok], np.array(clusters)[ok], n_boot=1000).__dict__
                m["reader_after_gemini"] = residual_auroc(rd[ok], gl[ok], y[ok])
            means[rdname] = m
        summary[f] = {"per_template": rows, "mean_of_selected": means}
        print(f"\n== {f}")
        for r in rows:
            gtxt = f"{r['textnla_gemini_auroc']['auroc']:.2f}" if "textnla_gemini_auroc" in r else "  - "
            print(f"{r['reader']:3s} {r['role'][:9]:9s} reader {r['reader_auroc']['auroc']:.2f} | "
                  f"textQwen {r['textnla_local_auroc']['auroc']:.2f} | gemini {gtxt} | rho {r['spearman_reader_vs_textlocal']:+.2f} | "
                  f"reader|text {r['reader_auroc_after_regressing_text']:.2f}  {r['cand_a'][:50]}")
        for k, m in means.items():
            print(f"MEAN3 {k}: reader {m['reader']['auroc']:.2f} {m['reader']['lo']:.2f}-{m['reader']['hi']:.2f} | "
                  f"textQwen {m['textnla_local']['auroc']:.2f} {m['textnla_local']['lo']:.2f}-{m['textnla_local']['hi']:.2f} | "
                  f"gemini {m.get('textnla_gemini', {}).get('auroc', float('nan')):.2f} | rho {m['spearman']:+.2f} | "
                  f"reader|text {m['reader_after_text']:.2f} | reader|gemini {m.get('reader_after_gemini', float('nan')):.2f}")
    summary["_gemini"] = {"spend_usd_this_run": round(g.spend_usd, 4),
                          "key_usage_delta_usd": round(key_usage() - KEY_BASELINE_USD, 4), "calls": g.n_calls,
                          "failed": g.n_failed, "unparsed": g.n_unparsed}
    print(summary["_gemini"])
    (RES / "textnla_col_summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
