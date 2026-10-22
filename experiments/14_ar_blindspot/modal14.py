"""Exp 14 Modal stages (app `kq2`; functions r1-r4):
r1 = fresh Oracle Lens readouts at 44/40 for all 200 items (H100),
r3 = fresh NLA readouts + NLA-AR reconstructions + NLA statement reconstructions (H200),
r2 = Oracle Lens AR (the ruler OLens was trained against) on true spans, OL bullets, NLA
     explanations and statement pairs, tags 44/40 (H100; after r1 and r3),
r4 = CPU analysis (probe retention, error term, gap retention, statement shifts).

    modal run modal14.py --stage gen_ol
    modal run modal14.py --stage gen_nla
    modal run modal14.py --stage recon
    modal run modal14.py --stage analyze
"""
import json
import time
from pathlib import Path

import modal

from env import OUT, SRC, VOLUMES, gpu_image
from spec14 import (AR_REPO, AR_SUBDIR, K_GENS, LONG_SPAN_TOKENS, MAX_TEXT_TOKENS, NLA_LAYER,
                    OL_AR_LAYERS, ROWS, SPAN_TOKENS)

app = modal.App("kq2")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
GPU = dict(volumes=VOLUMES, timeout=3 * 3600, memory=131072, single_use_containers=True, max_containers=2)
GPU1 = {**GPU, "max_containers": 1}  # the two generation stages run side by side: one GPU each
BASE = "Qwen/Qwen3.6-27B"


# ============================================================================ shared
def _row(feature: str):  # noqa: ANN202
    """Items (with token ids and labels) and captured activations {L: [N, d]} for one row."""
    import torch

    src = SRC[ROWS[feature]["src"]]
    items = json.load(open(f"{src}/gen_ol_{feature}.json"))["items"]
    blob = torch.load(f"{src}/acts_{feature}.pt", weights_only=False)
    assert blob["ids"] == [it["id"] for it in items], "activation order != item order"
    return items, {int(L): a.float() for L, a in blob["acts"].items()}


def _encode_cap(tok, text: str, n: int) -> list[int]:  # noqa: ANN001
    return tok.encode(text or "", add_special_tokens=False)[:n]


# ============================================================================ r1: OL readouts (H100)
@app.function(image=gpu_image, gpu="H100", **GPU1)
def r1(feature: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mats.readers.olens import WSBENCH_SAMPLING, load_olens

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    Path(OUT).mkdir(parents=True, exist_ok=True)
    path = f"{OUT}/gen14_ol_{feature}.json"
    if Path(path).exists():
        return json.load(open(path))
    items, acts = _row(feature)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
    ol = load_olens(model, tok)
    gens = {}
    for L in OL_AR_LAYERS:
        g = ol.verbalize(acts[L], L, k=K_GENS, sampling=WSBENCH_SAMPLING, batch=32)
        gens[str(L)] = {it["id"]: g[i] for i, it in enumerate(items)}
    out = {"feature": feature, "ids": [it["id"] for it in items], "gens": gens, "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(path, "w"))
    VOLUMES["/results"].commit()
    return out


# ============================================================================ r3: NLA readouts + NLA-AR (H200)
@app.function(image=gpu_image, gpu="H200", **GPU1)
def r3(feature: str, pairs: list[list[str]]) -> dict:
    """pairs = [[template_id, statement_A, statement_B], ...] in the NLA's explanation register."""
    import torch

    from mats.readers import nla as NL

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    Path(OUT).mkdir(parents=True, exist_ok=True)
    path = f"{OUT}/recon14_nla_{feature}.pt"
    if Path(path).exists():
        return {"feature": feature, "skipped": True}
    items, acts = _row(feature)
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=True)
    raw = nla.verbalize_raw(acts[NLA_LAYER], k=K_GENS, batch=64)
    expl = [[(NL.extract_explanation(t) or t.split("</br>")[0].strip()) for t in row] for row in raw]
    flat = [e for row in expl for e in row]
    rec = nla.reconstruct(flat, batch=16).reshape(len(items), K_GENS, -1)
    stmt = nla.reconstruct([s for _, a, b in pairs for s in (a, b)], batch=16).reshape(len(pairs), 2, -1)
    torch.save({"ids": [it["id"] for it in items], "expl": expl, "recon": rec,
                "stmt_ids": [p[0] for p in pairs], "stmt_recon": stmt}, path)
    json.dump({"feature": feature, "expl": {it["id"]: expl[i] for i, it in enumerate(items)}},
              open(f"{OUT}/gen14_nla_{feature}.json", "w"))
    VOLUMES["/results"].commit()
    return {"feature": feature, "n": len(items), "wall_s": round(time.time() - t0, 1)}


# ============================================================================ r2: Oracle Lens AR (H100)
class OLensAR:
    """The ptag reconstructor: ``[Layer NN] <span>`` -> predicted block-NN activation at the span's
    last token. Backbone = Qwen3.6-27B text model + the shipped LoRA (compile-wrapper keys renamed),
    head = norm (no affine) + linear. Which norm the head uses and whether the backbone's final norm
    is applied are not documented; both are calibrated on true spans (retrieval@1) before use."""

    def __init__(self, device: str = "cuda") -> None:
        import shutil
        import tempfile

        import torch
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from safetensors.torch import load_file, save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ar_lib import rename_lora_keys
        from mats.capture import decoder_blocks

        root = Path(snapshot_download(AR_REPO, allow_patterns=[f"{AR_SUBDIR}/*"])) / AR_SUBDIR
        self.tok = AutoTokenizer.from_pretrained(BASE)
        full = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map=device).eval()
        text = full.model
        tmp = Path(tempfile.mkdtemp())
        sd = rename_lora_keys(load_file(str(root / "lora" / "adapter_model.safetensors")))
        save_file(sd, str(tmp / "adapter_model.safetensors"))
        shutil.copy(root / "lora" / "adapter_config.json", tmp / "adapter_config.json")
        self.model = PeftModel.from_pretrained(text, str(tmp)).eval()
        # the adapter must be live: one loaded B matrix equals the checkpoint's
        key = next(k for k in sd if k.endswith("lora_B.weight"))
        live = dict(self.model.named_parameters())[key.replace("lora_B.weight", "lora_B.default.weight")]
        assert torch.allclose(live.float().cpu(), sd[key].float()), "AR LoRA did not load"
        assert sd[key].abs().sum() > 0, "AR LoRA B matrix is zero"
        heads = torch.load(root / "heads.pt", map_location="cpu", weights_only=False)
        self.W = heads["head"]["linear.weight"].float().to(device)
        self.b = heads["head"]["linear.bias"].float().to(device)
        self.tag_ids = {int(L): heads["tag_ids"][i].tolist() for i, L in enumerate(heads["layers"])}
        self.final_norm = self.model.base_model.model.norm
        self.blocks = decoder_blocks(self.model)
        meta = json.load(open(root / "meta.json"))["config"]
        # ptag reads target layer li at backbone block layer_indices[li] (12 blocks, 5..16)
        self.read_block = {int(L): int(meta["layer_indices"][i]) for i, L in enumerate(heads["layers"])}
        self.device = device
        self.variant = ("li", "ln")
        self._cur = {}
        for i, blk in enumerate(self.blocks):
            blk.register_forward_hook(lambda _m, _i, o, i=i: self._cur.__setitem__(i, (o[0] if isinstance(o, tuple) else o)))

    @staticmethod
    def _head_norm(x, kind: str):  # noqa: ANN001
        import torch

        if kind == "ln":
            return torch.nn.functional.layer_norm(x, (x.shape[-1],))
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def reconstruct(self, rows: list[list[int]], layer: int, variant=None, batch: int = 16):  # noqa: ANN001
        """[N, d] fp32 CPU predictions for token-id rows under the given (final_norm, head_norm)."""
        import torch

        from ar_lib import right_pad

        fn, hn = variant or self.variant
        tag = self.tag_ids[layer]
        out = []
        with torch.no_grad():
            for s in range(0, len(rows), batch):
                chunk = [tag + list(r) for r in rows[s : s + batch]]
                ids, mask = right_pad(chunk, self.tok.pad_token_id or 0)
                ids, mask = ids.to(self.device), mask.to(self.device)
                self.model(input_ids=ids, attention_mask=mask)
                h = self._cur[self.read_block[layer] if fn == "li" else len(self.blocks) - 1]
                last = h[torch.arange(len(chunk)), mask.sum(1) - 1]
                if fn == "last_post":
                    last = self.final_norm(last)
                x = self._head_norm(last.float(), hn)
                out.append((x @ self.W.T + self.b).cpu())
        return torch.cat(out)

    def calibrate(self, rows: list[list[int]], gold, layer: int) -> dict:  # noqa: ANN001
        from ar_lib import fve_raw, retrieval_at_1

        res = {}
        for fn in ("li", "last_pre", "last_post"):
            for hn in ("ln", "rms"):
                p = self.reconstruct(rows, layer, (fn, hn))
                res[f"{fn}/{hn}"] = {"ret1": retrieval_at_1(p, gold), "fve": fve_raw(p, gold)}
        best = max(res, key=lambda k: res[k]["ret1"])
        self.variant = tuple(best.split("/"))
        return {"scores": res, "chosen": best}


@app.function(image=gpu_image, gpu="H100", **GPU1)
def r2(feature: str, pairs_ol: list[list[str]], pairs_nla: list[list[str]], calib: bool = False, calib_only: bool = False) -> dict:
    import torch

    from ar_lib import span_crop

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    path = f"{OUT}/recon14_olar_{feature}.pt"
    if Path(path).exists():
        return {"feature": feature, "skipped": True}
    items, acts = _row(feature)
    if calib_only:
        ar = OLensAR()
        rows = [span_crop(it["ids"], SPAN_TOKENS) for it in items]
        return {"feature": feature, "calib": {str(L): ar.calibrate(rows, acts[L], L) for L in OL_AR_LAYERS},
                "wall_s": round(time.time() - t0, 1)}
    ol_gens = json.load(open(f"{OUT}/gen14_ol_{feature}.json"))["gens"]
    nla_blob = torch.load(f"{OUT}/recon14_nla_{feature}.pt", weights_only=False)
    assert nla_blob["ids"] == [it["id"] for it in items]
    ar = OLensAR()
    tok = ar.tok
    span32 = [span_crop(it["ids"], SPAN_TOKENS) for it in items]
    span128 = [span_crop(it["ids"], LONG_SPAN_TOKENS) for it in items]
    out = {"ids": [it["id"] for it in items], "variant": None, "calib": None, "layers": {}}
    if calib:
        out["calib"] = ar.calibrate(span32, acts[OL_AR_LAYERS[0]], OL_AR_LAYERS[0])
    out["variant"] = "/".join(ar.variant)
    out["note"] = ("head output is in the AR's whitened space (moments gated: agu18dec/local-workspace); "
                   "analysis compares reconstructions inside that space")
    print("AR variant", out["variant"], flush=True)
    for L in OL_AR_LAYERS:
        bul = [[_encode_cap(tok, t, MAX_TEXT_TOKENS) for t in ol_gens[str(L)][it["id"]]] for it in items]
        expl = [[_encode_cap(tok, t, MAX_TEXT_TOKENS) for t in row] for row in nla_blob["expl"]]
        d = {"span32": ar.reconstruct(span32, L), "span128": ar.reconstruct(span128, L),
             "ol_bullets": ar.reconstruct([r for row in bul for r in row], L).reshape(len(items), K_GENS, -1),
             "nla_expl": ar.reconstruct([r for row in expl for r in row], L).reshape(len(items), K_GENS, -1)}
        stmt = {}
        for tag, pairs in (("ol", pairs_ol), ("nla", pairs_nla)):
            rows = [_encode_cap(tok, s, MAX_TEXT_TOKENS) for _, a, b in pairs for s in (a, b)]
            rec = ar.reconstruct(rows, L).reshape(len(pairs), 2, -1)
            stmt.update({f"{tag}:{p[0]}": rec[i] for i, p in enumerate(pairs)})
        d["stmt"] = stmt
        out["layers"][L] = d
        print(f"{feature} L{L} done {time.time() - t0:.0f}s", flush=True)
    torch.save(out, path)
    VOLUMES["/results"].commit()
    return {"feature": feature, "variant": out["variant"], "calib": out["calib"], "wall_s": round(time.time() - t0, 1)}



# ============================================================================ r5: AR loader diagnostics (H100)
@app.function(image=gpu_image, gpu="H100", **GPU1)
def r5(feature: str) -> dict:
    """One model load; many read/pool/norm/target variants on true 32-token spans. Also checks the
    adapter is live, decodes the tag, and tests a held-out ridge map from head output to activation."""
    import torch

    from ar_lib import fve_raw, retrieval_at_1, right_pad, span_crop

    VOLUMES["/results"].reload()
    items, acts = _row(feature)
    ar = OLensAR()
    tok = ar.tok
    rows = [span_crop(it["ids"], SPAN_TOKENS) for it in items]
    out = {"tag_decode": {L: tok.decode(t) for L, t in ar.tag_ids.items()}, "read_block": ar.read_block, "variants": {}}
    nb = len(ar.blocks)

    def forward(L, with_tag=True, adapter=True):
        feats = {}
        with torch.no_grad():
            for s0 in range(0, len(rows), 16):
                chunk = [(ar.tag_ids[L] if with_tag else []) + list(r) for r in rows[s0:s0 + 16]]
                ids, mask = right_pad(chunk, tok.pad_token_id or 0)
                ids, mask = ids.to(ar.device), mask.to(ar.device)
                ctx = ar.model.disable_adapter() if not adapter else torch.no_grad()
                with ctx:
                    ar.model(input_ids=ids, attention_mask=mask)
                last_idx = mask.sum(1) - 1
                for b in {ar.read_block[L] - 1, ar.read_block[L], ar.read_block[L] + 1, nb - 1}:
                    h = ar._cur[b].float()
                    m = mask.unsqueeze(-1).float()
                    feats.setdefault((b, "last"), []).append(h[torch.arange(len(chunk)), last_idx].cpu())
                    feats.setdefault((b, "mean"), []).append(((h * m).sum(1) / m.sum(1)).cpu())
        return {k: torch.cat(v) for k, v in feats.items()}

    for L in OL_AR_LAYERS:
        gold = acts[L]
        zgold = (gold - gold.mean(0)) / gold.std(0).clamp_min(1e-6)
        f_on, f_off = forward(L), forward(L, adapter=False)
        b0 = ar.read_block[L]
        out["variants"][L] = {"adapter_rel_change": float((f_on[(b0, "last")] - f_off[(b0, "last")]).norm() / f_off[(b0, "last")].norm())}
        f_notag = forward(L, with_tag=False)
        for name, feats in (("tag", f_on), ("notag", f_notag)):
            for (b, pool), x in feats.items():
                for hn in ("ln", "rms", "none"):
                    xx = x if hn == "none" else OLensAR._head_norm(x, hn)
                    pred = xx.to(ar.device) @ ar.W.T + ar.b
                    pred = pred.cpu()
                    out["variants"][L][f"{name}/b{b}/{pool}/{hn}"] = {
                        "ret1_raw": retrieval_at_1(pred, gold), "fve_raw": fve_raw(pred, gold),
                        "ret1_z": retrieval_at_1(pred, zgold), "fve_z": fve_raw(pred, zgold),
                        "pred_norm": float(pred.norm(dim=-1).mean()), "gold_norm": float(gold.norm(dim=-1).mean())}
        # held-out ridge map from the default head output to the activation (does the output carry it linearly?)
        x = OLensAR._head_norm(f_on[(b0, "last")], "ln").to(ar.device) @ ar.W.T + ar.b
        x = x.cpu(); n = len(x) // 2
        for lam in (1e2, 1e4):
            A = x[:n]; G = gold[:n]
            M = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1]), A.T @ G)
            out["variants"][L][f"ridge_lam{lam:g}_heldout_ret1"] = retrieval_at_1(x[n:] @ M, gold[n:])
        best = max((k for k in out["variants"][L] if "/" in k), key=lambda k: out["variants"][L][k]["ret1_raw"])
        out["variants"][L]["best_raw"] = (best, out["variants"][L][best])
        print(L, "best", best, out["variants"][L][best], "adapter", out["variants"][L]["adapter_rel_change"], flush=True)
    return out

# ============================================================================ r4: analysis (CPU)
@app.function(image=gpu_image, volumes=VOLUMES, timeout=3600, memory=32768, cpu=8)
def r4(feature: str) -> dict:
    from analyze14 import analyze_row

    VOLUMES["/results"].reload()
    items, acts = _row(feature)
    res = analyze_row(feature, items, acts, OUT)
    json.dump(res, open(f"{OUT}/analysis14_{feature}.json", "w"), indent=1)
    VOLUMES["/results"].commit()
    return res


# ============================================================================ entrypoint
def _pairs(feature: str, reader: str) -> list[list[str]]:
    """[id, A, B] statement pairs: the frozen fixed label plus the selected mined templates."""
    exp = {"13": "13_new_rows", "07": "07_template_mining"}[ROWS[feature]["src"]]
    t = json.loads((HERE.parent / exp / "templates_frozen.json").read_text())["templates"][reader][feature]
    vals = ROWS[feature]["values"]
    rows = [t["fixed_label"]] + t["selected"]
    return [[r["id"], r["candidates"][vals[0]], r["candidates"][vals[1]]] for r in rows]


@app.local_entrypoint()
def main(stage: str = "gen_ol", features: str = "") -> None:
    feats = features.split(",") if features else list(ROWS)
    RES.mkdir(exist_ok=True)
    if stage == "gen_ol":
        for r in r1.map(feats, return_exceptions=True):
            _save(r, "gen14_ol")
    elif stage == "gen_nla":
        for r in r3.map(feats, [_pairs(f, "nla") for f in feats], return_exceptions=True):
            _save(r, "recon14_nla")
    elif stage == "recon":
        for r in r2.map(feats, [_pairs(f, "ol") for f in feats], [_pairs(f, "nla") for f in feats], return_exceptions=True):
            _save(r, "recon14_olar")
    elif stage == "debug":
        r = r5.remote(feats[0])
        (RES / "debug14.json").write_text(json.dumps(r, indent=1, default=str))
        for L, d in r["variants"].items():
            top = sorted((k for k in d if "/" in k), key=lambda k: -d[k]["ret1_raw"])[:8]
            print(L, "adapter_rel_change", d["adapter_rel_change"], "ridge", {k: v for k, v in d.items() if k.startswith("ridge")})
            for k in top:
                print("  ", k, {m: round(v, 3) for m, v in d[k].items()})
        print("tags", r["tag_decode"], "read_block", r["read_block"])
    elif stage == "calib":
        r = r2.remote(feats[0], [], [], True, True)
        print(json.dumps(r, indent=1))
        (RES / "calib14.json").write_text(json.dumps(r, indent=1))
    elif stage == "analyze":
        for r in r4.map(feats, return_exceptions=True):
            _save(r, "analysis14")


def _save(r, kind: str) -> None:  # noqa: ANN001
    if isinstance(r, Exception):
        print(f"[{kind}] FAILED: {r!r}")
        return
    (RES / f"{kind}_{r['feature']}.json").write_text(json.dumps(r, default=str))
    print(f"[{kind}] {r['feature']} done: {json.dumps({k: v for k, v in r.items() if k not in ('gens', 'expl')}, default=str)[:600]}")
