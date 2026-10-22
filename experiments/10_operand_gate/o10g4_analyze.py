"""Stage 2a-G4 (CPU): render results/stage2a_g4/stage2a_g4.json as markdown tables.

    ../../.venv/bin/python o10g4_analyze.py > results/stage2a_g4/analysis.md
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results" / "stage2a_g4"
o = json.loads((R / "stage2a_g4.json").read_text())


def z(b: dict) -> str:
    return f"{b['z']:+.1f}" if b["z"] == b["z"] else "nan"


def bands(d: dict, key: str) -> str:
    b = d["band_" + key]
    return f"[jc {z(b['jc'])} / iso {z(b['iso'])}]"


leak = o["leak"]


def clean(name: str) -> str:
    c, l = leak["content"].get(name), leak["label"].get(name)
    if c is None:
        return "n/a"
    fired = c["fires95"] or l["fires95"]
    return f"{'**VOID**' if fired else 'clean'} (content z {c['z_mean_shift']:+.1f}/{c['z_auroc']:+.1f}, label z {l['z_mean_shift']:+.1f}/{l['z_auroc']:+.1f})"


print(f"One-digit bank: n = {o['n']}, model correct unsteered {o['model_correct_unsteered']:.3f}, first token is the space {o['first_token_is_space']:.3f}; "
      f"median ‖h‖ {o['median_norm']:.1f}; swap displacement / ‖h‖: raw {o['swap_displacement_over_norm']['raw']:.3f}, J-removed {o['swap_displacement_over_norm']['jrem']:.3f}; "
      f"{o['n_conditions']} conditions per item.\n")
e = o["dir_enrichment"]
print(f"Direction enrichment (n1000): class-means raw r* {min(e['cm_raw']):.2f}–{max(e['cm_raw']):.2f}, raw 4r* {min(e['cm_raw_1536']):.2f}–{max(e['cm_raw_1536']):.2f}, "
      f"J-removed 4r* {max(e['cm_jrem_1536']):.2f}; magnitude raw r* {e['mag_raw_384']:.2f}, J-removed 4r* {e['mag_jrem_1536']:.2f}.\n")

print("### Directions (centred ±Δ_i at r11 / r22; z vs K = 20 J-complement and 20 isotropic directions at the same per-item norm)\n")
print("| condition | flip + / − | flip mean [z] | first-digit log-odds signed (+ − −)/2 [z] | log-odds drop [z] | value signed effect [z] | KL digit pos. (null jc/iso) | leak |")
print("|---|---|---|---|---|---|---|---|")
for dose in ("r11", "r22"):
    nk = o["directions"][f"null_kl_digit@{dose}"]
    for dn in ("cm_own_jrem", "cm_own_raw", "mag_jrem", "mag_raw"):
        k = f"{dn}@{dose}"
        d = o["directions"][k]
        print(f"| {k} | {d['flip_plus']:.2f} / {d['flip_minus']:.2f} | {d['flip_mean']:.2f} {bands(d, 'flip_mean')} | {d['lo_signed_effect']:+.2f} {bands(d, 'lo_signed_effect')} | "
              f"{d['lo_drop']:+.2f} {bands(d, 'lo_drop')} | {d['value_signed_effect']:+.2f} {bands(d, 'value_signed_effect')} | {d['kl_digit']:.3f} ({nk['jc']:.3f}/{nk['iso']:.3f}) | {clean(k)} |")

print("\n### Class-mean subspace swap (9-dim a_units span; target a' = a + 3 wrapped; z vs 20 + 20 random 9-dim subspaces at the same displacement)\n")
print("| swap | follow exact (= a' ± b) [z] | moved toward [z] | normalised shift mean [z] | flip rate [z] | log-odds drop | KL digit (null jc/iso) | leak |")
print("|---|---|---|---|---|---|---|---|")
nk = o["swap"]["null_kl_digit"]
for nm in ("swap_raw", "swap_jrem"):
    s = o["swap"][nm]
    print(f"| {nm} | {s['follow_exact']:.2f} {bands(s, 'follow_exact')} | {s['moved_toward']:.2f} {bands(s, 'moved_toward')} | {s['norm_shift_mean']:+.2f} {bands(s, 'norm_shift_mean')} | "
          f"{s['flip_rate']:.2f} {bands(s, 'flip_rate')} | {s['lo_drop']:+.2f} | {s['kl_digit']:.3f} ({nk['jc']:.3f}/{nk['iso']:.3f}) | {clean(nm)} |")
print(f"\nWall time: G4 {o.get('t_g4_s')} s, total {o['t_total_s']} s.")
