"""s6_two_checks (exps 24 + 25): two independent checks that the readers' words carry the structural state.
Left: a second decoder (Oracle Lens's own reconstructor, exp 24). Right: the independent public bank through the
critic (exp 25). Data and checks come from g4_s6_second_decoder.gather and g4_s6_public_carry.gather unchanged."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g4_common as G  # noqa: E402
import style  # noqa: E402
from style import C, plt  # noqa: E402
import g4_s6_public_carry as P25  # noqa: E402
import g4_s6_second_decoder as P24  # noqa: E402

# Shared row axis: the six readable natural rows, then the language control (public bank only).
# Exp 24's table column / parentheses and both Markdown rows are in the backup s6_second_decoder.
ROWS = ["py_string", "js_string", "latex_math", "json_depth", "html_depth", "py_arg", "code_lang"]
LABEL = dict(G.SHORT, code_lang="code language\n(control)")


def dot(ax, a, lo, hi, y, col, mk, filled=True, ms=9):
    ax.plot([lo, hi], [y, y], color=col, alpha=0.35, lw=2, solid_capstyle="butt", zorder=2)
    ax.plot(a, y, mk, ms=ms, mfc=col if filled else "white", mec=col, mew=1.8, zorder=3)


def main():
    style.slide_style()
    d24, s24, c24 = P24.gather()
    d25, s25, c25 = P25.gather()
    keep = set(ROWS)
    checks = [c for c in c24 + c25 if c["quantity"].split(" ")[0] in keep or not c["quantity"].startswith(
        tuple(P24.ROWS) + tuple(P25.ROWS))]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=G.FIG, sharey=True, layout="constrained")
    ypos = {r: (len(ROWS) - 1 - i) + (0 if r == "code_lang" else 0.35) for i, r in enumerate(ROWS)}
    for ax in (a1, a2):
        ax.axvline(0.5, color=C["line"], lw=1.5, ls="--", zorder=0)
        ax.axvline(0.8, color=C["line"], lw=1.5, ls="--", zorder=0)
        ax.set_xlim(0.3, 1.03)
        ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        top = max(ypos.values()) + 0.45
        ax.text(0.5, top, "chance", ha="center", va="bottom", color=C["muted"], fontsize=15)
        ax.text(0.8, top, "0.80", ha="center", va="bottom", color=C["muted"], fontsize=15)
        ax.set_ylim(-0.5, top + 0.55)
        ax.set_xlabel("state recovered (AUROC)")
    # left: exp 24, frozen probe from the true span's rebuilds (filled), NLA by a probe of its own (hollow)
    for r in ROWS:
        y = ypos[r]
        if r not in d24:
            a1.text(0.65, y, "not in this bank", ha="center", va="center", color=C["muted"], fontsize=15)
            continue
        dot(a1, *d24[r][0], y + 0.18, C["ol"], "D")
        dot(a1, *d24[r][2], y - 0.02, C["nla"], "o")
        dot(a1, *d24[r][3], y - 0.22, C["nla"], "o", filled=False)
    a1.set_title("second decoder (exp 24)", loc="left")
    # right: exp 25 public bank through the critic
    for r in ROWS:
        y = ypos[r]
        if r not in d25:
            a2.text(0.65, y, "not in this bank", ha="center", va="center", color=C["muted"], fontsize=15)
            continue
        dot(a2, *d25[r]["ol_readout"], y + 0.18, C["ol"], "D")
        dot(a2, *d25[r]["nla_readout"], y - 0.02, C["nla"], "o")
        dot(a2, *d25[r]["bag_32"], y - 0.22, C["muted"], "X", ms=8)
    a2.set_title("public bank (exp 25)", loc="left")
    a1.set_yticks([ypos[r] for r in ROWS])
    a1.set_yticklabels([LABEL[r] for r in ROWS])
    a1.get_yticklabels()[-1].set_fontstyle("italic")
    hs = [a1.plot([], [], "D", ms=9, color=C["ol"])[0], a1.plot([], [], "o", ms=9, color=C["nla"])[0],
          a1.plot([], [], "o", ms=9, mfc="white", mec=C["nla"], mew=1.8)[0],
          a1.plot([], [], "X", ms=8, color=C["muted"])[0]]
    fig.legend(hs, ["Oracle Lens readout", "NLA explanation", "NLA, probe fit on its own rebuilds (left)",
                    "true tokens shuffled (right)"], loc="outside upper center", ncol=2, handletextpad=0.2,
               columnspacing=1.0)
    style.save(
        fig, "s6_two_checks",
        title="Two independent checks agree: Oracle Lens's words carry the state through a second decoder and on a public bank",
        caption=("Left, exp 24: each exp 20 readout (k = 1) is fed to Oracle Lens's own activation reconstructor (layer tag "
                 "44) and scored by a probe fit on the rebuilds of the true 32-token span, applied frozen: Oracle Lens 0.73 "
                 "to 0.80 on five of six readable rows (call argument 0.61), the NLA at chance; a probe fit on the NLA's own "
                 "rebuilds (grouped 5-fold CV, hollow) separates it at 0.81 to 0.90. Right, exp 25: exp 20's protocol on the "
                 "independent public bank through the NLA's critic: Oracle Lens 0.84 to 0.97 on string, math and JSON "
                 "depth (call argument 0.51); shuffled tokens keep only the language control. 160 items per row; bars "
                 "Hanley-McNeil 95% (no per-item scores saved; ignores clustering)."),
        section="S6", role="main", sources=sorted(set(s24 + s25)), checks=checks,
        notes=("Combines the backups s6_second_decoder (all ten exp 20 rows, both probes in separate panels, shuffled "
               "reference) and s6_public_carry (all eight public-bank rows, critic on the true span too). Rows shown here: "
               "the readable natural rows shared by the claims plus the public bank's language control; exp 24's table "
               "column (Oracle Lens frozen 0.57) and parentheses (0.60) and the two Markdown rows (not linearly present at "
               "layer 42) are in the backups. Exp 24 has no language row and exp 25 has no HTML row ('not in this bank')."),
    )
    print("checks", len(checks), "mismatch", [c for c in checks if not c["match"]])


if __name__ == "__main__":
    main()
