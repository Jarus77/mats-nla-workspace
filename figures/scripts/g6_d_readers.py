"""Diagram: the three readers, each as input -> carrier -> output (talk S1/S2 glossary)."""
from __future__ import annotations

from g6_common import C, SOFT, WHITE, arrow, box, canvas, text
from style import save

NAME = "s1_d_readers"


def activation_icon(ax, x, y, w, h, n=9):
    """A column vector drawn as n stacked cells with slate shades."""
    shades = ["#2F4858", "#5E6E78", "#9FB0BA", "#2F4858", "#C9D3D9", "#5E6E78", "#2F4858", "#9FB0BA", "#5E6E78"]
    ch = h / n
    for i in range(n):
        box(ax, x, y + i * ch, w, ch, fc=shades[i % len(shades)], ec=WHITE, lw=1.0, radius=0.2)


def main():
    fig, ax = canvas()  # 92.4 x 42 units, 10 per inch

    # source activation
    activation_icon(ax, 4.3, 15, 3.2, 17)
    text(ax, 5.9, 35, "h", fs=20, weight="bold", color=C["probe"])
    text(ax, 5.9, 10.2, "layer 42\nactivation", fs=15, color=C["probe"], linespacing=1.15)

    rows = [  # y0, colour, pale, name, line 2, line 3 (how h enters), output
        (29.0, C["nla"], C["nla_pale"], "NLA  (RL adapter, step 400)",
         "prompt: ... <concept>[h]</concept>", "h is added at a marker token", "explanation"),
        (16.6, C["ol"], C["ol_pale"], "Oracle Lens  (LoRA, same model)",
         "carrier: <activation>[h]</activation>", "h replaces one character's embedding", "'- ' bullets"),
        (4.2, C["jl"], C["jl_pale"], "J-lens  (published n1000 lens)",
         "no carrier, no generation", "scores every token by W_U · J · h", "top tokens"),
    ]
    cx0, cw, hgt = 13, 43.5, 11.0
    ox0, ow = 58.6, 13.6
    for y0, col, pale, name, line2, how, out in rows:
        yc = y0 + hgt / 2
        box(ax, cx0, y0, cw, hgt, fc=SOFT, ec=col, lw=2.0)
        text(ax, cx0 + 1.0, y0 + hgt - 2.0, name, fs=15, weight="bold", color=col, ha="left")
        text(ax, cx0 + 1.0, yc - 0.2, line2, fs=15, ha="left", color=C["ink"])
        text(ax, cx0 + 1.0, y0 + 2.2, how, fs=15, ha="left", color=col, style="italic")
        box(ax, ox0, yc - 3.2, ow, 6.4, out, fc=pale, ec=col, lw=1.8, fs=15)
        arrow(ax, (cx0 + cw, yc), (ox0, yc), color=col, lw=2.2)
        arrow(ax, (8.2, 23.5), (cx0, yc), color=C["probe"], lw=1.8)

    # NLA critic / reconstructor loop
    y0 = rows[0][0]
    kx = 74.2
    box(ax, kx, y0 - 1.6, 92.2 - kx, hgt + 2.4, "", fc=WHITE, ec=C["nla"], lw=1.8, ls="--")
    text(ax, (kx + 92.2) / 2, y0 + hgt - 1.5, "critic", fs=15, weight="bold", color=C["nla"])
    text(ax, (kx + 92.2) / 2, y0 + 3.0, "rebuilds h\nfrom the words\n= RL reward", fs=15, linespacing=1.15)
    arrow(ax, (ox0 + ow, y0 + hgt / 2), (kx, y0 + hgt / 2), color=C["nla"], lw=2.0)

    # footer
    text(ax, 52, 1.7, "No reader ever sees the prompt text: each gets only h.", fs=15, weight="bold",
         color=C["probe"])

    sources = ["src/mats/readers/nla.py", "src/mats/readers/olens.py", "src/mats/readers/jlens.py",
               "notes/wsbench_internals.md", "notes/doc_part2_draft.md", "notes/decisions.md"]
    checks = [
        {"quantity": "NLA carrier is the actor prompt (meticulous AI researcher ... <concept>)", "computed":
         "nla.py: 'Prompt = nla_meta.yaml actor template'; wsbench_internals.md 1.5 quotes it", "record":
         "talk script glossary quotes the warm-start / TextNLA instruction ('A language model needs to predict...')",
         "where": "talk script, 'three readers' table", "match": False},
        {"quantity": "NLA injection", "computed": "norm-matched ADD at block-1 output at marker id 158983",
         "record": "talk script: 'added at a marker token'", "where": "nla.py docstring", "match": True},
        {"quantity": "Oracle Lens carrier", "computed": "olens.py PROMPT (verbatim, layer=42)",
         "record": "talk script glossary carrier text", "where": "olens.py", "match": True},
        {"quantity": "Oracle Lens trained layers", "computed": "TRAINED_LAYERS = range(20, 61, 4)",
         "record": "20 to 60 step 4", "where": "olens.py", "match": True},
        {"quantity": "J-space rank at layer 42", "computed": "1,536 = 4 r*, r* = 384", "record": "rank 1,536",
         "where": "notes/doc_part2_draft.md; exp 15 RESULTS (4r* = 1536)", "match": True},
        {"quantity": "J-lens readout", "computed": "s_t = (W_U J_L h)_t / ||J_L^T W_U[t]||, top-k; no generation",
         "record": "talk script: 'top tokens of the unembedding applied to the Jacobian image'", "where": "jlens.py",
         "match": True},
        {"quantity": "NLA adapter", "computed": "DEFAULT_ADAPTER av_rl_adapters/iter_000400", "record": "iter_000400",
         "where": "nla.py", "match": True},
    ]
    notes = ("Diagram, no data. NLA actor prompt (from notes/wsbench_internals.md 1.5, abbreviated there with '...'; "
             "nla_meta.yaml is not in the repo): 'You are a meticulous AI researcher ... We will pass the vector enclosed "
             "in <concept> tags ... The explanation consists of 2-3 text snippets ... <concept>[marker U+321C]</concept> "
             "Please provide an explanation.' The NLA's h is norm-matched and added at the marker token after block 1. The marker glyph U+321C is not in the figure font, so it is drawn as [h]. "
             "The talk script's NLA carrier text quotes experiments/06_resolving_decomposition/data/nla_instruction.txt, "
             "which is the TextNLA / warm-start instruction, not the actor prompt: fix the script. Oracle Lens prompt, "
             "verbatim: 'An activation vector from layer 42 of a language model is enclosed in activation tags: "
             "<activation>X</activation>. Produce distinct concepts that encode this activation, each as a '- ' bullet on "
             "its own line.' (slot = an enclosed ideograph, id 158983 for Qwen3.6, embedding replaced by alpha*unit(h), "
             "alpha 16000); Oracle Lens was trained at layers 20 to 60 in steps of 4, so layer 42 is off its grid. J-lens: the J-space is the top 1,536 directions at layer 42 (= 4 r*, r* = 384); W_U · J · h is cosine-normalised. Critic input (nla.py): 'Summary of the following text: <text>{e}</text> <summary>'.")
    save(fig, NAME, title="Three readers get the same activation and never the prompt",
         caption=("Each reader receives only the layer-42 activation h at the read token. The NLA has h added at a "
                  "marker inside its actor prompt and writes an explanation; its critic rebuilds h from those words, "
                  "and RL rewards words that survive the rebuild. Oracle Lens has h written into one slot character's "
                  "embedding and writes bullets; the J-lens generates nothing and lists the top tokens of W_U·J_42·h."),
         section="S1", role="main", sources=sources, checks=checks, notes=notes)


if __name__ == "__main__":
    main()
