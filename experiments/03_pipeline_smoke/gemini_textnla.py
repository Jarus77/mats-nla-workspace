"""Local: TextNLA via Gemini 3.8 Flash (OpenRouter), paired elicited-probability LO.

Prompt mode on all smoke texts; explanation mode (re-solving ceiling) on the NLA's natural
explanations from stage B (first of k=4 samples). Key from OPENROUTER_API_KEY / repo .env
(never printed). Output: results/gemini_textnla.json.

    ../../.venv/bin/python gemini_textnla.py
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from mats.textnla import TextNLAGemini  # noqa: E402
from smoke_items import GEMINI_EXPL_QUESTION, GEMINI_QUESTION, LANG_A, LANG_B, items  # noqa: E402


def main() -> None:
    res_dir = HERE / "results"
    texts = [x["text"] for x in items()]
    g = TextNLAGemini()
    t0 = time.time()
    lo_prompt = g.paired_logodds(texts, LANG_A, LANG_B, GEMINI_QUESTION)
    t_prompt = time.time() - t0
    out = {"lo_prompt": lo_prompt.tolist(), "t_prompt_s": t_prompt}
    nla_path = res_dir / "stage_nla.pt"
    if nla_path.exists():
        nla = torch.load(nla_path, weights_only=False)
        expl = [row[0] or "" for row in nla["natural_explanations"]]
        out["lo_expl"] = g.paired_logodds(expl, LANG_A, LANG_B, GEMINI_EXPL_QUESTION,
                                          context_label="Description").tolist()
        out["expl_items"] = [int(i) for i in nla["fve_items"]]
    out.update(n_calls=g.n_calls, n_failed=g.n_failed, n_unparsed=g.n_unparsed, spend_usd=g.spend_usd,
               s_per_call=t_prompt / max(1, 2 * len(texts)))
    (res_dir / "gemini_textnla.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if not k.startswith("lo_")})


if __name__ == "__main__":
    main()
