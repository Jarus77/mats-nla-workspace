# Inputs for the cross-scoring with Guiv's bank (data files are git-ignored; only this README is tracked)

Two files, dropped here as they arrive. The stages `xs_ol` then `xs_nla` skip (exit 0) while they are absent.

**A. `guiv_items_{row}.json`** for `row` in `tb_tablecol`, `tb_paren`: a JSON list of
`{"id": str, "text": str, "value": str, "cluster": str}`. `value` is one of our two class names
(`first` / `late`, `inside` / `outside`). The activation is read at the last token of `text` (no chat
template, no special tokens), the same convention as our items. `cluster` groups items that share a source
passage (used for the bootstrap); use the id if there is no grouping.

**B. `guiv_phrasings.json`**: `{row: {"nla": [...], "ol": [...]}}`, each entry
`{"id": str, "template": "... {v} ...", "fillers": {"first": "...", "late": "..."}}` (the exp 13 template
shape). OLens phrasings start with `- `.

Outputs: `/results/17_harden/xs_{ol,nla}_{row}.pt`, `results/analysis17_xs_{row}.json`, `results/xs17.md`:
a 2×2 per reader (his items × our phrasings, his items × his phrasings, our items × his phrasings; ours × ours
is exp 13), AUROC with cluster-bootstrap CIs.
