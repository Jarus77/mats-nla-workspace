**Accuracy over 70 scored cells: 0.63 [0.51, 0.73]** (declared bar 0.80 over at least 36 cells).

| row | nla_prompted | nla_states | nla_names_complement | ol_prompted | ol_exhibits | ol_states | jlens_prompted |
|---|---|---|---|---|---|---|---|
| json_depth | H / 1.00 ok | L / 0.25 MISS | L / 0.85 MISS | H / 0.99 ok | H / 0.00 MISS | L / 0.03 ok | L / 0.87 MISS |
| latex_math | H / 0.95 ok | H / 0.00 MISS | L / 0.00 ok | H / 1.00 ok | H / 0.00 MISS | L / 0.00 ok | L / 1.00 MISS |
| py_string | H / 1.00 ok | H / 0.03 MISS | L / 0.00 ok | H / 1.00 ok | H / 0.00 MISS | L / 0.00 ok | L / 0.95 MISS |
| call_arg | H / 1.00 ok | H / 0.33 ok | L / 1.00 MISS | H / 1.00 ok | H / 0.00 MISS | L / 0.00 ok | L / 0.99 MISS |
| md_heading | H / 0.92 ok | L / 0.00 ok | L / 0.96 MISS | L / 0.65 ok | L / 0.00 ok | L / 0.00 ok | L / 0.74 MISS |
| csv_col | H / 0.96 ok | L / 0.13 ok | L / 1.00 MISS | L / 0.89 MISS | L / 0.00 ok | L / 0.00 ok | L / 0.84 MISS |
| html_depth | H / 0.84 MISS | L / 0.00 ok | L / 0.21 MISS | L / 0.57 ok | L / 0.00 ok | L / 0.00 ok | L / 0.58 ok |
| stance_word | L / 0.55 ok | L / 0.00 ok | L / 0.00 ok | L / 0.59 ok | L / 0.00 ok | L / 0.00 ok | H / 0.52 MISS |
| ij2_lang | H / 1.00 ok | L / 0.95 MISS | L / 0.97 MISS | H / 1.00 ok | H / 0.72 ok | L / 0.00 ok | H / 0.98 ok |
| bk_quote | H / 0.79 MISS | H / 0.08 MISS | L / 0.04 ok | L / 0.62 ok | L / 0.00 ok | L / 0.00 ok | H / 0.55 MISS |

Confusion per reader (predicted x observed):

- nla: HH 8, HL 5, LH 8, LL 9
- ol: HH 6, HL 4, LH 1, LL 19
- jlens: HH 1, HL 2, LH 6, LL 1
