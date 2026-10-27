**Accuracy over 56 scored cells: 0.79 [0.66, 0.87]** (declared bar 0.80 over at least 36 cells).

| row | nla_prompted | nla_states | nla_names_complement | ol_prompted | ol_exhibits | ol_states | jlens_prompted |
|---|---|---|---|---|---|---|---|
| py_string | H / 0.84 MISS | L / 0.19 ok | L / 0.00 ok | H / 0.52 MISS | H / 0.00 MISS | L / 0.00 ok | L / 0.40 ok |
| js_string | H / 0.92 ok | L / 0.08 ok | L / 0.00 ok | H / 0.82 ok | H / 0.00 MISS | L / 0.00 ok | L / 0.51 ok |
| latex_math | H / 0.84 MISS | L / 0.15 ok | L / 0.41 MISS | H / 0.86 ok | H / 0.00 MISS | L / 0.00 ok | L / 0.63 ok |
| py_arg | H / 0.79 MISS | L / 0.03 ok | H / 0.74 ok | H / 0.86 ok | H / 0.00 MISS | L / 0.00 ok | L / 0.57 ok |
| json_depth | H / 0.91 ok | L / 0.14 ok | H / 0.66 ok | H / 0.64 MISS | H / 0.00 MISS | L / 0.03 ok | L / 0.68 ok |
| html_depth | H / 0.93 ok | L / 0.00 ok | H / 0.94 ok | L / 0.51 ok | L / 0.00 ok | L / 0.00 ok | L / 0.62 ok |
| md_heading | L / 0.65 ok | L / 0.04 ok | H / 1.00 ok | L / 0.58 ok | L / 0.00 ok | L / 0.00 ok | L / 0.58 ok |
| md_list | L / 0.59 ok | L / 0.04 ok | H / 0.00 MISS | L / 0.53 ok | L / 0.00 ok | L / 0.00 ok | L / 0.47 ok |

Confusion per reader (predicted x observed):

- nla: HH 7, HL 4, LH 1, LL 12
- ol: HH 3, HL 7, LH 0, LL 14
- jlens: HH 0, HL 0, LH 0, LL 8
