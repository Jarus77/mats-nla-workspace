**Accuracy over 56 scored cells: 0.79 [0.66, 0.87]** (declared bar 0.80 over at least 36 cells).

| row | nla_prompted | nla_states | nla_names_complement | ol_prompted | ol_exhibits | ol_states | jlens_prompted |
|---|---|---|---|---|---|---|---|
| py_string | H / 0.88 ok | L / 0.12 ok | L / 0.00 ok | H / 0.91 ok | H / 0.00 MISS | L / 0.00 ok | L / 0.56 ok |
| js_string | H / 0.90 ok | L / 0.14 ok | L / 0.00 ok | L / 0.87 MISS | L / 0.00 ok | L / 0.00 ok | L / 0.66 ok |
| latex_math | H / 0.82 MISS | L / 0.09 ok | L / 0.40 MISS | H / 0.93 ok | H / 0.00 MISS | L / 0.00 ok | L / 0.56 ok |
| py_arg | H / 0.65 MISS | L / 0.02 ok | H / 0.71 ok | H / 0.82 ok | H / 0.03 MISS | L / 0.00 ok | L / 0.56 ok |
| json_depth | H / 0.96 ok | L / 0.25 MISS | H / 0.68 ok | H / 0.88 ok | H / 0.00 MISS | L / 0.01 ok | L / 0.79 MISS |
| md_heading | L / 0.76 ok | L / 0.04 ok | H / 1.00 ok | L / 0.58 ok | L / 0.00 ok | L / 0.00 ok | L / 0.53 ok |
| md_list | L / 0.69 ok | L / 0.01 ok | H / 0.00 MISS | L / 0.49 ok | L / 0.25 ok | L / 0.00 ok | L / 0.51 ok |
| code_lang | H / 1.00 ok | L / 0.70 MISS | H / 0.78 ok | H / 0.99 ok | H / 0.78 ok | L / 0.00 ok | H / 0.93 ok |

Confusion per reader (predicted x observed):

- nla: HH 8, HL 3, LH 3, LL 10
- ol: HH 6, HL 4, LH 1, LL 13
- jlens: HH 1, HL 0, LH 1, LL 6
