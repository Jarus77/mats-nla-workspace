Frozen-probe AUROC against the true layer-42 activations (160 selection items per row):

| row | span_32 | span_128 | bag_32 | ol_readout | nla_readout |
|---|---:|---:|---:|---:|---:|
| py_string | 0.84 | 0.86 | 0.46 | 0.86 | 0.99 |
| js_string | 0.77 | 0.78 | 0.46 | 0.84 | 0.97 |
| latex_math | 0.85 | 0.87 | 0.50 | 0.87 | 1.00 |
| py_arg | 0.88 | 0.86 | 0.49 | 0.51 | 0.93 |
| json_depth | 0.92 | 0.98 | 0.76 | 0.97 | 0.99 |
| md_heading | 0.62 | 0.69 | 0.55 | 0.65 | 0.70 |
| md_list | 0.50 | 0.58 | 0.44 | 0.48 | 0.49 |
| code_lang | 0.99 | 1.00 | 0.98 | 0.98 | 0.98 |

Cosine of each rebuild's class direction with the true one:

| row | span_32 | span_128 | bag_32 | ol_readout | nla_readout |
|---|---:|---:|---:|---:|---:|
| py_string | 0.45 | 0.44 | -0.10 | 0.54 | 0.90 |
| js_string | 0.38 | 0.37 | -0.03 | 0.58 | 0.88 |
| latex_math | 0.49 | 0.65 | 0.03 | 0.68 | 0.89 |
| py_arg | 0.63 | 0.67 | 0.09 | 0.36 | 0.85 |
| json_depth | 0.48 | 0.66 | 0.23 | 0.57 | 0.78 |
| md_heading | 0.42 | 0.46 | 0.12 | 0.43 | 0.86 |
| md_list | 0.22 | 0.35 | -0.00 | 0.30 | 0.80 |
| code_lang | 0.59 | 0.66 | 0.36 | 0.61 | 0.80 |
