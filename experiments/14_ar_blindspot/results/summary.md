## Reconstruction and concept retention

| concept | ruler | layer | input | FVE | ret@1 | probe on true | probe on recon | probe on error | probe retrained on recon | gap kept |
|---|---|---|---|---|---|---|---|---|---|---|
| table column | NLA-AR | 42 | NLA explanation | 0.50 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 | 0.56 |
| table column | OL-AR | 44 | true text, last 32 tokens | nan | nan | 1.00 | nan | nan | 1.00 | nan |
| table column | OL-AR | 44 | true text, last 128 tokens | nan | nan | 1.00 | 1.00 | nan | 1.00 | nan |
| table column | OL-AR | 44 | OLens bullets | -0.06 | 0.18 | 1.00 | 0.66 | nan | 0.82 | 0.12 |
| table column | OL-AR | 44 | NLA explanation | -0.80 | 0.00 | 1.00 | 0.55 | nan | 0.95 | 0.01 |
| table column | OL-AR | 40 | true text, last 32 tokens | nan | nan | 1.00 | nan | nan | 1.00 | nan |
| table column | OL-AR | 40 | true text, last 128 tokens | nan | nan | 1.00 | 1.00 | nan | 1.00 | nan |
| table column | OL-AR | 40 | OLens bullets | 0.02 | 0.26 | 1.00 | 0.61 | nan | 0.69 | 0.11 |
| table column | OL-AR | 40 | NLA explanation | -0.73 | 0.00 | 1.00 | 0.58 | nan | 0.95 | 0.01 |
| inside a parenthesis | NLA-AR | 42 | NLA explanation | 0.71 | 1.00 | 0.92 | 0.89 | 0.76 | 0.91 | 0.64 |
| inside a parenthesis | OL-AR | 44 | true text, last 32 tokens | nan | nan | 0.94 | nan | nan | 0.93 | nan |
| inside a parenthesis | OL-AR | 44 | true text, last 128 tokens | nan | nan | 0.94 | 0.85 | nan | 0.88 | nan |
| inside a parenthesis | OL-AR | 44 | OLens bullets | -0.08 | 0.42 | 0.94 | 0.60 | nan | 0.90 | 0.18 |
| inside a parenthesis | OL-AR | 44 | NLA explanation | -0.53 | 0.00 | 0.94 | 0.56 | nan | 0.89 | 0.01 |
| inside a parenthesis | OL-AR | 40 | true text, last 32 tokens | nan | nan | 0.94 | nan | nan | 0.94 | nan |
| inside a parenthesis | OL-AR | 40 | true text, last 128 tokens | nan | nan | 0.94 | 0.86 | nan | 0.90 | nan |
| inside a parenthesis | OL-AR | 40 | OLens bullets | -0.03 | 0.54 | 0.94 | 0.56 | nan | 0.83 | 0.20 |
| inside a parenthesis | OL-AR | 40 | NLA explanation | -0.53 | 0.00 | 0.94 | 0.61 | nan | 0.90 | 0.02 |
| language (FR/DE) | NLA-AR | 42 | NLA explanation | 0.61 | 0.99 | 1.00 | 0.99 | 0.76 | 1.00 | 0.79 |
| language (FR/DE) | OL-AR | 44 | true text, last 32 tokens | nan | nan | 1.00 | nan | nan | 0.91 | nan |
| language (FR/DE) | OL-AR | 44 | true text, last 128 tokens | nan | nan | 1.00 | 0.92 | nan | 0.97 | nan |
| language (FR/DE) | OL-AR | 44 | OLens bullets | -0.25 | 0.13 | 1.00 | 0.92 | nan | 0.98 | 0.28 |
| language (FR/DE) | OL-AR | 44 | NLA explanation | -0.65 | 0.01 | 1.00 | 0.64 | nan | 0.98 | 0.03 |
| language (FR/DE) | OL-AR | 40 | true text, last 32 tokens | nan | nan | 1.00 | nan | nan | 0.94 | nan |
| language (FR/DE) | OL-AR | 40 | true text, last 128 tokens | nan | nan | 1.00 | 0.94 | nan | 0.98 | nan |
| language (FR/DE) | OL-AR | 40 | OLens bullets | -0.19 | 0.22 | 1.00 | 0.91 | nan | 0.97 | 0.32 |
| language (FR/DE) | OL-AR | 40 | NLA explanation | -0.67 | 0.00 | 1.00 | 0.63 | nan | 0.98 | 0.03 |
| bullet index | NLA-AR | 42 | NLA explanation | 0.56 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 0.51 |
| bullet index | OL-AR | 44 | true text, last 32 tokens | nan | nan | 1.00 | nan | nan | 1.00 | nan |
| bullet index | OL-AR | 44 | true text, last 128 tokens | nan | nan | 1.00 | 1.00 | nan | 1.00 | nan |
| bullet index | OL-AR | 44 | OLens bullets | -0.20 | 0.22 | 1.00 | 0.54 | nan | 0.84 | 0.07 |
| bullet index | OL-AR | 44 | NLA explanation | -1.02 | 0.00 | 1.00 | 0.52 | nan | 0.99 | 0.01 |
| bullet index | OL-AR | 40 | true text, last 32 tokens | nan | nan | 1.00 | nan | nan | 1.00 | nan |
| bullet index | OL-AR | 40 | true text, last 128 tokens | nan | nan | 1.00 | 1.00 | nan | 1.00 | nan |
| bullet index | OL-AR | 40 | OLens bullets | -0.04 | 0.33 | 1.00 | 0.53 | nan | 0.85 | 0.10 |
| bullet index | OL-AR | 40 | NLA explanation | -0.83 | 0.00 | 1.00 | 0.50 | nan | 0.98 | 0.01 |

## Statement shifts

| concept | ruler | layer | statement pair | shift along concept (1 = true gap) | cosine |
|---|---|---|---|---|---|
| table column | NLA-AR | 42 | nla:tb_tablecol:fixed | 0.07 | 0.10 |
| table column | NLA-AR | 42 | nla:tb_tablecol:m1 | 0.14 | 0.12 |
| table column | NLA-AR | 42 | nla:tb_tablecol:m3 | 0.10 | 0.11 |
| table column | NLA-AR | 42 | nla:tb_tablecol:m0 | 0.06 | 0.10 |
| table column | OL-AR | 44 | ol:ol:tb_tablecol:fixed | 0.03 | 0.01 |
| table column | OL-AR | 44 | ol:ol:tb_tablecol:m3 | 0.01 | 0.01 |
| table column | OL-AR | 44 | ol:ol:tb_tablecol:m0 | 0.01 | 0.01 |
| table column | OL-AR | 44 | ol:ol:tb_tablecol:m1 | 0.04 | 0.02 |
| table column | OL-AR | 44 | nla:nla:tb_tablecol:fixed | 0.00 | 0.00 |
| table column | OL-AR | 44 | nla:nla:tb_tablecol:m1 | -0.00 | -0.00 |
| table column | OL-AR | 44 | nla:nla:tb_tablecol:m3 | -0.10 | -0.02 |
| table column | OL-AR | 44 | nla:nla:tb_tablecol:m0 | 0.07 | 0.02 |
| table column | OL-AR | 40 | ol:ol:tb_tablecol:fixed | 0.06 | 0.02 |
| table column | OL-AR | 40 | ol:ol:tb_tablecol:m3 | 0.02 | 0.02 |
| table column | OL-AR | 40 | ol:ol:tb_tablecol:m0 | 0.05 | 0.04 |
| table column | OL-AR | 40 | ol:ol:tb_tablecol:m1 | 0.03 | 0.01 |
| table column | OL-AR | 40 | nla:nla:tb_tablecol:fixed | 0.00 | 0.01 |
| table column | OL-AR | 40 | nla:nla:tb_tablecol:m1 | -0.01 | -0.00 |
| table column | OL-AR | 40 | nla:nla:tb_tablecol:m3 | -0.11 | -0.03 |
| table column | OL-AR | 40 | nla:nla:tb_tablecol:m0 | 0.09 | 0.02 |
| inside a parenthesis | NLA-AR | 42 | nla:tb_paren:fixed | -0.05 | -0.04 |
| inside a parenthesis | NLA-AR | 42 | nla:tb_paren:m1 | -0.08 | -0.05 |
| inside a parenthesis | NLA-AR | 42 | nla:tb_paren:m3 | 0.19 | 0.10 |
| inside a parenthesis | NLA-AR | 42 | nla:tb_paren:m4 | -0.05 | -0.02 |
| inside a parenthesis | OL-AR | 44 | ol:ol:tb_paren:fixed | 0.10 | 0.04 |
| inside a parenthesis | OL-AR | 44 | ol:ol:tb_paren:m1 | -0.17 | -0.02 |
| inside a parenthesis | OL-AR | 44 | ol:ol:tb_paren:m0 | 0.14 | 0.02 |
| inside a parenthesis | OL-AR | 44 | ol:ol:tb_paren:m2 | 0.44 | 0.06 |
| inside a parenthesis | OL-AR | 44 | nla:nla:tb_paren:fixed | -0.10 | -0.06 |
| inside a parenthesis | OL-AR | 44 | nla:nla:tb_paren:m1 | 0.00 | 0.00 |
| inside a parenthesis | OL-AR | 44 | nla:nla:tb_paren:m3 | -0.65 | -0.09 |
| inside a parenthesis | OL-AR | 44 | nla:nla:tb_paren:m4 | -0.15 | -0.07 |
| inside a parenthesis | OL-AR | 40 | ol:ol:tb_paren:fixed | 0.12 | 0.04 |
| inside a parenthesis | OL-AR | 40 | ol:ol:tb_paren:m1 | -0.20 | -0.03 |
| inside a parenthesis | OL-AR | 40 | ol:ol:tb_paren:m0 | 0.14 | 0.02 |
| inside a parenthesis | OL-AR | 40 | ol:ol:tb_paren:m2 | 0.33 | 0.04 |
| inside a parenthesis | OL-AR | 40 | nla:nla:tb_paren:fixed | -0.09 | -0.05 |
| inside a parenthesis | OL-AR | 40 | nla:nla:tb_paren:m1 | 0.09 | 0.01 |
| inside a parenthesis | OL-AR | 40 | nla:nla:tb_paren:m3 | -0.87 | -0.12 |
| inside a parenthesis | OL-AR | 40 | nla:nla:tb_paren:m4 | -0.21 | -0.10 |
| language (FR/DE) | NLA-AR | 42 | nla:ij2_lang:fixed | 0.72 | 0.52 |
| language (FR/DE) | NLA-AR | 42 | nla:ij2_lang:m0 | 0.56 | 0.50 |
| language (FR/DE) | NLA-AR | 42 | nla:ij2_lang:m3 | 0.57 | 0.53 |
| language (FR/DE) | NLA-AR | 42 | nla:ij2_lang:m1 | 0.59 | 0.52 |
| language (FR/DE) | OL-AR | 44 | ol:ol:ij2_lang:fixed | 0.10 | 0.05 |
| language (FR/DE) | OL-AR | 44 | ol:ol:ij2_lang:m2 | 1.15 | 0.18 |
| language (FR/DE) | OL-AR | 44 | ol:ol:ij2_lang:m1 | 1.39 | 0.21 |
| language (FR/DE) | OL-AR | 44 | ol:ol:ij2_lang:m4 | 0.87 | 0.11 |
| language (FR/DE) | OL-AR | 44 | nla:nla:ij2_lang:fixed | 0.58 | 0.15 |
| language (FR/DE) | OL-AR | 44 | nla:nla:ij2_lang:m0 | 0.05 | 0.05 |
| language (FR/DE) | OL-AR | 44 | nla:nla:ij2_lang:m3 | 0.08 | 0.04 |
| language (FR/DE) | OL-AR | 44 | nla:nla:ij2_lang:m1 | 0.05 | 0.03 |
| language (FR/DE) | OL-AR | 40 | ol:ol:ij2_lang:fixed | 0.15 | 0.07 |
| language (FR/DE) | OL-AR | 40 | ol:ol:ij2_lang:m2 | 1.30 | 0.21 |
| language (FR/DE) | OL-AR | 40 | ol:ol:ij2_lang:m1 | 1.42 | 0.22 |
| language (FR/DE) | OL-AR | 40 | ol:ol:ij2_lang:m4 | 1.03 | 0.13 |
| language (FR/DE) | OL-AR | 40 | nla:nla:ij2_lang:fixed | 0.62 | 0.17 |
| language (FR/DE) | OL-AR | 40 | nla:nla:ij2_lang:m0 | 0.05 | 0.06 |
| language (FR/DE) | OL-AR | 40 | nla:nla:ij2_lang:m3 | 0.10 | 0.05 |
| language (FR/DE) | OL-AR | 40 | nla:nla:ij2_lang:m1 | 0.07 | 0.04 |
| bullet index | NLA-AR | 42 | nla:bk_listidx:fixed | 0.23 | 0.29 |
| bullet index | NLA-AR | 42 | nla:bk_listidx:m0 | 0.15 | 0.21 |
| bullet index | NLA-AR | 42 | nla:bk_listidx:m2 | 0.13 | 0.14 |
| bullet index | NLA-AR | 42 | nla:bk_listidx:m1 | 0.14 | 0.24 |
| bullet index | OL-AR | 44 | ol:ol:bk_listidx:fixed | 0.03 | 0.01 |
| bullet index | OL-AR | 44 | ol:ol:bk_listidx:m5 | 0.11 | 0.07 |
| bullet index | OL-AR | 44 | ol:ol:bk_listidx:m1 | 0.08 | 0.05 |
| bullet index | OL-AR | 44 | ol:ol:bk_listidx:m4 | -0.02 | -0.01 |
| bullet index | OL-AR | 44 | nla:nla:bk_listidx:fixed | 0.00 | 0.01 |
| bullet index | OL-AR | 44 | nla:nla:bk_listidx:m0 | 0.01 | 0.02 |
| bullet index | OL-AR | 44 | nla:nla:bk_listidx:m2 | 0.15 | 0.03 |
| bullet index | OL-AR | 44 | nla:nla:bk_listidx:m1 | 0.03 | 0.02 |
| bullet index | OL-AR | 40 | ol:ol:bk_listidx:fixed | -0.03 | -0.01 |
| bullet index | OL-AR | 40 | ol:ol:bk_listidx:m5 | 0.13 | 0.07 |
| bullet index | OL-AR | 40 | ol:ol:bk_listidx:m1 | 0.07 | 0.04 |
| bullet index | OL-AR | 40 | ol:ol:bk_listidx:m4 | 0.02 | 0.01 |
| bullet index | OL-AR | 40 | nla:nla:bk_listidx:fixed | -0.00 | -0.01 |
| bullet index | OL-AR | 40 | nla:nla:bk_listidx:m0 | 0.03 | 0.04 |
| bullet index | OL-AR | 40 | nla:nla:bk_listidx:m2 | -0.07 | -0.01 |
| bullet index | OL-AR | 40 | nla:nla:bk_listidx:m1 | 0.02 | 0.01 |

## OL-AR note

```
{
 "tb_tablecol": "head output is in the AR's whitened space (moments gated: agu18dec/local-workspace); analysis compares reconstructions inside that space",
 "tb_paren": "head output is in the AR's whitened space (moments gated: agu18dec/local-workspace); analysis compares reconstructions inside that space",
 "ij2_lang": "head output is in the AR's whitened space (moments gated: agu18dec/local-workspace); analysis compares reconstructions inside that space",
 "bk_listidx": "head output is in the AR's whitened space (moments gated: agu18dec/local-workspace); analysis compares reconstructions inside that space"
}
```
