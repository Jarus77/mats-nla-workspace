| arm | typo_mt | multilingual_mt | multihop_mt | basic_readout_mt | mean abs diff vs frozen |
|---|---|---|---|---|---|
| frozen floor (empty cells) | 0.67 (3) | 0.42 (4) | 0.69 (1) | 0.59 (1) | – |
| floor_think_off (empty / </think> closed) | 0.83 (0 / 0) | 0.47 (0 / 0) | 0.84 (0 / 0) | 0.80 (0 / 0) | 0.143 |
| floor_think_on (empty / </think> closed) | 0.64 (0 / 0) | 0.59 (0 / 0) | 0.78 (0 / 0) | 0.54 (0 / 0) | 0.085 |
| floor_think_on_answer_only (empty / </think> closed) | 0.00 (100 / 0) | 0.00 (100 / 0) | 0.00 (100 / 0) | 0.00 (100 / 0) | 0.593 |
| floor_raw (empty / </think> closed) | 0.80 (0 / 50) | 0.45 (0 / 53) | 0.79 (0 / 41) | 0.65 (0 / 53) | 0.080 |

Cells hitting the 256-token cap: {"think_off": {"typo_mt": 98, "multilingual_mt": 90, "multihop_mt": 92, "basic_readout_mt": 84}, "think_on": {"typo_mt": 100, "multilingual_mt": 100, "multihop_mt": 100, "basic_readout_mt": 100}, "raw": {"typo_mt": 96, "multilingual_mt": 100, "multihop_mt": 98, "basic_readout_mt": 93}}