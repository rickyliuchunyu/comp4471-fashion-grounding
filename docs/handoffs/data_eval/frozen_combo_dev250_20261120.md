# wkq receipt for the frozen S0 Fashionpedia development evaluation

On 2026-11-20 wkq verified the published inputs and all seven prediction/evaluation pairs on AutoDL. Grounding passed the pinned manifest as its explicit candidates argument. The serving threshold is 0.3 and the selected checkpoint starts with b7146787a1fc. The 250-image candidate set contains 1,320 masks. The query file matches the original reviewed package byte for byte, with 642 accepted queries on 246 images.

| Input | SHA-256 |
|---|---|
| queries | `ac8d6af9d1b854452627b2f6d72aed58a91bee0642344ac37932f8701fb841d9` |
| candidates | `4daf79030f88bc285d35272cc8e7160e0143d99216b05740a3569f946bb186bd` |
| annotations | `74837c386c4bbd2805f9976fcfd796931b2f6cfa72693b91d2f441bee473c635` |

| Method | Strict instance correct / 642 | Mask IoU >= 0.5 / 642 |
|---|---:|---:|
| B1 | 483 | 533 |
| B2-selected | 494 | 544 |
| M1-clean1000 | 492 | 554 |
| M2-clean1000 | 499 | 565 |
| M2-FP | 497 | 564 |
| M2-binding-pair | 512 | 565 |
| B0 | 10 | 10 |

All seven evaluations have 610/642 covered queries under same-class one-to-one mask matching. For M2-binding-pair, 32 targets lack an associated candidate and 98 queries select incorrectly despite coverage. The wkq evaluator independently reproduced the entire M2-binding-pair result from the archived predictions and the original accepted queries. For the six other methods wkq verified hashes and recounted archived per-query results.

wkq output directory: `/root/autodl-tmp/fashion-grounding/runs/data_eval/20261120-frozen-combo-dev250-r01`. The [machine-readable receipt](frozen_combo_dev250_20261120.json) records the absolute input paths, file hashes and upstream run. GT results can be reused for identical queries, annotations, model and scoring protocol. The MHP draft challenge retains its separate candidate version, and final test evaluation remains pending.
