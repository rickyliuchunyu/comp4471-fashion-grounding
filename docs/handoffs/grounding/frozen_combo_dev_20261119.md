# Fashionpedia dev250 组合分割候选签收

lcy 发布的组合分割权重 `b7146787a1fc8f16f4694f2a9d05c294a49090b095de901b63b963f162a640c1` 使用阈值 0.3。wyz 的 dev 配置固定该发布版本和以下候选，显式传入 `--candidates`。已验证误传旧版本会在创建运行目录前被拒绝。

```text
/root/autodl-tmp/fashion-grounding/shared/candidates/combo-u28318-dev250-20261119-r01/selected_candidates/candidates_predicted.json
4daf79030f88bc285d35272cc8e7160e0143d99216b05740a3569f946bb186bd
```

候选集 `combo-u28318-dev250-20261119-r01-trial-0.3` 包含 250 张图和 1320 个候选。642 条已审核中英文查询覆盖 246 图，与上一版查询及 GT 标注字节一致。按同类一对一 IoU ≥ 0.5 匹配，覆盖由 602/642 到 610/642。模型权重和 B2 权重 2 沿用原开发评价。所有结果均属开发集。

| 方法 | 旧 50k 严格选对 /642 | 新组合严格选对 /642 | 旧 50k 掩码达标 /642 | 新组合掩码达标 /642 |
|---|---:|---:|---:|---:|
| B1 | 478 | 483 | 519 | 533 |
| B2-selected | 490 | 494 | 531 | 544 |
| M1-clean1000 | 499 | 492 | 548 | 554 |
| M2-clean1000 | 503 | 499 | 554 | 565 |
| M2-FP | 503 | 497 | 552 | 564 |
| M2-binding-pair | 508 | 512 | 558 | 565 |
| B0 | 8 | 10 | 8 | 10 |

每种方法均产出完整 642 行输出。B0 不支持的文本仍计入总分母。复核了所有输出的候选集 ID、manifest 哈希和评价输入哈希。区域特征重编码 246 图，全部以新版候选 ID 和哈希为键。492 个不同文本与语言组合在编码器签名一致时从原缓存复用，未重新编码文本。

运行目录为 `/root/autodl-tmp/fashion-grounding/runs/grounding/combo-u28318-reviewed-dev-20261119-r01`。新区域特征在 `features/`，分数、预测与评价在 `predicted/`，B0 的输出在 `B0/`。既有 50k、GT 和 MHP 运行可从相应历史配置复现。

使用 grounding-v1 环境，从仓库根目录执行以下命令。复跑指定新的输出目录与 run ID。

```bash
export PYTHONPATH=src:. CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1
/root/autodl-tmp/fashion-grounding/envs/grounding-v1/bin/python scripts/evaluate_frozen_grounding_dev.py \
  --config configs/grounding_fashionpedia_dev_current.json \
  --candidates /root/autodl-tmp/fashion-grounding/shared/candidates/combo-u28318-dev250-20261119-r01/selected_candidates/candidates_predicted.json \
  --output "$NEW_RUN" --run-id "$NEW_RUN_ID"
```

72 项既有读取、排名和评价测试通过。按 lcy 的发布回执核对冻结权重与阈值。见 [冻结记录](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/341f4313fc95d55b929aecf3e97e55b91a660c8a/docs/handoffs/segmentation/current_candidates.md) 和 [运行证据](frozen_combo_dev_20261119_evidence.json)。
