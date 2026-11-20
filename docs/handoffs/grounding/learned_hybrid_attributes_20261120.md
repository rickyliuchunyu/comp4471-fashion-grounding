# 方向与掩码颜色的独立学习和组合推理

保持原 1523 图 / 5220 条训练划分、64 张 MHP 内部选模图和同一份冻结的 M2-binding 主排序权重。额外训练文本方向、类别和颜色置信度头。方向/颜色存在的弱标签由训练查询词表产生，类别来自训练标注。颜色排序只用 MHP 同图两目标的掩码内颜色直方图学习。训练不读取外部 MHP dev 或 Fashionpedia dev 答案。

训练方向标签 {'1': 1059, '2': 990, '0': 3070, '-100': 60, '3': 17, '4': 24}（0/1/2/3/4 对应无方位/左/右/上/下，−100 代表跳过）。颜色学习使用 89 个配对组。种子42，32轮。加分参数只用64图选定，位置、类别、颜色权重为 [4.0, 2.0, 8.0]。充分方向解析时用固定规则，否则用学习的属性分数。该组合方案由开发结果启发，比分量结果更偏探索，仍需另留人审图复核。

| 方法 | FP 冻结候选严格选对 /642 | FP 掩码达标 /642 | MHP GT 方位四条全对 /200 | MHP 分割方位四条全对 /200 | MHP GT 颜色四条全对 /59 | MHP 分割颜色四条全对 /59 |
|---|---:|---:|---:|---:|---:|---:|
| M2-binding | 510 | 565 | 1 | 2 | 24 | 15 |
| B2-rule | 494 | 544 | 102 | 47 | 13 | 7 |
| M2-B2-rule | 522 | 577 | 180 | 96 | 24 | 15 |
| M2-learned-direction | 519 | 574 | 41 | 20 | 24 | 15 |
| M2-learned-attributes | 552 | 583 | 41 | 20 | 44 | 26 |
| M2-hybrid-attributes | 556 | 587 | 180 | 96 | 44 | 26 |

FP 使用 lcy 的冻结 S0 组合权重，候选覆盖 610/642。MHP 草稿使用其已发布候选，能同时匹配两件的方位对有130/200，颜色对有42/59。单条选对、同语双对、掩码成功和配对差值的区间见证据。纯学习属性推理不调用词表，组合方案的显式方位分单独记录。

逐条核对组合输出，命中规则的分数等于同输入 M2 加规则对照，其余分数等于学习属性对照。候选 ID 与排序顺序也一致。72 项原有测试、对向排序的手算检查和属性权重重载检查通过。两次运行均只复用已保存的区域和文本特征。

学习和对照目录 `/root/autodl-tmp/fashion-grounding/runs/grounding/learned-attributes-20261120-r01`，可直接复用的组合推理结果在 `/root/autodl-tmp/fashion-grounding/runs/grounding/hybrid-attributes-20261120-r03`。注：模型权重和逐条预测留在服务器。

从 wyz 工作区使用 grounding-v1 环境，配置固定原权重、属性头、分割候选及哈希。只复跑已审核的主 dev 可用：

```bash
export PYTHONPATH=src:. CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1
/root/autodl-tmp/fashion-grounding/envs/grounding-v1/bin/python scripts/evaluate_hybrid_attributes.py \
  --release configs/grounding_hybrid_attributes_20261120.json \
  --candidates /root/autodl-tmp/fashion-grounding/shared/candidates/combo-u28318-dev250-20261119-r01/selected_candidates/candidates_predicted.json \
  --conditions fp_pred --output "$NEW_RUN"
```

输入、配置、命令、权重和每项对照见 [运行证据](learned_hybrid_attributes_20261120_evidence.json)。
