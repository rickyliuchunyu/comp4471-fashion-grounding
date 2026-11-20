# wkq → wyz：候选 Transformer 三轮实验与排序端接入

**2026-11-20 · 接收方 wyz（grounding）· 开发集实验，待接收方联调。** 建议将“属性监督 Transformer＋类别与水平位置校准”作为独立排序方法接入，以种子 42 复现；同时保留“只加预测类别”的消融。前者在 MHP 分割候选上的左右双问最好，后者在颜色双问和 Fashionpedia 总体上略好。三轮实验均冻结 SigLIP 2，使用 CPU。主结果保存在 wkq 的独立工作区。

## 三轮实验分别回答了什么

| 实验 | 对照与协议 | 主要发现 |
|---|---|---|
| 1 · 候选之间是否需要交互 | 相同的 5,604 条训练查询清单、采样、配对损失、490 步；用 Fashionpedia GT dev 选权重。对照原 M2-binding、成对 MLP、一层候选 Transformer。 | 种子 42 的 MHP GT 左右双问分别为 32、10、323 /400。Transformer 在种子 43、44 为 257、68 /400，初始化敏感。 |
| 2 · 能否稳住方位并保住颜色 | 从训练包按图留出 64 张 MHP 图、384 条查询，其余 1,523 张有查询图、5,220 条用于拟合。每 98 步用内部 GT 的方位与颜色双问准确率等权选模。原 Transformer 在同一新划分上重训。 | 加入查询对候选 token 的调节，并用训练图目标与干扰项的水平排名、mask 颜色差监督属性映射。三个种子平均，MHP GT 左右由 77.6% 到 80.7%，预测颜色由 31.9% 到 39.5%，预测左右由 44.8% 到 44.9%。 |
| 3 · 怎样改善真实分割候选上的排序 | 固定第 2 轮三个属性监督检查点。从拟合集训练文本类别映射；用旧 50k 分割器在同一批内部 64 图上真实推理，在 64 组预设系数中选择三个种子共用的一组。 | 最终系数按“类别、置信度、绝对水平位置”的顺序为 `[5, 0, 10]`。类别信息贡献最大；加入水平位置后，左右继续提升，颜色和总体分数有小幅取舍。 |

第 1、2 轮使用不同拟合划分和选模标准，跨轮差值包含协议变化。第 3 轮为推理分数校准，正文中的“完整版”指标包含这一层。模型通过训练数据学习类别和方向，推理输入为原查询与候选特征；没有用关键词解析器或人工目标筛候选。

## 同一批输入下的第 3 轮成绩

三个种子的准确率算术平均。MHP 每个同语言双问要求同图两条相反描述各自选中正确目标。方向分母为 400 组（200 图 × 2 种语言）；颜色为 118 组（59 图 × 2 种语言）。Fashionpedia 一列是 642 条查询的单条严格选对率。

| 第 2 轮属性监督模型上的校准 | MHP GT 左右 | MHP 预测左右 | MHP GT 颜色 | MHP 预测颜色 | Fashionpedia 预测严格选对 |
|---|---:|---:|---:|---:|---:|
| 未校准 | 80.67% | 44.92% | 74.29% | 39.55% | 83.54% |
| 仅置信度 [0, 1, 0] | 80.67% | 43.17% | 74.29% | 42.09% | 86.24% |
| 仅类别 [5, 0, 0] | 86.58% | 52.33% | **76.84%** | **46.61%** | **92.00%** |
| 完整版 [5, 0, 10] | **90.67%** | **55.08%** | 74.29% | 44.35% | 91.69% |

完整版的 Fashionpedia 严格选对依次为 **592、586、588 /642**（种子 42、43、44）。种子 42 在同类多件子组由 46/76 提升到 61/76。在 MHP 实际分割候选上，它在左右双问答对 222/400；140 组缺至少一个匹配候选，另外 38 组候选齐全但排序失败。颜色答对 53/118；34 组候选不全，31 组候选齐全但排序失败。

## 固定输入与验收数字

- **Fashionpedia** 使用 `/root/autodl-tmp/fashion-grounding/shared/candidates/combo-u28318-dev250-20261119-r01/selected_candidates/candidates_predicted.json`，SHA-256 为 `4daf79030f88bc285d35272cc8e7160e0143d99216b05740a3569f946bb186bd`。S0 权重前缀 `b7146787a1fc`，候选阈值 0.3。原 642 条已审核查询的 SHA-256 为 `ac8d6af9d1b854452627b2f6d72aed58a91bee0642344ac37932f8701fb841d9`。候选覆盖 **610/642**；种子 42 完整版严格选对 **592/642**、选中 mask IoU ≥ 0.5 为 **606/642**。[wkq 冻结签收](frozen_combo_dev250_20261120.md)
- **MHP** 使用 `shared/processed/mhp2_binding_dev_v2/queries_draft.jsonl` 中 1,036 条查询。GT 使用该包的 `candidates_gt.json`；预测使用 `shared/candidates/wkq-mhp2-binding-dev-lcy50k-t03-v2/candidates_predicted.json`，对应旧 50k 权重、阈值 0.3。种子 42，官方候选左右/颜色双问应为 **363/400、88/118**；旧 50k 候选应为 **222/400、53/118**。wkq 已反馈人工核对无问题，本次评分沿用归档的草稿查询文件和草稿模式。[MHP 数据交接](mhp2_binding_expansion_20261118.md)

## wyz 接入入口

AutoDL 上的公共目录前缀为 `/root/autodl-tmp/fashion-grounding/workspaces/data_eval/outputs/`。下列 A、S、C 表示该前缀下的三个目录。

| 目录 | 文件与用途 |
|---|---|
| A = `20261120-ranker-architecture-r01` | `architectures.py` 定义基础候选 Transformer；`signature.json` 为冻结编码器签名，`inputs.json` 为输入路径，`source/` 是 grounding 源码快照（`4bf19fe2cbcd8ebb94ebb5ed4912646be73c6638`）。 |
| S = `20261120-ranker-stability-r01` | `models.py` 中的 `AttributeTransformer` 是待加载类。三个检查点在 `seed-{42,43,44}/supervised/selected.pt`，`selection.json` 记录各自的选模依据。 |
| C = `20261120-candidate-context-r01` | **从 `deploy.json` 开始**：包含精确路径、三个权重哈希、类别映射哈希及校准权重。使用 `category_fit.npy`、`rerank.py`。完整消融在 `summary.json`，内部选参在 `selection.json`，统一评价输出在 `eval-{fp_combo,mhp_gt,mhp_pred}/`。 |

**接入顺序。** 在独立 run 中加载 `deploy["upstream"]["42"]["path"]`，核对 SHA-256 为 `9344be674728684cb84fce555b7f3ff0ad466c606b5b3ab19771e5fa7850e129`。用 `AttributeTransformer(state["dimension"])` 初始化，加载 `state["model"]` 后调用 `eval()`。将 A、S、C 纳入模块搜索路径，使用与 A 中签名一致的冻结编码器。读取类别映射 `np.load(deploy["category_map"], allow_pickle=False)`。类别 ID 范围为 1–8，候选顺序贯穿整个计算。

下面是单条查询的评分代码。使用 `grounding.predict.rank_scores`、`grounding.spatial.score_features`（均位于 `fashion_grounding` 包下）以及 C 中的 `rerank.candidate_features/calibrated_scores`。序列为空时沿用现有无候选处理。其余情况下，`visual` 为按候选框裁剪得到的视觉特征，`text` 为形状 [1, D] 的文本向量，`geometry` 为 `binding_geometry` 生成的 [N, 24] 数组，依次包含原 12 维几何与 mask 内 12 维颜色直方图。`category_ids`、`detector_scores` 按 manifest 顺序取自候选。最终按最高分返回原候选 ID。

```python
with torch.inference_mode():
    base = rank_scores(visual, text, signature["logit_scale"], signature["logit_bias"])
    raw = score_features(model, visual, text, geometry, base)
    features = candidate_features(text, geometry, category_ids, detector_scores,
                                  category_map, model.cue_map.cpu().numpy())
    scores = calibrated_scores(raw, features, deploy["weights"])
```

**接收检查。** wkq 已核对 `deploy.json` 中三个检查点、类别映射、推理函数的 SHA-256；用上述入口和已有冻结特征重放 Fashionpedia 种子 42 的 642 条查询，所有候选顺序及第一名与归档一致，分数最大绝对误差小于 0.000001。wyz 可先在自己的新输出目录复现此结果，再做只加类别的对照。校准分最多加一次。新候选 manifest 需要重新建立区域特征和预测记录。

**待验证。** MHP 换用冻结组合 S0 后的端到端收益尚未测量。建议固定语义权重和上述校准系数，仅替换 MHP 的分割候选，重新统计双目标覆盖、两件都检出时的选对率、全量双问选对率。外部 MHP dev 已在多轮研究中用于诊断，最终独立测试仍待团队冻结后开展。
