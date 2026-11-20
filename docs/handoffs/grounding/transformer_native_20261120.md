# 候选 Transformer 排序端接入与开发集验收

2026-11-20，wyz（grounding），P6 开发集联调。上游用 wkh 的 [Transformer 交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/8945d0b00c66a4ef3c474ca621b80bfb17b2ff7b/docs/handoffs/data_eval/transformer_ranker_to_wyz_20261120.md)。原生入口是 `python -m fashion_grounding.grounding.predict`。方法 `transformer-full` 使用属性 Transformer、预测类别和水平位置校准，固定系数 [5, 0, 10]。方法 `transformer-category-only` 使用同一排序权重、系数 [5, 0, 0]，用来量化水平位置校准的贡献。主方案候选默认使用交接指定的种子 42。

模型结构和打分在仓库中。三个种子的检查点与类别映射已按哈希快照到 grounding 运行目录，推理入口不导入其他成员工作区的 Python 代码。冻结 SigLIP 2 revision 为 `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`。类别与位置校准读取查询的冻结文本向量及候选字段。评价单独接收 GT 注释。真实空缓存检查从一张图和两条查询写入了 3 项特征。修改这两条查询的人工答案、目标类别和查询类型后，输出选择及分数逐条相同。

| 方法及候选，种子 42 | 严格选对 | 掩码 IoU ≥ 0.5 | 方位双问全对 | 颜色双问全对 |
|---|---:|---:|---:|---:|
| Fashionpedia 冻结 S0 · 上一版混合排序 | 556/642 | 587/642 | | |
| Fashionpedia 冻结 S0 · Transformer 完整版 | 592/642 | 606/642 | | |
| Fashionpedia 冻结 S0 · 仅类别校准 | 593/642 | 607/642 | | |
| MHP 官方 GT 候选 · Transformer 完整版 | 968/1036 | 968/1036 | 363/400 | 88/118 |
| MHP 新版组合 S0 · 上一版混合排序 | 758/1036 | 800/1036 | 247/400 | 68/118 |
| MHP 新版组合 S0 · Transformer 完整版 | 797/1036 | 831/1036 | 250/400 | 70/118 |
| MHP 新版组合 S0 · 仅类别校准 | 778/1036 | 808/1036 | 232/400 | 70/118 |

双问全对按同图、同语言的两条相反描述统计。Transformer 完整版在 Fashionpedia 三个种子 42、43、44 的严格选对分别为 592、586、588 / 642，平均 91.69%。和上一版在相同输入上的逐图配对比较显示，主 dev 提升 5.61 个百分点（95% 区间 +3.12 至 +8.33）。MHP 新版候选双问差值的区间跨零。该开发诊断沿用有版本记录的 MHP 草稿查询，正式保留集待团队冻结。

Fashionpedia 候选的 SHA-256 为 `4daf79030f88bc285d35272cc8e7160e0143d99216b05740a3569f946bb186bd`，MHP 新版组合候选为 `20ac7a9a49f40ddb3345c7456695123021294e37748c7cc4f1e60e8588ca599c`。两者的 S0 权重摘要为 `b7146787a1fc`、候选阈值 0.3。开发集配置绑定候选、查询和注释字节。模型发布配置记录检查点、编码器签名、类别映射及打分系数。

## 复跑

从以下工作区在 CPU 运行，命令使用已安装的 `grounding-v1`。开发集脚本调用标准预测入口和团队统一评价器。范围可以用 `--conditions fp_combo` 缩小，或用 `--seed 43` 指定已发布的种子。

```bash
cd /root/autodl-tmp/fashion-grounding/workspaces/grounding
export PYTHONPATH=src:. HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
NEW_RUN="/root/autodl-tmp/fashion-grounding/runs/grounding/transformer-recheck-$(date +%Y%m%d-%H%M%S)"
/root/autodl-tmp/fashion-grounding/envs/grounding-v1/bin/python -u scripts/evaluate_transformer_dev.py \
  --config configs/grounding_transformer_dev_20261120.json --output "$NEW_RUN"
```

模型配置在 [冻结排序发布](../../../configs/grounding_transformer_20261120.json)，输入在 [开发集配置](../../../configs/grounding_transformer_dev_20261120.json)。逐条预测和运行证据保存在 `/root/autodl-tmp/fashion-grounding/runs/grounding/transformer-native-dev-20261120-r01`，其中 `verification.json` 记录逐条复核、运行参数与测试摘要。另两个种子在 `/root/autodl-tmp/fashion-grounding/runs/grounding/transformer-native-seeds-20261120-r01`。CPU 测试在图文与数据环境分别运行，178 项中 173 项通过、5 项跳过。图文环境的 NumPy 1.26 跑数据构建测试时缺少 `bitwise_count`，相关模块用团队现有的 NumPy 2 数据环境验证。

## 团队复核

- wkh 用相同候选、查询和预测文件核对统一评价，继续冻结最终测试查询和标注。开发结果及正式测试分别登记。
- lcy 如需改进 MHP，优先查看新版候选下方位双问的漏候选组。完整版 150 组方位失败中有 110 组缺至少一个匹配候选，40 组候选齐全但排序失败。
- wyz 的排序接口开发验收已完成。最终测试和三方方法冻结待团队确认。
