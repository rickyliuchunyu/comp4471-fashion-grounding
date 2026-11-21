# 50k 初始化的全量分割单线训练准备

2026-11-18，lcy / segmentation。已生成固定取图顺序和两阶段命令。**当前训练更新数为 0，权重仍用已交付的 50k。** 下次获得启动授权后，一条训练线完成 28,318 次成功更新，半程与末端各评一次完整 dev250。[运行计划](../../../configs/runs/segmentation/full_single_plan.json) · [训练配置](../../../configs/runs/segmentation/full_single_train.json)

## 输入与优化器

- 数据版本为 `fp_train_full_eligible_v1`，42,476 图、152,848 个实例。训练及 dev 入口核对同一份输入绑定，复用旧 10k 与 32,476 张新增图的验收。合并的 146 张曝光图保护清单 SHA256 为 `6cecceaf3215830f2cb88626f97fa8ee5b29ec5049d9824c1187aedb94fd86fd`。采样排除了其项目 ID、官方源键、分组和文件 SHA。[输入绑定](../../../configs/runs/segmentation/full_single_inputs.json) · [历史组裁决](full_train_protection_20261118.md)
- 起点为旧 50k 的 `checkpoint_050000.pth`，SHA256 为 `1ed0f913602746e027ae84ec9a84c98a80306b34820a824d39ddafba1671208a`。严格加载模型与八类头参数；新运行从 0 步建立 AdamW、学习率调度、AMP、采样和随机状态。初始配方选用基础 LR 2×10⁻⁵、backbone 倍率 0.1、500 步 warmup 后 cosine 衰减至基础 LR 的 2%。这组值用于成熟模型短微调，收益待测。沿用 768、LSJ 0.5–2、水平翻转、有效 batch 2。[模型来源](../../../configs/runs/segmentation/full_single_model_only.json)

## 采样结果

基础流无放回打乱全部训练图，额外流轮流从五个源细类取图。入池条件为非 crowd 且原尺寸 mask 面积在 1–1,024 像素。每个额外位置优先挑该池内全局额外次数最少的组，同一图上的全部八类标注照常进入原训练 mapper。种子固定为 42；每图组额外最多 4 次。根据两条流的实际长度均匀穿插，两半各含 21,238 个基础位置和 7,080 个额外位置。

| 源细类 | 合格图片 | 独立组 | 定向取图次数 |
| --- | ---: | ---: | ---: |
| 袜 | 964 | 964 | 2,832 |
| 腰带 | 1,338 | 1,337 | 2,832 |
| 鞋 | 6,543 | 6,542 | 2,832 |
| 头饰（headband/head covering/hair accessory） | 1,504 | 1,504 | 2,832 |
| 手表 | 2,386 | 2,385 | 2,832 |

实测 56,636 个位置覆盖 42,476 张不同图。14,160 个额外位置来自 8,416 张不同图片、8,414 个独立组。一图可能符合多个池条件，配额按本次取图所属的池统计。这些数字描述取图时原始标注的覆盖，随机增强后的目标可见性及模型效果尚未测量。[CPU 采样回执](evidence/full_single_line_20261118/sampling_receipt.json)

服务器固定顺序位于 `/root/autodl-tmp/fashion-grounding/runs/segmentation/full-single-sampling-20261118-r01/order.json`，SHA256 为 `4f4933a5739f954a9941efa450192cb0491fe051398b49e0667f790f5a45d58d`。运行时从顺序文件把真实图片 ID 映射到训练记录；checkpoint 保存顺序摘要和游标。CPU 已按实际半程位置验证后半段逐项一致、AMP 重放时的下一批一致，以及末尾越界拒绝。另有三个仅针对采样器的边界案例，覆盖小目标面积边界、crowd、同图多实例、同组跨图限次和禁止组碰撞。[采样实现](../../../src/fashion_grounding/segmentation/full_sampling.py) · [采样验证](../../../scripts/prepare_full_single_sampling.py) · [新增用例](../../../tests/test_full_sampling.py)

## 评价、选择与时间

第 14,159 次成功更新保存权重并立即评 dev250，然后从这个 checkpoint 恢复所有训练状态。第 28,318 步保存末端权重，评同一 dev250。两份新报告均采用原 250 图 / 1,008 个 GT、800/1333、FP32、上游实例解码、top100 和无分数阈值的 COCO mask AP 协议。旧 50k 报告直接复用，AP 69.0313、AP_small 23.1082、AP75 73.4854。以完整 dev mask AP 最大者选权重，新旧同分留旧，新组内同分取较早步数。一并记录所选模型相对 50k 的 AP_small、AP75 变化。失败或超时保留已完成阶段及回执，并标记整次运行失败。[旧 50k 实测](evidence/tenk_completion_20261118/epoch_10/evaluation_report.json)

总硬上限 **18,000 秒（5 小时）**，从启动入口计时，覆盖初始化、两段训练、两次 dev、选模及退出；半程恢复沿用同一绝对截止，不续时。按旧末段 10,000 步约 4,252 秒估算，纯训练约 3.35 小时，实际取决于实例密度和 I/O。CPU 准备及锁检查时数据盘可用 130 GiB，RTX 3090 显存占用 0 MiB。本次未读取 checkpoint 权重，未运行 GPU、推理、A/B 或 smoke。

只查看准备配置，在仓库根目录执行：

```sh
PYTHONPATH=src python scripts/run_tenk_adaptation.py --plan configs/runs/segmentation/full_single_plan.json
```

该入口默认只打印计划。后续明确启动时从相同版本的服务器分割 checkout 使用同一入口和显式 `--execute`；启动器才在独立输出目录创建授权配置，并在共享 GPU 锁内运行。正式训练的输出位置见计划中的 `future_output`。本次只产生采样准备目录。

开工已核对三名成员的分支、全部五个 PR 及讨论。同步前 wkq 发布了训练划分内的同类监督审计，wyz 发布配对目标函数对照；两份更新均未改变本轮的训练图、保护清单或分割权重。[GitHub 快照](evidence/full_single_line_20261118/github_snapshot.json) · [wkq 交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/ae08c8de8491256724123f5f2b126b0a7faa74b2/docs/handoffs/data_eval/binding_supply_20261118.md) · [wyz 交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/2c47c3609080a5a1f35914a0ed2269fcba47817d/docs/handoffs/grounding/pair_objective_20261118.md)
