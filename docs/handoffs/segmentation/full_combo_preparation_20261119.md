# 全量分割组合微调准备

2026-11-19，lcy / segmentation，Codex 执行。状态为 **prepared_not_started**。服务器已生成新取图顺序，实际入口已解析两个训练与完整 dev 阶段。训练更新、整模型前向和 dev 评价次数均为 0。[准备回执](evidence/full_combo_20261119/preparation.json) · [训练计划](../../../configs/runs/segmentation/full_combo_plan.json)

## 输入、学习率与评价

起点是已交付的完整 dev250 最佳权重，28,318 步，整体 mask AP 71.3205、AP75 76.3129、小目标 AP 23.4459。服务器文件实算 SHA-256 为 `aaff8d9dd89e5aad4be0d6304eaa12dc0c8bb880395722220fa1f12af33f7809`。首段使用 `--model-only` 严格加载模型参数及八类头。从新阶段第 0 步建立 AdamW、学习率调度器、AMP、采样器及随机状态。第二段从本轮半程 checkpoint 完整恢复。[源权重与身份](../../../configs/runs/segmentation/full_combo_model_only.json) · [已有完整 dev 报告](full_single_completion_20261118.md)

工程初值为头部峰值 LR 1×10⁻⁵、主干 1×10⁻⁶，300 次更新预热后余弦衰减到峰值的 2%。有效 batch 为 2。规划新增 28,318 次成功更新，第 14,159、28,318 步各用原协议评完整 dev250（800/1333、FP32、top100）。两段训练、评价、选模及收尾共享 28,800 秒硬截止，失败即停。比较两份新报告与缓存的当前最佳报告，按整体 mask AP 选权重，同分优先现有版本，新权重之间同分选较早步数。通过文件路径和 SHA-256 区分新旧两个“28,318 步”。实际效果、混合尺寸的吞吐和峰值显存待运行后记录。[训练配置](../../../configs/runs/segmentation/full_combo_train.json)

## 取图与尺寸

复用既有全量包、八类映射、dev 验收及 146 张曝光保护清单。每张基础图取一次，按种子 43 打乱。每个定向位置须包含相应源细类的非 crowd 实例，原图掩膜面积 1–1,024 像素。同一组最多额外取 4 次。真实顺序共有 56,636 个取图位置，定向部分来自 8,345 张不同图片。[采样回执](evidence/full_combo_20261119/sampling_receipt.json)

| 训练边长 | 基础取图 | 定向小鞋 | 定向袜子 | 定向腰带 |
| --- | ---: | ---: | ---: | ---: |
| 768 | 21,209 | 3,552 | 1,782 | 1,775 |
| 896 | 21,267 | 3,528 | 1,758 | 1,765 |
| 合计 | 42,476 | 7,080 | 3,540 | 3,540 |

每两个成功更新各用一次 768 和 896，批内两图同尺度，每对的次序由种子和绝对更新号确定。每种尺寸共 14,159 步。沿用原 LSJ、水平翻转和零填充。顺序文件位于 `/root/autodl-tmp/fashion-grounding/runs/segmentation/full-combo-sampling-20261119-r01/order.json`，实算 SHA-256 为 `07a51a52b1a1cb8a5468c542bc3a5a3f5adec88ae977bb46a3a0942cd14bdde6`。表中次数表示原图被取到，增强后实际可见性由训练记录说明。

## 采点观测与候选评分

每 250 步观察开头连续两步，覆盖两种尺寸。各阶段的 `train/metrics.jsonl` 记录实际训练边长；观测行的 `point_observations` 按最终解码层实际匹配的实例记录图片、标注 ID、源细类、原图与增强后面积、总点数、前景点和边界点数。前景采用真实双线性 GT 标签 ≥0.5；边界按采点所在像素，用训练尺寸下的 8 邻域一像素内外带计算，两类计数可重叠。分母为增强后存留且参与匹配的实例。只在成功更新行汇总，AMP 失败尝试独立记录。当前仅观测采点，未更改损失函数。[观测实现](../../../src/fashion_grounding/segmentation/point_observations.py)

新导出的阈值前记录及离线候选包保留真实 decoder query ID、原有 `score`，并增加 `class_probability` 与 `mask_quality`。前者是含无目标类别的 softmax 后选中前景类的概率，后者是正掩膜像素 sigmoid 值之和除以正像素数加 1e-6。原有 float32 打分顺序、阈值边界和排序保留。历史缓存仍可按其原配置离线过滤，原分项无法从乘积唯一还原。当前 GUI 和下游交付仍指向已发布版本。末程选模后，用 `export_candidates bind-selected` 将[分项导出模板](../../../configs/runs/segmentation/candidate_u28318_components.json)绑定到获选报告，再从新缓存导出；绑定会校验权重摘要、路径、训练身份和完整 dev 报告。

## 核对与复现

- 开工拉取 7 个远端分支并核对 5 个 PR 及讨论。主分支 `7064ebb`，原分割分支 `5ed9db1`。wkq 的 MHP 数据交接为 `b82fc99`，wyz 的 MHP 评价为 `1e0a457`。本轮只使用 Fashionpedia 分割工作区。[wkq #4](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/4) · [wyz #5](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/5)
- 开工时数据盘可用 121 GiB，系统盘 27 GiB，RTX 3090 显存 0/24,576 MiB，共享锁空闲。新增 6 项 CPU 检查全部通过，覆盖配方与比较、确定性尺寸、真实 mapper、可手算的点位、上游最终与辅助 mask loss 及 RNG 等价、分项缓存与离线导出。张量夹具只调用损失原语和候选解码。训练输出目录尚未创建。[新增检查](../../../tests/test_full_combo.py) · [准备回执](evidence/full_combo_20261119/preparation.json)
- 在分割 checkout 执行下面的准备命令可查看两个阶段的完整参数。实际启动需要另行明确使用执行模式。

```bash
PYTHONPATH=src /root/autodl-tmp/fashion-grounding/envs/segmentation-v1/bin/python scripts/run_tenk_adaptation.py --plan configs/runs/segmentation/full_combo_plan.json
```
