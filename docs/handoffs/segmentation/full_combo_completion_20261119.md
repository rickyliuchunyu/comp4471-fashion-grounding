# 全量分割组合训练完成与权重选择

11 月 19 日 05:38:15（香港时间）完成全部 28,318 次新增更新、两次完整 dev250 评价及选模。正式入口耗时 3 小时 30 分 56 秒，8 小时硬上限内正常结束，协调进程退出码为 0。按预定的整体 mask AP 最大规则，选中组合训练的末程权重。[运行回执](evidence/full_combo_completion_20261119/run_metadata.json) · [阶段回执](evidence/full_combo_completion_20261119/stages.json) · [选模回执](evidence/full_combo_completion_20261119/selection.json)

## 三份完整 dev250 报告

三次评价使用同一批 250 张图、同一标注摘要和同一 COCO 掩膜评价协议，均有 25,000 条 top100 预测，推理错误为 0。AP 按 0–100 标度，差值从未舍入数值计算。新阶段从旧最佳仅加载模型参数，更新数从零重新计。[旧最佳报告](evidence/full_single_completion_20261118/update_028318/dev/update_028318/evaluation_report.json) · [组合半程报告](evidence/full_combo_completion_20261119/update_014159/dev/update_014159/evaluation_report.json) · [组合末程报告](evidence/full_combo_completion_20261119/update_028318/dev/update_028318/evaluation_report.json)

| 指标 | 旧最佳 28,318 步 | 组合 +14,159 步 | 组合 +28,318 步 | 末程较半程 | 末程较旧最佳 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 整体 mask AP | 71.3205 | 71.7278 | 71.8911 | +0.1633 | +0.5706 |
| 小目标 AP | 23.4459 | 23.4985 | 28.7762 | +5.2777 | +5.3303 |
| 鞋 AP | 60.0969 | 59.4262 | 60.3814 | +0.9551 | +0.2844 |
| 配饰 AP | 47.8811 | 46.7598 | 48.1669 | +1.4071 | +0.2859 |
| 中目标 AP | 54.0114 | 49.4473 | 51.0998 | +1.6524 | -2.9116 |
| 大目标 AP | 80.3632 | 80.8849 | 81.1020 | +0.2171 | +0.7389 |
| AP50 | 86.4098 | 87.0355 | 87.1864 | +0.1509 | +0.7766 |
| AP75 | 76.3129 | 76.1933 | 76.4803 | +0.2870 | +0.1674 |

小目标 AP 到半程接近旧最佳，提升主要来自后半程。鞋和配饰在半程均低于旧最佳，末程均略高于旧最佳。中目标 AP 回升到 51.0998，仍低于旧最佳 54.0114。鞋和配饰两行统计该类全部尺寸的实例。本次为组合训练，结果支持整套配方在这个开发集上的表现，无法分摊各项改动的贡献。

## 选中权重和当前交付

服务器权重为 `/root/autodl-tmp/fashion-grounding/runs/segmentation/full-combo-u28318-20261119-r01/update_028318/train/checkpoint_028318.pth`，SHA-256 为 `b7146787a1fc8f16f4694f2a9d05c294a49090b095de901b63b963f162a640c1`。末程训练报告记录 6 次 AMP 溢出重试，优化器跳步为 0；两段各完成 14,159 次成功更新。[半程训练报告](evidence/full_combo_completion_20261119/update_014159/train/train_report.json) · [末程训练报告](evidence/full_combo_completion_20261119/update_028318/train/train_report.json)

当前 GUI 和下游已交付候选引用前一版权重 `aaff8d9dd89e`。本轮交接发布已有报告与新权重的选择结果，本次汇总新增训练和推理次数均为 0。[现行候选索引](current_candidates.md)
