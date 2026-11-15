# min_scale 单因素对照结果

2026-11-15；segmentation / lcy，Codex 执行。状态 **completed / gpu_verified / evaluated**。同一18k父权重续训到20k，min_scale 0.1→0.5，新增2000次成功更新；随后只对B评一次完整dev250并离线比较。总耗时 **958.213秒（15分58秒）**，未触及2700秒硬限、未延长或自动重启。执行源 e73ef4ffa9d1a72cee041e23d9b49b673c23bdad；运行 ID segmentation-min-scale05-20261115-r01。

**按最高完整dev mask AP规则选中 B20k，AP=63.7929502**，比复用A20k高0.2334197点、比旧最佳18k高0.2137899点。这是单seed、低学习率短对照的小幅增益，不是统计显著性证据；小目标和边界问题没有获得明确改善。

## 固定比较

所有值为0–100尺度AP点。A和旧最佳直接读取保存报告，没有重跑推理/AP/smoke。B包含250/250图、25,000条阈值前top-100预测、0错误；没有读取语言查询或test，评价未修改模型状态。

| 权重 / 条件 | mask AP | AP50 | AP75 | AP_small | AP_medium | AP_large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A20k，min_scale=0.1，复用 | 63.5595 | 79.1353 | 68.3130 | 10.6641 | 40.2686 | 71.6934 |
| 旧最佳18k，复用 | 63.5792 | 79.2945 | 68.4563 | 10.7389 | 39.8875 | 71.7116 |
| **B20k，min_scale=0.5，本次** | **63.7930** | **79.3888** | **68.4059** | **10.7525** | **40.0568** | **71.7558** |
| B−A | +0.2334 | +0.2535 | +0.0929 | +0.0884 | −0.2117 | +0.0625 |
| B−旧最佳 | +0.2138 | +0.0943 | −0.0504 | +0.0136 | +0.1693 | +0.0442 |

| 八类 | A20k AP | B20k AP | B−A |
| --- | ---: | ---: | ---: |
| top | 70.5177 | 70.8382 | +0.3206 |
| pants | 81.6106 | 82.1560 | +0.5454 |
| skirt | 66.1074 | 66.8058 | +0.6984 |
| outerwear | 70.1096 | 70.1136 | +0.0039 |
| dress | 83.4987 | 83.5324 | +0.0336 |
| shoes | 50.8256 | 50.7912 | −0.0345 |
| bag | 48.1471 | 48.4233 | +0.2762 |
| accessory | 37.6595 | 37.6833 | +0.0237 |

相对A，增益主要来自skirt、pants、top与bag；AP_small仅+0.0884，accessory基本持平、shoes略降，AP_medium下降。相对旧最佳，AP_small仅+0.0136且AP75低0.0504，因此不能写成“小目标漏检/边界已解决”。AP75是高IoU总体指标，本轮未做新的边界专门诊断。按预登记整体AP规则选B，不因偏好子指标事后改规则。

## 恢复与选中权重

共同父权重：

`/root/autodl-tmp/fashion-grounding/runs/segmentation/segmentation-s0-20000-20261115-r01/train/checkpoint_018000.pth`

SHA256：207e2dbe1dcc8b9d0ef936d683d25af33c210347e34506914bbf6dad3426ee01。model、optimizer、scheduler、scaler、sampler、RNG六项恢复指纹完全一致，见[实际resume_check](evidence/min_scale_trial_20261115/train/resume_check.json)。

**选中B权重：**

`/root/autodl-tmp/fashion-grounding/runs/segmentation/segmentation-min-scale05-20261115-r01/train/checkpoint_020000.pth`

SHA256：**87c18e83bf024596a8c2020785ac352900c30d526704c018efdbe40b23617112**。

[comparison.json](evidence/min_scale_trial_20261115/comparison.json)记录新最佳；原S0的18k最佳及历史报告不改写。现有0.3候选和B1联调交付仍对应18k，本轮没有为B生成候选或替换队友缓存。

科学差异只有min_scale。train5000/dev250、seed42、768边长、max_scale=2.0、AdamW、AMP恢复策略及调度均固定；主干外LR约1e-6、backbone约1e-7，没有重置。执行元数据仅调整run_scope、2100秒训练限时及20k保存点。见[实际配置](evidence/min_scale_trial_20261115/train/resolved_config.json)、[配方恢复记录](evidence/min_scale_trial_20261115/train/recipe_transition.json)与[准备方案](min_scale_preparation_20261115.md)。历史policy的reason保留“preparation only”原文以维持hash；**启动授权来自本轮用户明确指令**，在运行元信息单独登记，不从旧计划推断授权。

## 稳定性、时间与资源

- 成功更新18001–20000连续，无缺失/重复，2000次成功更新loss/梯度有限、成功步跳过数0。前100/后100步平均loss为11.2714/11.1180，只描述本次训练，不把不同增强下的loss当AP证据。
- **3次AMP溢出尝试**：18823步一次（512→256），19129步两次（256→128→64）。失败尝试不推进optimizer/scheduler，重放同一batch并恢复sampler/RNG/buffers；[原始事件](evidence/min_scale_trial_20261115/train/amp_overflow.jsonl)保留。有限性结论仅覆盖成功更新，不能写“所有尝试梯度均有限”。
- 成功步裁剪前最大梯度范数13325.7959，裁剪后最大0.0100000016。训练峰值allocated **4.39694 GiB**、reserved **5.16797 GiB**；没有临时改裁剪或AMP策略。
- 训练828.660秒；完整dev内部计时117.889秒；总阶段958.213秒，包含启动核对、进程切换和离线比较。22:46:25 HKT启动，23:02:24左右完成，早于23:31:25硬截止。23:06快照GPU为0 MiB/0%、无计算进程，协调状态completed。

整组GitHub、五个PR、活跃讨论及交接已核对：wkq f5787f4、wyz 103dcc5、main 7064ebb无增量。启动前GPU空闲，数据盘约157.84 GiB空闲；只把自己干净checkout从994c1e5快进到e73ef4f。没有改队友工作区、共享数据、环境或排序代码，没有运行test、语言评价、A/旧最佳推理、旧smoke或已通过的11项准备检查。

## 证据与复现

实际命令见[launch.json](evidence/min_scale_trial_20261115/launch.json)，顺序仅train_B、dev250_B、compare_cached_A，三个阶段exit code均为0。完整权重与预测留服务器上述运行目录；dev预测SHA256为481cda8a905bd17c38bdf7150bf8d412971f432a44484b2b2b38421367c79312。

- [B完整dev报告](evidence/min_scale_trial_20261115/dev/update_020000/evaluation_report.json)、[选择结果](evidence/min_scale_trial_20261115/comparison.json)、[保存日志派生汇总](evidence/min_scale_trial_20261115/result_summary.json)
- [阶段状态](evidence/min_scale_trial_20261115/stage_status.json)、[各阶段记录](evidence/min_scale_trial_20261115/phases.json)、[原始日志](evidence/min_scale_trial_20261115/stage.log)
- [GitHub快照](evidence/min_scale_trial_20261115/github_preflight.json)、[启动资源](evidence/min_scale_trial_20261115/server_preflight.json)、[结束资源与锁状态](evidence/min_scale_trial_20261115/server_postflight.json)
- [运行元信息](evidence/min_scale_trial_20261115/run_metadata.json)、[26份原始小文件清单](evidence/min_scale_trial_20261115/evidence_manifest.json)

归档SHA256：df95fe4e8a27036f5aa6000053699f968457b9df29c2eb35bc97a42280bd68dd。本地归档及26份原始成员指纹一致。归档阶段两次SSH连接失败，确认未生成归档后重连完成收集；GPU阶段此前已成功结束，未因此重启训练/推理。registry分别登记本次train与B20k dev行，A与旧最佳不登记为新测。

本轮到此结束，不自动延长训练、改学习率或启动10k训练。结果只支持这个低LR短对照的小幅总体增益，小目标、边界和弱细类仍待后续分割优化；数据构建与排序方法仍由对应负责人推进。
