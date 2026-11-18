# 全量分割单线训练启动记录 · 2026-11-18

11 月 18 日 18:26:59 HKT，lcy 在分割工作区启动 `full-single-50k-20261118-r01`，执行版本为 `9f1ca245fbde243df17ec711854c803e52892617`。18:30:59 的服务端快照记录了前 545 次成功更新。服务器运行目录为 `/root/autodl-tmp/fashion-grounding/runs/segmentation/full-single-50k-20261118-r01/`。[启动命令](evidence/full_single_launch_20261118/launcher.json) · [运行元信息](evidence/full_single_launch_20261118/run_metadata.json)

## 启动前核对

开工时已核对 7 个远端分支、5 个 PR 和队友交接。wkq 分支为 `ae08c8d`，wyz 两个分支均为 `2c47c36`，与准备阶段最后一次记录一致。18:25:07 HKT，数据盘可用 129.14 GiB，系统盘可用 26.93 GiB，RTX 3090 占用 0 / 24,576 MiB，共享 GPU 锁空闲，计算进程为空。分割工作区随后快进到准备版本，运行期间固定在该提交。[分支、PR 和资源快照](evidence/full_single_launch_20261118/preflight.json) · [准备记录](full_single_line_20261118.md)

## 实际启动状态

- 初始化从旧 50k 加载模型参数及八类分类头，优化器、学习率调度、AMP、采样和随机状态重新初始化，从更新 0 开始。[初始化回执](evidence/full_single_launch_20261118/initialization.json)
- 全程计划取图 56,636 次，包含 42,476 张基础图各一次和 14,160 次弱类、小目标定向取图，每步 2 张。训练配置绑定了 146 张已曝光图的保护清单及已验证的顺序文件。启动快照读取了实际训练日志，前 1,090 个图片 ID 与该顺序逐项一致。[运行配置](evidence/full_single_launch_20261118/resolved_config.json) · [已有采样回执](evidence/full_single_line_20261118/sampling_receipt.json)
- 18:30:59 HKT，更新 1–545 连续完成，各步 loss、裁剪前后梯度范数均有限，AMP 溢出 0 次，优化器跳步 0 次。第 545 步总 loss 为 9.1305，裁剪前后梯度范数分别为 124.4165 和 0.0100。该时刻 GPU 占用 5,461 / 24,576 MiB。[首批更新快照](evidence/full_single_launch_20261118/startup_snapshot.json) · [共享 GPU 协调记录](evidence/full_single_launch_20261118/coordination_at_start.json)

## 完整 dev 与选模

第 14,159 步和第 28,318 步各评一次完整 dev250（250 图、1,008 个 GT）。半程评价后从该 checkpoint 恢复本轮训练状态。按完整 dev 的 mask AP 从两份新权重和旧 50k（69.0313）中选最高者；新旧同分保留 50k，两份新权重同分取较早步数。启动快照时两次评价均待执行，供下游使用的候选版本为 50k＋0.3。[执行方案](../../../configs/runs/segmentation/full_single_plan.json) · [旧 50k 的完整 dev 报告](evidence/tenk_completion_20261118/epoch_10/evaluation_report.json)

初始化、两段训练、两次评价和选模共用 5 小时。工作进程截止约为 23:26:44 HKT，总硬截止约为 **23:26:59 HKT**，最后 15 秒预留清理。失败或超时即停止，保留已有产物，自动重启和延时均已关闭。启动确认使用本次训练自然产生的日志，复用了既有输入和采样验收记录。[运行元信息](evidence/full_single_launch_20261118/run_metadata.json)
