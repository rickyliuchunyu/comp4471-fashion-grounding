# min_scale 单因素对照实际运行

2026-11-15；segmentation / lcy，Codex 执行。用户本轮明确授权启动已准备的对照，状态 **running**，尚无 B 组 AP。实际执行源 `e73ef4ffa9d1a72cee041e23d9b49b673c23bdad`，运行 ID `segmentation-min-scale05-20261115-r01`；不修改队友工作区、共享数据或排序代码。

已复核整组 GitHub 分支、五个 PR、活跃讨论及队友交接：wkq `f5787f4`、wyz `103dcc5`、main `7064ebb` 无增量。服务器启动前 RTX 3090 0 MiB/0%、无计算进程/活动协调记录；数据盘剩余 169,476,403,200 bytes，系统盘剩余 28,930,568,192 bytes。只将自己的干净 checkout 从 `994c1e5` 快进至 `e73ef4f`；传输 Git bundle SHA256 `574bc3390c48819bd8ba864dcc83d5742bb388bdd8f112c24b0519515a1620c0`。

按[准备方案](min_scale_preparation_20261115.md)执行：同一完整18k父权重到20k，科学因素仅 min_scale 0.1→0.5；保持 train5000/dev250、seed42、优化器与学习率、已验证 AMP 策略。A20k 和旧最佳18k 直接引用已有结果。B 结束时仅评一次完整 dev250，按整体 mask AP 比较选择。

已于 **22:46:25 HKT** 启动 `scripts/run_min_scale_trial.py --execute`，总硬上限2700秒，最迟约 **23:31:25 HKT** 结束；实际 worker 截止更早10秒以保留清理时间，不自动延长或重启。共享锁已登记。恢复检查已通过，18000步六项状态 fingerprint 完全相同，实际记录见服务器 `train/resume_check.json`。没有重跑11项准备检查、旧 smoke、A组或旧最佳推理。

运行目录：`/root/autodl-tmp/fashion-grounding/runs/segmentation/segmentation-min-scale05-20261115-r01`。阶段日志 `stage.log`，阶段状态 `stage_status.json`，更新日志 `train/metrics.jsonl`；完成后接 `dev/update_020000/` 和 `comparison.json`，未完成部分不填结果。

- [GitHub 启动快照](evidence/min_scale_trial_20261115/github_preflight.json)
- [服务器启动快照](evidence/min_scale_trial_20261115/server_preflight.json)
- [运行登记](evidence/min_scale_trial_20261115/run_metadata.json)

本文件先如实登记 running，完成后更新实际状态、指标与选择；旧18k＋0.3候选及联调交付保持原版本。
