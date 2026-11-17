# 10k 模型参数热启动执行

2026-11-17，lcy / segmentation。用户在执行器准备完成后明确要求：“做好之后直接开始训练，训练开始后你不用一直盯着”。本记录在启动时补齐实际状态，不把配置授权视为训练完成。

已再次核对整组GitHub，wkq `b24515b`、wyz `103dcc5`、main `7064ebb`与全部PR/活跃讨论均无新变化，交接复用。[GitHub快照](evidence/tenk_launch_20261117/github_preflight.json)。17:21 HKT服务器约150.6GiB数据盘可用，RTX3090显存0、GPU0%、无计算/活动协调记录、锁空闲，本人checkout `1ec9723` clean。[启动前资源](evidence/tenk_launch_20261117/server_preflight.json)。

使用[已验证执行器](tenk_execution_preparation_20261117.md)和独立的[授权计划](../../../configs/runs/segmentation/tenk_adaptation_launch_20261117.json)、[授权训练配置](../../../configs/runs/segmentation/tenk_adaptation_train_authorized_20261117.json)。默认准备文件继续禁用，授权副本只改变执行标记与引用路径，不改变训练配方。新resolved训练配置SHA `2437450750f2d3b0114e3a670e7c919d45496f1bfb8eba4fc157757607925f2c`。[授权阶段命令](evidence/tenk_launch_20261117/authorized_plan.json)。

- 仅加载 C5k20k 模型参数，重置优化器/调度/采样等状态；新update从0起。
- 最多10等效epoch/50000成功updates；每2epoch/10000updates完整dev250评价。
- 总硬上限8小时，单段训练＋dev＋选择合计2小时；连续3次完整dev未超过已有最佳早停，失败即停，不自动重试/延长。
- 复用C5k20k AP64.6957缓存基线。正式新AP、选择结果均等待运行产物；没有重复旧权重推理或smoke。
- 共享锁覆盖所有段落，启动使用独立会话，不依赖聊天持续在线。用户不要求持续盯守；只确认实际训练已开始后交还控制，执行器自行评价、选择、停止并记录结果。

运行输出预定为 `/root/autodl-tmp/fashion-grounding/runs/segmentation/tenk-adaptation-20261117-r01`。当前记录状态：**authorized_not_started，部署与启动进行中**。未声称训练稳定性、吞吐、显存峰值或新性能已通过。
