# 10k 模型参数热启动执行

2026-11-17，lcy / segmentation。用户在执行器准备完成后明确要求：“做好之后直接开始训练，训练开始后你不用一直盯着”。本记录在启动时补齐实际状态，不把配置授权视为训练完成。

已再次核对整组GitHub，wkq `b24515b`、wyz `103dcc5`、main `7064ebb`与全部PR/活跃讨论均无新变化，交接复用。[GitHub快照](evidence/tenk_launch_20261117/github_preflight.json)。17:21 HKT服务器约150.6GiB数据盘可用，RTX3090显存0、GPU0%、无计算/活动协调记录、锁空闲，本人checkout `1ec9723` clean。[启动前资源](evidence/tenk_launch_20261117/server_preflight.json)。

使用[已验证执行器](tenk_execution_preparation_20261117.md)和独立的[授权计划](../../../configs/runs/segmentation/tenk_adaptation_launch_20261117.json)、[授权训练配置](../../../configs/runs/segmentation/tenk_adaptation_train_authorized_20261117.json)。默认准备文件继续禁用，授权副本只改变执行标记与引用路径，不改变训练配方。新resolved训练配置SHA `2437450750f2d3b0114e3a670e7c919d45496f1bfb8eba4fc157757607925f2c`。[授权阶段命令](evidence/tenk_launch_20261117/authorized_plan.json)。

- 仅加载 C5k20k 模型参数，重置优化器/调度/采样等状态；新update从0起。
- 最多10等效epoch/50000成功updates；每2epoch/10000updates完整dev250评价。
- 总硬上限8小时，单段训练＋dev＋选择合计2小时；连续3次完整dev未超过已有最佳早停，失败即停，不自动重试/延长。
- 复用C5k20k AP64.6957缓存基线。正式新AP、选择结果均等待运行产物；没有重复旧权重推理或smoke。
- 共享锁覆盖所有段落，启动使用独立会话，不依赖聊天持续在线。用户不要求持续盯守；只确认实际训练已开始后交还控制，执行器自行评价、选择、停止并记录结果。

运行输出预定为 `/root/autodl-tmp/fashion-grounding/runs/segmentation/tenk-adaptation-20261117-r01`。当前记录状态：**running，已启动**（启动观察时间为17:24:31 HKT）。真实执行源码固定为 `0541df4f7abbb7bc569bbbc59a4b9ed52f1a3c2f`，通过git bundle将本人干净checkout从1ec9723快进部署；后续GitHub状态记录提交不拉入正在运行的服务器checkout。没有修改队友工作区。

17:24:13 HKT独立会话启动；监督器PID230160，共享锁wrapper230228，worker230234，trainer230302。启动观察已有3个成功updates，最后loss12.8249，梯度裁剪前234.5435/后0.01，AMP重试0，显存5455MiB。模型初始化记录确认只加载model，优化状态重置、八类头未重新初始化。此处仅确认已经开始，不作为长程稳定性、吞吐或性能验收。

[启动快照](evidence/tenk_launch_20261117/startup_snapshot.json)、[运行元信息](evidence/tenk_launch_20261117/run_metadata.json)、[模型初始化](evidence/tenk_launch_20261117/initialization.json)、[真实resolved配置](evidence/tenk_launch_20261117/resolved_config.json)、[复用数据验收记录](evidence/tenk_launch_20261117/data_check.json)、[共享锁记录](evidence/tenk_launch_20261117/coordination_at_start.json)。registry追加本次running训练行；AP保持空白。

绝对总截止约为11月18日01:24 HKT，可能因每段2小时限制、失败或早停提前结束，不自动延长。之后按用户要求不持续轮询。服务端会自行写入各段日志、dev预测和evaluation_report、selection.json、worker_result.json及最终run_metadata；GitHub本轮同步的是启动快照，结束结果下次读取后再同步，不声称后台会自动推送GitHub。
