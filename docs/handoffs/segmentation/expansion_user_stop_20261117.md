# 按用户要求停止 5k/10k 对照

2026-11-17 15:08 HKT，lcy / segmentation。用户因耗时明确要求停止当前训练；该指令取代此前“按原上限完成”的继续运行安排。

已核对实时命令、父子进程与 PID start time，仅向本组控制器发送 SIGINT；原受控清理逻辑退出本组 wrapper/worker/trainer。15:08:43复核无本组残留进程、无 GPU 计算进程，GPU 0 MiB/0%，共享锁可取得，协调记录无 running。本组标为 interrupted / explicit_user_stop，不自动继续或重启，未关闭共享服务器、未终止队友进程。

停止前 C5k 已完成20,000 updates及10,000/20,000的完整dev评价，组内选20,000步，mask AP=64.695685。四份5k/10k/15k/20k checkpoint、预测及日志保留。以上仅读取既有选择记录，未重跑评价或推理。

D10k 于15:04:43开始，本次停止时尚无保存的checkpoint；已有初始化配置、输入检查和metrics日志保留。D10k未完成，不能填有效配对终点或据此判断5k/10k数据量收益。完整数据100epoch仍未启动。

服务器stage_status、D10k run_metadata/phases及本组coordination已记录用户中断，保留原KeyboardInterrupt和状态历史；独立user_stop记录保留信号、进程身份和产物清单。未改训练源码、数据、checkpoint和原计划预算。[停止及复核证据](evidence/expansion_user_stop_20261117/verification.json)。原启动交接和14:58仍在运行的快照保留为历史，当前状态以此记录为准。

后续如需恢复，须由用户另行明确启动；本轮不自动接续10k，也不改为启动100epoch。
