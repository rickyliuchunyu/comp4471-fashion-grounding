# 5k/10k 对照：验收证据复用与统一硬超时入口

2026-11-16，lcy / segmentation。**入口已实现并完成新增 CPU 检查，未启动训练或推理。** 本轮承接 [10k 技术验收](train10000_acceptance_20261116.md) 和 [受控方案](train10000_control_plan_20261116.md)，只补原方案列明的两处接线；旧验收、COCO 加载、头适配、AMP/恢复及模型 smoke 不重跑。

## 整组核对与输入范围

开工 lcy=`cbbc67d`，wkq=`4b0584a`，wyz=`103dcc5`，main=`7064ebb`。所有远端分支、五个 PR、活跃评论/review 和增量交接已核对；队友无新增提交或交接。#2/#4 open/draft，#5 open/non-draft 且 base 为 wkq 分支，#1/#3 closed-unmerged；无新讨论、均未合并。见 [GitHub 快照](evidence/expansion_execution_preparation_20261116/github_preflight.json)。

本轮服务器预查 RTX3090 0 MiB/0%、无计算进程和活动协调任务；数据盘约 153.5 GiB 空闲，两个计划 run ID 均尚不存在。仅更新 segmentation checkout，未修改队友环境、数据包或排序接口。输入仍是上一轮固定的 train5000、定向 train10000 和 dev250，未消费任何新数据版本。

## 组合验收入口

新增 `fashion_grounding.segmentation.input_receipts` 及评价参数 `--input-validation-bundle`；与旧 `--reuse-validated-inputs` 互斥。

- **C5k**：[固定旧报告的 bundle](../../../configs/runs/segmentation/expansion_input_train5000.json)，调用原 unchanged-train 复用路径。原入口继续拒绝训练指纹变化，不放宽旧门禁。
- **D10k**：[组合 bundle](../../../configs/runs/segmentation/expansion_input_train10000.json)，固定旧 B20k dev 报告、新 10k acceptance、当时 acceptance_plan 及 run_metadata 的 hash。运行记录必须绑定这两份原始验收文件，生产源/接收源、父 5k 身份、新 10k 身份、保护 dev 身份、旧/新数量和八类映射均需一致。
- 隔离记录必须明确包含 parent/dev/test 的 image ID、namespaced source identity、group、SHA 四项零交集；缺字段、非零、未完成验收或缺少四方向近重复证据均拒绝。旧 dev 对旧 5k 的隔离与新验收对新增 5k 的隔离共同覆盖完整 10k。
- 调用时只核当前 train/dev manifest、annotation 文件的 hash，10k 发布校验目录的 hash，以及 dev 元数据分母 250 图/1008 GT。**不打开图片、不解码 mask、不重新计算 pHash、不读取 test 文件。** 原始报告不修改，`formal_three_party_freeze=false` 原样传递，不代签人工审核。

新的 data_check 记录 bundle 和每份证据 hash、复用模式、真实训练数据身份及 `image_and_mask_reaudit=false`。后续 evaluator 仍严格匹配 checkpoint 的 training_data；新报告不能伪装成旧 5k 训练结果。

## 执行入口与硬上限

入口为 [run_expansion_control.py](../../../scripts/run_expansion_control.py)，配置仍为 [同一结构化方案](../../../configs/runs/segmentation/expansion_control_plan.json)。默认只解析配置、核固定证据链、展示命令，不创建训练输出、不申请 GPU 锁、不调用 trainer/evaluator。

显式 `--execute` 仅供未来用户明确启动后调用：

1. 顺序执行 C5k、D10k；各自从相同官方 COCO 权重、seed42 冷启动，独立 optimizer/sampler/输出。两个训练配置仅训练数据 hash 不同，固定 20k 成功更新、batch2、min_scale0.5。
2. 每组保存 5k/10k/15k/20k 恢复点；训练完成后在同一评价进程中依次评 **保存的 10k、20k 权重**，每份都是完整 dev250。评价节点、数量和科学配方不因运行耗时改变。
3. **每组从开始即固定一个 monotonic 截止时间，最多 10800 秒；两组从入口启动起累计最多 21600 秒。** 证据核对、训练、两点评价、组内选择及第二组后的配对比较都计入；阶段切换不重置计时。训练内部 10000 秒、评价内部 600 秒仍受更严格的外层剩余时间约束。
4. 复用现有共享 GPU lock 包装器，GPU 占用/活动记录不满足时直接退出，不排队抢占。外层监督进程、worker 内同一绝对截止时间和 Linux parent-death 保护共同覆盖子进程树；预留 15/10/5 秒收尾余量，使强杀/记录不成为额外训练预算。只终止本次隔离进程组，不操作队友进程。
5. 任何阶段或任一组失败立即停止；C5k 失败不会自动启动 D10k。超时/失败保留已经写完的 checkpoint 和日志，不选择部分结果、不自动恢复/重启/延长。输出和协调 run ID 必须全新；执行需 Linux 固定分割环境、干净且两组一致的源码、至少 20 GiB 可用空间。

每组输出 input_receipts、phases、supervisor 日志和 run_metadata，整对输出 resolved_plan、stage_status；真实训练/评价报告由现有入口生成。本轮未产生任何训练 run 或协调任务，未将准备记录登记成训练完成。

## 选择与配对比较

各组先用原严格 selector 比较 10k/20k：完整 dev AP 最大，平手取较早。新对照入口额外要求 fresh 0→20k、40000 次图片曝光、成功训练记录、相同源码、正确 checkpoint/config/data 身份及两个完整评价点。配对比较还核对两组初始 backbone/classifier digest 相同。

跨数据集不调用原“同一次训练”选择器混选；另生成 20k 终点 D10k−C5k 的 AP、AP_small、AP75 及弱类差值，应用先前固定的实用筛选线。旧 B20k 的固定报告仅作为历史参考，不推理、不重读权重。输出 `best_by_dev_AP` 和建议，**不会复制/替换部署权重、候选或 wyz 缓存**。若没有超过历史最佳或未通过筛选线，保持历史交付。小目标/边界细诊断继续用未来保存的预测离线分析，不在此入口追加推理。

## 本轮实际验证

服务器 CPU 准备源 `8b522a21c987a6c4ad60cb55cc801f3680e43e7d`；最终代码 `e38b8369262b18c6a0ca2bc77314c90a855e337a` 只补预算已耗尽时的失败记录分支，单独新增 1 项检查通过。证据复用与 Linux 保护实现未变，因此不重跑已通过的服务器检查。

- 新增 11 个 CPU 案例：本地 10 项通过，Linux 专属 parent-death 案例在服务器单独通过。覆盖旧门禁保留、无像素依赖、证据/当前元数据漂移、错误父数据/隔离冲突、执行记录绑定、默认准备模式、固定参数、共享 deadline 杀子进程、失败停止、两个评价点齐备和相同初始化。添加每组 metadata 后仅重跑受影响的失败处理案例 1 项，没有重复全套。
- 新 timeout 案例用短寿命 CPU 子进程模拟两个阶段；第二阶段及后代在共同截止时间终止。Linux 专属案例验证锁包装器父进程死亡后 worker 与后代退出。它们不导入模型、不申请 GPU，也不是旧训练 smoke。
- 服务器对真实 train5000、train10000 和 dev250 执行一次 `--check-inputs`：两个模式均通过，250 图/1008 GT 元数据分母正确；只核证据和文件身份。未重跑上一轮新增 5k 解码或旧 dev 像素检查。
- 准备报告显示 model_loads=0、inference_images=0、training_updates=0；计划训练目录和两个 GPU 协调 run ID 均未创建。当前没有新的 AP、训练耗时或 GPU 稳定性结论。3 小时长任务本身未执行。

服务器 CPU 准备证据位于：

`/root/autodl-tmp/fashion-grounding/runs/segmentation/segmentation-expansion-preparation-20261116-r01`

仓库小型证据：[本地检查](evidence/expansion_execution_preparation_20261116/local_checks.json)、[服务器准备结果](evidence/expansion_execution_preparation_20261116/preparation_report.json)、[Linux 案例](evidence/expansion_execution_preparation_20261116/linux_guard_test.log)、[结束快照](evidence/expansion_execution_preparation_20261116/server_postflight.json)。

## 命令与下一步

在服务器 segmentation checkout，准备模式（下次无需无变化重跑）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
/root/autodl-tmp/fashion-grounding/envs/segmentation-v1/bin/python \
scripts/run_expansion_control.py \
  --output /root/autodl-tmp/fashion-grounding/runs/segmentation/segmentation-expansion-control-20261116-r01 \
  --check-inputs
```

未来明确启动时，在同一入口去掉 `CUDA_VISIBLE_DEVICES=''` 和 `--check-inputs`，加 `--execute`；由入口管理共享锁和各组硬截止时间。当前计划 IDs 分别为 `segmentation-expansion-5000-20261116-r01`、`segmentation-expansion-10000-20261116-r01`。开训前只核最新团队/资源变化、正式数据使用记录及现有证据是否仍适用，不再安排模型 smoke。

本轮用户明确要求“先不启动训练”；执行入口已交付，停在准备完成状态。排序、语言数据、test 评价及数据生产均不在本次范围。
