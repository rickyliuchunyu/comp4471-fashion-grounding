# 10k 分段执行、早停与权重选择准备

2026-11-17，lcy / segmentation，Codex 执行。**implemented / CPU_checked / prepared_not_started**。代码提交 `6b5fc8c51da6a643cd47e68897be006ce0be0264`。本轮补齐上一轮10k方案的执行器，没有启动训练、分割推理或重复 smoke；不接管 wkq / wyz 的数据与排序任务。

后续用户追加“做好之后直接开始训练”授权；以下保留准备阶段的事实与禁用默认配置，实际启动另见[10k执行交接](tenk_launch_20261117.md)。

## 开工依据和复用范围

- 已 fetch 整组分支，并核对全部5个PR及活跃#2/#4/#5讨论。开工本分支 `dac74b4`，wkq `b24515b`、wyz `103dcc5`、main `7064ebb`，两个独立 full-data 分支均无变化。队友交接按相同commit复用；wkq语言增量没有新增分割监督。[GitHub快照](evidence/tenk_execution_preparation_20261117/github_preflight.json)。
- 17:07 HKT只读检查：数据盘约151GiB可用，系统盘约27GiB；RTX3090显存0/24576MiB、GPU0%、无计算进程，共享锁空闲。17:17补记无活动协调记录，服务器本人checkout仍为 `1ec9723` clean；没有部署本轮代码。[资源](evidence/tenk_execution_preparation_20261117/server_preflight.json) / [checkout及协调记录](evidence/tenk_execution_preparation_20261117/server_checkout.json)。
- 复用已通过的10k增量/旧5k保留/隔离、dev验收、COCO/分类头、AMP和恢复、model-only/cosine及增强可见性证据。上一轮裁剪诊断负结果有效，仍不启用定向裁剪；不再次读取真实图片或加载checkpoint。
- 现有 C5k20k 完整dev AP **64.6957** 仍是最佳已测结果；没有新的10k AP。旧18k/0.3候选继续保留。

## 本轮新增行为

[入口](../../../scripts/run_tenk_adaptation.py)默认仅准备；[计划](../../../configs/runs/segmentation/tenk_adaptation_plan.json)的 `training_authorized`、`execution_enabled`、`budget_proposal.authorized` 和训练配置的 `training_authorized` 四项均为 **false**。`runtime_ready=true`只表示执行器实现和CPU验证完成，不表示训练已获授权、已部署或GPU实跑通过。直接加 `--execute` 也会在创建输出/子进程前拒绝。

已准备的配方保持不变：C5k20k仅加载模型参数，优化器/学习率/采样器等重置；R50八类、768、LSJ0.5–2、batch2，LR5e-5、backbone5e-6和cosine。第一段从新update0开始，后续段从上一段的新checkpoint恢复完整状态，不重复重置优化器。

| 等效epoch | 新run累计成功updates | 行为 |
| --- | --- | --- |
| 2 | 10000 | 首段model-only训练 → 完整dev250 → 选择与停止判断 |
| 4 | 20000 | 同run完整状态恢复 → 完整dev250 → 选择与停止判断 |
| 6 | 30000 | 同上；若前三次均未超过已有最佳，此处最早早停 |
| 8 | 40000 | 仅上段完整成功且未早停时进入 |
| 10 | 50000 | 最多到此结束，不自动延长 |

每段顺序固定为训练、训练/恢复记录核对、完整dev、评价记录核对、权重选择，然后才决定是否进入下一段。核对包括成功update及曝光数、model-only来源、恢复组件指纹、配置/数据/源码身份、checkpoint sidecar、250图完整评价和新预测hash。真实checkpoint完整hash与模型指纹仍由原trainer/evaluator负责，不为核对记录重复加载真实权重。

新run内部继续调用原严格 `choose_checkpoint`，要求相同训练/评价身份；不把5k和10k混入该选择器。再以同一dev数据、评价配置/协议/图像数独立比较C5k缓存基线；正式启动还要求评价器及模型推理源码与该基线一致。跨run同分保留C5k，新run同分取较早update。主指标是完整dev mask AP，连续3次未严格超过基线或新最佳则早停；改善会清零计数。选择记录同时写AP_small/AP75差值及八类AP。小目标IoU匹配、边界诊断留待训练后复用新预测离线分析，不冒称本执行器已经计算。

## 时间、锁和失败处理

**拟总硬上限8小时，每个“2epoch训练＋完整dev＋选择”段硬上限2小时；此预算尚未授权。** 全部五段共用从launcher启动起计算的绝对截止时间，换段不重置总时钟。worker提前15秒结束，为包装器落盘和释放锁留余量；这15秒在8小时内，不是延长。训练内部7100秒及评价内部600秒仍受该段剩余预算约束，不能相加突破2小时。

复用已有共享GPU锁、独立进程组硬定时器、Linux父进程死亡保护和外层截止包装器。一次持锁覆盖所有段落；超时只结束本次worker及其后代，不处理队友进程。源代码或输入pin变化、训练/评价/选择失败均终止；不重试、不自动进入下一段，不复用已存在输出或协调run ID。保留失败日志、已保存checkpoint和最后一次完整dev选择。状态/选择JSON原子写入，主监督器在worker失败后记录failed；强制kill时阶段记录可能仍为running，必须结合外层失败记录，不能当作阶段完成。

硬上限不是保证跑满10epoch；没有完整dev的checkpoint不参与选择。CPU验证不能证明新训练吞吐、显存或精度收益。

## 必要验证及停止点

新增 **15项CPU案例**全部通过：默认禁用/四重授权和pin、预算/早停门禁、首段与后续恢复记录、错误或不完整dev、基线/新run选择及同分、早停计数、阶段顺序、各阶段失败保留上次选择、总计时不重置、段内训练和dev合计超时、真实CPU睡眠子进程组按段截止结束、外层失败落盘且不重试/覆盖。夹具checkpoint只是字节文件，没有反序列化真实模型。

先运行新增文件14项（2.458秒）；补齐pin完整性和外层失败门禁后只重验受影响的4项门禁（0.625秒），最终15个不同案例。未扩大到旧套件；旧Linux父死亡保护测试证据直接复用，新增进程组接线在本机POSIX CPU验证。仅CPU工程验证不登记成新的S0训练实验。[验证范围和代码hash](evidence/tenk_execution_preparation_20261117/new_logic_validation.json)。

默认准备命令已生成[可复核的阶段计划](evidence/tenk_execution_preparation_20261117/prepared_plan.json)：

```sh
PYTHONPATH=src .venv-segmentation/bin/python scripts/run_tenk_adaptation.py
```

本轮模型加载、训练updates、新推理图数、旧smoke均为0。没有创建未来服务器训练目录或追加训练registry行。

## 下一步与GitHub同步边界

下一步是用户明确授权后的10k实际训练。启动时只复核最新整组进度、磁盘/GPU/锁和输入/源码身份，按批准预算启用四项标记并更新训练配置hash，提交后同步到本人干净服务器checkout；使用上述执行器的 `--execute`。复用本轮及历史证据，不另加一轮smoke。训练后依据完整dev实际结果选择，若未超过C5k则保留旧最佳。

本轮代码、计划、证据和状态表推送到原个人分支，继续由 draft PR #2 承载，未合并。PR #2标题/正文仍是历史full-data提案：上一轮已确认GitHub连接器更新返回403（`Resource not accessible by integration`），本轮没有重复无效权限/UI尝试。以本交接和状态表为当前10k路线依据，PR diff随分支正常更新。
