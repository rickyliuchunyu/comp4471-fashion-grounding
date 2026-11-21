# 5k/10k 分割对照：C5k完成，D10k中断（2026-11-17）

lcy / segmentation，执行源 `dc2fba429cdc4ffaed8a345f8ff5cd859cec1750`。**C5k完成20k更新及10k/20k完整dev评价，选C5k 20k AP64.6957；D10k在159步主动中断，整对对照未完成。** 没有10k模型性能或扩充收益结论，不自动恢复、重启或追加预算。当前已完成权重中C5k 20k最高；候选、部署文件和wyz缓存未替换。

## 实际结果与中断

- C5k：12:40:10–15:04:43 HKT，全链路8674.208秒（2小时24分34秒）/10800秒；训练8400.824秒，20000次成功更新、40000次图片曝光，训练/评价/选择三个阶段exit0。
- 19125步一次AMP溢出，既定策略将scale512降为256，恢复sampler/RNG/buffers后成功重放同一有效批次；失败尝试未执行优化器/调度器步进。成功更新loss和梯度均有限、跳过更新0，峰值allocated4.389GiB。4份5k/10k/15k/20k checkpoint保留。
- 两份完整dev均250图/1008 GT、0推理错误。10k报告elapsed135.228秒、20k报告264.223秒是**同一评价进程的累计时间**，不得相加成两点评价耗时。
- D10k：15:04:43–15:08:06 HKT，202.252秒后中断，最后159次成功更新/318次图片曝光、AMP溢出0。没有到达首个5k保存点，**没有D10k checkpoint、完整训练报告或dev评价**，不能从159步续训或当20k对照。
- 服务器[停止原始记录](evidence/expansion_control_20261117/user_stop_20261117.json)标记`explicit_user_stop`，原因是耗时过长并要求不继续后续组；记录向本次controller发送SIGINT，捕获KeyboardInterrupt。原failed状态和修订为interrupted的历史完整保留。这不是本轮观察到的数值失败或硬超时。监控发现后未重新启动。
- 整对运行8877.877秒后结束；C5k缓存诊断另耗4.063秒，历史配对计算0.008秒，都在原C5k三小时截止前完成，0新增推理/训练。只读复用现有诊断函数，没有重跑旧smoke、数据验收或测试套件。
- [结束资源快照](evidence/expansion_control_20261117/server_postflight.json)：GPU0MiB/0%，无计算/本次存活进程、无活动协调任务，共享锁空闲；数据盘约150.8GiB空闲。未关机或释放实例，未终止队友任务。

## 完整 dev AP 与权重选择

所有AP均为0–100尺度，FP32、800/1333、upstream instance top100，同一dev250。

| 权重 | AP | AP50 | AP75 | AP_small | AP_medium | AP_large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 历史B20k（缓存复用） | 63.7930 | 79.3888 | 68.4059 | 10.7525 | 40.0568 | 71.7558 |
| C5k 10k | 55.8283 | 72.4617 | 59.9869 | 7.3199 | 28.0885 | 64.9227 |
| C5k 20k（组内选中） | 64.6957 | 80.3897 | 68.4916 | 11.0483 | 31.3090 | 73.7301 |

C5k 20k较历史B20k AP+0.9027、AP_small+0.2958、AP75+0.0857；AP_medium下降8.7478，不能只报告改善指标。两者训练配方/轨迹不同，这是历史参照，不是纯min_scale或扩数据因果对照。D10k−C5k主要终点和预登记筛选线**不可计算**，不得用历史B替代缺失D10k。

| 类别 | 历史B20k | C5k 20k | 差值 |
| --- | ---: | ---: | ---: |
| top | 70.8382 | 70.1122 | -0.7260 |
| pants | 82.1560 | 81.7679 | -0.3881 |
| skirt | 66.8058 | 67.5248 | +0.7190 |
| outerwear | 70.1136 | 69.3756 | -0.7380 |
| dress | 83.5324 | 83.8757 | +0.3434 |
| shoe | 50.7912 | 53.7159 | +2.9247 |
| bag | 48.4233 | 52.2789 | +3.8556 |
| accessory | 37.6833 | 38.9146 | +1.2314 |

选择：`C5k/train/checkpoint_020000.pth`，完整路径为
`/root/autodl-tmp/fashion-grounding/runs/segmentation/segmentation-expansion-control-20261117-r01/C5k/train/checkpoint_020000.pth`。
SHA256：`d9deacc85d1bf402cf6417a71226bd51e1fc98b631dcd30c9afcc22d51b51044`。
此为已完成dev报告中的权重选择，未导出新候选、未替换旧18k＋0.3下游交付，也不是最终test结论。

## 小目标与边界：新预测离线分析

同一138个小目标：历史B20k→C5k20k，IoU≥.5匹配102→99（新增4、丢失7），≥.75匹配35→41（新增12、丢失6）。保存预测没有IoU≥.1几何重叠的小目标10→6，但同类.5匹配仍下降；不能把几何改善等同于完整召回改善。

原图1像素内边界IoU均值.178115→.184525；两者共同匹配的95个GT，薄边界.210430→.221611、像素precision .817789→.822609、recall .843420→.853963。是固定分母和共同集合的描述性诊断，非标准Boundary AP或统计显著性，不用GT修改预测。

| C5k 20k 小目标细类 | GT数 | IoU≥.5 | IoU≥.75 |
| --- | ---: | ---: | ---: |
| bag, wallet | 2 | 2 | 0 |
| belt | 11 | 5 | 1 |
| glasses | 8 | 8 | 3 |
| glove | 2 | 1 | 0 |
| hat | 1 | 0 | 0 |
| headband, head covering, hair accessory | 14 | 10 | 5 |
| shoe | 68 | 50 | 19 |
| skirt | 1 | 0 | 0 |
| sock | 13 | 5 | 2 |
| watch | 18 | 18 | 11 |

袜子5/13、腰带5/11的.5匹配未超过历史B；小包仅2例、skirt仅1例，不作普遍结论。小目标漏检、严格形状和outerwear仍有不足，整体AP增益不能代替分项验收。

## 启动依据

开工源 `4d60fcd`，wkq 新增 `b986630`、`c514a4f`：完整 dev250 语言视觉初稿和精简 v4（330 中英组/660 条，accepted=0），以及训练文字覆盖分析；未改变本轮分割 train/dev 输入。wyz `103dcc5`、main `7064ebb` 未变。所有远端分支、五个 PR 和活跃讨论已核；#2/#4 draft open，#5 open 且依赖 #4；无合并、新 PR 或新讨论。PR #4 正文仍是旧摘要，以上进度依据其最新 commit 的三份交接，不替队友更新 PR。

wkq 标注了原 dev 若干类别/源 mask 碎片疑点，尚无裁决或修订。本轮两组使用同一原 dev250（250 图/1008 GT），保留这些评价局限；不改标注、删样本或重建数据。语言 v4 不进入本轮训练/评价。

资源快照：RTX3090 0 MiB/0%，无计算进程、无活动协调任务，共享锁可用；数据盘空闲 164682031104 bytes（约153.4 GiB）。新输出和两组 run ID 均不存在。见 [GitHub快照](evidence/expansion_control_20261117/github_preflight.json)、[服务器快照](evidence/expansion_control_20261117/server_preflight.json)。

复用 [上一轮入口准备及11项CPU检查](expansion_execution_preparation_20261116.md)、旧 COCO/八类头/AMP/稳定性/恢复、旧 dev 验收和新10k接收/隔离证据。本轮只登记启动授权和新 run ID，训练/评价科学配置及代码未修改；不重跑准备模式、完整数据审计、测试套件或 smoke。正常 trainer/evaluator 的身份门禁继续执行。

## 执行

输出：`/root/autodl-tmp/fashion-grounding/runs/segmentation/segmentation-expansion-control-20261117-r01`。

- C5k：`segmentation-expansion-5000-20261117-r01`
- D10k：`segmentation-expansion-10000-20261117-r01`

固定 segmentation 环境及干净源码，通过已实现的共享锁/独立进程组/绝对截止时间入口执行；仅本角色 checkout。本轮实际启动命令（历史复现记录，不授权重跑）：

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/autodl-tmp/fashion-grounding/envs/segmentation-v1/bin/python \
scripts/run_expansion_control.py \
  --output /root/autodl-tmp/fashion-grounding/runs/segmentation/segmentation-expansion-control-20261117-r01 \
  --execute
```

比较规则沿用 [受控方案](train10000_control_plan_20261116.md)，主要终点为同预算20k；10k/20k组内选最大完整dev AP，平手取早。旧B20k AP63.79295仅复用报告比较，不重推理；不自动替换现有候选、wyz缓存或部署权重。这里只完成分割对照，不扩展至语言、排序或test。

## 证据与后续边界

- [C5k训练报告](evidence/expansion_control_20261117/C5k/train/train_report.json)、[1000步窗口日志摘要](evidence/expansion_control_20261117/C5k/train/training_summary.json)、[AMP原记录](evidence/expansion_control_20261117/C5k/train/amp_overflow.jsonl)。
- [10k评价](evidence/expansion_control_20261117/C5k/dev/update_010000/evaluation_report.json)、[20k评价](evidence/expansion_control_20261117/C5k/dev/update_020000/evaluation_report.json)、[组内选择](evidence/expansion_control_20261117/C5k/selection.json)、[历史比较](evidence/expansion_control_20261117/completed_results_comparison.json)。
- [小目标诊断](evidence/expansion_control_20261117/C5k/diagnostic20000/summary.json)、[D10k中断分母](evidence/expansion_control_20261117/D10k_targeted/train/interrupted_summary.json)、[最终阶段状态](evidence/expansion_control_20261117/stage_status.json)。
- 输入复用回执分别为C5k unchanged_train、D10k accepted_expansion；COCO完整性/适配记录表明同一官方文件和同一seed配置被实际加载。D10k未生成完成报告或checkpoint，其初始参数digest未落盘，因此原定最终双digest配对守卫未执行，不能宣称整对已验收。
- 图片、checkpoint、32MB训练逐步日志、完整预测和逐GT账本留服务器；Git仅存配置、小型原始报告、hash与交接。C5k历史对照不是新的AP推理。

保持本轮中断状态；后续是否重启D10k或改用更短设计需新的明确指令，不自动重跑C5k、不从不存在的D10k checkpoint恢复。无需再做已通过的COCO/头适配/AMP/恢复/smoke。本次没有完成等预算数据扩充对照，也不宣称10k改善或无效。
