# 10k训练完成与最终权重选择

2026-11-18，lcy / segmentation，Codex汇总既有结果。**completed / final checkpoint selected**：本次10k训练已于 **2026-11-17 23:30:37 HKT** 正常结束，达到10等效epoch/50000次成功update，总21985.538秒（6小时6分26秒），未触发早停或硬超时，未自动延长或重启。执行源固定 `0541df4f7abbb7bc569bbbc59a4b9ed52f1a3c2f`。

本轮是结束核查和离线汇总，新增模型加载、训练update、推理、smoke均为0；复用原训练、恢复、评价和数据验收记录。“最终”指本次10k实验的最终权重，不等于全组P7方法冻结或最终test已完成。

## 五个阶段的完整dev结果

统一dev250/1008GT、800/1333、FP32、upstream instance、top100、无score阈值；五份报告均完整、零推理错误且可用于选择。AP按0–100标度报告。新run从C5k20k只加载模型参数后重新从update0计数，下表update为新run累计成功更新。

| checkpoint | AP | AP50 | AP75 | AP_small | AP_medium | AP_large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧最佳 C5k20k | 64.6957 | 80.3897 | 68.4916 | 11.0483 | 31.3090 | 73.7301 |
| 新2epoch / 10000updates | 63.2350 | 80.5538 | 67.9905 | 12.5737 | 32.0876 | 73.5781 |
| 新4epoch / 20000updates | 66.3351 | 82.0282 | 70.1123 | 19.8113 | 53.5313 | 74.7239 |
| 新6epoch / 30000updates | 66.7770 | 82.7211 | 71.5677 | 17.1241 | 47.2974 | 75.5711 |
| 新8epoch / 40000updates | 68.5897 | 83.8415 | 73.2629 | 21.3105 | 47.0885 | 77.7153 |
| 新10epoch / 50000updates | 69.0313 | 84.2177 | 73.4854 | 23.1082 | 45.8002 | 77.6325 |
| 最终50k相对旧C5k | +4.3356 | +3.8280 | +4.9937 | +12.0599 | +14.4911 | +3.9024 |

按预定的**完整dev mask AP最大值**选择，最终为新 **10epoch/50000updates**，AP69.0313。与旧C5k相比+4.3356点，与40k相比+0.4417点；小目标AP23.1082，比旧C5k +12.0599点；AP75为73.4854，比旧C5k +4.9937点。新run同分取较早update、跨run同分保留C5k的规则未变，未按某个类别或其他指标临时改选。

五份新run报告先沿用原严格组内身份规则选择，再单独与相同dev协议的C5k缓存基线比较。离线汇总与服务器selection.json、worker_result.json及最后阶段selection完全一致。2/4epoch和6/8epoch原报告复用已有GitHub证据，没有重新推理旧权重。

## 逐类表现与仍存在的不足

| 类别 | 旧C5k AP | 最终50k AP | 差值 |
| --- | ---: | ---: | ---: |
| top（上衣） | 70.1122 | 74.7557 | +4.6435 |
| pants（裤子） | 81.7679 | 82.7867 | +1.0188 |
| skirt（裙子） | 67.5248 | 74.2462 | +6.7214 |
| outerwear（外套） | 69.3756 | 74.6426 | +5.2670 |
| dress（连衣裙） | 83.8757 | 83.2181 | -0.6576 |
| shoes（鞋子） | 53.7159 | 58.4935 | +4.7776 |
| bag（包包） | 52.2789 | 57.7063 | +5.4275 |
| accessory（配饰） | 38.9146 | 46.4015 | +7.4869 |

七类高于旧最佳，dress略降0.6576点，不能写成八类全面提升。最终50k的skirt AP比40k低约0.9877点；AP_medium/large较40k分别低1.2883/0.0828点，但总体AP更高，仍按既定主指标选50k。小目标、鞋/包/配饰有所改善，绝对AP仍显著低于大目标和强类别。

AP75改善是高IoU匹配表现的证据，不是逐实例边界误差分析；未新增同类小目标漏检/分类分解或边界诊断。本次是C5k热启动后的10k性能优化，额外训练、配方和数据均不同于旧C5k，不是同预算5k/10k因果消融，也未做多种子显著性验证。

## 最终权重与复用入口

[最终权重索引](evidence/tenk_completion_20261118/final_checkpoint.json)固定服务器位置、SHA、源码/config/data身份与dev指标：

```text
/root/autodl-tmp/fashion-grounding/runs/segmentation/tenk-adaptation-20261117-r01/epoch_10/train/checkpoint_050000.pth
SHA256: 1ed0f913602746e027ae84ec9a84c98a80306b34820a824d39ddafba1671208a
```

- checkpoint文件528479885字节，原训练完整状态文件；未复制、重命名、裁剪权重或改变格式。
- 训练配置沿用[真实resolved配置](evidence/tenk_launch_20261117/resolved_config.json)，SHA `2437450750f2d3b0114e3a670e7c919d45496f1bfb8eba4fc157757607925f2c`。
- 评价配置：[expansion_control_dev250.json](../../../configs/runs/segmentation/expansion_control_dev250.json)，SHA `4d732d3d4cd81f12d9f6636c5674b4792bf6536ef8f7fe346ecd3e97a08feaff`。
- 10k与dev数据hash、五份报告位置及服务器原报告hash见[汇总JSON](evidence/tenk_completion_20261118/summary.json)；不访问test。
- 原训练过程已完成checkpoint完整hash和模型状态校验；本轮仅核对小报告hash、记录一致性、文件存在和sidecar对应关系，没有为复核再加载或重哈希真实checkpoint。

50k成为本人当前分割最佳权重。旧C5k、所有阶段checkpoint/预测保留；已交付的18k/0.3候选继续保留原身份，**没有自动替换候选或重跑wyz排序**。若下一阶段需要新版候选，应单独明确范围后执行；本轮不追加该工作。

## 完成、预算与资源证据

- 新run累计50000成功updates、100000次图像曝光；五个训练段全部完成，4次完整状态恢复记录passed，优化器/调度/采样/RNG等指纹一致。
- 五段成功更新的loss/梯度均有限，跳过optimizer update为0。累计14次AMP溢出均按原策略重放；无失败重启。
- 训练峰值显存allocated约4.44GiB、reserved约5.17GiB，不将训练峰值当作整个评价链路的最大显存。
- 五段训练＋dev＋选择分别75.50/73.26/72.50/72.05/73.04分钟，均低于2小时；整体6小时6分26秒低于8小时。10个checkpoint均保留。
- 11月18日00:25:40 HKT复核：GPU0MiB/0%，无计算进程、无活动协调记录，共享锁可获取；监督器、wrapper和worker均已退出；数据盘余145.27GiB。服务器本人checkout仍是0541df4 clean，未修改队友工作区。

[最终run_metadata](evidence/tenk_completion_20261118/run_metadata.json)、[worker结果](evidence/tenk_completion_20261118/worker_result.json)、[共享锁完成记录](evidence/tenk_completion_20261118/coordination_final.json)、[阶段索引](evidence/tenk_completion_20261118/stages.json)、[资源复核](evidence/tenk_completion_20261118/server_postflight.json)、[结束核对记录](evidence/tenk_completion_20261118/completion_receipt.json)。旧启动/运行中快照保持历史，未覆盖为completed。

## 全组GitHub与同步范围

开工已fetch整组分支、检查全部5个PR和活跃#2/#4/#5讨论。本人读取时a0b47f2；wkq b24515b、wyz103dcc5、main7064ebb及两条历史full-data分支均未变，队友交接按固定commit复用；没有跨角色新增依赖。[GitHub快照](evidence/tenk_completion_20261118/github_preflight.json)。

本轮同步个人分支的README、状态表、交接、完成证据与registry：训练行改为completed，新增10epoch完整dev行，既有2/4/6/8epoch评价行保持。选中50k及其hash明确记录。PR #2继续draft未合并；已知连接器编辑PR正文403，因此未重复无效权限尝试，当前状态以分支文档与diff为准。没有新增训练/推理、修改模型实现、复跑旧测试或执行最终test。
