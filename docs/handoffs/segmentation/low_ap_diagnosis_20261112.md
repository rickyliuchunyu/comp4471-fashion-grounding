# 低 AP 诊断：训练增广补边污染，并非只有步数不足

后续执行：[修复后1000步与完整dev选择](postfix1000_dev_ap_20261112.md)现已完成，AP35.3047/42.8038，选择1000步；下文保留诊断阶段事实与当时尚未启动的计划，不改写为配对因果实验。

结论：发现已复现的 **LSJ 二值实例 mask 补边错误**，必须先修复；100 updates 不足也客观存在，但不能据原运行推断“只要加训即可”。原始数据与 dev AP 评价没有发现相应错误。执行者：Codex，lcy 工作区；本轮没有模型加载、推理、训练、完整验收重跑或 COCOeval 重算。

## 直接证据与责任

安装的 Detectron2 `FixedSizeCrop` 默认 `seg_pad_value=255`。本项目 `map_training_record` 把 RLE mask 直接送入 `transforms.apply_segmentation(mask).astype(bool)`，导致补边 255 变为 True。图像缩小或长宽比导致 padding 时，每个实例都额外包含同一片右侧/下方背景。源数据没有这些错误区域，污染发生在 **训练适配器**。

两个新增 CPU 回归用例在旧源码上都失败：

1. 32×32 图中 [8:24,8:24] 的方形实例缩到 8×8 后，应只剩 [2:6,2:6] 的 16 个前景像素，补回 32×32 的区域必须全为背景；旧实现把补边标成前景。
2. 缩放后消失的单像素对象应被过滤；旧实现由于补边非空，仍保留一个假实例。

旧 LSJ 测试只检查无缩放的几何及随机状态重放一致性，没有对缩小补边给出独立的正确答案；有限 loss/梯度和恢复一致性同样不能证明监督 mask 正确。**这是 Codex 编写训练适配器及测试覆盖的缺陷，不是用户要求不当，也不是 wkq 原始数据包的问题。**

修复提交 `ba94e5b835530069f12e7069c0df896a21a7e0c1`：显式传入 `seg_pad_value=0`，保留图像 pad=128 及其他配方。只运行上述两个 CPU 用例，旧版 2 failed，修复后 2 passed（0.632 秒），未重跑旧测试套件、权重加载、恢复或 GPU smoke。

## 本次 100-step 运行受影响范围

以旧训练配置、NumPy seed42、日志中的实际图片顺序和尺寸重建增广；不读取图片像素、不加载模型。所有 200 张图重建的旧实例数量均与日志吻合。数值来自确定性重建，原运行并未保存逐张增广 mask，不能冒称直接读取了历史增广张量。

| 重建统计 | 结果 |
| --- | ---: |
| 200 张训练图中需要补边 | 138（69%） |
| 补边面积 / 768×768 的中位数（含无补边图） | 25.39% |
| 错误补边像素 / 所有实例目标前景像素之和 | 88.94% |
| 旧实现保留实例 / 去掉错误补边后实例 | 683 / 654 |
| 本应为空却被补边“保留”的实例 | 29 |

88.94% 是按实例累加的前景像素比例，**不是图片错误率，也不是已量化的 AP 损失贡献**。重建脚本和逐图记录见 [证据目录](evidence/low_ap_diagnosis_20261112/)。该缺陷足以使现有权重不宜续训或作为正常 S0 基线。

## 训练不足的独立证据

- 100×batch2 仅曝光 200 张不同图片，占 train5000 的 4%，约 0.04 epoch；原八类（含 no-object）分类头重新初始化。
- 500-step warmup 未走完。主 LR 从 `1e-7` 到第100步 `1.98802e-5`，目标为 `1e-4`；这100步主 LR 累计 / 目标 LR = 9.9901。此值仅比较学习率积分，不能声称等价于10步 AdamW 更新。
- 前10步到后10步平均总 loss 从 400.14 到53.89；加权主/辅助 CE 合计80.05→34.16，mask BCE合计282.78→10.06，Dice合计37.32→9.67。loss下降包含对错误补边的拟合，不构成正确分割已学会的证据。

## 已保存 dev 预测说明了什么

只对原25,000条预测/保存点与1,008个GT做 RLE IoU 配对统计，未重新推理、重新评分AP或改阈值。下列覆盖率是每个GT是否存在任意匹配，不做一对一去重，**不是 COCO AP/AR**。

| 指标 | 第50步 | 第100步 |
| --- | ---: | ---: |
| 原报告 mask AP（0–100） | 0.1007 | 0.4585 |
| 忽略类别，GT有mask IoU≥0.5的比例 | 27.78% | 33.43% |
| 同类别，GT有mask IoU≥0.5的比例 | 7.34% | 18.55% |
| 同类别，GT有mask IoU≥0.75的比例 | 3.17% | 5.75% |
| 预测 mask 面积 / 原图面积中位数 | 0.37% | 4.22% |

可见类别识别和 mask 定位都不足，AP上升不能洗清训练污染。第100步 shoes 只有25/327个GT存在任意类别 IoU≥0.5匹配，accessory为20/200；小目标问题突出。预测面积明显增大与错误大面积监督相容，但没有干净对照，不能单凭这一变化断言因果贡献。AP使用全部top100，不是被0.5语言阈值误删；低置信度本身也不必然导致低AP，排序与重叠才决定分数。

## 为什么目前不归因于源数据或评价器

复用三方已经通过的源 mask 对照、RGB/尺寸、ID/hash/类别/几何与分区检查，本轮不重复。源码逐项追踪：train项目类别1–8减1，输出加1；原图RLE与GT同尺寸；训练和评价均为RGB且使用上游RGB均值/方差；COCOeval用segm、IoU .50–.95、maxDets100，百分数转换一致。保存结果覆盖250/250图、零推理错误，两个checkpoint对应正确train/hash。

没有发现证据指向类别偏移、错图片关联、RGB/BGR反转、mask尺寸错位或AP单位错误；这不是对所有潜在数据/评价缺陷的绝对排除。新增缺陷位于训练增广路径，原始输入验收不覆盖它，不能要求 wkq 无依据地返工数据。

## 旧结果处置与下一步

- 原日志、AP、checkpoint、selection.json全部保留；AP仍是这些受污染模型的真实测量值。两份checkpoint标记 **不得续训、不得导出生产候选**，见 [处置记录](evidence/low_ap_diagnosis_20261112/checkpoint_disposition.json)。旧200-update LSJ工程权重同样受该代码影响；数值稳定/状态恢复证据保留其原本含义，不当作训练语义正确的证明。
- 下一项有意义的实验：使用已校验的官方COCO重新初始化（不接着第100步），保持现有配方，只修正mask补边；限定1000 updates，跨过500步warmup，在500/1000各评完整250图dev。建议硬上限训练600秒、评价600秒；不搜索学习率/阈值、不增加seed、不做test。
- 此方案用于观察正确监督下的学习趋势，仍不是完整S0收敛试验；如需单独量化补边修复在第100步的影响，另行定义同预算配对对照，不能拿未来1000步与旧100步直接归因。
- **本轮只完成诊断和最小修复，下一训练实验尚未启动。** 不重跑已通过的环境/数据全验收、COCO下载、模型smoke、200-update稳定性或恢复测试。下一次运行必须避开旧污染checkpoint。

## 版本与复现

整组开工检查：lcy=`de49ae6`，wkq=`f5787f4`，wyz=`103dcc5`；PR2/4/5状态与交接无新增变化，均未合并，无新评论。服务器系统/数据盘约27/165GiB空闲，GPU空闲；本轮CPU设置 `CUDA_VISIBLE_DEVICES=""`。

- [只读诊断脚本](evidence/low_ap_diagnosis_20261112/analyze_existing_outputs.py)；服务器输出 `runs/segmentation/low-ap-diagnosis-20261112-r01/`，9.42秒，未重跑COCOeval。
- [分析摘要](evidence/low_ap_diagnosis_20261112/diagnosis.json)、[逐图增广重建](evidence/low_ap_diagnosis_20261112/augmentation_replay.json)、[已有loss/LR聚合](evidence/low_ap_diagnosis_20261112/log_analysis.json)。预测文件hash与原AP报告一致。
- [失败复现](evidence/low_ap_diagnosis_20261112/low-ap-padding-before-20261112.log)、[修复后两用例](evidence/low_ap_diagnosis_20261112/low-ap-padding-after-20261112.log)。修复后实际命令：`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/fashion-grounding/envs/segmentation-v1/bin/python -m unittest discover -s tests -p test_segmentation_lsj_padding.py -v`。
