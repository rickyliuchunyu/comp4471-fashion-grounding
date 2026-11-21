# 3000→5000步dev回落诊断与候选交付准备

日期：2026-11-14（Asia/Shanghai）。角色segmentation / lcy；Codex执行；PR #2。状态：离线分析completed；候选交付prepared_not_exported。

**保留第3000步权重。总体AP50.5603→48.8844的回落主要集中于skirt的相对排序/错类问题和部分高IoU几何退步，outerwear兼有少量漏检与错类。AP-small实际上7.5811→8.4758，并非小目标整体变差。** 单seed、相邻保存点不能证明过拟合或收敛；本轮没有新增数据或评价实现错误证据。

## 开工核对与执行边界

已fetch并检查实际远端heads、全部5个PR、活跃PR的评论/review及固定提交交接：main=`7064ebb`，lcy开工=`a69af74`，wkq=`f5787f4` / #4 open-draft，wyz=`103dcc5` / #5 open且base=#4；#2 open-draft，均未合并。#1/#3 closed-unmerged，旧分支在实际远端已删除（本地残余tracking ref不是新进度）。没有新增分支/PR、队友接口变化或活跃PR评论。继续沿用用户已确认的分割数据复核；未把语言needs_revision自动提升为accepted。[GitHub快照](evidence/dev_analysis_20261114/github_preflight.json)。

开工服务器系统盘约27GiB、数据盘约163GiB可用，RTX3090 0MiB/0%，无GPU计算进程。CPU分析设CUDA_VISIBLE_DEVICES为空、BLAS/OMP单线程、600秒硬上限；不持GPU锁。没有重新下载权重、加载模型、运行训练或推理、重跑旧smoke/全套测试。结束盘点及共享文件hash见[交付回执](evidence/dev_analysis_20261114/delivery_receipt.json)。

## 输入、方法与可复核性

只读取固定250图/1008GT dev及 `segmentation-resume5000-ap-20261113-r01/update_003000`、`update_005000` 的既有predictions.json。每份25000条官方query×class top100结果。报告SHA、预测SHA、标注/manifest SHA与已归档证据一致；没有重审全部图片、重做包验收或重评1000步。

新增脚本：[analyze_saved_segmentation_predictions.py](../../../scripts/analyze_saved_segmentation_predictions.py)，配置：[dev_error_analysis_3000_5000.json](../../../configs/runs/segmentation/dev_error_analysis_3000_5000.json)。仅在CPU重算COCO匹配，取得旧汇总中没有的逐GT/逐预测及分数排序诊断；两份AP/AP50/AP75/各尺寸/各类均与原报告一致（误差<1e-8）。这项离线派生不是重新推理，也不是语言候选评价。

- COCO匹配：使用其原始类别约束、score排序、IoU阈值及ignore规则。
- 独立最大IoU：每GT在已有top100里找同类/异类最大几何重叠，**不是一对一匹配，也不是query-weighted coverage**。
- FP描述：忽略/TP/同类重复/异类IoU达到阈值/最大IoU在0.1至阈值间/背景或未标注。优先级固定，类别可能混合多种误差，不能解读为TIDE式因果AP贡献。
- score≥0.5只是预先固定的诊断切片；不搜索阈值，不声称它是正式语言候选。合成score不能进一步拆成类别概率和mask质量，现有数据不支持归因到某一内部logit。
- 尺寸按原图面积；AP-small使用COCO按面积ignore后的匹配。下述小目标逐GT迁移则取all-area匹配后按area≤1024过滤，二者明确分开。

r01在已完成3000步匹配后，写JSON时遇到NumPy uint32序列化错误；属Codex新增脚本输出问题。原目录/日志保留，只将area转为Python int后执行r02，未重跑模型。r02源码=`a91c5811e7d82627d49adca5fbd1b98776d5bee8`，总CPU wall time=6.212秒，torch/checkpoint加载、推理与optimizer update均为0。失败与成功各有独立registry行。

## 各类对总体AP变化的贡献

总体AP在八类上取宏平均，最后一列只是类别AP差/8的代数分解，不是误差干预的因果贡献。

| 类别 | GT数 | AP3000 | AP5000 | 总体AP变化贡献 |
| --- | ---: | ---: | ---: | ---: |
| top | 140 | 56.5549 | 56.3289 | -0.0283 |
| pants | 91 | 69.3892 | 72.9278 | +0.4423 |
| skirt | 29 | 48.4508 | 30.6052 | -2.2307 |
| outerwear | 65 | 53.4679 | 47.4331 | -0.7543 |
| dress | 111 | 73.7763 | 77.8162 | +0.5050 |
| shoes | 327 | 39.9167 | 41.4998 | +0.1979 |
| bag | 45 | 38.1456 | 36.8632 | -0.1603 |
| accessory | 200 | 24.7811 | 27.6007 | +0.3525 |

skirt与outerwear合计−2.9850点，其他六类合计+1.3091点，净−1.6759点。所有GT在IoU.5的匹配数反而883→887；因此不能将整体AP回落解释为“所有类漏检增加”。

## skirt：主要是相对排序退步，伴有高IoU边界问题

29个GT来自29图，其中27个large、1个medium、1个small。

| 诊断 | 3000 | 5000 |
| --- | ---: | ---: |
| AP50 / AP75 | 51.3036 / 48.4710 | 35.1343 / 30.9218 |
| IoU.5匹配GT | 28/29 | 28/29 |
| IoU.75匹配GT | 26/29 | 25/29 |
| 同类最大IoU均值 | 0.9058 | 0.8780 |
| TP50合成score中位数 | 0.6660 | 0.2162 |
| score≥0.5的TP50 | 21 | 9 |
| 全dev该类最高20个score中的TP50 | 11 | 8 |

IoU.5匹配的28个GT全部保留，没有新增漏检；但相同召回上限下AP50下降16.17点，结合前20名TP减少，说明相对排序/precision-recall曲线明显变差。绝对score下降本身不能证明AP下降（单调整体重标定不改变AP），必须结合相对排名看。

高分错类重叠FP从36→12，并没有绝对数量增加：3000主要为pants/dress各17、top2；5000的12个均与dress GT重叠。但真阳性分数下降更明显，剩余错类结果仍排在真阳性之前。另有IoU.75匹配3个丢失、2个获得，large AP51.5529→31.6745，存在局部边界/完整性损失。

例：图1000018166、GT1000002616，同类最大IoU0.962→0.579，score0.488→0.0017；5000步异类dress预测score0.963、与该skirt IoU0.692。图1000012154同类IoU0.972→0.746，mask包含了更多上身区域。案例取自已保存预测，不改GT或预测。

## outerwear：分类混淆、少量漏检与边界同时变化

65个GT/63图，63个large、2个medium、0个small。AP53.4679→47.4331；AP50 72.6360→64.9979，AP75 58.8136→53.4222。IoU.5匹配64→63，IoU.75匹配58→56；同类最大IoU均值0.8707→0.8448。

TP50 score中位数0.8281→0.7534；score≥0.5的TP从50→44，高分异类重叠FP从20→25。其中5000的25个有24个重叠top GT、1个重叠dress GT；需要优先看top/outerwear边界，而不是追加全类数据验收。

图1000017379的手提外套：3000同类IoU0.972但score仅0.011；5000没有有效同类重叠，但bag预测IoU0.945、score0.967。几何区域仍存在，错误主要表现为类别；不据此认定源标签错误。图1000019876的外套同类IoU0.824→0.537，叠图显示掩膜覆盖了更多内搭区域，属于另一个几何问题。

## 小目标：总体小幅改善，内部表现不同且样本不均衡

138个GT/96图：shoes68、accessory67、bag2、skirt1；top/pants/outerwear/dress无小目标。COCO AP-small只对有正例的四类取宏平均，稀有类影响很大，不能把它当成按138个实例加权的准确率。

| 小目标类别 | GT数 | AP3000 | AP5000 |
| --- | ---: | ---: | ---: |
| shoes | 68 | 12.7493 | 16.5155 |
| accessory | 67 | 12.2898 | 9.5609 |
| bag | 2 | 5.2854 | 7.8267 |
| skirt | 1 | 0 | 0 |

all-area匹配后的小目标子集：IoU.5为83→81（保留71、丢失12、新增10、两次均漏45）；IoU.75为27→38（保留23、丢失4、新增15）。独立同类最大IoU均值0.5127→0.5259。较严格几何匹配改善与部分IoU.5漏检并存，符合AP-small略升，不能写成统一退步。

选取的失败例：图1000015687的210像素鞋子可见区域，最佳同类IoU0.653→0.005；图1000011092的453像素眼镜，0.621→0。二者在3000步分数也只有0.0205/0.0092，说明“原始AP top100里找得到”不等于已通过语言候选阈值或能用于排序。

## 已有证据支持的下一步

1. 保留3000步作为当前交付权重；不要直接续训或换超参。总体mask AP选择规则未变，5000不替代它。
2. 先补正式候选导出的编排入口并取得一次必要推理的授权，再利用同一前向保存阈值前、带真实query ID的候选缓存，避免为三档阈值重复推理。该工作本轮未执行。
3. wkq继续accepted语言dev的独立审核，wyz继续M2/正式训练入口；候选准备不替代语言数据和核心消融。
4. 若以后考虑训练调整，先围绕skirt/dress、outerwear/top混淆及小配饰做单因素设计；本轮仅提出待验证方向，没有证明学习率、类别损失或分辨率哪个应改，也未做阈值调优/重标数据。

## 候选交付和证据位置

[3000步候选交付准备](candidate_u3000_preparation_20261114.md)已明确权重SHA、解码、train/dev manifest、wyz消费接口、共享准备目录与尚缺项。**prepared_not_exported**：现有AP输出缺真实query ID/logits，不能无损还原“一query一候选”包，正式阈值与新包A/B验收仍待完成。

原始逐GT/逐预测诊断留服务器 `runs/segmentation/segmentation-dev-analysis-20261114-r02/`，完整路径/hash/实际命令在[metadata](evidence/dev_analysis_20261114/run_metadata.json)。[分析汇总](evidence/dev_analysis_20261114/analysis.json)、[错类关联和六个案例](evidence/dev_analysis_20261114/case_details.json)、[失败记录](evidence/dev_analysis_20261114/failed_attempt_metadata.json)已归档。案例渲染脚本：[render_saved_segmentation_cases.py](../../../scripts/render_saved_segmentation_cases.py)，源`91e0bfa`；PNG留服务器 `focused_cases/paired_cases.png` 和本地outputs，不上传原图到GitHub。

六图按每组同类最大IoU下降最多的两例确定，是刻意选择的失败案例，不是随机质量样本；GT裁剪与叠图仅诊断，不进入候选payload。报告不把这些例子扩大为数据质量认证，也不把原始AP预测当生产候选。
