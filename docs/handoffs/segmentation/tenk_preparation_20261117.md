# 10k 模型参数热启动与增强可见性准备

2026-11-17，lcy / segmentation，Codex 执行。**prepared_not_started**：真实 train-only CPU 可见性诊断、模型参数热启动与 cosine 支持、10k 配置及每2epoch评价规划完成；没有启动训练、分割推理或旧 smoke。只处理分割，不接语言训练/排序，不访问 final test。

当前方案取代“等待完整 Fashionpedia 包再训练”的个人下一步；旧完整数据/100epoch/1–2epoch提案保持历史，不合并其运行器。C5k20k AP64.6957 与旧18k/0.3候选分别保留。此次是模型性能优化，不是已完成的5k/10k同预算消融。

## 开工与证据复用

- 已 fetch 全部远端；本分支起点 `30baf267be48df93e631edd676f6d0d29982d46d`。wkq `b24515b`、wyz `103dcc5`、main `7064ebb` 未变；两个独立 full-data 分支也检查。全5个PR元信息与活跃#2/#4/#5讨论已核对，讨论为空；#1/#3关闭未合并，#2/#4 draft、#5 open且依赖#4。详见 [GitHub快照](evidence/tenk_preparation_20261117/github_preflight.json)。
- wkq 最新149张语言训练增量均在现有10k内，没有新增分割标签；新39张语言困难dev未接收。本轮10k不等待语言审核或尚未发布的完整分割包。
- 16:30 HKT：系统盘剩余28,916,768,768字节，数据盘161,685,753,856字节（约150.6GiB）；RTX3090显存0MiB、GPU0%、无计算进程，锁空闲、无活动协调记录。16:45结束快照仍空闲；正式服务器segmentation checkout保持 `1ec9723` clean，未更新训练部署。CPU代码在独立运行目录的快照执行。[前](evidence/tenk_preparation_20261117/server_preflight.json) / [后](evidence/tenk_preparation_20261117/server_postflight.json)。
- 复用 COCO/八类头、200-update、AMP恢复、旧5k/dev像素几何、10k增量/旧5k保留/隔离验收及所有历史AP/预测。只新增增强可见性、跨数据model-only入口、cosine及其必要检查。

## 新诊断：问题与方法

未回答的问题是：真实小目标经当前训练增强后是否仍可见，目标保留裁剪是否牺牲其他实例。新入口 [visibility.py](../../../src/fashion_grounding/segmentation/visibility.py) 与训练 mapper 共用几何变换，不运行模型。

固定train10000 manifest `9966383f85e1a4223a261f023eb80359b945815dff42ad4965c66c26225250d0`、annotation `780b55fd3965606b41efeee62a7e708fcd45d3bd114b198d70f813f9f93f9f54`。seed20261117，先均匀抽128图，再按旧/新增及小目标源细类补256图，共384图；每图4个确定性随机视图。两个方案使用同一清单与逐视图seed。均匀组与富集组单独报告，不把后者冒充全10k分布。

读取真实RLE与已验收的图像尺寸；缩放/翻转/裁剪随机参数只依赖尺寸和RNG，因此用空RGB画布生成与训练相同的几何。没有重新解码图片、核验全包像素或进行人工语义复核。原图面积≤1024为small，≤256为extreme；裁剪保留比例以裁剪前同尺度前景为分母，不能用缩放后面积直接除原面积。重复视图不是独立图片样本。此诊断未测实际loss点命中、AP或训练收益。

原方案：768、LSJ0.5–2、水平翻转、固定随机裁剪。候选方案：符合条件的图以概率0.25换成目标上下文裁剪，保留完整目标框，裁剪尺度至少原图长边的一半、目标框长边的3倍，再缩放/补背景到768；同图所有实例用同一变换。实际1536个视图中269个进入该分支。

| 同一组实例/视图 | 原LSJ | 25%候选裁剪 | 解释 |
| --- | ---: | ---: | --- |
| 均匀组small消失率，81实例/324视图 | 20.68% | 19.44% | 改善有限，来自49张含small图片 |
| 均匀组small保留不足一半 | 23.77% | 22.22% | 同分母 |
| 均匀组small输入面积<64像素 | 28.70% | 25.00% | 包含完全消失 |
| 富集组small消失率，605实例/2420视图 | 10.50% | 11.86% | 其他小目标受损 |
| 全部非small消失率，1488实例/5952视图 | 8.28% | 10.84% | 全部实例必须计入 |
| 全部非small保留不足一半 | 14.03% | 18.18% | 上下文/完整实例损失增加 |
| 小袜消失率，119实例/476视图 | 16.18% | 17.86% | 不支持弱细类系统改善 |
| 被选中的目标消失率，269视图 | 不适用 | 0% | 不能据此忽略同图其他实例 |

**决策：本轮10k配置不加入定向裁剪，保留原LSJ。** 候选放大了选中目标，但未通过“同时检查其他实例”的准备判断；不把选中目标100%保留写成模型收益，也不继续搜索裁剪变体。这是本样本上的保守工程选择，不是统计显著性或所有定向裁剪均无效的结论。

基线CPU127.348秒，候选115.336秒，各自900秒硬限，均exit0。两次是不同增强方案的新增诊断，没有重复旧检查。[基线](evidence/tenk_preparation_20261117/visibility_baseline.json)、[候选](evidence/tenk_preparation_20261117/visibility_target_context.json)、[决定](evidence/tenk_preparation_20261117/visibility_decision.json)。

## 准备好的训练配置

- [训练配置](../../../configs/runs/segmentation/tenk_adaptation_train.json)：同R50/八类，768，scale0.5–2，batch2/accum1，seed42；不改变loss/匹配代价/类别权重，不启用target crop。
- [仅模型参数来源](../../../configs/runs/segmentation/tenk_model_only_source.json)：C5k20k，checkpoint SHA `d9deacc85d1bf402cf6417a71226bd51e1fc98b631dcd30c9afcc22d51b51044`，原训练源`dc2fba4`。新`--model-only`路径严格核对hash、源数据/config/update/source/model指纹及所有tensor键/形状/dtype，保留训练好的八类头；不走COCO重新建头适配。
- 新优化器、scheduler、scaler、sampler、RNG、AMP恢复计数；新update从0开始，历史20k只作为来源。后续同一新run的分段继续才用`--resume`恢复完整新状态。
- AdamW head/decoder峰值LR5e-5、backbone5e-6；500个新成功update warmup，之后cosine平滑衰减，末端factor0.02。旧12000/16000里程碑不沿用，旧终末LR不继承。
- 一等效epoch=10000次图片曝光=5000次成功update。均匀无放回epoch采样，**最多10epoch=100000曝光=50000新update**。AMP失败重放不推进成功update；每1epoch存权重，每2epoch评完整dev后才进入下一训练段。
- 新`--input-validation-bundle`训练读取路径绑定已接收10k清单、标注和发布catalog，复用旧验收；不重做图像hash/解码/mask几何。真实元数据读取10000图/39699实例通过，4.436秒。[记录](evidence/tenk_preparation_20261117/accepted_training_load.json)。复用以不可变发布包为前提。

| 新训练等效epoch | 新update | 累计新图片曝光 | 评价 |
| --- | ---: | ---: | --- |
| 2 | 10000 | 20000 | 完整dev250 |
| 4 | 20000 | 40000 | 完整dev250 |
| 6 | 30000 | 60000 | 完整dev250 |
| 8 | 40000 | 80000 | 完整dev250 |
| 10 | 50000 | 100000 | 完整dev250，结束 |

固定dev250/1008GT、800/1333、upstream instance、FP32、top100、无候选分数阈值。每次保存预测，后续小目标/边界分析复用。以完整dev mask AP选择，既有C5k20k作为缓存基线参与比较；跨run同分保留基线，新run同分取较早点。不能混入不同训练身份去绕过原严格组内选择器。辅助观察AP_small、AP75、同类小目标IoU.5/.75、bag/shoes/accessory及已有边界口径；极小分母细类只作诊断。

至多10epoch；拟至少4epoch后才适用平台期判断，连续3次完整dev未超过当前最佳则停止（最早6epoch触发）；异常即停，不自动重启/延长。历史吞吐推算10epoch纯训练约5.83小时，非新配方实测。**拟总硬限8小时、每2epoch训练+dev段2小时**，覆盖加载/保存/评价/选择，总期限不随段落重置；本轮没有授予该训练预算。

## 入口与验证边界

[准备入口](../../../scripts/prepare_tenk_adaptation.py)只校验和打印阶段命令，没有execute模式：

```bash
PYTHONPATH=src .venv-segmentation/bin/python scripts/prepare_tenk_adaptation.py
```

[计划](../../../configs/runs/segmentation/tenk_adaptation_plan.json)及训练JSON均禁用执行。实际trainer在读取数据、创建输出或加载模型之前拒绝禁用配置。阶段模板见[准备结果](evidence/tenk_preparation_20261117/prepared_plan.json)。**未来启动仍需明确授权总预算，将既有共享锁/统一硬截止包装接入这五个段落及早停/跨run选择；本轮未实现或验证一个自动多段执行器。** 不直接执行裸阶段模板，不将准备状态写成已部署或GPU实跑。

仅新增12项CPU案例：本地5项model-only/cosine/旧调度兼容（合成状态，无前反向），服务器4项几何/采样，本地3项计划/禁用门禁；全部通过。另做一次新的真实元数据loader消费。历史测试套件不重复；未重新下载权重、加载真实C5k模型、运行训练/推理或评dev。[验证范围](evidence/tenk_preparation_20261117/new_logic_validation.json)。

原始运行目录：`/root/autodl-tmp/fashion-grounding/runs/segmentation/tenk-preparation-20261117-r01/`。保留两份逐实例/逐视图账本、原始日志、各版本源码tar；源码快照hash和产物hash见[清单](evidence/tenk_preparation_20261117/artifact_sha256.json)。仓库只保存小型摘要/配置/证据，不提交图片、权重或全量预测。诊断代码快照早于最终准备配置，分别固定，不冒称都是最终commit执行。

下一步仅限本人：按此配置提出有界10k训练启动；保留本轮负裁剪结果，不再重跑诊断或smoke。新10k性能仍未知，旧最佳及下游候选不替换。
