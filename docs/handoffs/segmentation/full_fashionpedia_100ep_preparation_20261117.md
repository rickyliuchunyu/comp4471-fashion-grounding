# 完整 Fashionpedia / Mask2Former-R50：100-epoch 准备

2026-11-17，lcy / segmentation。状态：**preparation_checked / not_executable**。
用户授权本轮核对整组 GitHub，准备完整数据冻结清单、现有 R50 的100-epoch配置及评价/恢复方案，并同步 GitHub；不启动新训练，不改动正在执行的5k/10k对照。

## 交付与实际边界

- [冻结清单](../../../configs/runs/segmentation/full_fashionpedia_freeze_checklist.json)：字段、责任人、证据要求、旧输入保护及已知源标注问题。
- [训练配方](../../../configs/runs/segmentation/full100_train_recipe.json)：同一Mask2Former-R50，完整数据，100 epochs，batch16，768输入。
- [评价方案](../../../configs/runs/segmentation/full100_evaluation_plan.json)：每2 epochs完整dev、固定AP口径、缓存身份及checkpoint选择。
- [恢复方案](../../../configs/runs/segmentation/full100_recovery_plan.json)：完整状态恢复、同配置同数据、失败评价复用及执行前缺口。
- [总准备计划](../../../configs/runs/segmentation/full100_preparation_plan.json)：上述文件与旧运行关键文件的SHA256绑定。
- [CPU准备入口](../../../scripts/prepare_full_fashionpedia.py)：生成配置预览和累计epoch节点；没有`--execute`选项，不调用GPU/服务器/子进程训练。
- [本轮实际输出](evidence/full100_preparation_20261117/configuration/preparation.json)和[新增验证](evidence/full100_preparation_20261117/verification.json)。

完整包尚无本轮核实的发布清单、实际N、标注数、哈希和三方冻结记录；这些字段保持null/pending，**不是已冻结训练集**。准备输出故意保持`ready_for_training=false`，原trainer也会拒绝其null预算/未绑定输入。此次没有准备或执行可绕过这些条件的启动命令。

本轮在独立worktree、`segmentation/full-fashionpedia-100ep-prep`分支操作；原checkout停留在`segmentation/pilot-adapter-v1`。不修改原5k/10k配置、trainer、读取器、AMP恢复、共享环境、服务器数据或进程。没有连接服务器，因此不把历史空闲快照当作当前资源状态。

## 全组 GitHub 核对

开工源`308492f4ec19426845a5aa63f46f58639db2872f`；`git fetch origin`后读取全部远端分支、五个PR元信息与三个活跃PR的评论/review时间线。见[快照](evidence/full100_preparation_20261117/github_preflight.json)。

| 成员 | 固定head | 本轮事实 |
| --- | --- | --- |
| lcy | `308492f4ec19426845a5aa63f46f58639db2872f` | PR#2 open/draft，5k/10k启动交接已发布；其PR正文仍是较早准备说明，以实际启动交接为准，本轮不改原运行 |
| wkq | `4bbd7f3f30e92df8c7d23beabc5ce0ba814716e0` | PR#4 open/draft；新增train1000视觉草稿v3，分割10k及dev/test保持；未发现完整分割训练包的冻结交接 |
| wyz | `103dcc546714dd7c7866b4259d96a9825d8090ef` | PR#5 open/non-draft，依赖wkq分支，未合并；原消费接口和弱试训交接复用 |
| main | `7064ebb933584f2922607e51ce392e1572fd7dbb` | 未变化 |

PR#1/#3 closed-unmerged；无新PR、合并或活跃讨论。wkq最新[固定交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/4bbd7f3f30e92df8c7d23beabc5ce0ba814716e0/docs/handoffs/data_eval/train1000_visual_draft_20261117.md)是语言初查，不能据此宣告完整分割数据就绪；没有消费新语言查询或替队友改accepted状态。

## 完整数据冻结要求

生产方wkq维护转换与发布，lcy完成分割消费方接收，三方冻结确认分别留痕。`fp_train_full8_v1`只是拟议包名，实际路径尚未指定。

1. 使用全部符合协议的官方train，不混入官方validation或项目test。通过源清单给出“原始 → jumpsuit整图排除 → 非法/无整件实例排除 → 泄漏隔离 → 实际N”的数量账，不能把45,623直接填为训练N。
2. 固定现有八类映射与内部0–7映射；保留源细类、source ID、group ID、原图尺寸与实例定义。部件只删除该annotation；不合并鞋、不拆原始多区域mask；mask几何和版本变更必须可追溯。
3. 旧10k身份和已通过证据优先复用。若完整包修订旧记录，列出逐项变更和新版本，不能再声称旧记录完全保持，也不能覆盖正在执行的旧包。
4. 跨分区ID、source identity、group、SHA重叠为零；新近重复候选保留人工处置记录。对已完成且输入不变的源像素/pHash检查不重复执行，只处理新增范围或失效证据。
5. 保护现有dev250/test400。扩容发现新的train与冻结评价组冲突时，先隔离相关新增训练记录。现有`docs/data_protocol.md`第4节的初始划分规则优先隔离validation；本次扩容的“保护既有评价集”规则须在**新冻结协议**中明确记录，三方复核后采用，不静默更改旧协议或旧划分。
6. 对漏标、错类、源mask碎片、无效几何、空图/crowd作显式处置。当前训练读取器跳过crowd；若全量新增出现crowd或仅crowd图片，须在冻结时明确忽略/排除策略，不能默认当成负背景。有效小目标不能为美化指标而删除。
7. wkq报告的source image38309 dress被分成top/skirt，以及29246/31540训练内近重复，列入裁决/分组来源清单；前者隔离语言查询并未修复分割GT，后者本身不是跨分区泄漏证据。
8. 统计八类、源细类、尺寸、每图实例数、crowd、隔离原因和剩余疑点；人工复核来源保留，不把Codex视觉初查改成人工认证。旧5k用户确认不能自动覆盖全量新增图。
9. 发布schema3 `image_manifest.json`、`annotations_coco.json`、校验目录、源/转换/映射版本、隔离/质量/复核记录与RELEASE。真实N以接收方实际可训练记录数核对，不能使用只有图片文件计数的N。
10. 原dev250继续作为首选参照。若裁决需要修订GT，单独发布dev v2，在新实验前固定；旧运行不改，新旧成绩不能混用。独立确认集只从未分配官方validation组建立，规模待实际可用池确定；它不是首轮训练启动的语言审核依赖。

本轮只准备上述清单，没有下载、转换、扫描或代签全量数据。test只绑定既有manifest哈希，没有打开其图片、mask、标签或结果。

## 100-epoch首轮训练配方

模型、backbone和损失结构不变：Mask2Former-R50、现有八类适配与官方COCO初始化；复用`envs/segmentation/upstream.json`记录的权重SHA256。重新初始化本轮训练，不从旧20k/当前5k10k权重隐式续训。

| 配置 | 首轮值 |
| --- | --- |
| seed | 42；最终配方再登记43/44 |
| 图像增强 | 768画布、min_scale0.5/max_scale2.0、水平翻转；已有mask背景padding=0修复保留 |
| batch | microbatch2 × accumulation8 = effective16，当前实现为单GPU |
| AdamW | LR1e-4、backbone倍率0.1、weight_decay0.05、norm/embed decay0 |
| warmup | 500次有效更新，起始倍率0.001 |
| 学习率 | 80、90 epoch累计节点后各乘0.1；复用现有UpdateSchedule |
| 梯度/精度 | AMP、初始scale1.0、全模型clip0.01，复用原溢出重放策略 |
| 训练长度 | 100 epochs；不继承20k步或3小时上限 |
| 保存 | 每epoch累计节点和最终节点；记录实际checkpoint hash |
| 完整dev | 每2 epochs及最终节点，共50个计划评价点 |

`effective_batch_images=16`描述每次成功更新消费的图片数。当前原生损失在各microbatch计算并平均，不声称与一次性16图前向在所有损失归一化/随机操作上严格相同。此变化及更长训练是新配方差异，不能把相对旧5k/10k的全部收益归因于数据量。

令N为实际冻结训练图数，B=16，累计节点`u(e)=ceil(e*N/B)`。StatefulSampler连续跨epoch、无每轮丢尾；LR下降在完成`u(80)`/`u(90)`后作用于下一次更新。最终最多多消费15张来自下一轮的图片，必须报告该取整误差，不使用`ceil(N/B)*100`。例如**纯算术示例**N=40,001时，最终250,007次更新，LR节点200,005/225,006，最终比100N多12张曝光；不是全量包的实际数量。

只统计成功optimizer更新。AMP失败尝试按原实现恢复sampler/RNG后重放，不增加有效epoch或推进scheduler。将来若采用重复采样，需要新配方和曝光量定义，不能继续称每图均匀遍历100遍。

## 评价、选择与诊断

- 固定完整dev250 v1：250图/1008 GT，manifest和annotation哈希见评价配置；FP32，短边800/最长1333，原生instance解码，maxDets100。
- COCO mask AP@[.50:.05:.95]为主指标，另报AP50/AP75、小中大、逐类AP及AR100；标度0–100，无正例类为N/A。选最高完整dev AP，同分取更早update，不用最后一步代替最佳权重。
- 每次报告绑定权重SHA、训练输入SHA、dev SHA、评价配置SHA和评价器commit。只有这些身份一致且完整成功的缓存可以复用；失败/部分结果不参与选模，只补缺失评价，不重做已成功的相同预测。
- 标准AP不使用语言候选0.3/0.5/0.7过滤，不让语言结果决定分割checkpoint。候选缓存与阈值选择在权重冻结后独立进行。
- 计划固定train-only 256图分层诊断清单，seed42，每10 epochs使用相同推理口径观察拟合；不选checkpoint。当前清单尚未生成，样本必须在看预测前确定。
- Boundary AP拟采用公开`boundary-iou-api`固定版本和dilation_ratio0.02；接入尚未实现，不冒称现有thin-boundary指标就是Boundary AP。
- loss/梯度/学习率/AMP重放/显存/吞吐使用已有日志；由有效更新和sampler状态确认进度。后期dev稳定后再考虑更长调度，不能靠无限追加更新代替诊断。
- 独立确认集供最终少数配置确认；最终配方报告三种子均值/std及图片级paired bootstrap，最终test400在全组方法锁定后评价。此次不产生任何AP、确认集或test结果。

## 恢复方案与执行前必须补齐的接线

同一100-epoch run的恢复必须保持完整配置、训练输入、映射、源码commit和绝对update计划一致。使用已有checkpoint hash与组件fingerprint检查恢复model、optimizer、scheduler、AMP scaler、sampler、RNG、AMP recovery state；不重置warmup/学习率，不把旧20k延长门禁误用于本实验。

从最新完整且校验通过的恢复点继续优化，不能自动从dev最佳权重恢复并遗忘后续更新。每次进程使用新的attempt目录；中断后未持久化的部分梯度不能恢复，只能从上一个完整optimizer边界重放并留痕。不存在文件不等于零更新完成；失败状态不得登记为完成。

现有代码经只读检查存在以下长运行接线缺口，**本轮没有修改旧执行路径**：

1. `training_state.read_config`和`train.check_budget`要求正整数`budget_seconds`，旧launcher也强制3小时。新长训练需要独立的明确“无旧总墙钟上限”执行策略，不能往旧入口填一个巨大的假预算。当前预览保留null，故不能直接执行。
2. 原trainer不做inline dev。未来采用每2 epoch安全边界“训练保存 → 独立评价 → 原状态恢复”或等价隔离方案，要求恢复后RNG/sampler与未评价训练一致，不能让评价消耗训练随机状态。计划不等于该编排器已实现。
3. 原`load_training_package`每次恢复会读取所有图片并解码mask。完整包消费验收后应接入绑定哈希的metadata-only复用路径，按既有验收证明范围复用，不每2 epoch重做全量审计；新完整包也需独立评价输入绑定，不能伪装成旧10k接受记录。
4. 原trainer把每步`loss_rows`保留到进程结束。长任务需有界分段或流式聚合，保留原始JSONL，不因100-epoch增长内存。分段本身不得重置优化状态。
5. 启动前按实际权重/optimizer大小估算100个checkpoint、数据增量和预测缓存所需磁盘；未获清理授权不删除旧产物。每次评估后恢复也要核查共享GPU锁，不能占用队友运行。
6. 现有累积实现可复用；只有新增接线使原验证失效时才补受影响的CPU/最小运行检查。没有理由重跑COCO下载、五图推理、200-step稳定性、旧恢复或重复smoke。

用户当前未授权启动新训练；未来启动时须刷新团队GitHub、服务器磁盘/GPU/锁及运行登记，并满足全量数据冻结与消费证据。无旧预算限制不等于允许自动延长epoch、自动重试或动当前5k/10k实验。

## CPU准备命令（不训练）

在本分支repo根目录执行，输出必须是新目录：

```bash
PYTHONPATH=src python3 scripts/prepare_full_fashionpedia.py \
  --output /tmp/fashionpedia-full100-preparation
```

默认只检查准备文件和保护文件哈希，真实N未绑定，update保持null。未来拿到完整包元数据，可追加`--manifest <image_manifest.json> --annotations <annotations_coco.json>`生成按实际N计算的预览；仅JSON读取不构成像素验收、近重复审核或正式冻结，输出仍不允许训练。

`--illustrative-image-count 40001`只用于算术示例；不能与真实元数据混用，也不能产生数据哈希。输入变更/旧配置漂移/已有输出目录/`--execute`均拒绝，不覆盖原产物。

新增检查仅运行：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_full_data_preparation.py' -v
```

9项通过，覆盖累计取整、未知N不伪造、示例身份隔离、合法评价节点、元数据绑定不等于冻结、错误分区/身份拒绝、旧文件漂移、禁止覆盖与禁止执行。没有模型导入、服务器任务、图片读取、mask解码、推理或优化更新。

## 后续顺序与升级模型条件

先由数据方补全量发布/裁决/冻结，再补上述独立长训练接线，最后才能按未来启动指令运行R50完整基线。现有5k/10k对照按其原授权完成；本方案不能追溯性取消它的截止时间或失败停止规则。

基线后依次比较学习率、768/1024/必要时1280训练分辨率、小目标/长尾采样及边界问题。完整数据、可靠标注、合理优化和足够训练尚未到位时，不把AP平台认定为模型容量上限。训练很好/dev差优先解决泛化；小目标因缩放改善则继续现有模型。只有残余瓶颈经过控制实验仍持续，才登记更强Mask2Former backbone比较，进一步架构替换属于可选后续，不是本次依赖或承诺。
