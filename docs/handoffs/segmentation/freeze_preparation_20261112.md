# 分割 train/dev 冻结补齐与限定 dev 实验准备

后续更新：用户已明确确认“现在发的就是经过人工复核的，你先用”。因此以下“人工复核未取得”是准备阶段历史状态，不再阻塞当前已授权的限定实验；执行依据见 [固定版本与用户确认记录](segmentation_train_dev_freeze_20261112.json)。原始观察与 pending 模板保留，不伪造逐对审核结论。

日期：2026-11-12 21:28 HKT。执行者：Codex（lcy 工作区）。状态：`technical_evidence_complete_human_review_pending`；**尚未正式冻结，未执行真实 dev AP 或 checkpoint 选择**。

## 本轮结论

三方技术接收已有证据，后续复用，不重新跑 smoke。已补充已知近重复的分区处置核对、每源每类 20 例可视化抽样、可填写的人工复核材料，以及固定输入的限定实验配置。缺少实际人工近重复复核和团队冻结结论，不能把 Codex 观察改记为人工验收，也没有改写 wkq 的不可变发布文件。

开工 GitHub 快照：main `7064ebb`；lcy `0c10d1f` / PR #2 open/draft；wkq `f5787f4` / PR #4 open/draft；wyz `103dcc5` / 新 PR #5 open/non-draft，base 为 wkq 分支。均未合并，所查评论为空。服务器 own checkout `dd64016`，grounding `103dcc5`，data_eval `f5787f4`，均 clean。本次没有改动队友工作区。

服务器执行前系统/数据盘约 27/166 GiB 可用；GPU 0/24576 MiB、0%，无计算进程、无 running 协调记录。本轮仅 CPU 元数据与图片材料处理，没有启动 GPU 任务。

## 冻结条件与证据

| 条件 | 状态 | 证据 / 边界 |
| --- | --- | --- |
| 生产方完整转换、来源、映射、统计与源 mask 对照 | 复用 wkq 发布结果 | [f5787f4 大包交接][wkq]；不重跑其 93 项测试 |
| lcy train/dev 实际读取 | 已通过，直接复用 | [5000 train / 250 dev 接收](large_input_acceptance_20261112.md)，5250 图、18783 实例；本轮只核对元数据指纹仍相同 |
| wyz 实际消费 | 已通过，引用队友证据 | [103dcc5 交接][wyz]：六包消费通过；含 train/dev 与 heldout 完整性检查，不是 test 模型评价 |
| 已知近重复分区处置 | 本轮补齐 | [freeze_preflight.json](evidence/freeze_preparation_20261112/freeze_preflight.json)：53 对全部同组，无已知跨已选 split 配对；10 张跨官方源的可疑 val 图全部隔离 |
| 每源每类至少 20 例可视化 | 抽样完成，有局限 | 复用 64 例，新增 96 例，合计 160；新增 95 train / 1 dev，按 Fashionpedia 来源抽样，**不是每 split 每类 20 例**；[抽样清单](evidence/freeze_preparation_20261112/visual_sample_manifest.json)、[Codex 观察](evidence/freeze_preparation_20261112/visual_review.json) |
| 近重复人工复核记录 | **未取得** | [data_protocol](../../data_protocol.md) 第 110、176 行明确要求人工复核；保守归组不等于实际人工确认 |
| 三成员冻结结论 | **待闭合** | 技术消费者通过不等于三个人签署；第 112 行要求“三位成员复核后冻结”。可新增独立 freeze 记录引用原发布 hash，不篡改原 RELEASE |

视觉样本总体与可见服饰吻合，未见系统性的 mask 平移。小配饰及遮挡不能凭拼图保证精细边界；新增裙装 `127856`、重遮挡连衣裙 `271653`、小配饰 `248669/22054` 记录了不确定性，原 `16468/42660` 的局限保留。未擅自重标、删样或宣称全量质量正确。团队冻结结论需说明如何处理这些局限。

语言 dev 的人工描述审核独立跟踪，**不是本轮分割冻结的阻塞项**。已知近重复处置无跨区泄漏不代表发现全部同人物/裁剪重复。

## 人工复核材料与本轮产物

服务器根：`/root/autodl-tmp/fashion-grounding`。

本轮目录：`runs/segmentation/segmentation-freeze-preparation-20261112-r01/`。

- `freeze_preflight.json`：原接收报告 SHA、当前输入和发布元数据 SHA、53 对归组处置、10 张隔离图片。此文件是新增视觉抽样之前的原始快照，内部“20 例待补”项已由后续 visual_review.json 补充，保留原始证据不改写。
- `visual/`：只新增的 8 张类别拼图、精确 sample_manifest。已有 64 例不重新渲染或重复验收。
- `duplicate_review/index.html` 与 `duplicate_review.zip`：53 对图片及现有分区并排显示，人工选择 same/near、different、uncertain，并填写理由及真实姓名，下载 `duplicate_human_review.json`。初始全部 pending，不自动完成冻结；[可填写记录模板](evidence/freeze_preparation_20261112/duplicate_review_template.json) 同步入 Git。涉及 heldout 的图片仅用于数据质量复核，不进行模型评分。
- [图片材料脚本](evidence/freeze_preparation_20261112/prepare_visual_review.py)、[近重复材料脚本](evidence/freeze_preparation_20261112/prepare_duplicate_review.py)：已在 segmentation-v1 Python、`CUDA_VISIBLE_DEVICES=""` 执行；只写上述新目录，不更改 shared 包。材料文件 hash 存在各清单中。

人工复核可确认保留现有保守分组/隔离，无须为无害的 false positive 拆组；任何确需修订的 split/标签应由生产方发布新版，保留旧版和变更原因，然后仅复核受影响输入。不可直接在本记录中把 pending 改成他人的签字。

## 条件满足后的限定实验（仅准备，尚未运行）

配置：[100-update 训练](../../../configs/runs/segmentation/s0_dev_selection_100.json)、[250-image dev AP](../../../configs/runs/segmentation/dev_ap_250_bounded.json)。它们不是冻结证明；现有 trainer/evaluator 不会自动验证人工签字，执行者必须先核对 freeze 记录及其输入 hash。

1. 复用 COCO/八类加载、200-update、恢复、AP 及选择器的已有工程证据。旧 pilot checkpoint 的训练数据不匹配，不拿来冒充新数据选模。
2. 新 train5000 上仅运行 seed 42、100 updates，batch=2，第 50/100 步保存 checkpoint。保留 S0 的 500-update warmup、LR、增强及既有 AMP scale=1；不为提高短跑 AP 改调度。锁内上限 300 秒，不另做恢复重放。
3. 对两个 checkpoint 各评完整 250 张 dev：官方实例 top100、FP32、短边 800/最长边 1333、原图 mask，合计最多 500 次图像推理，锁内上限 600 秒。不做 candidate 阈值网格或语言评价。
4. 复用离线选择器，按 **dev mask AP 最大、并列较早 update** 选择；保存原始 COCO 报告、预测 hash 和 checkpoint hash。仅是限定开发实验，不代表完整 S0 收敛效果或最终方法冻结。
5. 有非有限值/OOM/超时则保留失败日志，停止，不自动延长、换 seed 或开始长训练。若两个完整评价未齐，不宣称完成两 checkpoint 比较。

实际执行时继续使用 `scripts/with_gpu_lock.py`；先核对磁盘/GPU/锁，提交固定配置后以干净自己的 checkout 运行现有 `segmentation.train`、`segmentation.evaluate`、`segmentation.select_checkpoint`。目前未创建任何该实验运行、checkpoint 或正式指标，registry 不添加虚构运行。GPU 预算最多 300+600 秒，不包含完整长训练。

## 后续只补剩余项

取得上述固定版本的实际人工近重复复核与团队冻结结论后，直接进入限定 dev 实验。若包指纹不变，不重跑大包全读取、权重下载、模型 smoke、200-update 或恢复检查。本轮新增可视化与人工材料也直接复用。

[wkq]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/f5787f4ac7b6941abdadc43f9fbdc666eba93925/docs/handoffs/data_eval/large_dataset_20261112.md
[wyz]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/103dcc546714dd7c7866b4259d96a9825d8090ef/docs/handoffs/grounding/accepted_baselines_and_weak_trial_20261112.md

静态核对：新配置的 manifest/annotation SHA 与已有接收报告一致，100 updates / checkpoint 50 / 总 GPU 上限 900 秒；160 个 annotation ID 唯一、每类 20 例；53 对模板全部 pending；本地 review ZIP CRC 通过。未运行模型或测试套件。
