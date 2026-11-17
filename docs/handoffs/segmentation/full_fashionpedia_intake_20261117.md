# 完整 Fashionpedia：发布缺口与分割接收准备

2026-11-17，lcy / segmentation。**准备已整理；完整训练包尚未发布，未冻结、未接收、未启动训练。** 当前只继续现有 Mask2Former-R50 的完整数据准备；保留用户停止 5k/10k 的决定，不自动恢复 D10k，不启动 100 epoch。

## 当前证据及分支衔接

已核对整组远端分支、全部五个 PR 和三个活跃 PR 的评论/review；无新 PR、合并或讨论。wkq 为 `6eb7b923d32142e2510ffc87e6adf0e549a8f743`，wyz 为 `103dcc546714dd7c7866b4259d96a9825d8090ef`，main 为 `7064ebb`。wkq 的[最新交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/6eb7b923d32142e2510ffc87e6adf0e549a8f743/docs/handoffs/data_eval/reviewed_dev_eval_20261117.md)发布的是 642 条 accepted 语言 dev（321 双语组、246 张查询图，manifest 仍 250 图，9 组待处理），并复用旧候选完成 B1/B2/M1 开发评价，**不是完整分割 train 发布**。不以语言审核作为分割额外依赖，不替队友重跑评价或训练排序。

lcy 结果分支 `segmentation/pilot-adapter-v1@1ec9723` 已保存最终 C5k 结果和 D10k 中断，PR #2 正文也已更新。独立准备分支 `segmentation/full100-runtime@d80f33b` 已有执行入口，实际代码源 `ee7db25`；本轮在该准备分支整理接收需求，并在结果分支放固定链接。没有合并两条执行路径，也没有覆盖服务器的旧运行源码。

15:25 HKT 只读检查：GPU 0 MiB/0%，无计算进程、活动协调记录，共享锁空闲；数据盘空闲 161,686,339,584 bytes，约 150.6 GiB。共享 processed 目录有 train5000 和 targeted10000，没有已发布的完整分割训练包。包是否存在的结论限于已检查的发布目录和上述固定 GitHub 提交；不扫描私人目录或原始数据代替生产方发布。见 [GitHub 快照](evidence/full_fashionpedia_intake_20261117/github_preflight.json)、[服务器清单](evidence/full_fashionpedia_intake_20261117/server_readonly.json)、[当前缺口](evidence/full_fashionpedia_intake_20261117/readiness.json)。

## 已完成成果，直接复用

[主分支最终报告](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/1ec972363271b5ba70be565de71d1b638dc7477c/docs/handoffs/segmentation/expansion_control_20261117.md)是唯一对应这次运行的最终结果；本页只引用，没有重算 AP、读取新预测或重新推理。

| 同一 dev250 / 1008 GT，AP 为 0–100 | 历史 B20k | C5k 20k | 差值 |
| --- | ---: | ---: | ---: |
| mask AP | 63.7930 | 64.6957 | +0.9027 |
| AP75 | 68.4059 | 68.4916 | +0.0857 |
| AP small | 10.7525 | 11.0483 | +0.2958 |
| AP medium | 40.0568 | 31.3090 | -8.7478 |
| AP large | 71.7558 | 73.7301 | +1.9742 |

选 C5k 20k；checkpoint SHA256 为 `d9deacc85d1bf402cf6417a71226bd51e1fc98b631dcd30c9afcc22d51b51044`。已有诊断中，小目标 IoU≥0.5 同类匹配 102→99，IoU≥0.75 为 35→41；袜子 5/13、腰带 5/11 的 0.5 匹配仍弱。整体 AP 和部分边界有所改善，漏检与中目标没有全面改善；源标注争议也不能由模型结果裁决。D10k 停在 159 步且无权重/AP，无法得出扩到 10k 的收益结论。新完整数据及 batch/调度共同改变时，也不能把未来全部差值归因于数据量。旧 u18000/0.3 候选交付保持。

## 给 wkq 的具体发布需求

以现有[冻结清单](../../../configs/runs/segmentation/full_fashionpedia_freeze_checklist.json)为字段依据，不再另造第二套冻结规则。拟议名 `fp_train_full8_v1` 不是实际发布路径。当前数量、路径、hash 全部待生产方提供。

| 材料 | 必须提供的内容 | 当前缺口与主责 |
| --- | --- | --- |
| 真实完整包 | schema 3 的 image_manifest.json、annotations_coco.json、images、mapping、RELEASE.json、SHA256SUMS.json；真实路径、版本、转换 commit、源文件 hash、实际 N/实例数 | 未发布；wkq |
| 全来源数量账 | 全部官方 train → 版本化排除 → 实际可训练 N；source=retained+excluded；逐图保留/排除原因，不能用名义官方总数或只数图片文件 | 未发布；wkq |
| 相对旧 10k 的增量 | unchanged / added / modified / removed 图片与 annotation ID 清单；修改前后 hash/原因；原10k记录及资源不变才能复用其接收结论 | 未发布；wkq，lcy 消费 |
| 数据质量与隔离 | 新增或失效范围的源 mask/框/尺寸检查；ID/source/group/SHA 与固定 dev/test 交集；近重复候选和真实人工处置；保留来源而非填 passed 即验收 | 未发布；wkq/复核者 |
| 分布与争议 | 八类、源细类、小中大、每图实例数、crowd/排除统计；38309 类别/碎片疑点和 29246/31540 组关系的处置记录 | 尚无完整包裁决；wkq/复核者 |
| 协议与 dev 版本 | 本次扩容保护既有 dev250/test400 的明确版本化补充；有 GT 修订须另发 dev v2，不能覆盖旧 v1 | 尚未确认；wkq 汇总、成员复核 |
| 接收/冻结/binding | 真实完整包增量接收 receipt；三方对相同 manifest/annotation hash 的真实确认；最终 freeze 和 binding 引用 | 等上游材料；lcy 技术接收，成员各自确认 |

图片保持原图 RGB、真实尺寸/相对路径、稳定 image/source/group ID 与 SHA；项目类别 1–8，训练内部 0–7，保留源细类。对现有训练 worker 发布原尺寸**压缩 RLE，counts 为字符串**；框是由有效 mask 得到的 COCO xywh，转换 xyxy 时右下界排他。jumpsuit 仍整图排除，部件只移除对应 annotation，不合并左右鞋或拆分同一原始实例。crowd/仅 crowd 图明确计数和处置，困难但合法的小目标不为提高指标而删除。

可直接转发的简版需求：

> 我这边 C5k 20k 已完成，dev mask AP 64.6957；D10k 已按要求停止，暂时没有 5k/10k 数据量对照结论。接下来准备现有 R50 的完整 Fashionpedia 分割训练，请按现有八类 schema 发布全部合格官方 train 的版本化包，并给出实际图数/实例数、源与转换 hash、相对旧10k的新增/修改/删除清单、跨固定dev250/test400隔离及近重复人工处置、弱细类/尺寸统计和源标注疑点裁决。旧10k、dev/test不要覆盖，压缩RLE/原图尺寸/ID与映射保持。拿到真实包后我只补新增或变化输入的技术验收，复用已通过检查；语言数据不阻塞这一步。完整训练暂不启动，正式冻结仍保留各成员真实确认记录。

这段已写入 GitHub 交接，**未私信或替任何成员确认**。

## lcy 接收顺序与复用范围

1. 先接生产发布候选及源数量账，核定真实 N、旧10k保留范围、变化清单和资源增量；发布材料不等于 formal_freeze。无需先让生产方虚构消费方验收通过。
2. 现有 `scripts/accept_segmentation_expansion.py` 固定了 10000/5000/21924 数量及 targeted10k 的 sidecar/抽样格式，**不是完整数据通用接收器**。拿到实际增量格式后，复用其校验/真实 loader，只适配受影响的数量、成员清单和证据组合；完整 receipt 生成接线尚未完成。不能仅把旧 receipt 的 scope 改名。
3. 未变旧10k的源像素、旧资源字节和接收证据直接引用；新增/修改图才做读取、非空原图 mask、bbox/area、类别和 crowd 兼容检查。删除/替换旧记录必须逐项解释。近重复复核只覆盖新增/改变关系；无候选也须有绑定范围的零候选报告，不代签人工决定。完整包同一 train 内允许同 group 多图，约束是组不跨 dev/test，不能机械沿用定向10k脚本要求每图group唯一的条件。
4. 旧 dev250 v1 固定，复用原完整 dev 成功凭据；test 只复用隔离 manifest/hash，不打开 test 图片、mask、标签或模型结果。协议补充未确认前，新增且与既有评价组冲突的训练候选先保持 excluded/pending，不移动原 dev/test，也不把本建议当已生效的全组协议。
5. 冻结发布包后，消费 receipt 记录实际检查 commit、输入/catalog hash、复用/新检查范围、N/实例数、1–8→0–7 映射、dev/test 四项交集及复核来源。只有真实检查齐全后才能令完整 receipt 的 `image_and_mask_acceptance_complete` / `near_duplicate_review_complete` 为真；它不要求给每张图片另做新的语言审核。
6. 成员确认绑定同一 manifest/annotation，生成最终 freeze；再填[现有 binding 模板](../../../configs/runs/segmentation/full100_data_binding.template.json)。引用路径按所在 JSON 目录解析，移动文件后重新确认路径。准备命令只生成元数据 contract；身份匹配不代替首次输入验收。

发布顺序固定为：**不可变数据及生产证据 → 不含自身的 SHA256SUMS → 外置消费 receipt → 外置成员确认/最终 freeze → binding**。消费 receipt 引用 catalog，freeze 引用 receipt，binding 最后引用三者；catalog 不反向包含 receipt/freeze/binding，避免循环哈希。RELEASE 可以记录生产发布状态，不能提前写消费接收/三方冻结通过。

## 不增加本轮计算，以及后续预算

复用旧 COCO/八类适配/200-update/AMP/恢复、旧10k和dev验收、9项准备和21项runtime CPU案例，不重跑。此次仅 GitHub/元数据清单读取与文档更新；新模型加载、优化更新、推理、smoke、数据像素审计均为0。完整包未到，不生成假的 passed receipt、run 行或 full100 AP；`ready_for_training=false`。

已准备的100 epoch不代表短任务。C5k实际训练8400.824秒处理40000次图片曝光；按这一历史速度作**粗略算术外推**，100 epochs训练部分约为 `0.2100206 × 100 × N` 秒。若仅举例 N=40000，则约233小时，尚未加50点评价、保存和中断成本；不是实际全量N或新累积batch的实测速度。microbatch2×accumulation8的优化器开销和数据I/O会改变速度，拿到实际N后必须重新给出可审阅的时间/磁盘预算。不能把旧单组3小时误当100epoch总耗时，也不能因为入口存在就启动。

当前停止点：真实完整发布、增量接收及冻结材料未齐。下一项只接这些输入并更新数量/预算；不额外做分辨率扫描、恢复D10k、导出新候选或承担wyz的排序研究。
