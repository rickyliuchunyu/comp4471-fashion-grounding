# 当前进度与下一步

更新：2026-11-14；[3000→5000离线误差分析][dev-analysis]完成，复用已有250图dev预测，推理/模型加载/训练更新均为0。skirt AP48.45→30.61，但IoU.5匹配仍28/29，主要表现为相对排序退步及部分高IoU几何下降；outerwear AP53.47→47.43，兼有错类和少量漏检；AP-small实际7.58→8.48。继续选择第3000步（mask AP=50.5603）。

**[候选交付准备][candidate-prep]已发布到共享目录，正式候选payload尚未导出。** 权重/解码/manifest/wyz接口已固定；现有AP结果缺decoder query ID/logits，不能伪造为“一query一候选”包。正式导出编排、必要前向、accepted语言dev阈值选择和A/B新包验收仍待完成；本轮没有追加训练或重复smoke。

## GitHub 同步快照

以下为本轮开工检查的远端 head（2026-11-14，记录见最新交接）；PR #2/#4 为 open / draft / unmerged，PR #5 为 open / non-draft / unmerged（base `data-eval/p3-baseline-evaluation`），main 为 `7064ebb`。PR #1/#3 已关闭且未合并，旧远端分支已删除，其提交包含在统一 PR #4（base main）。lcy/wyz 已包含旧公共提交 `dc24f2d`，后续按 #4 新增接口审查集成，不再等待合并 #1。

| 成员 / GitHub | 已推送分支与 commit | PR / 最新交接 |
| --- | --- | --- |
| lcy / rickyliuchunyu | `segmentation/pilot-adapter-v1`：开工已发布 `a69af74`；离线分析源码 `a91c581`、案例/准备 `91e0bfa`，结果随PR同步 | [PR #2](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/2)；[离线分析][dev-analysis] / [候选准备][candidate-prep] |
| wyz / MayI2004 | `grounding/siglip-baseline-interfaces` @ `103dcc5` | [PR #5](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/5)；[实际数据接收、B1/B2 与 M1 交接][wyz-latest] |
| wkq / LE-brawl | `data-eval/p3-baseline-evaluation` @ `f5787f4`；扩充实现及发布交接已推送 | [统一 PR #4](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/4)；旧 #1/#3 已被替代；[大型数据交接][wkq-large] |

队友交接链接固定到本次读取的 commit，避免将未合并内容误认为当前 checkout 已包含。下次开工须重新检查远端增量，流程见 [AGENTS.md](../AGENTS.md)。

17:05 HKT 复核：所有队友分支 head 与上次一致，四份 PR 均无 review/comment。服务器发布的 `fp_pilot20_v2`（20 图）、`fp_grounded9_draft_v2`（9 图）、`fp_grounded9_reviewed_v1`（9 图）均仅有 train 图片；未发现已发布 dev 包，wkq 最新交接仍明确 P4 未完成。34 条 accepted 是训练语言查询，不表示分割 train/dev 已冻结。本轮未启动 AP、checkpoint 选择或训练，见 [就绪核查](handoffs/segmentation/freeze_readiness_20261112.md)。随后按 wkq 请求完成训练小包分割接口验收，见 [reviewed_v1 接收记录][reviewed-acceptance]；两项任务的前提不同。

## 1. 阶段状态

原范围审查核对服务器时只有三个 train 包，四个跨分支解析器/B2 CPU 案例通过；这些是历史证据。责任修正时读取 wkq `f5787f4` 大包交接，此前已完成 train/dev CPU 输入接收；本轮 wyz 消费证据也已齐，用户已确认当前交付数据经过人工复核，消费者固定版本与限定 dev 实验已完成。具体范围、责任修正和原代码审查的适用提交见[全组主线审查](handoffs/team_scope_audit_20261112.md)。上文 17:05 是历史快照，不再作为最新发布状态。

状态：`not_started` 无执行证据；`in_progress` 有明确活动与产物；`blocked` 已开始但受阻；`ready_for_review` 待接收方验收；`done` 满足该阶段放行条件。文档已写不等于该阶段 done。

| 阶段 | 当前状态 | 已有证据 | 尚缺的放行证据 | 主责 |
| --- | --- | --- | --- | --- |
| P0 接口确认 | in_progress | [文件契约](artifact_contract.md)、[合成示例](../examples/README.md)、[解析约定](rule_parser_contract.md) | 三人消费方复核与实际读写器检查 | 全员，B 汇总 |
| P1 环境与工作区 | in_progress | [lcy 工程检查](handoffs/segmentation/pilot_adapter_20261112.md)、[COCO 八类加载/五图推理](handoffs/segmentation/coco_weights_20261112.md)、[200-update 与跨进程恢复](handoffs/segmentation/stability_200_20261112.md)；[wyz 图文环境与 GPU 编码][wyz-handoff]；wkq 独立 CPU 环境；[lcy AP/候选读取实测][lcy-ap] | 全组交付的接收方复核与环境复现验收；wyz 环境只读复用 base torch，base 变化需重验；非长训练验收 | lcy 协调，全员 |
| P2 真实 GT 小包 | in_progress | [原包 20 图/145 GT](handoffs/data_eval/remote_pilot_review_20261112.md)，lcy 加载与 wyz 145 个裁剪消费通过；[新审核包 9 图/59 候选/18 语义查询/36 双语记录][wkq-handoff]，35 测试、59 mask 与源一致 | [新版 reviewed_v1][wkq-p3] 已有 17 组/34 条 accepted；1 组/2 条 pending；lcy 新包加载/前反向/保存恢复已通过，wyz 实际消费、B1/B2 与统一评价已通过 [103dcc5][wyz-latest]；5 组/10 条 accepted 为已暴露答案的训练再审，非独立测试 | B，全员标注 |
| P3 基线与评价器 | in_progress | [wyz 已接收共享解析器；34 条同包 train 上 B0/B1/B2=12/18/22 选对，输出完整，缓存重放一致][wyz-latest]；wkq 统一评价核心已交付 | 下游 PR 审阅、正式 accepted dev 评价与 B2 lambda 选择仍待完成；训练诊断不是泛化成绩 | A/B |
| P4 数据扩充冻结 | in_progress | [wkq 大包交接][wkq-large]：5000/250/400 分割图已发布；[lcy train/dev CPU 读取验收通过][large-inputs]；1000 图训练初标和 116 图 dev 草稿待审 | 三方技术消费已通过；[每类 20 例抽样及 53 对保守归组核对完成][freeze-prep]；用户已确认交付数据经过人工复核，已形成消费者固定版本记录；原发布 formal_freeze=false 保留作历史，语言审核/最终冻结独立跟踪 | B，全员 |
| P5-S 分割训练 | in_progress | [续训及dev选择][resume5000]完成，选择3000步；[离线误差分析][dev-analysis]与[候选准备][candidate-prep]完成 | 生产候选编排/实际导出、语言dev阈值选择及A/B接收未完成；完整S0未宣告收敛 | lcy |
| P5-L 排序训练 | in_progress | [M1 seed42 机器弱标签试训：5 epochs / 490 updates，已保存权重][wyz-latest]，不属于正式训练结果 | 人工审核 dev、M2 实现与 M1/M2 三种子公平对照、真实 dev 选择 | A |
| P6 联调与对照 | not_started | E1 与集成规范 | dev 全流程、E1、demo 和扩展取舍 | 全员 |
| P7 方法锁定 | not_started | freeze 清单 | 权重/词表/阈值/数据及最终运行清单 | B 汇总，全员核对 |
| P8 最终测试 | not_started | 评价协议 | 完整预测、指标与失败记录 | 全员 |
| P9 报告与演示 | not_started | 写作分工；此行指最终交付 | 最终实验对应报告、演示与渲染检查 | 全员 |
| P10 归档与提交 | not_started | 归档要求 | 复现索引、课程回执、授权算力收尾 | 全员 |

本表不判断已有 milestone 文件或课程提交状态，最终报告与 milestone 是不同交付物。

## 2. 每人的下一项交付

| 负责人 | 下一项任务 | 可立即做 | 依赖/阻塞 | 交付位置与证据 |
| --- | --- | --- | --- | --- |
| lcy | 准备正式候选导出编排；保持3000步权重 | 已有AP逐实例诊断、权重/解码配置/消费接口清单；复用既有检查 | 现有AP缺真实query ID，正式payload尚无；本轮不授权新增推理，正式阈值仍需accepted语言dev | [分析][dev-analysis] / [共享准备包][candidate-prep] |
| wyz（A）/ grounding | 推进 M2 与正式可比训练入口、交付 PR #5 审阅 | 复用已完成 B0/B1/B2 与 M1 弱试训；改善困难样本/模板重复，准备三种子比较 | 正式语言 dev 未人工验收；预测候选待 lcy 固定版本；M2 尚未实现 | [最新交接][wyz-latest]、PR #5 |
| wkq（B）/ data_eval | 组织新包接收与冻结，审核/补写 dev 查询 | 分割大包与语言初标已发布；接收消费方检查及 B1/B2 输出 | 新包三方技术验收已齐，用户已确认分割数据人工复核；生产方可归档同步冻结记录；语言机器草稿不自动 accepted | [大包交接][wkq-large]、PR #4 |

无 GPU 不阻塞 CPU 接口开发。使用者可用 lcy/wyz/wkq 自报身份；对应角色已绑定，不根据作者顺序推断全名或账号。三人的工作区与代码分支继续遵守 [团队说明](team_workflow.md)。

## 3. 已确认与待确认

| 事项 | 当前值/状态 | 证据或补齐责任 |
| --- | --- | --- |
| 研究与暂定分工 | documented | README、三份角色卡和 workflow |
| 文件交接例及检查器 | locally_checked，仅合成夹具 | [规范补齐交接](handoffs/contract_readiness.md) |
| 模型代码/数据转换器 | trainer/AP/选择及有记录的预算扩展续训已实现 | 1000→5000真实恢复、训练及dev选择完成；完整S0、生产候选及最终test仍待完成，队友状态按各自交接 |
| 实验结果 | 已有dev AP46.0636/50.5603/48.8844，选3000步；新增离线诊断解释回落 | 本轮新增2条派生运行（r01序列化失败、r02完成），不增加模型性能测量；无最终test成绩 |
| GPU/环境/付费预算 | 本轮离线分析r02 CPU 6.212秒，600秒上限；未用GPU或升级环境 | 结束GPU0MiB/0%，无计算进程/running记录，系统/数据盘约27/162GiB空闲；[回执](handoffs/segmentation/evidence/dev_analysis_20261114/delivery_receipt.json) |
| 成员简称与角色 | lcy / wyz（A）/ wkq（B）已确认 | [绑定记录](handoffs/team_aliases.md) |
| GitHub 账号与服务器访问 | lcy：rickyliuchunyu；wkq：LE-brawl；wyz：MayI2004；lcy SSH 已验证 | lcy/wkq 对应 PR 作者；wkq 授权记录、wyz 本人登录/接受邀请见各自交接。两位队友远端运行已有记录，本次未独立验证其各自 SSH 凭据/权限 |
| 公共权重 revision 与数据 manifest | 分割源码、pilot manifest、COCO 文件 hash，以及 SigLIP 2 revision/权重 hash 已固定；两模型实际加载有证据 | [分割来源记录](../envs/segmentation/upstream.json)；[SigLIP 2 来源与校验][wyz-handoff]（镜像传输、官方 API 核对）；train/dev 指纹及用户人工复核确认已登记消费者 freeze；原生产方历史 RELEASE 不改写 |
| 课程实际截止与提交回执 | 未重新核验 | 提交负责人查课程系统；历史日期不是当前确认 |

## 4. 范围检查后的优先顺序

保持 Fashionpedia 八类 + 完整双语单目标 + S0/B0/B1/B2/M1/M2/E1；不默认混训、部件定位或其他扩展。旧分割dev结果来自受增广缺陷影响的权重；修复后1000→5000续训与真实dev选择现已完成，依据选定权重准备候选交付及后续S0预算；wyz 的 M2/正式训练准备并行。M1/M2 是核心研究，不能用更多 smoke 替代。

**责任修正：reviewed_v1 接收时重复安排两步训练/保存恢复，是 Codex 未充分复用已有证据的决策问题，不是用户要求不当或 lcy 让项目跑偏。** 后续执行 AGENTS.md 的验证复用规则：具体变化使原证据失效时才补最小检查；没有新增风险就推进下一项交付。新包不自动触发训练 smoke；该次责任修正仅修改文档，未运行模型或训练验证。

后续大包接收按该规则只执行 CPU 输入检查（52.41 秒），模型加载/训练/恢复均为 0。此结果不等于正式冻结或 dev AP，也不依赖 wyz 的语言 dev 审核完成。

## 5. 更新规则

每次开工先按 AGENTS.md 检查整组远端分支、PR 和最新交接，再确定自己的下一步。每次工作交接在同一 PR 更新相关阶段/个人任务和证据链接，注明实际未验证项；跨分支汇总须标明来源 commit、未合并状态及是否仅引用对方验证。状态与证据不一致时以可复核产物为准，先修正文档；不覆盖他人的完成记录。阶段 done 需对应 workflow 放行条件，训练开始不等于完成，代码完成不等于 GPU 验证。

历史交接文件保留，状态表只展示当前摘要。公开索引不放密码、私钥、token 或私有下载凭据；服务器地址与访问方式由受控渠道交接。

[wyz-handoff]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/66c0da7e066c1235ee618ea7015171e73b82c914/docs/handoffs/grounding/siglip_baselines_20261112.md
[wkq-handoff]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/75bf2ef50f80bc939c35ba15afe6dc6efb15f3b1/docs/handoffs/data_eval/grounded_query_review_20261112.md

[lcy-ap]: handoffs/segmentation/dev_ap_20261112.md
[wkq-p3]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/66dd07c56fb02ae90af7a12de3524650d2fe6455/docs/handoffs/data_eval/p3_reviewed_baseline_20261112.md

[reviewed-acceptance]: handoffs/segmentation/reviewed_pilot_acceptance_20261112.md

[wkq-large]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/f5787f4ac7b6941abdadc43f9fbdc666eba93925/docs/handoffs/data_eval/large_dataset_20261112.md

[large-inputs]: handoffs/segmentation/large_input_acceptance_20261112.md

[freeze-prep]: handoffs/segmentation/freeze_preparation_20261112.md
[wyz-latest]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/103dcc546714dd7c7866b4259d96a9825d8090ef/docs/handoffs/grounding/accepted_baselines_and_weak_trial_20261112.md

[real-dev]: handoffs/segmentation/real_dev_ap_20261112.md

[low-ap]: handoffs/segmentation/low_ap_diagnosis_20261112.md

[postfix1000]: handoffs/segmentation/postfix1000_dev_ap_20261112.md

[resume5000]: handoffs/segmentation/resume5000_dev_ap_20261113.md

[dev-analysis]: handoffs/segmentation/dev_error_analysis_20261114.md
[candidate-prep]: handoffs/segmentation/candidate_u3000_preparation_20261114.md
