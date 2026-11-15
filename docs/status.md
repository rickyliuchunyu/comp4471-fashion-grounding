# 当前进度与下一步

更新：2026-11-15；用户明确确认当前语言dev可直接用，已按固定116图/145双语组/290查询完成[18k候选离线coverage与阈值选择][u18000-selection]。0.3/0.5/0.7覆盖258/256/250条，本轮选0.3（88.9655%）。实际CPU阶段7.044秒/360秒硬上限，0推理/权重加载/训练/旧smoke；原始needs_revision及机器来源保留，结果标记user_authorized_development。

**当前开发组合：18000步权重＋0.3候选包＋本次用户授权固定语言dev。** [独立selection记录][u18000-selection]已发布，下游可直接用于开发联调；原trial包的历史selected_threshold=null不改写。此前[完整dev250缓存及三包交付][u18000-candidates]、[旧3000包][real-candidates]均保留。wyz本人接收/实际排序、train候选、完整250图语言覆盖和最终test未冒称完成。

## GitHub 同步快照

以下为本轮开工检查的远端head（2026-11-15，见[本轮选择交接][u18000-selection]）：lcy=`8e9a2b7`、wkq=`f5787f4`、wyz=`103dcc5`、main=`7064ebb`；队友分支/交接无增量，活跃PR无评论/review。PR #2/#4 open/draft/unmerged，#5 open/non-draft/unmerged（base `data-eval/p3-baseline-evaluation`）；#1/#3 closed-unmerged。没有合并队友工作。

| 成员 / GitHub | 已推送分支与 commit | PR / 最新交接 |
| --- | --- | --- |
| lcy / rickyliuchunyu | `segmentation/pilot-adapter-v1`：开工`8e9a2b7`；本轮CPU选择执行源`bb39822`，候选源`00b408d`，结果随本PR同步 | [PR #2](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/2)；[18k阈值选择][u18000-selection] / [候选交付][u18000-candidates] / [正式S0][s0-recovery] |
| wyz / MayI2004 | `grounding/siglip-baseline-interfaces` @ `103dcc5` | [PR #5](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/5)；[实际数据接收、B1/B2 与 M1 交接][wyz-latest] |
| wkq / LE-brawl | `data-eval/p3-baseline-evaluation` @ `f5787f4`；扩充实现及发布交接已推送 | [统一 PR #4](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/4)；旧 #1/#3 已被替代；[大型数据交接][wkq-large] |

队友交接链接固定到本次读取的 commit，避免将未合并内容误认为当前 checkout 已包含。下次开工须重新检查远端增量，流程见 [AGENTS.md](../AGENTS.md)。

历史快照（2026-11-12 17:05 HKT）：当时所有队友分支head与前次一致，四份PR均无review/comment。服务器当时只有 `fp_pilot20_v2`（20图）、`fp_grounded9_draft_v2`（9图）、`fp_grounded9_reviewed_v1`（9图）三个train包；34条accepted是训练语言查询。该次未启动AP、checkpoint选择或训练，见[就绪核查](handoffs/segmentation/freeze_readiness_20261112.md)；随后完成[reviewed_v1接收][reviewed-acceptance]。这不是本日S0运行或当前大包状态。

## 1. 阶段状态

原范围审查核对服务器时只有三个 train 包，四个跨分支解析器/B2 CPU 案例通过；这些是历史证据。责任修正时读取 wkq `f5787f4` 大包交接，此前已完成 train/dev CPU 输入接收；本轮 wyz 消费证据也已齐，用户已确认当前交付数据经过人工复核，消费者固定版本与限定 dev 实验已完成。具体范围、责任修正和原代码审查的适用提交见[全组主线审查](handoffs/team_scope_audit_20261112.md)。上文 17:05 是历史快照，不再作为最新发布状态。

状态：`not_started` 无执行证据；`in_progress` 有明确活动与产物；`blocked` 已开始但受阻；`ready_for_review` 待接收方验收；`done` 满足该阶段放行条件。文档已写不等于该阶段 done。

| 阶段 | 当前状态 | 已有证据 | 尚缺的放行证据 | 主责 |
| --- | --- | --- | --- | --- |
| P0 接口确认 | in_progress | [文件契约](artifact_contract.md)、[合成示例](../examples/README.md)、[解析约定](rule_parser_contract.md) | 三人消费方复核与实际读写器检查 | 全员，B 汇总 |
| P1 环境与工作区 | in_progress | [lcy 工程检查](handoffs/segmentation/pilot_adapter_20261112.md)、[COCO 八类加载/五图推理](handoffs/segmentation/coco_weights_20261112.md)、[200-update 与跨进程恢复](handoffs/segmentation/stability_200_20261112.md)；[wyz 图文环境与 GPU 编码][wyz-handoff]；wkq 独立 CPU 环境；[lcy AP/候选读取实测][lcy-ap] | 全组交付的接收方复核与环境复现验收；wyz 环境只读复用 base torch，base 变化需重验；非长训练验收 | lcy 协调，全员 |
| P2 真实 GT 小包 | in_progress | [原包 20 图/145 GT](handoffs/data_eval/remote_pilot_review_20261112.md)，lcy 加载与 wyz 145 个裁剪消费通过；[新审核包 9 图/59 候选/18 语义查询/36 双语记录][wkq-handoff]，35 测试、59 mask 与源一致 | [新版 reviewed_v1][wkq-p3] 已有 17 组/34 条 accepted；1 组/2 条 pending；lcy 新包加载/前反向/保存恢复已通过，wyz 实际消费、B1/B2 与统一评价已通过 [103dcc5][wyz-latest]；5 组/10 条 accepted 为已暴露答案的训练再审，非独立测试 | B，全员标注 |
| P3 基线与评价器 | in_progress | [wyz已实现B0/B1/B2与M1弱试训][wyz-latest]；本轮用户授权dev可用、候选coverage及0.3选择完成 | 各排序方法的真实dev预测、统一评价及B2 lambda选择尚未运行；旧train诊断不当作泛化成绩 | A/B |
| P4 数据扩充冻结 | in_progress | 分割5000/250已固定并使用；当前116图/290条语言dev按用户授权固定版本用于开发选择，原标注来源保留 | 其余134张dev语言及独立审核/最终测试描述另行完善；不再阻塞当前已授权开发联调 | B，全员 |
| P5-S 分割训练 | in_progress | S0选18k AP=63.5792；dev250候选和wyz读取技术验证完成；[用户授权dev选0.3][u18000-selection] | A/B本人接收及最小排序联调、后续train候选仍待推进 | lcy |
| P5-L 排序训练 | in_progress | [M1 seed42机器弱标签试训][wyz-latest]已完成；本轮固定语言dev可用于明确标记的开发对照 | M2实现及M1/M2三种子公平对照、实际dev选模尚未完成；使用查询须保留本次授权依据及原始来源 | A |
| P6 联调与对照 | not_started | E1 与集成规范 | dev 全流程、E1、demo 和扩展取舍 | 全员 |
| P7 方法锁定 | not_started | freeze 清单 | 权重/词表/阈值/数据及最终运行清单 | B 汇总，全员核对 |
| P8 最终测试 | not_started | 评价协议 | 完整预测、指标与失败记录 | 全员 |
| P9 报告与演示 | not_started | 写作分工；此行指最终交付 | 最终实验对应报告、演示与渲染检查 | 全员 |
| P10 归档与提交 | not_started | 归档要求 | 复现索引、课程回执、授权算力收尾 | 全员 |

本表不判断已有 milestone 文件或课程提交状态，最终报告与 milestone 是不同交付物。

## 2. 每人的下一项交付

| 负责人 | 下一项任务 | 可立即做 | 依赖/阻塞 | 交付位置与证据 |
| --- | --- | --- | --- | --- |
| lcy | 接通18k＋0.3候选到wyz排序接口和原mask/框输出，取得接收反馈 | 已有完整候选缓存、选择记录及固定授权查询，可直接准备最小联调 | 复用现有候选；不重复分割推理/smoke，不追加训练 | [本轮选择][u18000-selection] / [候选交付][u18000-candidates] |
| wyz（A）/ grounding | 接收18k＋0.3包并建立独立predicted特征缓存，继续M2/可比训练 | 当前语言dev已由用户授权可用，连同原审核状态和授权记录传递给消费入口 | 实际排序编码/预测及本人接收仍待执行；不能改写reviewer或冒称新增独立审核 | [选择记录][u18000-selection]、[wyz交接][wyz-latest]、PR #5 |
| wkq（B）/ data_eval | 接收当前候选/查询固定版本，完善剩余dev语言及独立审核记录，准备统一评价 | 当前116图/290查询已按用户授权完成coverage，原文件仍needs_revision；查询/图片/GT指纹和分母齐全 | 完整250图语言覆盖、正式独立审核来源和最终test描述仍不冒称齐备；不阻塞本轮开发选择 | [本轮授权与结果][u18000-selection]、PR #4 |

无 GPU 不阻塞 CPU 接口开发。使用者可用 lcy/wyz/wkq 自报身份；对应角色已绑定，不根据作者顺序推断全名或账号。三人的工作区与代码分支继续遵守 [团队说明](team_workflow.md)。

## 3. 已确认与待确认

| 事项 | 当前值/状态 | 证据或补齐责任 |
| --- | --- | --- |
| 研究与暂定分工 | documented | README、三份角色卡和 workflow |
| 文件交接例及检查器 | locally_checked，仅合成夹具 | [规范补齐交接](handoffs/contract_readiness.md) |
| 模型代码/数据转换器 | 训练/AP/cache/export已运行；新增显式用户授权dev路径，原accepted默认门禁与匹配器不变，3项必要CPU验证通过 | 已完成本轮0.3选择；实际排序联调、M2及最终test仍待完成 |
| 实验结果 | S0选18k AP=63.5792；本轮0.3覆盖258/290=88.9655%，0.5为256/290，0.7为250/290 | 仅当前用户授权116图语言dev子集的候选coverage，不是排序准确率、完整250图语言评价或新的独立审核证明；AP未重测 |
| GPU/环境/付费预算 | 本轮纯CPU离线选择7.044秒，内部300秒/外层360秒硬限，0模型加载/推理/训练 | 20:38结束GPU0MiB/0%、无进程或active记录，数据盘约157.84GiB空闲；[执行回执](handoffs/segmentation/evidence/u18000_coverage_20261115/execution_receipt.json) |
| 成员简称与角色 | lcy / wyz（A）/ wkq（B）已确认 | [绑定记录](handoffs/team_aliases.md) |
| GitHub 账号与服务器访问 | lcy：rickyliuchunyu；wkq：LE-brawl；wyz：MayI2004；lcy SSH 已验证 | lcy/wkq 对应 PR 作者；wkq 授权记录、wyz 本人登录/接受邀请见各自交接。两位队友远端运行已有记录，本次未独立验证其各自 SSH 凭据/权限 |
| 公共权重 revision 与数据 manifest | 分割源码、pilot manifest、COCO 文件 hash，以及 SigLIP 2 revision/权重 hash 已固定；两模型实际加载有证据 | [分割来源记录](../envs/segmentation/upstream.json)；[SigLIP 2 来源与校验][wyz-handoff]（镜像传输、官方 API 核对）；train/dev 指纹及用户人工复核确认已登记消费者 freeze；原生产方历史 RELEASE 不改写 |
| 课程实际截止与提交回执 | 未重新核验 | 提交负责人查课程系统；历史日期不是当前确认 |

## 4. 范围检查后的优先顺序

保持Fashionpedia八类、完整双语单目标和S0/B0/B1/B2/M1/M2/E1主线。S0训练/AP、18k候选和本轮用户授权dev的0.3阈值选择已完成。下一项为实际排序联调和共同候选比较；当前语言dev可用，不因原审核字段再次无意义等待。原字段/来源继续保留，完整dev语言及独立审核记录另行完善。min_scale优化仍为未执行提案，不自动追加训练；wyz继续M2，wkq继续数据与统一评价。

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

[candidate-export]: handoffs/segmentation/candidate_export_20261114.md

[real-candidates]: handoffs/segmentation/real_candidates_dev250_20261114.md

[language-ready]: handoffs/segmentation/language_dev_readiness_20261114.md

[offline-coverage]: handoffs/segmentation/offline_coverage_20261115.md

[s0-preparation]: handoffs/segmentation/s0_20000_preparation_20261115.md

[s0-run]: handoffs/segmentation/s0_20000_20261115.md

[s0-recovery]: handoffs/segmentation/s0_recovery20000_20261115.md

[small-analysis]: handoffs/segmentation/small_object_analysis_18000_20000_20261115.md

[u18000-candidates]: handoffs/segmentation/u18000_candidates_dev250_20261115.md

[u18000-selection]: handoffs/segmentation/u18000_language_dev_selection_20261115.md
