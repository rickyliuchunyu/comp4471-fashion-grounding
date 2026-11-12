# 当前进度与下一步

更新：2026-11-12 17:14 HKT（lcy 已接收 wkq reviewed_v1 并通过分割接口检查；冻结分割 train/dev 仍未发布）。本表汇总已发布证据，包括尚未合并的队友分支；不推断成员未交接的工作。lcy 验证见本轮交接；队友结果按各自记录引用，未重跑其模型实验。路线见 [P0–P10](workflow.md)，本分支运行见 [registry](../experiments/registry.csv)，交接见 [handoffs](handoffs/README.md)。

**P0/P1/P2/P3 继续推进；正式分区与长训练尚未开始。** lcy 完成可恢复 trainer、200-update 检查、官方 AP/离线 checkpoint 选择入口，并以已有 100/200 步权重验证 wyz 候选读取接口；wyz 完成 SigLIP 2/B1 草稿验证。wkq 新发布 17 组/34 条 accepted 训练查询，以及共享解析器/B0/统一语言评价核心。正式 train/dev 冻结、B2 实际联调、真实 dev 权重选择仍待完成；训练小包诊断不作为泛化成绩。

## GitHub 同步快照

以下为本次检查的远端 head；四份 PR 均 open / draft / unmerged，main 为 `7064ebb`。PR #4 接续 PR #3，后者及 lcy/wyz 分支均基于 PR #1 的公共数据接口，集成时需保留依赖。

| 成员 / GitHub | 已推送分支与 commit | PR / 最新交接 |
| --- | --- | --- |
| lcy / rickyliuchunyu | `segmentation/pilot-adapter-v1` @ `cb1e77f`（本轮接收检查运行提交；交接另随本 PR 更新） | [PR #2](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/2)；[AP 与候选消费交接][lcy-ap] |
| wyz / MayI2004 | `grounding/siglip-baseline-interfaces` @ `66c0da7` | 本次全仓 PR 查询未发现对应 PR；[编码器与 B1/B2 交接][wyz-handoff] |
| wkq / LE-brawl | `data-eval/p3-baseline-evaluation` @ `66dd07c`；上游 `grounded-query-pilot` @ `75bf2ef`、`remote-pilot-review` @ `dc24f2d` | [PR #1](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/1)、[PR #3](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/3)、[PR #4](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/4)；[P2/P3 最新交接][wkq-p3] |

队友交接链接固定到本次读取的 commit，避免将未合并内容误认为当前 checkout 已包含。下次开工须重新检查远端增量，流程见 [AGENTS.md](../AGENTS.md)。

17:05 HKT 复核：所有队友分支 head 与上次一致，四份 PR 均无 review/comment。服务器发布的 `fp_pilot20_v2`（20 图）、`fp_grounded9_draft_v2`（9 图）、`fp_grounded9_reviewed_v1`（9 图）均仅有 train 图片；未发现已发布 dev 包，wkq 最新交接仍明确 P4 未完成。34 条 accepted 是训练语言查询，不表示分割 train/dev 已冻结。本轮未启动 AP、checkpoint 选择或训练，见 [就绪核查](handoffs/segmentation/freeze_readiness_20261112.md)。随后按 wkq 请求完成训练小包分割接口验收，见 [reviewed_v1 接收记录][reviewed-acceptance]；两项任务的前提不同。

## 1. 阶段状态

状态：`not_started` 无执行证据；`in_progress` 有明确活动与产物；`blocked` 已开始但受阻；`ready_for_review` 待接收方验收；`done` 满足该阶段放行条件。文档已写不等于该阶段 done。

| 阶段 | 当前状态 | 已有证据 | 尚缺的放行证据 | 主责 |
| --- | --- | --- | --- | --- |
| P0 接口确认 | in_progress | [文件契约](artifact_contract.md)、[合成示例](../examples/README.md)、[解析约定](rule_parser_contract.md) | 三人消费方复核与实际读写器检查 | 全员，B 汇总 |
| P1 环境与工作区 | in_progress | [lcy 工程检查](handoffs/segmentation/pilot_adapter_20261112.md)、[COCO 八类加载/五图推理](handoffs/segmentation/coco_weights_20261112.md)、[200-update 与跨进程恢复](handoffs/segmentation/stability_200_20261112.md)；[wyz 图文环境与 GPU 编码][wyz-handoff]；wkq 独立 CPU 环境；[lcy AP/候选读取实测][lcy-ap] | 全组交付的接收方复核与环境复现验收；wyz 环境只读复用 base torch，base 变化需重验；非长训练验收 | lcy 协调，全员 |
| P2 真实 GT 小包 | in_progress | [原包 20 图/145 GT](handoffs/data_eval/remote_pilot_review_20261112.md)，lcy 加载与 wyz 145 个裁剪消费通过；[新审核包 9 图/59 候选/18 语义查询/36 双语记录][wkq-handoff]，35 测试、59 mask 与源一致 | [新版 reviewed_v1][wkq-p3] 已有 17 组/34 条 accepted；1 组/2 条 pending；lcy 新包加载/前反向/保存恢复已通过，wyz 模型验收待完成；5 组/10 条 accepted 为已暴露答案的训练再审，非独立测试 | B，全员标注 |
| P3 基线与评价器 | in_progress | [wyz B1 草稿 80/80 合法输出、0 推理错误，暖缓存重放一致；B2 位置评分与接口已实现][wyz-handoff] | [wkq 已交付共享解析/B0/统一评价核心][wkq-p3]（74 CPU 测试、34 条训练 GT 诊断）；wyz 需接收新解析器并实际验证 B2，成员交叉验收及正式评价待完成 | A/B |
| P4 数据扩充冻结 | not_started | 数量与划分规则；[17:05 发布包核查](handoffs/segmentation/freeze_readiness_20261112.md)仅见 train 小包 | 冻结分割 train/dev 包、manifest/annotation hash、官方来源边界与跨组重复审查 | B，全员 |
| P5-S 分割训练 | not_started | 可复用 trainer / S0 配置模板；200-update 通过；[AP/离线选择与候选消费检查通过][lcy-ap] | 冻结 train/dev、真实 dev AP / checkpoint 选择、完整训练与固定候选 | lcy |
| P5-L 排序训练 | not_started | M1/M2 规划配置 | 各 seed 日志、权重和 dev 结果 | A |
| P6 联调与对照 | not_started | E1 与集成规范 | dev 全流程、E1、demo 和扩展取舍 | 全员 |
| P7 方法锁定 | not_started | freeze 清单 | 权重/词表/阈值/数据及最终运行清单 | B 汇总，全员核对 |
| P8 最终测试 | not_started | 评价协议 | 完整预测、指标与失败记录 | 全员 |
| P9 报告与演示 | not_started | 写作分工；此行指最终交付 | 最终实验对应报告、演示与渲染检查 | 全员 |
| P10 归档与提交 | not_started | 归档要求 | 复现索引、课程回执、授权算力收尾 | 全员 |

本表不判断已有 milestone 文件或课程提交状态，最终报告与 milestone 是不同交付物。

## 2. 每人的下一项交付

| 负责人 | 下一项任务 | 可立即做 | 依赖/阻塞 | 交付位置与证据 |
| --- | --- | --- | --- | --- |
| lcy | 接收冻结分割 train/dev，完成真实 dev AP 与选择验证 | AP/选择入口已实现；新版 reviewed_v1 的 9 图/59 实例分割消费检查通过，可接收后续扩充包 | 17:05 复核仍未发布冻结 train/dev；现有 100/200 步为 pilot，正式选择入口会拒绝；后续需与冻结 train 匹配的短训练 checkpoint | [AP 交接][lcy-ap]、[新版小包验收][reviewed-acceptance] |
| wyz（A）/ grounding | 提交 PR 与交付验收；接收 PR #4 解析器并验证完整 B2 | 将干净实现提交复跑的 `B1-draft-20261112-r02` 交给 wkq 验收；准备预测候选接口联调 | wkq PR #4 解析器与 accepted 包现已可接收；B2 实际联调/正式 dev 仍待完成；正式预测候选等待 lcy 固定版本 | [已发布交接][wyz-handoff]；本次未发现 PR |
| wkq（B）/ data_eval | 接收模型消费方验收，处理余下 1 组，发布冻结分割 train/dev | lcy 已确认 reviewed_v1 分割接口跑通，可按同 schema 扩充；接收 wyz 新包模型验收及 B1/B2 输出 | 正式分区、全量转换/去重与模型交叉验收未完成 | [最新交接][wkq-p3]、PR #4 |

无 GPU 不阻塞 CPU 接口开发。使用者可用 lcy/wyz/wkq 自报身份；对应角色已绑定，不根据作者顺序推断全名或账号。三人的工作区与代码分支继续遵守 [团队说明](team_workflow.md)。

## 3. 已确认与待确认

| 事项 | 当前值/状态 | 证据或补齐责任 |
| --- | --- | --- |
| 研究与暂定分工 | documented | README、三份角色卡和 workflow |
| 文件交接例及检查器 | locally_checked，仅合成夹具 | [规范补齐交接](handoffs/contract_readiness.md) |
| 模型代码/数据转换器 | 试点转换/复核、分割 trainer/AP/离线选择、wyz 编码/B1/B2 评分与 wkq 共享解析/B0/语言评价核心已实现 | 见三方交接；完整 S0、真实 dev 选择、全量转换、B2 实际联调与统一评价交叉验收未完成 |
| 实验结果 | 无正式 dev/test 成绩；有训练试点诊断 | 本分支 registry：14 条 plan + 9 条工程运行（4 failed / 5 completed）；新增 AP 接口检查不填正式指标栏。wyz B1 的 80/80 是合法输出数；wkq B0 训练 GT 诊断为 12/34 选对，均非泛化成绩 |
| GPU/环境/付费预算 | 新版小包接口检查占锁 12.67 秒、2 次工程更新；完整长训练未启动 | [本轮结束盘点](handoffs/segmentation/evidence/reviewed_pilot_20261112/postflight.log)：系统/数据盘约 27/170 GiB 可用，GPU 0 MiB/0%，无计算进程；后续使用前重查 |
| 成员简称与角色 | lcy / wyz（A）/ wkq（B）已确认 | [绑定记录](handoffs/team_aliases.md) |
| GitHub 账号与服务器访问 | lcy：rickyliuchunyu；wkq：LE-brawl；wyz：MayI2004；lcy SSH 已验证 | lcy/wkq 对应 PR 作者；wkq 授权记录、wyz 本人登录/接受邀请见各自交接。两位队友远端运行已有记录，本次未独立验证其各自 SSH 凭据/权限 |
| 公共权重 revision 与数据 manifest | 分割源码、pilot manifest、COCO 文件 hash，以及 SigLIP 2 revision/权重 hash 已固定；两模型实际加载有证据 | [分割来源记录](../envs/segmentation/upstream.json)；[SigLIP 2 来源与校验][wyz-handoff]（镜像传输、官方 API 核对）；正式冻结数据 manifest 待补齐 |
| 课程实际截止与提交回执 | 未重新核验 | 提交负责人查课程系统；历史日期不是当前确认 |

## 4. 更新规则

每次开工先按 AGENTS.md 检查整组远端分支、PR 和最新交接，再确定自己的下一步。每次工作交接在同一 PR 更新相关阶段/个人任务和证据链接，注明实际未验证项；跨分支汇总须标明来源 commit、未合并状态及是否仅引用对方验证。状态与证据不一致时以可复核产物为准，先修正文档；不覆盖他人的完成记录。阶段 done 需对应 workflow 放行条件，训练开始不等于完成，代码完成不等于 GPU 验证。

历史交接文件保留，状态表只展示当前摘要。公开索引不放密码、私钥、token 或私有下载凭据；服务器地址与访问方式由受控渠道交接。

[wyz-handoff]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/66c0da7e066c1235ee618ea7015171e73b82c914/docs/handoffs/grounding/siglip_baselines_20261112.md
[wkq-handoff]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/75bf2ef50f80bc939c35ba15afe6dc6efb15f3b1/docs/handoffs/data_eval/grounded_query_review_20261112.md

[lcy-ap]: handoffs/segmentation/dev_ap_20261112.md
[wkq-p3]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/66dd07c56fb02ae90af7a12de3524650d2fe6455/docs/handoffs/data_eval/p3_reviewed_baseline_20261112.md

[reviewed-acceptance]: handoffs/segmentation/reviewed_pilot_acceptance_20261112.md
