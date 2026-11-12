# 当前进度与下一步

更新：2026-11-12（已 fetch 全组远端分支，核对 PR #1/#2/#3 与最新交接）。本表汇总已发布证据，包括尚未合并的队友分支；不推断成员未交接的工作。本次同步未重跑队友实验，验证结论按各自交接记录引用。路线见 [P0–P10](workflow.md)，本分支运行见 [registry](../experiments/registry.csv)，交接见 [handoffs](handoffs/README.md)。

**P0/P1/P2 仍在推进，P3 已有实际基线实现与草稿输出。** lcy 完成 COCO 八类适配、五图推理、可恢复训练入口和 200-update 工程检查；wyz 完成 SigLIP 2 实际编码、B1 草稿预测及缓存重放；wkq 发布看图初标与人工审核包。人工查询 accepted=0；共享解析器、B0、统一模型评价器、冻结 train/dev 和完整长训练尚未完成。工程小包结果不作为正式性能成绩。

## GitHub 同步快照

以下为本次检查的远端 head；三份 PR 均 open / draft / unmerged，main 为 `7064ebb`。PR #3、lcy 与 wyz 分支均基于 PR #1 的公共数据接口，集成时需保留依赖。

| 成员 / GitHub | 已推送分支与 commit | PR / 最新交接 |
| --- | --- | --- |
| lcy / rickyliuchunyu | `segmentation/pilot-adapter-v1` @ `dab212d`（本次文档同步前） | [PR #2](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/2)；[200-update 交接](handoffs/segmentation/stability_200_20261112.md) |
| wyz / MayI2004 | `grounding/siglip-baseline-interfaces` @ `66c0da7` | 本次全仓 PR 查询未发现对应 PR；[编码器与 B1/B2 交接][wyz-handoff] |
| wkq / LE-brawl | `data-eval/remote-pilot-review` @ `dc24f2d`；`data-eval/grounded-query-pilot` @ `75bf2ef` | [PR #1](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/1)、[PR #3](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/3)；[最新审核包交接][wkq-handoff] |

队友交接链接固定到本次读取的 commit，避免将未合并内容误认为当前 checkout 已包含。下次开工须重新检查远端增量，流程见 [AGENTS.md](../AGENTS.md)。

## 1. 阶段状态

状态：`not_started` 无执行证据；`in_progress` 有明确活动与产物；`blocked` 已开始但受阻；`ready_for_review` 待接收方验收；`done` 满足该阶段放行条件。文档已写不等于该阶段 done。

| 阶段 | 当前状态 | 已有证据 | 尚缺的放行证据 | 主责 |
| --- | --- | --- | --- | --- |
| P0 接口确认 | in_progress | [文件契约](artifact_contract.md)、[合成示例](../examples/README.md)、[解析约定](rule_parser_contract.md) | 三人消费方复核与实际读写器检查 | 全员，B 汇总 |
| P1 环境与工作区 | in_progress | [lcy 工程检查](handoffs/segmentation/pilot_adapter_20261112.md)、[COCO 八类加载/五图推理](handoffs/segmentation/coco_weights_20261112.md)、[200-update 与跨进程恢复](handoffs/segmentation/stability_200_20261112.md)；[wyz 图文环境与 GPU 编码][wyz-handoff]；wkq 独立 CPU 环境 | 全组交付的接收方复核与环境复现验收；wyz 环境只读复用 base torch，base 变化需重验；非长训练验收 | lcy 协调，全员 |
| P2 真实 GT 小包 | in_progress | [原包 20 图/145 GT](handoffs/data_eval/remote_pilot_review_20261112.md)，lcy 加载与 wyz 145 个裁剪消费通过；[新审核包 9 图/59 候选/18 语义查询/36 双语记录][wkq-handoff]，35 测试、59 mask 与源一致 | accepted=0；wkq 人工审核导出、旧分歧裁决与消费方验收；新包仅四类目标，6 张旧图已有答案暴露，不计独立盲审 | B，全员标注 |
| P3 基线与评价器 | in_progress | [wyz B1 草稿 80/80 合法输出、0 推理错误，暖缓存重放一致；B2 位置评分与接口已实现][wyz-handoff] | 共享 `data.query_rules.parse_query`、B0、统一模型评价器未交付；B2 完整自然语言路径受解析器阻塞；accepted 查询及正式评价待完成 | A/B |
| P4 数据扩充冻结 | not_started | 数量与划分规则 | 实际分区、哈希、复核与重复组报告 | B，全员 |
| P5-S 分割训练 | not_started | 可复用 trainer / S0 配置模板；200-update 工程检查通过 | 冻结 train/dev、dev AP / checkpoint 选择、完整训练与固定候选 | lcy |
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
| lcy | 复核 wyz 候选读取接口；准备 dev AP 与 checkpoint 选择接入 | 阅读 wyz 已发布的消费接口，对齐固定预测候选交接；与 wkq 统一评价接口后准备小样本检查 | 正式训练等待冻结 train/dev、dev evaluator 与完整训练预算；200-update 和第100步跨进程恢复已通过 | [200-update 交接](handoffs/segmentation/stability_200_20261112.md)、[wyz 消费接口交接][wyz-handoff] |
| wyz（A）/ grounding | 提交 PR 与交付验收；接入共享解析器后验证完整 B2 | 将干净实现提交复跑的 `B1-draft-20261112-r02` 交给 wkq 验收；准备预测候选接口联调 | B2 等待 wkq 解析器；正式评价等待 accepted 查询/评价器；预测候选等待 lcy 固定版本 | [已发布交接][wyz-handoff]；本次未发现 PR |
| wkq（B）/ data_eval | 人工审核新包并发布 accepted 版本；交付共享解析/B0/评价器 | 审核页与审计工具已就绪；处理旧分歧及答案暴露记录；接收 wyz r02 预测接口 | 人工审核/导出与消费方验收待完成；正式分区、全量转换/去重未完成 | [9 图审核包交接][wkq-handoff]、PR #1/#3 |

无 GPU 不阻塞 CPU 接口开发。使用者可用 lcy/wyz/wkq 自报身份；对应角色已绑定，不根据作者顺序推断全名或账号。三人的工作区与代码分支继续遵守 [团队说明](team_workflow.md)。

## 3. 已确认与待确认

| 事项 | 当前值/状态 | 证据或补齐责任 |
| --- | --- | --- |
| 研究与暂定分工 | documented | README、三份角色卡和 workflow |
| 文件交接例及检查器 | locally_checked，仅合成夹具 | [规范补齐交接](handoffs/contract_readiness.md) |
| 模型代码/数据转换器 | 试点转换/复核、分割适配/候选、可恢复 trainer，以及 wyz 分支的 SigLIP 编码/缓存/B1/B2 评分已实现 | 见三方交接；共享解析器、B0、完整 S0 训练、dev AP、全量转换和统一模型评价器未完成 |
| 实验结果 | none_evaluated；仅工程检查与草稿诊断 | 本分支 registry：14 条 plan + 7 条工程运行（4 failed / 3 completed）；wyz 分支另有编码/B1 草稿运行，r02 为规范复跑。B1 的 80/80 是合法输出数，不是准确率；正式性能指标均未提供 |
| GPU/环境/付费预算 | lcy 分割短检查、wyz 编码/B1 检查已完成；完整长训练未启动 | [lcy 结束盘点](handoffs/segmentation/stability_200_20261112.md)：系统/数据盘约 27/170.3 GiB 可用；这是历史运行快照，本次文档同步未检查服务器，实际使用前重查磁盘/GPU/锁 |
| 成员简称与角色 | lcy / wyz（A）/ wkq（B）已确认 | [绑定记录](handoffs/team_aliases.md) |
| GitHub 账号与服务器访问 | lcy：rickyliuchunyu；wkq：LE-brawl；wyz：MayI2004；lcy SSH 已验证 | lcy/wkq 对应 PR 作者；wkq 授权记录、wyz 本人登录/接受邀请见各自交接。两位队友远端运行已有记录，本次未独立验证其各自 SSH 凭据/权限 |
| 公共权重 revision 与数据 manifest | 分割源码、pilot manifest、COCO 文件 hash，以及 SigLIP 2 revision/权重 hash 已固定；两模型实际加载有证据 | [分割来源记录](../envs/segmentation/upstream.json)；[SigLIP 2 来源与校验][wyz-handoff]（镜像传输、官方 API 核对）；正式冻结数据 manifest 待补齐 |
| 课程实际截止与提交回执 | 未重新核验 | 提交负责人查课程系统；历史日期不是当前确认 |

## 4. 更新规则

每次开工先按 AGENTS.md 检查整组远端分支、PR 和最新交接，再确定自己的下一步。每次工作交接在同一 PR 更新相关阶段/个人任务和证据链接，注明实际未验证项；跨分支汇总须标明来源 commit、未合并状态及是否仅引用对方验证。状态与证据不一致时以可复核产物为准，先修正文档；不覆盖他人的完成记录。阶段 done 需对应 workflow 放行条件，训练开始不等于完成，代码完成不等于 GPU 验证。

历史交接文件保留，状态表只展示当前摘要。公开索引不放密码、私钥、token 或私有下载凭据；服务器地址与访问方式由受控渠道交接。

[wyz-handoff]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/66c0da7e066c1235ee618ea7015171e73b82c914/docs/handoffs/grounding/siglip_baselines_20261112.md
[wkq-handoff]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/75bf2ef50f80bc939c35ba15afe6dc6efb15f3b1/docs/handoffs/data_eval/grounded_query_review_20261112.md
