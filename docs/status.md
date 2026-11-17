# 当前进度与下一步

更新：2026-11-17（wkq 语言困难增量已发布）。本表反映仓库中可核查的证据，不推断服务器或成员电脑上的未交接工作。路线见 [P0–P10](workflow.md)，运行见 [registry](../experiments/registry.csv)，交接见 [handoffs](handoffs/README.md)。

**wkq 已导入dev人工审核并完成现有模型两种候选条件的评价。** 321组/642条accepted、246张查询图，9组待处理。M1在GT候选选对91.59%；lcy u18000+0.3预测候选下Success@0.5=83.80%、同类关联选择准确率71.50%。B1/B2同清单对照、子组/错误分析和图片级区间见[交接](handoffs/data_eval/reviewed_dev_eval_20261117.md)。135项CPU测试及4项适配检查通过。没有训练或占用GPU，原train/dev/test划分不变；该次模型评价未扩充数据。

**wkq 已发布语言定向增量。** 最终新增149张训练图/298条AI逐图初标，合并包1149图/3296条；独立dev困难集39图/78条待审，原主dev/test不变。142项测试、7122个hash和5908个源mask/裁剪检查通过，wyz弱标注入口实际可读取。没有训练或GPU使用；[交接、路径、审核和下游步骤](handoffs/data_eval/language_targeted_20261117.md)。

wyz 已完成 B 六包技术消费验收，B1/B2 在34条已复核训练查询上选对18/22条；M1 seed42固定5轮弱标注试训完成，旧训练记录保留；B已发布新的dev评价，本轮正在接收并开发M2。见 [grounding 旧交接](handoffs/grounding/accepted_baselines_and_weak_trial_20261112.md)。

## 1. 阶段状态

状态：`not_started` 无执行证据；`in_progress` 有明确活动与产物；`blocked` 已开始但受阻；`ready_for_review` 待接收方验收；`done` 满足该阶段放行条件。文档已写不等于该阶段 done。

| 阶段 | 当前状态 | 已有证据 | 尚缺的放行证据 | 主责 |
| --- | --- | --- | --- | --- |
| P0 接口确认 | in_progress | [文件契约](artifact_contract.md)、[合成示例](../examples/README.md)、[解析约定](rule_parser_contract.md) | 三人消费方复核与实际读写器检查 | 全员，B 汇总 |
| P1 环境与工作区 | in_progress | [远程只读盘点](handoffs/segmentation/server_inventory.md)：GPU/磁盘/包版本/扩展导入 | 独立工作区、图文环境、GPU 锁与模型小批检查 | lcy 协调，全员 |
| P2 真实 GT 小包 | in_progress | 17组/34条accepted保留；lcy小包消费验收有工作区记录，用户报告wyz接口OK | 旧1组争议仍待处理；不同版本验收不能混用 | B，全员 |
| P3 基线与评价器 | in_progress | [P3交接](handoffs/data_eval/p3_reviewed_baseline_20261112.md)；共享解析/B0/统一评价核心，当前全套117测试通过 | B1/B2实际输出及评价交叉复核 | A/B |
| P4 数据扩充冻结 | in_progress | [10k交接](handoffs/data_eval/train10000_targeted_20261116.md)：新10k分割包，旧5000/250/400和语言包不变；定向补充/分组/hash/源mask检查通过 | 人工验收、dev草稿复核、独立测试描述、三方消费验收与冻结 | B，全员 |
| P5-S 分割训练 | in_progress | lcy [10k扩充请求](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/149ddf7/docs/handoffs/segmentation/train10000_expansion_request_20261115.md) 引用已有模型/候选；本轮未改动 | 新10k包消费验收与后续运行由lcy登记，本分支不代验 | lcy |
| P5-L 排序训练 | in_progress | 已核对wyz旧machine_v2、seed42、5轮M1权重；wkq本轮固定权重dev评价 | 新标注训练对照、M2、其他seed仍待完成 | A |
| P6 联调与对照 | in_progress | [已审核dev的GT/预测候选B1/B2/M1评价](handoffs/data_eval/reviewed_dev_eval_20261117.md) | 9组标注裁决、正式方法对照/M2/E1、接收方验收 | 全员 |
| P7 方法锁定 | not_started | freeze 清单 | 权重/词表/阈值/数据及最终运行清单 | B 汇总，全员核对 |
| P8 最终测试 | not_started | 评价协议 | 完整预测、指标与失败记录 | 全员 |
| P9 报告与演示 | not_started | 写作分工；此行指最终交付 | 最终实验对应报告、演示与渲染检查 | 全员 |
| P10 归档与提交 | not_started | 归档要求 | 复现索引、课程回执、授权算力收尾 | 全员 |

本表不判断已有 milestone 文件或课程提交状态，最终报告与 milestone 是不同交付物。

## 2. 每人的下一项交付

| 负责人 | 下一项任务 | 可立即做 | 依赖/阻塞 | 交付位置与证据 |
| --- | --- | --- | --- | --- |
| lcy | 验收新10k包，登记后续分割运行 | [10k交接](handoffs/data_eval/train10000_targeted_20261116.md)；旧dev/test和已有模型候选不变 | 实际加载器验收；训练配置与GPU协调由本人登记 | 新包 shared/processed/fp_train10000_targeted_v1 |
| wyz（A）/ grounding | 清理后1000 vs1149语言训练对照；空间模块 | [增量数据及入口已由wkq技术验证](handoffs/data_eval/language_targeted_20261117.md)，主dev642条accepted | 新增训练仍为AI弱标注；dev39待审；训练/GPU由本人登记 | 新包、hash、候选与训练输入见交接；不代签收或宣告训练完成 |
| wkq（B）/ data_eval | 审核新增149训练组、39dev困难组；原9组裁决 | [语言增量交接](handoffs/data_eval/language_targeted_20261117.md)，8页新增审核 | AI初标未人工accepted；新dev审核后单列评价 | fp_language1149_targeted_draft_v1 / fp_language_dev39_challenge_draft_v1；142测试和真实消费检查 |

无 GPU 不阻塞 CPU 接口开发。使用者可用 lcy/wyz/wkq 自报身份；对应角色已绑定，不根据作者顺序推断全名或账号。三人的工作区与代码分支继续遵守 [团队说明](team_workflow.md)。

## 3. 已确认与待确认

| 事项 | 当前值/状态 | 证据或补齐责任 |
| --- | --- | --- |
| 研究与暂定分工 | documented | README、三份角色卡和 workflow |
| 文件交接例及检查器 | locally_checked，仅合成夹具 | [规范补齐交接](handoffs/contract_readiness.md) |
| 模型代码/数据转换器 | wkq大包转换、GT源保真、初标及评价核心remote_cpu_checked | [大型数据交接](handoffs/data_eval/large_dataset_20261112.md)；模型训练由负责人另记 |
| 实验结果 | training_pilot_diagnostic | 保留14条计划；B0/B1/B2训练GT诊断与M1机器标签训练拟合，均非正式dev/test成绩 |
| GPU/环境/付费预算 | GPU 与基础依赖已盘点；完整模型/长期训练预算未验证 | [服务器检查](handoffs/segmentation/server_inventory.md)，系统盘仅余约 4.6G |
| 成员简称与角色 | lcy / wyz（A）/ wkq（B）已确认 | [绑定记录](handoffs/team_aliases.md) |
| wyz/wkq 的 GitHub/SSH 权限 | wkq SSH 已验证，GitHub LE-brawl 经本人确认授权；wyz SSH 与 MayI2004 GitHub push 已核验 | wkq 远端交接；其他成员分别授权 |
| 公共权重 revision 与数据 manifest | 大包manifest/hash已固定版本；三方正式freeze未完成 | 权重由方法负责人登记；数据见发布索引 |
| 课程实际截止与提交回执 | 未重新核验 | 提交负责人查课程系统；历史日期不是当前确认 |

## 4. 更新规则

每次工作交接在同一 PR 更新相关阶段/个人任务和证据链接，注明实际未验证项。状态与证据不一致时以可复核产物为准，先修正文档；不覆盖他人的完成记录。阶段 done 需对应 workflow 放行条件，训练开始不等于完成，代码完成不等于 GPU 验证。

历史交接文件保留，状态表只展示当前摘要。公开索引不放密码、私钥、token 或私有下载凭据；服务器地址与访问方式由受控渠道交接。
