# 当前进度与下一步

更新：2026-11-12（wkq 大包发布与机器初标）。本表反映仓库中可核查的证据，不推断服务器或成员电脑上的未交接工作。路线见 [P0–P10](workflow.md)，运行见 [registry](../experiments/registry.csv)，交接见 [handoffs](handoffs/README.md)。

**wkq 已推进至 P4 数据扩充：5,000 train / 250 dev / 400 test 留出图已版本化发布；1,000 图训练语言初标与 116 图 dev 草稿仍待人工审核。** 共享解析器、B0、统一评价核心和数据工具共 93 项 CPU 测试通过，见 [大型数据交接](handoffs/data_eval/large_dataset_20261112.md)。本轮没有 GPU 训练、正式 dev/test 语言评价或三方最终冻结；其他成员模型进度以其交接为准。

wyz 已完成独立图文环境、真实 GT 裁剪、SigLIP 2 logits 检查及 80 条 B1 草稿接口预测；36 项测试通过。共享解析与 accepted 小包已发布，正在接入及独立验收，未训练 M1/M2。见 [grounding 交接](handoffs/grounding/siglip_baselines_20261112.md)。

## 1. 阶段状态

状态：`not_started` 无执行证据；`in_progress` 有明确活动与产物；`blocked` 已开始但受阻；`ready_for_review` 待接收方验收；`done` 满足该阶段放行条件。文档已写不等于该阶段 done。

| 阶段 | 当前状态 | 已有证据 | 尚缺的放行证据 | 主责 |
| --- | --- | --- | --- | --- |
| P0 接口确认 | in_progress | [文件契约](artifact_contract.md)、[合成示例](../examples/README.md)、[解析约定](rule_parser_contract.md) | 三人消费方复核与实际读写器检查 | 全员，B 汇总 |
| P1 环境与工作区 | in_progress | [远程只读盘点](handoffs/segmentation/server_inventory.md)：GPU/磁盘/包版本/扩展导入 | 独立工作区、图文环境、GPU 锁与模型小批检查 | lcy 协调，全员 |
| P2 真实 GT 小包 | in_progress | 17组/34条accepted保留；lcy小包消费验收有工作区记录，用户报告wyz接口OK | 旧1组争议仍待处理；不同版本验收不能混用 | B，全员 |
| P3 基线与评价器 | in_progress | [P3交接](handoffs/data_eval/p3_reviewed_baseline_20261112.md)；共享解析/B0/统一评价核心，当前全套93测试通过 | B1/B2实际输出及评价交叉复核 | A/B |
| P4 数据扩充冻结 | in_progress | [大型数据交接](handoffs/data_eval/large_dataset_20261112.md)：5000/250/400图，1000图3160条训练初标，分组/hash/源mask检查通过 | 人工验收、dev补写、独立测试描述、三方消费验收与冻结 | B，全员 |
| P5-S 分割训练 | not_started | S0 规划配置 | 可恢复权重、dev 结果和固定候选 | lcy |
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
| lcy | 验收新train/dev包，准备自己的分割训练 | 小包模型smoke已有工作区交接；新包含5000/250图及源保真检查 | 新命名空间/加载器验收；训练配置与GPU协调由本人登记 | 见[大型数据交接](handoffs/data_eval/large_dataset_20261112.md)中的小包证据和新包路径 |
| wyz（A）/ grounding | 验收大包读取、缓存键和弱标注实验入口 | 用户报告小包接口OK；可使用明确标记的训练机器初标做实验 | 正式dev查询尚待人工验收；不得把needs_revision伪装accepted | 见[大型数据交接](handoffs/data_eval/large_dataset_20261112.md)，新包模型消费未由wkq代验 |
| wkq（B）/ data_eval | 审核dev草稿并组织补写，接收大包消费验收 | [已发布分区、初标和35页审核入口](handoffs/data_eval/large_dataset_20261112.md)；训练不属本轮职责 | dev仅116图有待审草稿，其余134图待补写；测试描述未生成 | 共享 processed/fp_data_eval_large_v1 与交接文档 |

无 GPU 不阻塞 CPU 接口开发。使用者可用 lcy/wyz/wkq 自报身份；对应角色已绑定，不根据作者顺序推断全名或账号。三人的工作区与代码分支继续遵守 [团队说明](team_workflow.md)。

## 3. 已确认与待确认

| 事项 | 当前值/状态 | 证据或补齐责任 |
| --- | --- | --- |
| 研究与暂定分工 | documented | README、三份角色卡和 workflow |
| 文件交接例及检查器 | locally_checked，仅合成夹具 | [规范补齐交接](handoffs/contract_readiness.md) |
| 模型代码/数据转换器 | wkq大包转换、GT源保真、初标及评价核心remote_cpu_checked | [大型数据交接](handoffs/data_eval/large_dataset_20261112.md)；模型训练由负责人另记 |
| 实验结果 | training_pilot_diagnostic | 保留14条计划，新增B0训练GT诊断；12/34选对，不是正式dev/test成绩 |
| GPU/环境/付费预算 | GPU 与基础依赖已盘点；完整模型/长期训练预算未验证 | [服务器检查](handoffs/segmentation/server_inventory.md)，系统盘仅余约 4.6G |
| 成员简称与角色 | lcy / wyz（A）/ wkq（B）已确认 | [绑定记录](handoffs/team_aliases.md) |
| wyz/wkq 的 GitHub/SSH 权限 | wkq SSH 已验证，GitHub LE-brawl 经本人确认授权；wyz SSH 与 MayI2004 GitHub push 已核验 | wkq 远端交接；其他成员分别授权 |
| 公共权重 revision 与数据 manifest | 大包manifest/hash已固定版本；三方正式freeze未完成 | 权重由方法负责人登记；数据见发布索引 |
| 课程实际截止与提交回执 | 未重新核验 | 提交负责人查课程系统；历史日期不是当前确认 |

## 4. 更新规则

每次工作交接在同一 PR 更新相关阶段/个人任务和证据链接，注明实际未验证项。状态与证据不一致时以可复核产物为准，先修正文档；不覆盖他人的完成记录。阶段 done 需对应 workflow 放行条件，训练开始不等于完成，代码完成不等于 GPU 验证。

历史交接文件保留，状态表只展示当前摘要。公开索引不放密码、私钥、token 或私有下载凭据；服务器地址与访问方式由受控渠道交接。
