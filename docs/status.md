# 当前进度与下一步

更新：2026-11-12（wkq CPU 交接；wyz grounding 编码与草稿预测验证）。本表反映仓库中可核查的证据，不推断服务器或成员电脑上的未交接工作。路线见 [P0–P10](workflow.md)，运行见 [registry](../experiments/registry.csv)，交接见 [handoffs](handoffs/README.md)。

**当前 P0/P1 继续推进；wkq 独立工作区与 CPU 环境已验证，P2 真实 GT 小包已在共享数据盘交接，查询修订与消费方验收待完成。** 研究计划、角色任务卡和交接规范已建立；试点数据转换、GT 验证器及复核审计已在远端运行；模型训练、统一模型评价器、正式查询与演示尚未完成。AutoDL 实例已创建，发起人 SSH 已验证，基础 PyTorch/CUDA 与分割扩展可导入；完整模型验证、组员访问与正式训练预算仍待确认。成员简称已确认：lcy = segmentation、wyz（A）= grounding、wkq（B）= data_eval；简称不代表账号权限。

wyz 已完成独立图文环境、真实 GT 裁剪、SigLIP 2 logits 检查及 80 条 B1 草稿接口预测；36 项测试通过。共享解析、accepted 查询与正式评价仍待交接，未训练 M1/M2。见 [grounding 交接](handoffs/grounding/siglip_baselines_20261112.md)。

## 1. 阶段状态

状态：`not_started` 无执行证据；`in_progress` 有明确活动与产物；`blocked` 已开始但受阻；`ready_for_review` 待接收方验收；`done` 满足该阶段放行条件。文档已写不等于该阶段 done。

| 阶段 | 当前状态 | 已有证据 | 尚缺的放行证据 | 主责 |
| --- | --- | --- | --- | --- |
| P0 接口确认 | in_progress | [文件契约](artifact_contract.md)、[合成示例](../examples/README.md)、[解析约定](rule_parser_contract.md) | 三人消费方复核与实际读写器检查 | 全员，B 汇总 |
| P1 环境与工作区 | in_progress | [远程只读盘点](handoffs/segmentation/server_inventory.md)：GPU/磁盘/包版本/扩展导入 | 独立工作区、图文环境、GPU 锁与模型小批检查 | lcy 协调，全员 |
| P2 真实 GT 小包 | in_progress | [wkq 远端交接](handoffs/data_eval/remote_pilot_review_20261112.md)：20 图、145 GT、19 测试通过 | 6 组复核分歧、2 组已示答案、语言设计修订与消费方验收；accepted=0 | B，全员标注 |
| P3 基线与评价器 | in_progress | wyz B1 草稿输出与 B2 评分检查；[证据](handoffs/grounding/siglip_baselines_20261112.md) | 共享解析、accepted 查询、统一评价与正式基线结果 | A/B |
| P4 数据扩充冻结 | not_started | 数量与划分规则 | 实际分区、哈希、复核与重复组报告 | B，全员 |
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
| lcy | 基础盘点完成；规划独立课程工作区，准备分割模型检查 | 核对公开源码/权重与候选接口，规划数据盘缓存/输出 | 模型小批验证需 B 的真实小包与具体运行配置 | [环境记录](handoffs/segmentation/server_inventory.md)；实现 PR 与模型检查待完成 |
| wyz（A）/ grounding | 接入 wkq 共享解析与正式查询，推进 B2 | 独立环境、768 维编码、缓存/B1 与 36 测试已验证 | accepted=0；共享 parse_query 和统一评价器未发布 | [远端 grounding 交接](handoffs/grounding/siglip_baselines_20261112.md) |
| wkq（B）/ data_eval | 修订语言试点，处理复核分歧；继续共享解析/B0/评价器 | 独立远端 CPU 环境、GT 小包与审计已运行 | 正式查询/消费方验收及全量划分未完成 | [远端交接](handoffs/data_eval/remote_pilot_review_20261112.md) |

无 GPU 不阻塞 CPU 接口开发。使用者可用 lcy/wyz/wkq 自报身份；对应角色已绑定，不根据作者顺序推断全名或账号。三人的工作区与代码分支继续遵守 [团队说明](team_workflow.md)。

## 3. 已确认与待确认

| 事项 | 当前值/状态 | 证据或补齐责任 |
| --- | --- | --- |
| 研究与暂定分工 | documented | README、三份角色卡和 workflow |
| 文件交接例及检查器 | locally_checked，仅合成夹具 | [规范补齐交接](handoffs/contract_readiness.md) |
| 模型代码/数据转换器 | 试点转换及 GT/复核审计 remote_cpu_checked；wyz 编码器/B1 接口 gpu_verified | 见 wkq 交接；正式全量转换和统一模型评价器未完成 |
| 实验结果 | 无正式成绩；B1 草稿诊断已运行 | registry 保留计划并追加 draft 运行，正式指标为空 |
| GPU/环境/付费预算 | GPU 与基础依赖已盘点；完整模型/长期训练预算未验证 | [服务器检查](handoffs/segmentation/server_inventory.md)，系统盘仅余约 4.6G |
| 成员简称与角色 | lcy / wyz（A）/ wkq（B）已确认 | [绑定记录](handoffs/team_aliases.md) |
| wyz/wkq 的 GitHub/SSH 权限 | wyz SSH 与 MayI2004 GitHub push 已核验；wkq 见其交接 | 不以简称推断权限；使用各自授权 |
| 公共权重 revision 与数据 manifest | wyz SigLIP 2 revision/权重 SHA 与 fp_pilot20_v2 已记录；正式数据未冻结 | [grounding 交接](handoffs/grounding/siglip_baselines_20261112.md) |
| 课程实际截止与提交回执 | 未重新核验 | 提交负责人查课程系统；历史日期不是当前确认 |

## 4. 更新规则

每次工作交接在同一 PR 更新相关阶段/个人任务和证据链接，注明实际未验证项。状态与证据不一致时以可复核产物为准，先修正文档；不覆盖他人的完成记录。阶段 done 需对应 workflow 放行条件，训练开始不等于完成，代码完成不等于 GPU 验证。

历史交接文件保留，状态表只展示当前摘要。公开索引不放密码、私钥、token 或私有下载凭据；服务器地址与访问方式由受控渠道交接。
