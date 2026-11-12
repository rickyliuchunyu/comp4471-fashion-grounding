# 当前进度与下一步

更新：2026-11-12（wkq 合格查询与 P3 CPU 验证）。本表反映仓库中可核查的证据，不推断服务器或成员电脑上的未交接工作。路线见 [P0–P10](workflow.md)，运行见 [registry](../experiments/registry.csv)，交接见 [handoffs](handoffs/README.md)。

**当前 P0/P1 继续推进；wkq 独立工作区与 CPU 环境已验证，P2 真实 GT 小包已在共享数据盘交接，查询修订与消费方验收待完成。** 研究计划、角色任务卡和交接规范已建立；试点数据转换、GT 验证器及复核审计已在远端运行；统一评价器核心已实现并在训练试点GT条件验证；模型训练、正式dev/test查询与演示尚未完成。AutoDL 实例已创建，发起人 SSH 已验证，基础 PyTorch/CUDA 与分割扩展可导入；完整模型验证、组员访问与正式训练预算仍待确认。成员简称已确认：lcy = segmentation、wyz（A）= grounding、wkq（B）= data_eval；简称不代表账号权限。

## 1. 阶段状态

状态：`not_started` 无执行证据；`in_progress` 有明确活动与产物；`blocked` 已开始但受阻；`ready_for_review` 待接收方验收；`done` 满足该阶段放行条件。文档已写不等于该阶段 done。

| 阶段 | 当前状态 | 已有证据 | 尚缺的放行证据 | 主责 |
| --- | --- | --- | --- | --- |
| P0 接口确认 | in_progress | [文件契约](artifact_contract.md)、[合成示例](../examples/README.md)、[解析约定](rule_parser_contract.md) | 三人消费方复核与实际读写器检查 | 全员，B 汇总 |
| P1 环境与工作区 | in_progress | [远程只读盘点](handoffs/segmentation/server_inventory.md)：GPU/磁盘/包版本/扩展导入 | 独立工作区、图文环境、GPU 锁与模型小批检查 | lcy 协调，全员 |
| P2 真实 GT 小包 | in_progress | [wkq 远端交接](handoffs/data_eval/remote_pilot_review_20261112.md)：20 图、145 GT、19 测试通过 | [合格查询包](handoffs/data_eval/p3_reviewed_baseline_20261112.md)：17组/34条accepted；1组待修订，消费方验收未完成 | B，全员标注 |
| P3 基线与评价器 | in_progress | [wkq交接](handoffs/data_eval/p3_reviewed_baseline_20261112.md)：共享解析/B0/统一评价核心，74测试，34条GT试点诊断 | B1/B2实际输出、消费方与评价交叉复核 | A/B |
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
| wyz（A）/ grounding | 审核候选读取与共享解析，准备 B1/B2 | 合成包读取、缓存键/预测输出接口、独立编码器入口 | GT 结果需真实图片/查询，GPU 使用需排队 | 个人 PR；特征与预测检查；角色交接文件，均待创建 |
| wkq（B）/ data_eval | 接收模型消费方验收，处理余下1组，扩充试点与分区 | 17组accepted；解析/B0/评价核心74测试通过，见 [P3交接](handoffs/data_eval/p3_reviewed_baseline_20261112.md) | 正式查询/消费方验收及全量划分未完成 | [远端交接](handoffs/data_eval/remote_pilot_review_20261112.md) |

无 GPU 不阻塞 CPU 接口开发。使用者可用 lcy/wyz/wkq 自报身份；对应角色已绑定，不根据作者顺序推断全名或账号。三人的工作区与代码分支继续遵守 [团队说明](team_workflow.md)。

## 3. 已确认与待确认

| 事项 | 当前值/状态 | 证据或补齐责任 |
| --- | --- | --- |
| 研究与暂定分工 | documented | README、三份角色卡和 workflow |
| 文件交接例及检查器 | locally_checked，仅合成夹具 | [规范补齐交接](handoffs/contract_readiness.md) |
| 模型代码/数据转换器 | 试点转换及 GT/复核审计 remote_cpu_checked；模型未实现 | 见 wkq 交接；正式全量转换未完成；统一评价核心已CPU验证 |
| 实验结果 | training_pilot_diagnostic | 保留14条计划，新增B0训练GT诊断；12/34选对，不是正式dev/test成绩 |
| GPU/环境/付费预算 | GPU 与基础依赖已盘点；完整模型/长期训练预算未验证 | [服务器检查](handoffs/segmentation/server_inventory.md)，系统盘仅余约 4.6G |
| 成员简称与角色 | lcy / wyz（A）/ wkq（B）已确认 | [绑定记录](handoffs/team_aliases.md) |
| wyz/wkq 的 GitHub/SSH 权限 | wkq SSH 已验证，GitHub LE-brawl 经本人确认授权；wyz 待核验 | wkq 远端交接；其他成员分别授权 |
| 公共权重 revision 与数据 manifest | not_pinned | 对应方法/数据负责人执行时补齐 |
| 课程实际截止与提交回执 | 未重新核验 | 提交负责人查课程系统；历史日期不是当前确认 |

## 4. 更新规则

每次工作交接在同一 PR 更新相关阶段/个人任务和证据链接，注明实际未验证项。状态与证据不一致时以可复核产物为准，先修正文档；不覆盖他人的完成记录。阶段 done 需对应 workflow 放行条件，训练开始不等于完成，代码完成不等于 GPU 验证。

历史交接文件保留，状态表只展示当前摘要。公开索引不放密码、私钥、token 或私有下载凭据；服务器地址与访问方式由受控渠道交接。
