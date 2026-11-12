# 当前进度与下一步

更新：2026-11-12（wkq CPU 交接 + lcy 分割工程检查）。本表反映仓库中可核查的证据，不推断服务器或成员电脑上的未交接工作。路线见 [P0–P10](workflow.md)，运行见 [registry](../experiments/registry.csv)，交接见 [handoffs](handoffs/README.md)。

**当前 P0/P1 继续推进；lcy 独立环境已完成真实小包加载、两次随机初始化 GPU 更新、保存恢复与空候选导出；P2 查询修订及其余消费方验收待完成。** 研究计划、角色任务卡和交接规范已建立；试点数据转换、GT 验证器及复核审计已在远端运行；正式模型训练、统一模型评价器、正式查询与演示尚未完成。AutoDL 实例已创建，发起人 SSH 已验证，基础 PyTorch/CUDA 与分割扩展可导入；公开预训练权重加载、完整训练验证、组员访问与正式训练预算仍待确认。成员简称已确认：lcy = segmentation、wyz（A）= grounding、wkq（B）= data_eval；简称不代表账号权限。

## 1. 阶段状态

状态：`not_started` 无执行证据；`in_progress` 有明确活动与产物；`blocked` 已开始但受阻；`ready_for_review` 待接收方验收；`done` 满足该阶段放行条件。文档已写不等于该阶段 done。

| 阶段 | 当前状态 | 已有证据 | 尚缺的放行证据 | 主责 |
| --- | --- | --- | --- | --- |
| P0 接口确认 | in_progress | [文件契约](artifact_contract.md)、[合成示例](../examples/README.md)、[解析约定](rule_parser_contract.md) | 三人消费方复核与实际读写器检查 | 全员，B 汇总 |
| P1 环境与工作区 | in_progress | [lcy 工程检查](handoffs/segmentation/pilot_adapter_20261112.md)：独立环境、sm_86 算子修复、GPU 锁、真实加载与两次更新/恢复 | 公开预训练加载、正式训练配置、图文环境；本轮非长训练验收 | lcy 协调，全员 |
| P2 真实 GT 小包 | in_progress | [wkq 远端交接](handoffs/data_eval/remote_pilot_review_20261112.md)：20 图、145 GT、19 测试通过 | 分割消费 20/145 已通过；6 组复核分歧、2 组已示答案、语言设计修订与其余验收；accepted=0 | B，全员标注 |
| P3 基线与评价器 | not_started | B0/B1/B2 设计 | 实现、可手算评价验证、真实试点输出 | A/B |
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
| lcy | 校验公开 COCO 权重与实际八类头加载，准备正式 trainer | 独立环境、数据适配、候选 API、短 GPU 恢复检查已完成 | 公开 checkpoint 文件/hash、冻结 train/dev、200-update 配置与预算 | [分割交接与证据](handoffs/segmentation/pilot_adapter_20261112.md) |
| wyz（A）/ grounding | 审核候选读取与共享解析，准备 B1/B2 | 合成包读取、缓存键/预测输出接口、独立编码器入口 | GT 结果需真实图片/查询，GPU 使用需排队 | 个人 PR；特征与预测检查；角色交接文件，均待创建 |
| wkq（B）/ data_eval | 修订语言试点，处理复核分歧；继续共享解析/B0/评价器 | 独立远端 CPU 环境、GT 小包与审计已运行 | 正式查询/消费方验收及全量划分未完成 | [远端交接](handoffs/data_eval/remote_pilot_review_20261112.md) |

无 GPU 不阻塞 CPU 接口开发。使用者可用 lcy/wyz/wkq 自报身份；对应角色已绑定，不根据作者顺序推断全名或账号。三人的工作区与代码分支继续遵守 [团队说明](team_workflow.md)。

## 3. 已确认与待确认

| 事项 | 当前值/状态 | 证据或补齐责任 |
| --- | --- | --- |
| 研究与暂定分工 | documented | README、三份角色卡和 workflow |
| 文件交接例及检查器 | locally_checked，仅合成夹具 | [规范补齐交接](handoffs/contract_readiness.md) |
| 模型代码/数据转换器 | 试点转换/复核与分割适配/候选/短 GPU 检查已实现 | 见 wkq/lcy 交接；正式 trainer、全量转换和统一模型评价器未完成 |
| 实验结果 | none_evaluated；工程运行已登记 | registry 保留 14 条 plan，追加 4 条随机初始化 smoke（3 failed / 1 completed）；性能指标均空 |
| GPU/环境/付费预算 | 分割短 GPU 检查通过；正式/长期训练未授权配置 | [本轮交接](handoffs/segmentation/pilot_adapter_20261112.md)；结束盘点系统/数据盘约 27/24 GB 可用，实际使用前重查 |
| 成员简称与角色 | lcy / wyz（A）/ wkq（B）已确认 | [绑定记录](handoffs/team_aliases.md) |
| wyz/wkq 的 GitHub/SSH 权限 | 待核验；lcy SSH 已验证 | 简称不是 GitHub 用户名，成员分别授权 |
| 公共权重 revision 与数据 manifest | 分割源码与 pilot manifest 已固定；COCO 权重文件未固定 | [分割来源记录](../envs/segmentation/upstream.json) 与交接 hash；正式数据及其他模型仍待补齐 |
| 课程实际截止与提交回执 | 未重新核验 | 提交负责人查课程系统；历史日期不是当前确认 |

## 4. 更新规则

每次工作交接在同一 PR 更新相关阶段/个人任务和证据链接，注明实际未验证项。状态与证据不一致时以可复核产物为准，先修正文档；不覆盖他人的完成记录。阶段 done 需对应 workflow 放行条件，训练开始不等于完成，代码完成不等于 GPU 验证。

历史交接文件保留，状态表只展示当前摘要。公开索引不放密码、私钥、token 或私有下载凭据；服务器地址与访问方式由受控渠道交接。
