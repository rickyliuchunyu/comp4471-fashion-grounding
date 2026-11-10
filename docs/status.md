# 当前进度与下一步

更新：2026-11-10。本表反映仓库中可核查的证据，不推断服务器或成员电脑上的未交接工作。路线见 [P0–P10](workflow.md)，运行见 [registry](../experiments/registry.csv)，交接见 [handoffs](handoffs/README.md)。

**当前可开始 P0 接口复核与 P1/P2 的准备工作。** 研究计划、角色任务卡和交接规范已建立；真实数据转换、查询复核、模型训练、评价和演示仍未实现/未验证。AutoDL 是已选平台，实例创建、账号访问和预算尚待确认。组员 A/B 身份待填写。

## 1. 阶段状态

状态：`not_started` 无执行证据；`in_progress` 有明确活动与产物；`blocked` 已开始但受阻；`ready_for_review` 待接收方验收；`done` 满足该阶段放行条件。文档已写不等于该阶段 done。

| 阶段 | 当前状态 | 已有证据 | 尚缺的放行证据 | 主责 |
| --- | --- | --- | --- | --- |
| P0 接口确认 | in_progress | [文件契约](artifact_contract.md)、[合成示例](../examples/README.md)、[解析约定](rule_parser_contract.md) | 三人消费方复核与实际读写器检查 | 全员，B 汇总 |
| P1 环境与工作区 | not_started | [AutoDL 规划](team_workflow.md#7-autodl-与多-codex-同时工作) | 实际环境清单、独立工作区、模型小批检查 | Liu Chunyu 协调，全员 |
| P2 真实 GT 小包 | not_started | [数据协议](data_protocol.md)；合成例不计入 | 真实图片、mask、复核查询及接收方验证 | B，全员标注 |
| P3 基线与评价器 | not_started | B0/B1/B2 设计 | 实现、可手算评价验证、真实试点输出 | A/B |
| P4 数据扩充冻结 | not_started | 数量与划分规则 | 实际分区、哈希、复核与重复组报告 | B，全员 |
| P5-S 分割训练 | not_started | S0 规划配置 | 可恢复权重、dev 结果和固定候选 | Liu Chunyu |
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
| Liu Chunyu | 审核候选接口，准备分割环境核查 | 复核 mask/框/ID/解码约定、核对公开实现与计划配置 | GPU 小批验证需已授权实例及 B 的真实小包 | 个人 PR；实际环境记录；角色交接文件，均待创建 |
| A / grounding | 审核候选读取与共享解析，准备 B1/B2 | 合成包读取、缓存键/预测输出接口、独立编码器入口 | GT 结果需真实图片/查询，GPU 使用需排队 | 个人 PR；特征与预测检查；角色交接文件，均待创建 |
| B / data_eval | 实现公共读写器、共享解析与真实小包转换 | 将契约落成代码、验证 RLE 与指标边界、安排标注试点 | 正式小包需取得原始数据；人工复核需另一成员 | 个人 PR；转换报告/版本 manifest；交接文件，均待创建 |

无 GPU 不阻塞 CPU 接口开发。A/B 实名与账号未绑定时，由使用者自报角色，不根据作者顺序猜测。三人的工作区与代码分支继续遵守 [团队说明](team_workflow.md)。

## 3. 已确认与待确认

| 事项 | 当前值/状态 | 证据或补齐责任 |
| --- | --- | --- |
| 研究与暂定分工 | documented | README、三份角色卡和 workflow |
| 文件交接例及检查器 | locally_checked，仅合成夹具 | [规范补齐交接](handoffs/contract_readiness.md) |
| 正式模型运行代码/数据转换器 | not_implemented | 无 src 运行入口；示例检查器不替代它们 |
| 实验结果 | none_recorded | registry 14 条计划，指标为空 |
| GPU/环境/付费预算 | unverified | Liu Chunyu 在取得实例后记录；不公开认证信息 |
| A/B 姓名、GitHub/SSH 权限 | unconfirmed | 成员确认，受控渠道授权 |
| 公共权重 revision 与数据 manifest | not_pinned | 对应方法/数据负责人执行时补齐 |
| 课程实际截止与提交回执 | 未重新核验 | 提交负责人查课程系统；历史日期不是当前确认 |

## 4. 更新规则

每次工作交接在同一 PR 更新相关阶段/个人任务和证据链接，注明实际未验证项。状态与证据不一致时以可复核产物为准，先修正文档；不覆盖他人的完成记录。阶段 done 需对应 workflow 放行条件，训练开始不等于完成，代码完成不等于 GPU 验证。

历史交接文件保留，状态表只展示当前摘要。公开索引不放密码、私钥、token 或私有下载凭据；服务器地址与访问方式由受控渠道交接。
