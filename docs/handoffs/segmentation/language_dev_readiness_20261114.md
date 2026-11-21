# 语言 dev 就绪核对：尚无 accepted 查询

2026-11-14 02:48 HKT（服务器记录 2026-11-13 18:48 UTC）；lcy / segmentation，Codex 执行。用户本轮授权为“若 accepted 语言 dev 就绪，使用现有三个候选包计算 matched query coverage 和选择阈值”。**前提未满足，因此未执行匹配/选择；不把空分母记成 coverage=0，selected_threshold 仍为 null。**

## 最新证据

已 fetch 并检查全部远端分支、PR 清单、活跃 PR 描述及评论/审阅。lcy 开工 head `39260b7` / #2 draft；wkq `f5787f4` / #4 draft；wyz `103dcc5` / #5（base data-eval）；main `7064ebb`。#1/#3 closed/unmerged；无新 PR、队友提交或活跃 PR 评论/审阅。队友相同 commit 的交接按增量复用，重新读取与语言 dev 有关的段落。

- wkq [大型数据交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/f5787f4ac7b6941abdadc43f9fbdc666eba93925/docs/handoffs/data_eval/large_dataset_20261112.md)仍明确语言 dev 116 图/290 条草稿，accepted=0，余下 134 图待补写。
- wyz [最新交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/103dcc546714dd7c7866b4259d96a9825d8090ef/docs/handoffs/grounding/accepted_baselines_and_weak_trial_20261112.md)仍未报告 accepted dev 或 M2 完成。
- 服务器 `shared/processed` 仍只有既有九个发布目录，没有新的语言 dev 包；只读清点发布目录和查询/RELEASE 元信息，没有加载图片或 mask。

| 已发布内容 | 本轮实际读取结果 |
| --- | --- |
| `fp_language_dev_draft_v2/queries_draft.jsonl` | 116 图、290 条；全部 split=dev、review_status=needs_revision |
| `fp_language_dev_draft_v2/queries_accepted.jsonl` | 0 字节、0 条 |
| `fp_language_dev_draft_v2/RELEASE.json` | accepted_language_queries=0；formal_freeze=false；Development draft only |
| `fp_dev250_v1/queries_accepted.jsonl` | 0 字节、0 条 |
| 旧 `fp_grounded9_reviewed_v1` accepted 文件 | 34 条、8 图，全部 split=train，不能代替 dev |
| 所有已发布 accepted 文件中的 accepted dev 条数 | 0 |

dev draft SHA256：`fa1419b5799fb6c5c5d0b49dc76f8014300b0c59ce7361007627c2da8fd1c395`；dev RELEASE SHA256：`27188eba64d1e869c58b6773d8d5b7a01f0cde039cff10e9f5c4bf7ed970bdca`。accepted 空文件 SHA256：`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

此前用户确认的分割 train/dev 人工复核继续有效，本次不重开分割冻结或重复数据验收；它与语言查询 accepted/冻结是两项独立条件。

## 复用与停止点

[u3000 dev250 真实候选交付](real_candidates_dev250_20261114.md)继续有效：三个包 0.3/0.5/0.7 的候选数为 1,565/1,258/980，已完成真实 query ID、全量读取与裁剪验证。本轮不重跑候选校验、导出、推理、AP、训练或任何旧 smoke/单测；不改查询审核状态或队友 checkout。服务器磁盘约 27/163 GiB 空闲，GPU 0 MiB/0%，无计算进程或 running 协调记录。

本轮只有就绪核查，没有产生新的科学实验或性能数据，不追加 registry run 行。matched query coverage 未计算，正式阈值未选择；不使用草稿、旧 accepted train 或原始 AP recall 替代冻结 dev 查询。

## 下一次的放行条件与执行规则

wkq 发布经过人工审核的 dev 查询清单及对应复核/冻结记录，固定包路径、query SHA256、图片与 GT annotation 版本和本次评价分母。需明确最终冻结的查询/图片范围；若是经确认的子集，应如实报告其覆盖范围，不能冒称完整 dev250 语言评价。机器初标或 Codex 视觉修改不能自动升级为 accepted。

就绪后只补新查询清单的必要版本/引用/审核兼容检查，复用现有三包和已有匹配器证据。依 [experiment_plan.md 的阈值选择与一对一关联规则](../../experiment_plan.md)，同一冻结查询分母、同类一对一关联，先最大化覆盖查询数；并列时最小化有查询的独立 dev 图片上的候选总数，再并列取更高阈值。零候选图片保留；报告 geometric recall、三档计数和选择理由。无需新的模型推理或训练。

## 证据与复现

- [GitHub / 分支 / 资源快照](evidence/language_dev_readiness_20261114/github_preflight.json)
- [服务器发布状态和查询计数、hash](evidence/language_dev_readiness_20261114/readiness.json)
- [仅元信息的只读检查脚本](evidence/language_dev_readiness_20261114/inspect_language_dev.py)：通过 SSH 将脚本作为标准输入交给现有 segmentation-v1 Python 执行，JSON 输出保存到本地；没有向共享数据包写入内容。

状态表、README 和本交接随 PR #2 同步；本 PR 保持 draft、未合并。本记录只说明检查时的已发布状态，不推断 wkq 尚未发布的工作进度。
