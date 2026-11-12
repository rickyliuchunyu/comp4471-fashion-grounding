# 冻结分割 train/dev 就绪核查

lcy / segmentation，2026-11-12 17:05 HKT。此次仅检查发布状态；没有新增模型运行或训练，不追加实验 registry 行。原始摘要见 [核查证据](evidence/freeze_readiness_20261112.json)。

## 结论

**尚未发现 wkq 发布的冻结分割 train/dev，不能启动真实 dev AP 与 checkpoint 选择。** 已检查整组 GitHub 远端分支、四份 PR 及讨论、最新交接，以及服务器 `shared/processed` 发布包；不推断队友尚未发布的工作。

- wkq 最新仍是 `data-eval/p3-baseline-evaluation` @ `66dd07c56fb02ae90af7a12de3524650d2fe6455` / [PR #4](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/4)。[该提交的交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/66dd07c56fb02ae90af7a12de3524650d2fe6455/docs/handoffs/data_eval/p3_reviewed_baseline_20261112.md)明确：P4 分区、扩充、全量重复审查未完成。
- wyz 分支仍为 `66c0da7`，本次未发现对应 PR；lcy 为 `754d543`。main 为 `7064ebb`。PR #1/#2/#3/#4 均 open、draft、未合并，讨论/评审记录为空。
- 服务器已发布包的实际图片 manifest 如下。语言 accepted 状态不能替代分割分区与冻结验收。

| 发布包 | 图片 | 分割实例 | 实际 split |
| --- | --- | --- | --- |
| fp_pilot20_v2 | 20 | 145 | train: 20 |
| fp_grounded9_draft_v2 | 9 | 59 | train: 9 |
| fp_grounded9_reviewed_v1 | 9 | 59 | train: 9 |

reviewed_v1 的 17 组/34 条 accepted 是训练语言查询；本次未重新裁决人类审核。三个包不代表互不重叠的数据，不能将图片数量相加作为训练集规模，也不能临时将其中一个改标为 dev。

## 后续接收条件

wkq 需发布版本化的分割 train/dev 图片 manifest 与 COCO 标注、明确冻结记录及 hash，并附官方来源边界、图片/group/hash 不重叠与近重复复核证据。接收时核验八类映射、全部图片和 mask 几何，再填写 dev 配置；不从训练试点或最终 test 临时抽取 dev。

另一个前提是可比较的 checkpoint：现有第 100/200 步权重来自 `pilot_200_update_stability`，正式选择入口会拒绝。冻结数据到位后，可按预登记的有界短训练配置产生匹配该 train 的保存点，用真实 dev 检查选择流程；不需要把完整 20k 长训练作为此小阶段的前提。当前未启动这项短训练，也未更改现有选择门禁或将 pilot 权重改标为 S0。

## 服务器与执行范围

系统盘约剩 27 GiB，数据盘约剩 171 GiB；RTX 3090 为 0 MiB / 0%，无计算进程，coordination 无 unresolved running 记录。资源并非当前阻塞，冻结数据尚未发布才是原因。本轮未获取 GPU 运行锁、未启动 AP、未选择权重、未训练；未修改任何共享数据、他人工作区或环境。
