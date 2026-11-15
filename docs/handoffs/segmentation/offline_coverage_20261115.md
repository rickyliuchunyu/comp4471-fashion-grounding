# 离线候选 coverage 与阈值选择入口

2026-11-15；lcy / segmentation，Codex 执行。**入口与新增专项验证完成；真实语言 dev 评价未运行，正式阈值未选择。** wkq 已发布 accepted dev 仍为 0，不能将缺失分母写成 coverage=0。此前分割 train/dev 的人工复核确认继续有效，本轮不重开分割冻结。

## 本轮实际进度

- 已 fetch 全部远端分支并读取 PR #1–#5 状态、活跃 PR 描述及评论/审阅。开工 lcy `6a6d5d9`；wkq `f5787f4`；wyz `103dcc5`；main `7064ebb`，均无队友增量。#2/#4 open/draft，#5 open/non-draft，base 为 data-eval；#1/#3 closed/unmerged。三个活跃 PR 的评论/审阅列表均为空。旧 data-eval 远端跟踪引用仍在，不将关闭 PR 误写为本轮已删除分支。
- 复用固定提交的 [wkq 大包交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/f5787f4ac7b6941abdadc43f9fbdc666eba93925/docs/handoffs/data_eval/large_dataset_20261112.md)及 [wyz 消费交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/103dcc546714dd7c7866b4259d96a9825d8090ef/docs/handoffs/grounding/accepted_baselines_and_weak_trial_20261112.md)。未发现 accepted dev、M2 或本人候选签收的新证据。
- 2026-11-15 14:52 HKT 服务器只读清点：仍为九个已发布目录；语言 dev 116 图/290 条，全为 `needs_revision`；所有发布包中的 accepted dev 共 0 条；旧 34 条 accepted 全部为 train。语言 RELEASE 仍为 `formal_freeze=false`。没有读取真实 mask、图片或 checkpoint。
- 系统盘约 27 GiB、数据盘约 163 GiB 空闲；GPU 0 MiB / 0%，无计算进程、无 running 协调记录。未执行 GPU 任务、候选重验、推理、AP 或训练；没有修改队友 checkout 和共享数据。

## 实现与复用边界

入口：[select_threshold.py](../../../src/fashion_grounding/segmentation/select_threshold.py)。运行配置：[candidate_u3000_coverage.json](../../../configs/runs/segmentation/candidate_u3000_coverage.json)，固定已选 u3000 checkpoint、源 dev manifest/COCO、阈值前 cache 及三个候选 JSON 的 SHA256。

共享匹配器 `evaluation/matching.py` 从 wkq `f5787f4` **逐字节引入同一路径**，没有复制出另一套关联算法或改变公共评价规则。来源及文件 SHA 见 [coverage_matcher_source.json](../../../configs/runs/segmentation/coverage_matcher_source.json)。当前分支仅接收这两个 evaluation 文件，不合并队友整个分支、不修改其模块。未来合并时相同文件内容可自然汇合。

执行流程：

1. 先检查 accepted dev 的独立审核、完整查询 ID、目标引用、split、冻结证据、精确分母和输入 hash。空查询、train/test、草稿、重复 ID、无独立 reviewer、缺冻结记录均拒绝，在打开候选 mask 前停止。
2. 只读取已有候选 JSON、必要的 RLE mask 和固定 COCO GT，按图处理；同图三个包重复引用的 mask 按 hash 复用解码。不加载 RGB、权重或特征编码器。
3. 对该图**全部同类非 crowd GT**执行既定分数/ID 排序的贪心一对一关联；不能只拿被查询的 GT 做匹配。关联仅写入评价结果，不回写预测包。
4. 按同一冻结查询清单计算 `matched_candidate_coverage`（分母是查询条数）及忽略类别约束的 `geometric_recall_at_0_5`。中英文各保留一个 query ID；零候选/推理失败仍在分母中。
5. 精确比较覆盖查询数；并列时比较**有查询的独立图片**上的候选总数，再并列选更高阈值。未被查询的 dev 图片不参与该计数，同图多条查询不重复计候选数。
6. 输出三档逐查询记录、关联、语言/类别/面积分组、比较 CSV 和独立 `selection.json`；不重写输入包的 `trial_unselected` 或原交付清单。下游以新的选择记录及其固定候选包 hash 为准。

输出目录不可覆盖；错误不会留下可消费的 `selection.json`。记录源码 commit/dirty 状态、入口及 matcher/schema 文件 hash、输入配置/冻结文件 hash、Python 版本与预算。内部预算默认 300 秒，服务器正式执行另加外层硬 timeout，超时输出不得作为完成结果消费。

## accepted dev 发布后的输入

[language_dev_freeze.template.json](../../../configs/runs/segmentation/language_dev_freeze.template.json)是**消费者侧运行回执模板**，目前 `formal_freeze=false`、确认信息与路径/hash 均为空；它不是 wkq 已发布的人工审核记录，也不要求修改其 schema-3 包。

接收 wkq 实际 accepted 发布物及人工审核/冻结记录后，在新的运行目录准备回执：

- `queries`：实际 accepted JSONL 的路径和原始字节 SHA256；不从草稿中自行筛选/修改 review_status。
- `image_manifest`、`annotations`：固定完整 `fp_dev250_v1` 的 manifest 和 COCO。新语言包若另带子集 manifest，先核实其图片/target 引用与这套固定 GT 一致，不换用另一套标注。
- `review_records`：实际人工复核/冻结证据的非空文件路径及 SHA256；`confirmed_by/confirmed_at` 来源于真实确认，不由 Codex 代签。程序检查引用/状态，不能自动证明人工语义判断正确。
- `scope.kind`：`full_dev` 或 `accepted_dev_subset`；`description` 明确语言覆盖范围，`query_count` 为全部查询数，`query_image_ids` 为完整升序唯一图片 ID。`full_dev` 要求所有 250 图都有查询；冻结子集则如实报告子集，不能冒称完整 dev250 语言评价。
- 只有真实证据满足后才将回执 `formal_freeze` 设为 true。模板不能直接运行，路径/hash 缺口不得编造。

这些回执和运行配置的引用是显式执行路径，可用绝对路径或相对回执路径；候选包内部 schema-3 引用仍沿用安全的包内相对路径。共享原始包保持不可变。

## 后续命令（本轮未运行真实评价）

在自己的干净 segmentation checkout 更新到含本入口的提交后执行；`FREEZE_RECEIPT` 必须指向上述真实填写的回执，`RUN_DIR` 为未存在的新输出目录。先按 AGENTS.md 检查 GitHub、资源和冻结增量。

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src \
timeout --signal=TERM --kill-after=5s 360s \
  /root/autodl-tmp/fashion-grounding/envs/segmentation-v1/bin/python \
  -m fashion_grounding.segmentation.select_threshold \
  --config configs/runs/segmentation/candidate_u3000_coverage.json \
  --freeze "$FREEZE_RECEIPT" --output "$RUN_DIR" --budget-seconds 300
```

退出码为 0、run_metadata.status=completed 且 selection.json 存在才可接收。真实运行后按三档结果追加 registry 记录并同步选择证据；不能将本文未执行命令或合成测试记为 dev 科学实验。

## 新增验证与停止条件

只执行 `tests/test_segmentation_threshold_selection.py` 的 10 项专项测试，最终 **10/10 通过，0.187 秒**。修改元信息记录与输出指标字段后仅重跑这份新增测试文件；没有运行旧 smoke、已有加载/恢复/候选导出套件。

覆盖：未查询 GT 的匹配竞争、同分/同 IoU ID 顺序、crowd 排除及 IoU=0.5 边界；错类几何重叠与 matched coverage 的区别；重复候选；同图双语分母与独立图片候选计数；空候选与全零 coverage 下的合法选择；三个 tie-break；查询/冻结/hash/图片版本门禁；破损 mask、不可覆盖输出、时间预算；完整 CLI 在主动阻止 torch/transformers/detectron2 导入且不存在 RGB 文件的合成夹具上成功，并保持输入文件不变。

**测试夹具中的 accepted 标志与 reviewer 是合成测试元数据，不是人工审核成果或项目成绩。** 实际三包仍复用 [既有交付与 wyz 接口证据](real_candidates_dev250_20261114.md)，没有重新读取其全量 mask 或裁剪。真实 coverage/阈值为 null；不追加实验 registry 行。

证据：

- [整组 GitHub 快照](evidence/offline_coverage_20261115/github_preflight.json)
- [服务器发布与资源核查摘要](evidence/offline_coverage_20261115/readiness.json)，使用[原只读检查脚本](evidence/language_dev_readiness_20261114/inspect_language_dev.py)
- [新增专项测试日志](evidence/offline_coverage_20261115/targeted_tests.log)
- [源码与验证摘要](evidence/offline_coverage_20261115/validation.json)

本轮同步 PR #2，保持 draft、未合并。下一步在 wkq 发布实际 accepted 且冻结的语言 dev 后运行本入口，并让 wyz 接收选定包开展图文排序评价。train5000 候选、完整 S0、M2、最终测试仍未宣告完成。
