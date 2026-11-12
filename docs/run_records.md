# 实验系列与实际运行记录

2026-11-10。[registry.csv](../experiments/registry.csv) 保留 14 条 `record_type=plan`。2026-11-12 新增 4 条分割 P1 工程 smoke run（随机初始化，3 failed / 1 completed）；它们不是正式 S0 训练或性能结果，所有性能指标仍为空，见 [交接](handoffs/segmentation/pilot_adapter_20261112.md)。

## 1. 一行代表什么

- `record_type=plan`：实验系列定义，experiment_id=S0/B1/M2 等；run_id、实际成员、数据 hash、测量值保持空。
- `record_type=run`：一个方法、seed、split、候选条件及固定配置的实际执行记录。不同 GT/predicted、dev/test、seed 或重试分别登记。训练 run 可 split=train；评价 run 必须 dev/test，并引用训练权重。
- 唯一 `run_id` 示例为 `M2-dev-predicted-s42-r01`，实际执行前确认不重名；重跑增加 r02，不覆盖旧产物。D1/D2 派生分析使用自己的 run ID，在 metadata 中列出来源运行。
- status 可为 queued/running/completed/failed/interrupted；completed 只说明该项执行结束，不自动说明评价完成或达到预期。无实验结果时不填 0。

计划行不改成运行行；运行人在启动时追加自己的 run 行，结束时只更新该行状态与产物引用。运行目录、原始日志与失败证据不可覆盖；状态修订用 Git 历史和 metadata 的 status_history 留痕。汇总统计另存文件，不伪装成某个 seed 的运行。

## 2. 必须区分的列

| 字段 | 规则 |
| --- | --- |
| experiment_id / owner / executed_by | 系列编号、角色 ID、实际执行成员简称 lcy/wyz/wkq；只有真实运行才填写 executed_by，不能因角色绑定预填计划行 |
| record_type / run_id / status | 计划或运行、唯一运行 ID、实际状态 |
| split / candidate_source | train/dev/test；gt/predicted/external/none，none 用于不依赖候选的训练/分割 AP |
| git_commit / config_path / metadata_path | 源码 commit、真正 resolved config、运行元信息索引；规划 JSON 不作为已解析运行配置 |
| data_manifest_sha256 / query_manifest_sha256 / candidate_manifest_sha256 | 对应文件 hash；不适用留空，适用时不得省略 |
| seed / weights_origin / run_dir | 实际 seed、权重来源、独立输出目录；公开索引不放凭据 |
| oracle_selection_accuracy | 仅 GT 候选诊断，值 0–1 |
| predicted_selection_accuracy / conditional_selection_accuracy / matched_candidate_coverage | 仅内部预测候选，值 0–1；coverage=0 时 conditional 留空并在 metadata 写原因 |
| geometric_recall_at_0_5 / mean_target_mask_iou / success_at_0_5 | 按指标定义填 0–1；不适用留空。E1 不填内部选择准确率 |
| mask_ap | COCO evaluator 的 0–100 标度；其他 AP 指标存 metrics 文件，不能混用 0–1 |
| mean_latency_ms / p95_latency_ms | 毫秒；首次/缓存/冷启动场景分别记录在 metadata/metrics，禁止混合平均 |

旧 `target_selection_accuracy` 列为兼容保留，**新 run 必须留空**，改用明确的 oracle/predicted 字段。CSV 空白表示未测或不适用，原因写 metadata；JSON 用 null，禁止 NaN。所有失败仍由全查询预测清单进入评价分母，不从 registry 的 completed 行数计算准确率。

## 3. 运行元信息与结果文件

实际运行生成 `run_metadata.json`，最小结构如下；它是模板，null 必须根据实际情况填写或注明不适用，不能直接当成完成记录。

```json
{
  "record_schema_version": 1,
  "run_id": null,
  "experiment_id": null,
  "owner": null,
  "executed_by": null,
  "status": "queued",
  "status_history": [],
  "git_commit": null,
  "split": null,
  "candidate_source": null,
  "seed": null,
  "artifacts": {},
  "parent_run_ids": [],
  "checkpoint": null,
  "environment": null,
  "command": null,
  "evaluation": null,
  "failure": null
}
```

`artifacts` 至少按任务列出 resolved_config、image_manifest、query_manifest、candidate_manifest、predictions、metrics、logs；每项为 `{path,sha256}`，不适用则 null 并附原因。公开仓库可保存不含凭据的受控存储逻辑路径，实际访问方式另行交接。checkpoint 包含来源/revision/hash，environment 引用真实依赖清单和硬件记录，command 是实际执行命令。

`evaluation` 包含 evaluator commit、查询总数、实际/缺失/失败记录数、分组定义、指标标度与 N/A 原因；分组指标另附图片数/查询数。metrics 文件分别保存 GT、预测或 E1 条件，混合条件不写进同一 registry 行。计划统计预算不是实际重复次数。

## 4. 状态与交接

项目级当前任务在 [状态表](status.md)，运行证据在 registry/metadata，交接使用 [模板](handoffs/README.md)。B 汇总但不替其他人宣告运行完成；方法负责人核对自己的日志与预测。无完整运行入口时写“尚不可运行”，不能填写未经执行的复现命令作为验证证据。
