# min_scale 单因素续训准备

2026-11-15；segmentation / lcy，Codex 执行。状态：**implemented / locally_checked，未启动训练、推理或服务器任务**。实现提交 `7c6ffce`。整组 GitHub、PR、评论/review 与交接核查见[快照](evidence/min_scale_preparation_20261115/github_preflight.json)：wkq `f5787f4`、wyz `103dcc5`、main `7064ebb` 无增量；自己的开工 head 为 `149ddf7`。未代做数据构建、排序训练或共享评价器修改。

## 固定对照与范围

| 项目 | A：直接复用已有结果 | B：本次准备，未运行 |
| --- | --- | --- |
| 起点 | 完整 u18000 checkpoint | 同一文件及 SHA256 |
| 更新区间 | 18000 → 20000，已有成功 2000 updates | 18000 → 20000，计划成功 2000 updates |
| 唯一科学变量 | min_scale=0.1 | min_scale=0.5 |
| 数据、seed、图像边长、max_scale | train5000 / dev250，seed42，768，2.0 | 完全相同 |
| 优化器、LR、scheduler、AMP 策略 | 已有 S0 AMP 修复版本 | 完全相同，不重置状态或学习率 |
| 评价 | 已保存完整 dev250 AP | 只在 B 的 20000 步评价一次完整 dev250 |
| 整体 mask AP / AP_small | 63.5595305 / 10.6641063，历史实测 | 未测，不填估计值 |

共同父权重：`/root/autodl-tmp/fashion-grounding/runs/segmentation/segmentation-s0-20000-20261115-r01/train/checkpoint_018000.pth`，SHA256 `207e2dbe1dcc8b9d0ef936d683d25af33c210347e34506914bbf6dad3426ee01`。父训练源码 `d8eb4ef`；A 实际执行源码 `e6637ff`。A 的 train_report、resume_check 和 AP 报告均已固定到阶段计划的文件 hash，确认其从同一父权重完整恢复六项状态并走完相同区间。

旧最佳 18k 的整体 AP=63.5791603，AP_small=10.7389194；当前 18k＋0.3 候选及 B1 联调交付不变。不下载权重，不重新跑 A 或旧最佳，不消费语言 dev，不引入 10k 数据。

## 新增入口和约束

- [resume_recipe.py](../../../src/fashion_grounding/segmentation/resume_recipe.py) 与 trainer `--resume-recipe`：独立于普通恢复、预算扩展和 AMP 修复入口；固定父权重、父/目标配置、源码 tree 与 A 源码。先复用历史 AMP 修复审计，再限定 A→B 科学差异仅为 min_scale 0.1→0.5。数据 hash、LR、batch、seed、其他增强及损失变化均拒绝。普通恢复规则未放宽。
- [B 配置](../../../configs/runs/segmentation/min_scale_05_20000.json)、[恢复记录](../../../configs/runs/segmentation/resume_recipe_18000_to_20000.json)、[阶段计划](../../../configs/runs/segmentation/min_scale_05_stage_plan.json)：model、optimizer、scheduler、scaler、sampler、RNG 原样交给现有恢复代码；运行时继续核对六项 fingerprint，不修改 checkpoint 元数据。新 lineage 单独保存为 `recipe_transition.json`。
- 与 A 相比的执行元数据变化仅为 `budget_seconds=2100`、新 `run_scope`、`checkpoint_updates=[20000]`；不延长 20000 上限。AP 直接复用原配置及实现，命令只列出 B20k 一份权重；原 AP 配置的历史 `run_scope` 不表示重评八点。
- [阶段入口](../../../scripts/run_min_scale_trial.py) 默认只输出准备方案，不创建运行目录、不连接服务器、不加载模型。显式 `--execute` 才检查至少 10 GiB 空闲并进入已有共享 GPU 锁；GPU 占用或未解决的 running 记录会拒绝启动。运行目录/协调记录不可覆盖，不自动排队、重试或续跑。
- **整个阶段硬上限 2700 秒（45 分钟）**，从入口开始计时，包含元数据核对、训练、dev 和离线比较；不是各阶段分别 45 分钟。训练内部上限 2100 秒；wrapper 与 worker 使用同一截止时间并预留清理时间。worker 自有进程组定时器在总截止前 10 秒结束自己的训练/评价进程组，外层另有截止兜底，不终止其他人的进程。失败保留日志与已有文件，不自动延长；未到 20k 或 dev 不完整时不出成功比较。
- [离线比较入口](../../../scripts/select_min_scale_trial.py) 核对 B 的训练完成状态、恢复审计、权重/预测 hash 和 250 图零错误评价；对 A/旧最佳仅读保存报告。输出 AP/AP50/AP75/AP_small/medium/large 和八类差值。选择仍按最高完整 dev mask AP，平分保留更早或已发布结果，不因只提高 AP_small 自动替换整体最佳。

## 判断提升与局限

B20k−A20k 是单因素对照；B20k−旧最佳18k 是是否更换交付权重的比较，两者分开记录。重点看 AP_small、AP75、shoe/accessory/bag 与整体 AP 是否同向改善；保留所有结果，不按一个好看的数字下结论。已通过的匹配诊断不重跑；若新 B 结果需要解释，再用现有离线工具分析新预测。

18k 后主干外 LR 约 `1e-6`、backbone 约 `1e-7`，两组均保持此值。这是单 seed、低 LR、2000 updates 的短期筛查；不能把很小的差值称为统计显著，也不能由阴性结果断言此增强永远无效。更大 LR、更多训练、更大数据均是另一个因素和后续决定，本轮不混入。

## 必要检查与证据复用

新增风险是配方放行范围、整阶段硬时间上限，以及不同配置身份下的完整 dev 选择。新增 [11 项 CPU 检查](../../../tests/test_min_scale_trial.py) 全通过（0.246 秒）：越界字段和来源拒绝、六状态不改写、旧 loader 数据门禁转交、AP 选择/完整性、只生成 B 训练和一次评价命令、失败停止、绝对截止及进程组要求。超时检查只启动 CPU sleep 子进程，无模型或优化步。首次测试的 `/tmp` 路径期望与 macOS `/private/tmp` 解析不一致，已用 `Path.resolve()` 修正。

提交后，默认准备入口以固定配置和已有 A 报告通过元数据审计，结果 `prepared_not_started`，源码 tree=`10539274179674320d1e2c5a74cbff3df6272185`。见[准备输出](evidence/min_scale_preparation_20261115/prepared_plan.json)、[测试日志](evidence/min_scale_preparation_20261115/new_logic_tests.log)、[验证记录](evidence/min_scale_preparation_20261115/verification.json)。日志里的 train_B 命令来自 mock，未执行。

复用 [S0 恢复及八点完整 dev](s0_recovery20000_20261115.md)、[小目标诊断](small_object_analysis_18000_20000_20261115.md)、[train5000 分布](train5000_distribution_20261115.md) 及原数据、COCO 加载、稳定性、AMP/恢复证据。未重跑原测试套件、模型 smoke、推理或 A 组。生产 checkpoint 本轮未加载；新增 B 的 GPU 路径仍待实际运行验证。未注册虚假的 completed 实验行。

## 后续启动方式（本轮未执行）

先重新核对整组 GitHub 增量、服务器磁盘/GPU/协调状态，将自己的服务器 checkout 同步到包含本准备的提交。保持队友工作区与环境不变；本轮没有读取服务器资源，旧空闲快照不能代替启动时检查。使用现有环境，不重装依赖。

在 `/root/autodl-tmp/fashion-grounding/workspaces/segmentation`：

```bash
PYTHONPATH=src /root/autodl-tmp/fashion-grounding/envs/segmentation-v1/bin/python \
  scripts/run_min_scale_trial.py \
  --plan configs/runs/segmentation/min_scale_05_stage_plan.json \
  --output /root/autodl-tmp/fashion-grounding/runs/segmentation/segmentation-min-scale05-20261115-r01
```

以上仍只准备。将来明确启动时才加 `--execute`；入口依次训练 B、评价 B20k 完整 dev250、离线比较历史 A 和旧最佳。45 分钟内未完成则结束并记录失败，不能自动改预算/换数据/重启。实际运行后再登记 run、指标与选择结论；准备状态不代表实测收益。
