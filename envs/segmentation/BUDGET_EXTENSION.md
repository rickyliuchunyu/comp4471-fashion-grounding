# 有记录的预算扩展续训

普通 `--resume` 仍要求完整配置 hash 和源码 commit 一致。预算扩展另需 `--resume-extension <JSON>`，不能用一个宽松的兼容开关跳过来源检查。

扩展记录固定父 checkpoint SHA256、父配置 SHA256、父源码 commit、已完成步数、目标配置 SHA256，以及父/目标 `src/fashion_grounding` Git tree。运行时核对父提交是当前提交的祖先，并核对两份源码 tree；这记录的是一次明确审查的源码过渡，不代表任意改动都能恢复。

允许变化仅为 `max_updates`、`budget_seconds`、`checkpoint_interval`、`checkpoint_updates` 和 `run_scope`。总步数必须增加，且实际差异必须与记录中的 `allowed_changes` 完全相符。数据、学习率/warmup/衰减、优化器、batch、seed、AMP、增广等必须保持一致；错误父权重、配置或源码直接拒绝。

`checkpoint_updates` 是严格递增、无重复、处于预算内的绝对 update 列表；存在时替代间隔保存，最终停止点仍保存。既有无此字段的配置继续按 `checkpoint_interval` 工作。

恢复沿用既有完整状态加载与六组件指纹核对，不重新载入 COCO、不重新初始化分类头，也不重置学习率、采样器或 RNG。输出 `budget_extension.json`，并在后续 checkpoint 和训练报告的 `resume_lineage` 中保留父权重与配置/源码过渡。实际恢复检查来自这次必要续训，不另起一次恢复 smoke。

## 本次已授权的扩展

- 父运行：`segmentation-postfix1000-train-20261112-r01/checkpoint_001000.pth`，SHA256 `ffd502fd566edbd4a716c7668f8763a752acefb5fb088642e9e7d05cfece64e6`，源 `89797754eb7442b76bf88e2dbfaa23136d90ee13`；旧补边污染权重不在允许范围。
- 训练配置：[s0_resume_5000.json](../../configs/runs/segmentation/s0_resume_5000.json)。从1000续至5000，只新增4000 updates，保存2000/3000/5000，训练硬上限2100秒。
- 扩展记录：[resume_1000_to_5000.json](../../configs/runs/segmentation/resume_1000_to_5000.json)。源码过渡仅新增扩展检查、保存点列表及来源记录；训练损失/反传/优化更新逻辑未变。
- dev配置：[dev_ap_250_resume_5000.json](../../configs/runs/segmentation/dev_ap_250_resume_5000.json)，三份完整250图dev合计上限600秒。沿用既有AP与选择入口；不重跑父1000步AP，不混入旧受污染模型。

示例参数（仍须使用共享GPU锁、干净运行源码及独立输出目录）：

```text
python -m fashion_grounding.segmentation.train
  --package <root>/shared/processed/fp_train5000_v1
  --source <root>/vendor/segmentation/Mask2Former
  --resume <root>/runs/segmentation/segmentation-postfix1000-train-20261112-r01/checkpoint_001000.pth
  --resume-extension configs/runs/segmentation/resume_1000_to_5000.json
  --config configs/runs/segmentation/s0_resume_5000.json
  --output <new-run-directory>
```

新增6个针对性CPU测试在原服务器环境通过，覆盖允许/拒绝边界、真实Git来源过渡、扩展后学习率/优化器/RNG及轨迹连续性；未重跑原测试套件或GPU smoke。CPU轨迹相同不扩大为GPU算子跨进程bitwise确定性。原数据接收、COCO来源、LSJ修复、数值稳定与评价证据继续复用。

实际运行是否完成、AP和停止状态以本次交接/registry为准；本功能不授权20k完整S0、其他seed或test。
