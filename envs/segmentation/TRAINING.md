# 分割训练入口与配置

入口：`python -m fashion_grounding.segmentation.train`。使用固定 Mask2Former R50 架构及其 Hungarian matcher、30 项主/辅助损失和点采样实现；训练循环负责可恢复采样、梯度累计、AdamW、AMP、梯度检查、调度、日志及 checkpoint。它不执行 dev AP 或 checkpoint 选择；完整 S0 的训练预算、冻结 train/dev 与评价接入仍需单独完成。

## 配置与执行范围

- [s0_train.json](../../configs/runs/segmentation/s0_train.json)：20k-update 的训练配置模板，LSJ 768、随机水平翻转、scale 0.1–2.0、有效 batch 2、AdamW LR 1e-4、backbone 0.1 倍、weight decay 0.05；norm/embedding 的 decay 按上游设为 0。前 500 updates 线性 warmup，从 0.001 倍 LR 起步，12k/16k 乘 0.1，full-model norm clip 0.01。
- [pilot_stability_200.json](../../configs/runs/segmentation/pilot_stability_200.json)：继承同一训练配方，只固定现有 pilot 数据 hash、将总更新数限为 200、checkpoint 间隔改为 100、每进程时限 900 秒。外层共享锁的 900 秒限制覆盖两个进程总时长。
- AMP 初始 scale 为 1。LSJ 首次试跑使用 128，在完成 6 updates 后出现非有限梯度；失败记录保留。训练器对非有限 loss/梯度、AMP 跳步均 fail-fast，不静默增加成功 update 计数或自动重试。
- 入口必须显式指定 `--config`；正式配置的两个 data hash 初始为空，未填写实际冻结训练包哈希时拒绝运行。配置模板不是已执行的完整训练。

样本按 image ID 排序后，用独立有状态 generator 打乱每个 epoch。数据读取与增广在主进程执行，无 worker 预取；恢复时不会丢失尚未使用的批次。RLE mask 使用与图片相同的上游 LSJ 几何变换；裁剪后空 mask 被移除，空目标图片继续进入分类损失，不重新抽样。输入始终为 Fashionpedia train，不读取查询或 dev/test。

`microbatch_images × accumulation_steps = effective_batch_images`，每个 microbatch 的损失除以 accumulation_steps 后反传，每个有效 update 仅做一次 unscale、全模型梯度裁剪、optimizer step 和 scheduler step。默认 microbatch 2、accumulation 1；microbatch 1 / accumulation 2 是显存不足时的可配置方案，尚未在本轮真实模型上验证，也不宣称 mask 损失归一化与物理 batch 2 完全等价。

## 断点与日志

checkpoint 保存 model、optimizer、scheduler、GradScaler、sampler 的 permutation/epoch/position/generator、Python/numpy/CPU/CUDA RNG、已完成 update、配置 hash、数据 hash、源码 commit 与原始 COCO 加载记录。先写临时文件并 fsync，再原子发布新文件和 SHA256 sidecar；旧 checkpoint 不覆盖。

恢复先验证文件 hash 和配置/数据/源码一致性，再严格加载模型，最后逐组件比较与 checkpoint 中一致的数值指纹。`--resume` 不重新载入 COCO 初始化，不重置 warmup。恢复要求 checkout 为运行时记录的源码 commit；仅有文档更新的最新 commit 也不能直接混入这次恢复，应先使用原执行 commit 的干净独立 checkout。

`metrics.jsonl` 每条为一个真正完成的 optimizer update，记录 30 项 weighted loss、total loss、裁剪前后 norm、AMP scale、实际 LR、图片 ID、增广后实例数、采样 epoch、用时及 allocated/reserved 显存。`gradient_failure.json`（若发生）记录未执行 optimizer step 的失败位置及非有限梯度参数。checkpoint 保存在 update 边界，不保存未完成的累计梯度。

GPU 自定义算子未证明跨进程 bitwise deterministic；真实 GPU 验收比较完整恢复状态，然后继续更新。CPU 小模型另行验证中断恢复与连续训练的后续 loss/参数完全一致，不能把 CPU 结果扩大为 GPU 轨迹完全一致。

## 有界 200-update 检查

从自己的干净 checkout 运行，先核查磁盘和 GPU；`RUN_ID` 必须是新值。下面只启动短稳定性检查，协调器在第 100 次更新正常保存/退出，再启动全新进程恢复至第 200 次。

```bash
export FG_ROOT=/root/autodl-tmp/fashion-grounding
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
SEG_PY="$FG_ROOT/envs/segmentation-v1/bin/python"
RUN_ID=segmentation-stability-200-YYYYMMDD-rNN
RUN="$FG_ROOT/runs/segmentation/$RUN_ID"
"$SEG_PY" scripts/with_gpu_lock.py --coordination "$FG_ROOT/coordination" \
  --run-id "$RUN_ID" --output "$RUN" --seconds 900 -- \
  "$SEG_PY" scripts/check_segmentation_stability.py \
  --package "$FG_ROOT/shared/processed/fp_pilot20_v2" \
  --source "$FG_ROOT/vendor/segmentation/Mask2Former" \
  --checkpoint "$FG_ROOT/shared/weights/mask2former-coco-r50-3c8ec9/model_final_3c8ec9.pkl" \
  --output "$RUN"
```

协调器把实际的两条 trainer 命令写入 `commands.json`，日志分为 `phase_1.log`、`phase_2.log`；训练产物分为 `updates_001_100/`、`updates_101_200/`。手动恢复时须用同样的共享锁、原配置和 `--resume <checkpoint.pth>`，`--output` 指向新目录；`--stop-after` 是绝对 update 编号，并受配置的 max_updates 限制。失败目录保留，新尝试另用 run ID。

当前训练 checkpoint 仅用于工程稳定性验证，没有正式 dev 选择；不能当作已验收的 S0 发布权重。正式训练应先落实冻结数据、dev evaluator/选择记录、计算预算，再独立登记运行。
