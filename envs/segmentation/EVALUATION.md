# Dev AP 与 checkpoint 选择接入

重要更新：[低AP诊断与修复](../../docs/handoffs/segmentation/low_ap_diagnosis_20261112.md)发现旧LSJ训练mask补边污染，已显式改为背景0并通过2个针对性CPU回归。旧50/100步及早期LSJ工程权重不得续训或导出生产候选；原AP/选择执行记录保留作故障证据。[修复后1000步训练与完整dev选择](../../docs/handoffs/segmentation/postfix1000_dev_ap_20261112.md)现已完成：500/1000步mask AP=35.3047/42.8038，选择1000步；不表示完整S0完成。

训练入口负责保存完整 checkpoint；`fashion_grounding.segmentation.evaluate` 独立读取这些 checkpoint 做只读评估，`fashion_grounding.segmentation.select_checkpoint` 读取完整 dev 报告选权重。两者不创建 optimizer、不继续训练，也不改写 checkpoint。当前未接入训练循环内定期评估 hook；可在保存边界退出训练、释放 GPU 后，按同一共享锁执行评估。恢复训练仍使用原训练源码及配置。

## 评价条件

- 使用固定 Mask2Former 的完整 `model.forward -> instance_inference`，query×class top-100，mask 恢复原图尺寸；FP32，不加语言候选的 0.5 阈值、单 query argmax 或 NMS。
- 调用 pycocotools COCOeval，segm IoU 0.50–0.95、maxDets=[1,10,100]。结果只传 mask/class/score，避免零框或其他 bbox 覆盖 mask area。输出 AP/AP50/AP75、small/medium/large、逐类 AP，单位为百分数；无正例为 null（N/A）。
- 每张 GT 图片必须有结果行，空输出保留；推理失败保存部分记录后失败退出，不发布可选权重的完整报告。CPU 评分接口也保留显式失败图片的分母，并标记 incomplete。
- 校验完整图片集合、原图 hash/尺寸、COCO mask/面积/框、八类映射及 checkpoint SHA256/model 指纹。训练图片 manifest/annotation hash 必须与 checkpoint 一致。
- `dev_ap.json` 是正式配置模板，数据 hash 为空时拒绝执行。须由 wkq 发布并冻结分割 train/dev 后，填入实际 dev hash。hash 防止数据漂移，本身不证明人工冻结、官方来源边界或近重复复核完成。
- 正式路径只接受 split=dev，禁止 test；与 checkpoint 的 train manifest 按图片 ID、文件 SHA256、source_image_id、group_id 检查交集。该检查不能发现尚未标记为同组的近重复图。工程 pilot checkpoint 不能进入正式选择。
- `pilot_ap_integration.json` 只接受现有 train 小包，报告 `selection_eligible=false`。这是用训练图片检查 AP 接口，不是 dev 结果；不能据此选 S0 权重。

## 有界只读检查命令

先检查磁盘、GPU、coordination 占用；从干净的 segmentation checkout 运行，使用新的 RUN_ID。源训练 checkpoint 位于先前的稳定性检查目录，不复制或覆盖。

```bash
export FG_ROOT=/root/autodl-tmp/fashion-grounding
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
SEG_PY="$FG_ROOT/envs/segmentation-v1/bin/python"
RUN_ID=segmentation-ap-integration-YYYYMMDD-rNN
RUN="$FG_ROOT/runs/segmentation/$RUN_ID"
TRAIN_RUN="$FG_ROOT/runs/segmentation/segmentation-stability-200-20261112-r02"
"$SEG_PY" scripts/with_gpu_lock.py --coordination "$FG_ROOT/coordination" \
  --run-id "$RUN_ID" --output "$RUN" --seconds 300 -- \
  "$SEG_PY" -m fashion_grounding.segmentation.evaluate \
  --package "$FG_ROOT/shared/processed/fp_pilot20_v2" \
  --training-package "$FG_ROOT/shared/processed/fp_pilot20_v2" \
  --source "$FG_ROOT/vendor/segmentation/Mask2Former" \
  --config configs/runs/segmentation/pilot_ap_integration.json --output "$RUN" \
  --checkpoints "$TRAIN_RUN/updates_001_100/checkpoint_000100.pth" \
                "$TRAIN_RUN/updates_101_200/checkpoint_000200.pth" \
  --export-diagnostic-candidates
```

每个 update 子目录保存原始 AP 预测、COCO 摘要、评价报告及 hash。可选诊断候选另走语言 decoder，带 `illustrative_only=true`，图片 manifest 也同步此标记；provenance 保留源图片 manifest hash，源包和旧产物均不改写。

## wyz 读取器验收

使用 wyz 已发布 commit `66c0da7e066c1235ee618ea7015171e73b82c914` 中的 `grounding/inputs.py` 原文，临时加载，不合并分支、不改队友工作区。只需要其 Git 对象可读取。Python 3.10+ 运行：

```bash
"$SEG_PY" scripts/check_grounding_consumer.py \
  --grounding-repo "$FG_ROOT/workspaces/grounding" \
  --candidates "$RUN/update_000100/candidate_package/candidates_predicted.json" \
               "$RUN/update_000200/candidate_package/candidates_predicted.json" \
  --output "$RUN/consumer_check"
```

检查 manifest/图片/mask hash、原图坐标、候选顺序、RGB 裁剪，并确认诊断包默认被拒绝、显式 opt-in 后才可读。此检查不加载 SigLIP，不证明语言选择准确率或预测候选已经正式验收。

## 正式选择入口

冻结数据与实际 S0 checkpoint 就绪后，换正式 dev 配置和包执行 evaluate，保存每个 checkpoint 的报告，再调用：

```bash
"$SEG_PY" -m fashion_grounding.segmentation.select_checkpoint \
  --reports <dev-run/update_002000/evaluation_report.json> \
            <dev-run/update_004000/evaluation_report.json> \
  --output <new-selection-directory>
```

选择器核验报告、预测文件与原 checkpoint hash；只比较同一评估配置/数据、训练配置/数据、训练及评估源码的完整 dev 结果。最大 mask AP 胜出，完全同分选最早 update；重复 update、无正例、失败或 pilot 诊断均拒绝。输出 `selection.json` 引用实际 checkpoint，不另存 `best.pth`。[修复后实际选择](../../docs/handoffs/segmentation/postfix1000_dev_ap_20261112.md)已完成：第500/1000步完整dev AP为35.3047/42.8038，选择第1000步；使用 `s0_postfix_1000.json` 和 `dev_ap_250_postfix_1000.json`。历史50/100步权重已隔离，不再作为可用选择。完整 S0、语言候选阈值选择和生产候选下游联调仍待完成。
