# Dev AP 与 checkpoint 选择接入

新增：[B20k分辨率对照](../../docs/handoffs/segmentation/resolution1024_20261116.md)完成：固定B20k，只新增1024/1706完整dev250推理，复用800/1333结果；总137.859秒/900秒，按整体AP保留800。入口 `scripts/run_resolution_trial.py` 默认只审核元信息，`--execute` 才运行，旧输出不覆盖。

`--reuse-validated-inputs <completed-dev-report>` 允许在固定不可变包上复用既有验收；调用方必须固定报告SHA256。当前入口核对dev/train manifest和annotation身份、完整评价及分区回执，不重新逐图片/mask认证。推理正常读图；版本或身份变化立即拒绝，默认无此参数的完整验收路径保持不变。此复用不是对已发布图片可能被外部修改的新审计。

最新：[lcy最小B1联调](../../docs/handoffs/segmentation/u18000_b1_pipeline_20261115.md)已完成，290查询均输出原mask/框，选对178/290；不改队友方法，不代表全组对照完成。

最新：[用户授权当前语言dev的离线选择](../../docs/handoffs/segmentation/u18000_language_dev_selection_20261115.md)已选18k+0.3。使用显式`--user-authorized-dev`和固定授权回执；原accepted默认路径及共享匹配器不变，原始审核状态不改写。

最新：[正式S0八点dev选择](../../docs/handoffs/segmentation/s0_recovery20000_20261115.md)已完成，6000–20000每2000步完整dev250；选择18000步mask AP=63.5792，旧3000步报告AP=50.5603只读复用，提升13.0188点。20000步AP=63.5595，差距约0.02点不作为显著退化证据。

新增：[18k/20k小目标离线诊断](../../docs/handoffs/segmentation/small_object_analysis_18000_20000_20261115.md)只读取已有预测：138例分解、严格IoU匹配及类别宏平均解释已完成；不改变COCO指标或18k选择。

最新候选：[18000步dev250交付](../../docs/handoffs/segmentation/u18000_candidates_dev250_20261115.md)已完成。配置`configs/runs/segmentation/candidate_u18000_export.json`固定权重/训练来源，复用`export_candidates cache/export`，完整阈值前缓存和三个离线trial均已实际生成；wyz固定接口全量读取和裁剪通过。本轮未重算AP、未选语言阈值，不重跑已完成候选推理。

重要更新：[低AP诊断与修复](../../docs/handoffs/segmentation/low_ap_diagnosis_20261112.md)发现旧LSJ训练mask补边污染，已显式改为背景0并通过2个针对性CPU回归。旧50/100步及早期LSJ工程权重不得续训或导出生产候选；原AP/选择执行记录保留作故障证据。[修复后1000步训练与完整dev选择](../../docs/handoffs/segmentation/postfix1000_dev_ap_20261112.md)现已完成：500/1000步mask AP=35.3047/42.8038，选择1000步；不表示完整S0完成。

训练入口负责保存完整 checkpoint；`fashion_grounding.segmentation.evaluate` 独立读取这些 checkpoint 做只读评估，`fashion_grounding.segmentation.select_checkpoint` 读取完整 dev 报告选权重。两者不创建 optimizer、不继续训练，也不改写 checkpoint。当前未接入训练循环内定期评估 hook；可在保存边界退出训练、释放 GPU 后，按同一共享锁执行评估。普通恢复训练仍使用原训练源码及配置；预算扩展须使用上文显式固定的过渡记录。

本次[正式S0恢复](../../docs/handoffs/segmentation/s0_recovery20000_20261115.md)复用6000–18000七份权重，仅新增20000权重。独立评价器仍保持原实现；阶段结束后调用 `scripts/select_s0_recovery.py --output <本轮运行目录> --plan configs/runs/segmentation/s0_recovery_20000_stage_plan.json`。该审计入口固定七份旧权重hash、同预算AMP修复的父/目标身份及源码文件差异，核对完整dev250和原AP实现，再跨两个已核准身份组选权重并与旧3000比较。原普通selector的严格同配置/源码要求不变；旧3000只读既有报告，不重推理。

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

选择器核验报告、预测文件与原 checkpoint hash；只比较同一评估配置/数据、训练配置/数据、训练及评估源码的完整 dev 结果。最大 mask AP 胜出，完全同分选最早 update；重复 update、无正例、失败或 pilot 诊断均拒绝。输出 `selection.json` 引用实际 checkpoint，不另存 `best.pth`。[修复后实际选择](../../docs/handoffs/segmentation/postfix1000_dev_ap_20261112.md)已完成：第500/1000步完整dev AP为35.3047/42.8038，选择第1000步；使用 `s0_postfix_1000.json` 和 `dev_ap_250_postfix_1000.json`。历史50/100步权重已隔离，不再作为可用选择。上述为历史1000步选择；最新正式S0训练/AP和18000步候选交付已完成，见顶部交接。当前用户授权语言dev的阈值已选0.3；接收方实际排序联调仍待完成。
