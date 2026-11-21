# u3000 正式候选入口与阈值前缓存

2026-11-14，lcy / segmentation；Codex 执行。实现提交 `c2c0645af62c455f9dae481f02a197d89768aa40`，本 PR 保持 draft、未合并。

**正式导出入口已实现，11 项新增 CPU 检查通过；本轮没有加载真实 checkpoint、运行模型推理或追加训练。真实 u3000 候选缓存/payload 尚未生成，正式语言阈值仍未选择。** 当前选中权重仍为第 3000 步，完整分割 dev mask AP=50.5603136（既有结果，非本轮测量）。

## 开工核对与验证边界

[GitHub/资源快照](evidence/candidate_export_20261114/github_preflight.json)：main `7064ebb`；lcy 开工 `ce3c865` / PR #2；wkq `f5787f4` / PR #4；wyz `103dcc5` / PR #5。#2/#4 draft，#5 base 为 data-eval 分支；#1/#3 closed/unmerged，无新增 PR 或队友提交。活跃 PR 的评论/审阅返回为空。队友交接按相同 commit 增量复用，重新读取 wyz 的 `inputs.py` 接口，未改队友 checkout。语言 accepted dev / M2 的缺口没有新增完成证据。

服务器开工系统盘约 27 GiB、数据盘约 163 GiB 空闲；GPU 0 MiB / 0%，无计算进程、无 running 协调记录。本轮只在独立测试目录复用 segmentation-v1 环境，设置 `CUDA_VISIBLE_DEVICES=""`，没有升级依赖或触碰既有训练运行。

复用 [COCO/八类加载](coco_weights_20261112.md)、[有效续训与选择](resume5000_dev_ap_20261113.md)、[数据接收](large_input_acceptance_20261112.md)及前次消费接口证据。未改原 `predict_instances` / `decode_candidates`、模型、训练、AP 路径；新风险只有缓存序列化、正式入口约束和离线过滤，因此仅测新增测试文件，不重复 smoke、全套单测或真实图片推理。

## 入口与输出

- [入口](../../../src/fashion_grounding/segmentation/export_candidates.py)：`python -m fashion_grounding.segmentation.export_candidates cache|export`。
- [固定配置](../../../configs/runs/segmentation/candidate_u3000_export.json)：锁定 u3000 SHA256 `e56eb7f46fab899d83f668b1878776db50e13cc273c4d4a7fbdc786692444148`、训练 config/source/data 指纹、原 decoder 源码指纹及 train5000/dev250 manifest；输入只允许这两套 train/dev。分割选重依据保持不变。
- `cache`：未来获授权后使用 GPU，strict 加载选中模型；只读 image manifest / RGB 像素，不读取 annotations 或 queries。一次处理一张图，调用既有 decoder，`score_threshold=0, max_candidates=100`，要求原输出恰为 100×9 类 logits、100×H×W mask logits。
- `export`：只读已完成缓存，在 CPU 上导出 schema 3 `candidates_predicted.json`、manifest、图像、RLE、resolved config 和运行元信息；不导入 PyTorch / model，不读取权重。命令必须显式选择 0.3、0.5 或 0.7，标记 `trial_unselected` / `selected_threshold=null`。这不是正式阈值选择。

缓存结构：`cache.json` + `cache.sha256`、`resolved_decoder_config.json`、`image_manifest.json`、逐图 `records/<image_id>.json`、`masks/<image_id>-<query_id>.json`、`images/*` 与元信息。原尺寸掩膜逐图写盘，不累积全数据集像素 mask。导出最多持有全包的候选元数据与单张 mask，不加载整包 mask 数组。

真实 `decoder_query_id` 从既有 decoder 的原 query 索引直接保存，ID 始终为 `image-{image_id}-query-{qid:03d}`。每 query 最多一个前景类别，过滤前保留全部非空 mask（包括 0 分），固定丢弃空 mask 并记录数量；候选按 score 降序、ID 升序。过滤只删候选，不重新编号，不按 AP 输出位置/相同二值 mask 猜 query 身份。原图 RLE、排他 xyxy 与 `source_annotation_id=null` 保持契约。

阈值比较沿用原 float32 tensor 的语义；尤其 float32 的 0.7 存为 Python float 时略低于十进制 0.7，不能直接用 Python 双精度阈值把边界候选错删。导出记录 `effective_score_cutoff`。缓存是**固定解码之后、score 阈值之前**的候选缓存，不含原始 logits；只能离线改注册 score 阈值。改变模型、输入、缩放、类别/mask 解码或 NMS，必须明确新版本，不能复用旧缓存。

完整覆盖所有 manifest 图片后才生成缓存完成标记；缺图、失败、超时或中断的部分目录不具备可导出完整缓存。空候选图片保留 `no_candidates`，推理异常留下失败记录并抛出，不伪装为空候选。离线校验 root/record/image/mask/config hash、query 唯一性、mask/框几何、权重及输入版本；失败不会发布最终 `candidates_predicted.json`。已存在目录拒绝覆盖；旧共享 preparation 包保持原样。

## 新增必要验证（已执行，停止点已满足）

[11 项测试日志](evidence/candidate_export_20261114/tests.log) / [实际命令、源码及输入指纹](evidence/candidate_export_20261114/run_metadata.json)。仅 `tests/test_segmentation_candidate_export.py`，1.870 秒，11 passed / 0 skipped。服务器目录：`runs/segmentation/segmentation-candidate-export-tests-20261114-r01`。

覆盖稀疏真实 query 索引与同分排序、float32 0.7 边界、逐阈值与旧 decoder 的合成 logits 对照、空图、逐图调度、失败/缺图/超时、hash 篡改/错权重、重复 ID/错误几何/非 100 queries、test 与路径穿越拒绝、训练元信息约束、目录不可覆盖，以及离线 CLI 不导入 torch/model。测试输入为 4×6 合成图、人工 fixture 候选或 logits，没有模型 forward。

wyz `103dcc5` 的真实 `CandidatePack(..., allow_illustrative=True).load()` 和 `crop_regions()` 已消费新格式的合成导出，保留 3 个候选与空图；只证明新增导出格式兼容，**不等同于 wyz 本人验收真实 u3000 大包**。全部通过即停止，未扩大检查范围。单测不新增 S0 科学实验 registry 行，也没有新增 AP/语言性能成绩。

## 后续复现命令（本轮未执行）

以下 GPU 命令是后续操作示例，不能作为本轮执行证据。运行前仍需核对最新全组状态/资源、获得对应推理范围与预算授权；使用包含本提交的干净 segmentation checkout。示例仅 dev250，硬上限 1800 秒，内部 1700 秒；train5000 需单独决定预算与运行 ID，不能沿用并默认扩大。

```bash
FG_ROOT=/root/autodl-tmp/fashion-grounding
PY="$FG_ROOT/envs/segmentation-v1/bin/python"
CFG=configs/runs/segmentation/candidate_u3000_export.json
RUN_ID=segmentation-u3000-dev-candidates-20261114-r01
CACHE_DIR="$FG_ROOT/runs/segmentation/$RUN_ID/cache"
mkdir "$FG_ROOT/runs/segmentation/$RUN_ID"
PYTHONPATH="$PWD/src" "$PY" scripts/with_gpu_lock.py \
  --coordination "$FG_ROOT/coordination" --run-id "$RUN_ID" \
  --output "$CACHE_DIR" --seconds 1800 -- \
  "$PY" -m fashion_grounding.segmentation.export_candidates cache \
  --config "$CFG" --split dev --package "$FG_ROOT/shared/processed/fp_dev250_v1" \
  --checkpoint "$FG_ROOT/runs/segmentation/segmentation-resume5000-train-20261113-r01/checkpoint_003000.pth" \
  --source "$FG_ROOT/vendor/segmentation/Mask2Former" \
  --run-id "$RUN_ID" --output "$CACHE_DIR" --budget-seconds 1700 \
  > "$FG_ROOT/runs/segmentation/$RUN_ID/cache.log" 2>&1

# 仅在上述缓存完整成功后；以下不再 forward，每个阈值输出到新目录。
for THRESHOLD in 0.3 0.5 0.7; do
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$PY" \
    -m fashion_grounding.segmentation.export_candidates export \
    --config "$CFG" --cache "$CACHE_DIR" --threshold "$THRESHOLD" \
    --candidate-set-id "u3000-dev-trial-$THRESHOLD" \
    --output "$FG_ROOT/runs/segmentation/$RUN_ID/trial-$THRESHOLD"
done
```

缓存没有自动续跑入口；中断后保留原部分目录，诊断后另定新 run ID，不把部分缓存冒充完成或重用同目录覆盖。GPU 锁与进程组硬终止复用 `with_gpu_lock.py`，内部逐图预算检查不能单独代替硬超时。

下一步是按后续授权范围生成一次真实候选缓存，供离线阈值试验与 A/B 接收。正式 threshold 仍须已冻结 accepted 语言 dev，按既定 matched query coverage → 较少候选数 → 较高阈值规则选择。当前语言草稿不能当 accepted，现有 AP 预测不能还原真实 query ID，亦不因本入口完成而追加训练。
