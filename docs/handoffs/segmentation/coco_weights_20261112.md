# COCO 权重校验、八类适配与五图短推理

任务 / 阶段：P1 公开权重与实际推理检查；角色 / 成员：segmentation / lcy；状态：`gpu_verified`，未进行性能评价。分支 `segmentation/pilot-adapter-v1`，沿用 [draft PR #2](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/2)。执行代码 commit：`c54b292ca119f33e8eb57113157151c3d27fd5f4`；本交接提交见 Git 历史。

## 来源、传输和完整性

- 官方 [固定 commit 的 Model Zoo](https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/MODEL_ZOO.md)：COCO instance segmentation、R50、50 epochs，model ID `47430277_2`，许可证 CC-BY-NC-4.0。源码与配置固定在 [upstream.json](../../../envs/segmentation/upstream.json)。
- 官方 HTTPS 下载端点在本机和服务器均报告证书过期；没有关闭 TLS 校验或修改系统时间。服务器直连镜像失败（Network is unreachable）。
- 本机通过正常 HTTPS 从 [固定镜像 revision](https://huggingface.co/egeozsoy/MM-OR/tree/9052e46f2c770601e17ebd797f2505cd54efaea3/panoptic_segmentation/mask2former) 下载，再通过 SSH 传到服务器。镜像不是 Meta 官方托管；完整 SHA256 来自镜像 LFS 元数据，不是发布者签名。
- 文件 `model_final_3c8ec9.pkl`：176,382,166 bytes；SHA256 `c18e6551f9c0e951e7ee3a63eb5402b901612e4e141965d4719d13c69a97ec2a`；MD5 `3c8ec9998b5769ed00abd9c5e51fcefb`。本机与服务器完全一致；MD5 前缀符合官方 Model Zoo 文件名规则。
- 权重位置：`shared/weights/mask2former-coco-r50-3c8ec9/model_final_3c8ec9.pkl`，相对于 `FG_ROOT=/root/autodl-tmp/fashion-grounding`。不提交大文件到 Git。
- [下载记录](evidence/coco_weights_20261112/download.json) 与 [服务器解析/加载记录](evidence/coco_weights_20261112/load_report.json) 保留校验值。读取器先校验同一份内存字节，再通过 numpy/OrderedDict 窄白名单解析，拒绝任意其他 pickle globals、非 float32 或非有限数组。`__author__` 仅是格式字段，不是认证证据。

## 实际加载与八类头

实际 checkpoint 含 558 个 state tensors。按固定上游的 v1→v2 规则将 `sem_seg_head.predictor.static_query.weight` 精确重命名为 `query_feat.weight`；没有模糊名称匹配。其他键必须完整一致，非分类头形状、dtype 和数值均检查。

| 项目 | 官方 COCO | 项目八类 | 实际处理 |
| --- | --- | --- | --- |
| class_embed.weight | 81 × 256 | 9 × 256 | 保留 seed 42 的 Xavier 初始化 |
| class_embed.bias | 81 | 9 | 保留零初始化 |
| criterion.empty_weight | 81 | 9 | 保留项目缓冲区；双方前景权重 1、no-object 权重 0.1 均核验 |
| 其他 tensors | 555 个 | 555 个 | 全部加载，并逐项 `torch.equal` 证明数值一致 |

9 个输出含 8 个服饰类别及 no-object；没有把 COCO 类别索引映射为服饰类别。保存 `adapted_initialization.pth` 后，在全新模型中 `strict=True` 重新加载并逐张量复核。该文件 SHA256 为 `4fc1c106e61dd8f72e09afa81af86ebcad2825ea0207d49996247c75a9e1d836`，与原始 COCO 文件哈希分开记录；候选 provenance 使用适配后的哈希。

## 输入与短推理结果

run ID：`segmentation-coco-inference-20261112-r01`。现有真实包 `shared/processed/fp_pilot20_v2` 的 image manifest SHA256：`1b1ec9f25c37cd7b910dfe2882cc33637be37f3fba32e6f7b0c32592b570f111`。按 image ID 升序固定取前五张 Fashionpedia train 图，逐图核验图片哈希及尺寸；本入口不读取 annotations 或 queries。

配置见 [coco_inference.json](../../../configs/runs/segmentation/coco_inference.json)：FP32、单图、短边 256 / 长边上限 384、阈值 0.5、不做 NMS、最多 100 candidates。未扫描或调节阈值。五张图恰好都是竖图，其他长宽比尚未在真实权重上验证。

| image ID | 原图 H × W | logits | 候选数 |
| --- | --- | --- | --- |
| 1643 | 1024 × 682 | 100 × 9，有限 | 43 |
| 3165 | 1024 × 683 | 100 × 9，有限 | 34 |
| 11778 | 1024 × 683 | 100 × 9，有限 | 48 |
| 13061 | 1024 × 683 | 100 × 9，有限 | 10 |
| 13898 | 1024 × 682 | 100 × 9，有限 | 19 |

全部 154 个预测 mask 的原图尺寸、排他 xyxy、RLE 编码/解码往返及 mask SHA256 检查通过；模型 state 在推理前后逐项不变。**八类分类头尚未训练，即使候选得分较高也不代表服饰类别正确。** 候选包 `illustrative_only=true`，只能验收接口，不得用于正式语言方法比较或当作 S0 效果。

检查主体耗时 6.91 秒（含 CPU 加载、保存/恢复、GPU 推理及导出），锁包装进程共 9.67 秒；5 次推理分别约 0.658 / 0.135 / 0.167 / 0.094 / 0.113 秒。首图含冷启动，未形成正式延迟指标。峰值 CUDA allocated 1,125,975,552 bytes（约 1.05 GiB），reserved 1,891,631,104 bytes（约 1.76 GiB）。

启动前 RTX 3090 为 0 MiB、无计算进程；系统盘约 27 GiB、数据盘约 24 GiB 可用。结束后 GPU 仍为 0 MiB、无计算进程；系统盘约 27 GiB、数据盘约 21.3 GiB 可用。本轮权重目录约 169 MiB，运行目录约 170 MiB；盘点是共享盘瞬时值，不将全部空间变化归因于本轮。见 [前盘点](evidence/coco_weights_20261112/preflight.log)、[后盘点](evidence/coco_weights_20261112/postflight.log)。

## 复现与证据

独立环境 `envs/segmentation-v1`，源码 `vendor/segmentation/Mask2Former`；[本轮环境](evidence/coco_weights_20261112/environment.json) 与 [原环境建立说明](../../../envs/segmentation/README.md)。本机和服务器 `PYTHONPATH=src python -m unittest discover -s tests -v` 均 31 项通过，包括完整性失败先于反序列化、非法 globals、旧 query 格式、loss buffer/no-object 权重和 dtype 拒绝检查；[服务器测试日志](evidence/coco_weights_20261112/tests.log)。

从自己的 checkout 执行；`RUN_ID` 必须换成新值。GPU 锁仍是 advisory，其他成员须采用相同入口。

```bash
export FG_ROOT=/root/autodl-tmp/fashion-grounding
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
SEG_PY="$FG_ROOT/envs/segmentation-v1/bin/python"
RUN_ID=segmentation-coco-inference-20261112-r01
RUN="$FG_ROOT/runs/segmentation/$RUN_ID"
"$SEG_PY" scripts/with_gpu_lock.py --coordination "$FG_ROOT/coordination" \
  --run-id "$RUN_ID" --output "$RUN" --seconds 180 -- \
  "$SEG_PY" -m fashion_grounding.segmentation.pretrained_check \
  --package "$FG_ROOT/shared/processed/fp_pilot20_v2" \
  --source "$FG_ROOT/vendor/segmentation/Mask2Former" \
  --checkpoint "$FG_ROOT/shared/weights/mask2former-coco-r50-3c8ec9/model_final_3c8ec9.pkl" \
  --output "$RUN"
```

完整产物在 `runs/segmentation/segmentation-coco-inference-20261112-r01/`：`adapted_initialization.pth`、`candidate_package/`、`input/`、模型与 resolved 配置、加载和推理报告。轻量证据见 [run_metadata.json](evidence/coco_weights_20261112/run_metadata.json)、[inference_report.json](evidence/coco_weights_20261112/inference_report.json)、[原始运行日志](evidence/coco_weights_20261112/run.log)；文件哈希见 [artifact_hashes.json](evidence/coco_weights_20261112/artifact_hashes.json)。

## 范围与下一步

本轮没有 optimizer、backward、参数更新或长训练，没有 AP、语言选择指标、dev/test 评价或模型选择。未改动旧项目、其他成员的环境/进程或共享数据。原来的随机初始化 smoke 保持原语义和历史证据。

P1 公开权重加载子项通过，但正式 trainer、冻结 train/dev 与训练配置/预算仍待完成；P5-S 保持 not_started。下游 grounding / data_eval 可验收这个非正式候选包的读取、尺寸和 provenance；接收方尚未复核。状态表与 registry 追加本次 completed 工程运行，所有性能指标留空。
