# Segmentation 环境与最小入口

lcy，2026-11-12。实际结果见 [随机 smoke 交接](../../docs/handoffs/segmentation/pilot_adapter_20261112.md) 与 [COCO 加载/五图推理](../../docs/handoffs/segmentation/coco_weights_20261112.md)。可复用训练入口与配置现已实现，见 [TRAINING.md](TRAINING.md)；[200-update 稳定性与恢复](../../docs/handoffs/segmentation/stability_200_20261112.md)已通过。完整 S0 训练及 dev 性能验收尚未执行。

## 独立运行环境

数据盘上的 `envs/segmentation-v1` 是 Python 3.10 venv：关闭 system-site-packages，依赖文件从旧 base 复制，复制后的旧项目 editable `.pth` 已禁用。torch、Detectron2 与自定义算子均从新 venv 加载，旧环境未安装或升级包。Python 可执行文件/标准库仍以 `/root/miniconda3` 为基础；这不是脱离原机器的完整可移植环境。

`conda create --offline --copy --clone` 首次失败：镜像不包含完整包缓存。失败副本已清理，失败日志保留于 `runs/segmentation/env-20261112/clone.log`。最终采用 venv 加文件复制；依赖快照与自定义编译日志在同一目录。这个快照是已观测环境，不能把 `pip freeze` 当作跨机器成功重建的证明。

源码使用独立 clone：`vendor/segmentation/Mask2Former`，固定 [upstream.json](upstream.json) 的 commit。CUDA Toolkit 12.1、PyTorch 2.1.2+cu121；复制来的旧扩展能导入但缺少 3090 的 sm_86 kernel，必须在**自己的环境/源码目录**重编译：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6 FORCE_CUDA=1 MAX_JOBS=2
cd "$FG_ROOT/vendor/segmentation/Mask2Former/mask2former/modeling/pixel_decoder/ops"
"$FG_ROOT/envs/segmentation-v1/bin/python" setup.py build_ext \
  --build-lib "$FG_ROOT/runs/segmentation/env-20261112/build-lib" \
  --build-temp "$FG_ROOT/runs/segmentation/env-20261112/build-temp"
# 将 build-lib 中的 MultiScaleDeformableAttention*.so 复制到自己的 venv site-packages。
```

`FG_ROOT=/root/autodl-tmp/fashion-grounding`。上述命令不带服务器地址或认证信息；跨机器使用时先核对 GPU compute capability，不能盲用 8.6。

## CPU 与短 GPU 检查

从自己的课程 checkout 根目录执行；每次 `RUN` 必须是新路径，入口拒绝覆盖。

```bash
export FG_ROOT=/root/autodl-tmp/fashion-grounding
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
SEG_PY="$FG_ROOT/envs/segmentation-v1/bin/python"
"$SEG_PY" -m unittest discover -s tests -v
"$SEG_PY" -m fashion_grounding.segmentation.smoke --data-only \
  --package "$FG_ROOT/shared/processed/fp_pilot20_v2" \
  --source "$FG_ROOT/vendor/segmentation/Mask2Former" --output "$RUN"
```

GPU 必须使用共用 `coordination/gpu.lock`；锁内检查计算进程、显存及未解决的 running 记录。锁是 advisory，其他角色仍须采用相同入口；现有 tmux 或自写脚本不会自动遵守它。CPU 数据工作不持锁。检查运行保存角色、成员、commit、PID、预算、输出与退出状态；异常只终止包装器自身启动的进程组。

```bash
"$SEG_PY" scripts/with_gpu_lock.py --coordination "$FG_ROOT/coordination" \
  --run-id "$RUN_ID" --output "$RUN" --seconds 180 -- \
  "$SEG_PY" -m fashion_grounding.segmentation.smoke \
  --package "$FG_ROOT/shared/processed/fp_pilot20_v2" \
  --source "$FG_ROOT/vendor/segmentation/Mask2Former" --output "$RUN"
```

运行配置：[pilot_smoke.json](../../configs/runs/segmentation/pilot_smoke.json)。固定两个 optimizer updates，单图 batch，短边 256、长边上限 384，随机初始化，AMP 初始 scale=128，种子 42；第一个 update 后重建模型/优化器并恢复，再执行第二个 update。保留完整官方 R50 架构、100 queries、10 次预测和点采样损失。这个小检查的 resize/StepLR 不代表规划中的 LSJ768/正式 warmup 训练。

真实输入必须为 Fashionpedia train；图片 hash、尺寸、COCO mask/面积/框与八类映射先检查，crowd 不进入训练。模型推理 API 只接收 RGB 像素和推理配置。公开 COCO 权重已通过固定镜像下载（官方端点证书过期），本机/服务器 SHA256 与官方文件名 MD5 前缀均核验；实际八类适配与五图短推理通过。原 `smoke` 入口仍只做随机初始化检查。独立 `pretrained_check` 入口不构造 optimizer，完成 555 tensors 加载一致性、旧 query 名转换、分类头/loss buffer 保留、适配权重严格恢复与原图 RLE 往返校验；完整命令见 COCO 交接。

## 候选接口

`predict_instances(image_rgb, model, config)` 将 decoder mask logits 先去 padding 并恢复原图尺寸，再用 `>0` 产生 mask。九类 softmax 含 no-object，八类前景中选一个类别；分数为类别概率乘 mask 前景平均 sigmoid（分母含官方 1e-6），阈值 0.5、最多 100、不做 NMS。

`export_package` 输出 schema 3、压缩 COCO RLE、原图排他 xyxy、稳定 query ID、严格同序候选和文件 SHA256。每图必须存在：正常空集 `no_candidates`，错误为带脱敏 code/message 的 `inference_error`。生产包不含 GT 目标 ID，拒绝覆盖旧输出。

随机初始化 smoke 的候选包强制 `illustrative_only=true`，即使输入图片是真实图。它只能检查格式，不能用于语言方法的正式比较。COCO 短推理的八类头尚未训练，其候选同样强制 illustrative_only=true；正式模型候选仍需训练及 dev 选择；标准 COCO AP 仍保留官方 instance inference，未由本入口计算。
