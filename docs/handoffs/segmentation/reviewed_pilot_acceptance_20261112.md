# lcy 接收 wkq 已审核训练小包

2026-11-12，segmentation / lcy，分支 `segmentation/pilot-adapter-v1` / PR #2。
实现与实际运行 commit：`cb1e77f3723c7c26daf0ab1806e3a1d30f1fea3e`。

**结论：`fp_grounded9_reviewed_v1` 的 lcy 分割消费接口验收通过，可按相同 schema 继续扩充数据。** 此结论覆盖文件完整性、图片/实例加载、类别映射和最小模型训练/恢复链路；不替代标注语义质量复核、wyz 的语言模型验收或正式 train/dev 冻结。上一轮未就绪结论针对正式 dev 选择，不阻塞本轮训练小包接口验收。

## 上游与输入

开工重新 fetch 全组远端；wkq 最新仍为 `66dd07c` / PR #4，wyz 为 `66c0da7`，没有新分支或 PR；PR #4 讨论为空。读取用户提供的服务器交接文件，与 [wkq 固定 GitHub 版本](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/66dd07c56fb02ae90af7a12de3524650d2fe6455/docs/handoffs/data_eval/p3_reviewed_baseline_20261112.md)一致。

数据位置：`shared/processed/fp_grounded9_reviewed_v1`，9 张 train 图片、59 个实例/GT 候选；34 条 accepted 双语记录。使用已有共享 schema 与自己的 segmentation loader 消费文件，没有合并/更改 wkq 分支、发布包或环境。

固定输入 hash 见 [配置](../../../configs/runs/segmentation/reviewed_pilot_acceptance.json)：

- image manifest：`1729076fac3d27862388ed8187a62ba7f89f90a1383873b81c73f0abf2590e2e`
- annotations：`552e0eb7942d706a85fc72f56d387a2b378991b76c57ed573cace55a728f7d17`
- GT candidates：`44969ca41c595d0a0c4ad32321ddceef677d0754b16d09eda6693e10b89be30f`
- accepted queries：`dda04845dfa71962a86bc94fccc0ff6d580dc27561a3001a777e9e4c94ae111a`

## 实际验收

| 检查 | 结果 |
| --- | --- |
| 发布文件校验 | `SHA256SUMS.json` 中 84 个文件全部匹配；运行后复查仍一致 |
| schema / 原图 / GT | 9 图、59 实例与候选；图片尺寸/RGB、COCO/RLE 面积/框及候选关联检查通过 |
| 类别映射 | 项目 1–8 → 内部 0–7；逐类实例数 5/5/2/10/2/14/9/12；crowd=0 |
| segmentation loader + mapper | 9 图全部加载，短边 256 / 长边 384 映射后保留 59 个实例 |
| GPU 前向/反向 | 图 4360、9080 各一次更新，每步 30 项 loss 均有限；梯度 norm 有限且非零 |
| 保存恢复 | 第 1 步保存，重新构建模型/optimizer/scheduler/scaler 后加载；严格模型加载、分类头相等和 scheduler/scaler 检查通过，继续完成第 2 步 |
| 独立 CPU checkpoint 检查 | 最终 update=2、scheduler=2，全部有状态 Adam 参数 step=2，model/optimizer tensor 均有限，文件 SHA256 一致 |
| 推理与输出 | 图 4360 推理得到 0 候选，保留 `no_candidates`；诊断包由 wyz 固定 reader 显式 opt-in 后成功读取 |

两个 total loss 为 131.3991 / 91.4899，裁剪前 norm 为 1382.3330 / 216.5582。它们来自不同图片，不能解释为收敛或性能收益。随机初始化、不加载 COCO，仅两次工程更新；AMP scale=1、full-model norm clip=0.01，复用原有 smoke 入口，没有执行正式 trainer 或长训练。

运行 ID：`segmentation-reviewed-pilot-20261112-r01`，GPU 锁上限 180 秒，实际占锁 **12.67 秒**，模型检查内部约 8.01 秒。peak allocated / reserved 为 **1.55 / 2.57 GiB**。本次保存恢复在同一进程重建对象，不能称为新的跨进程恢复验证；先前 200-update 跨进程检查有独立记录。

证据：[CPU 包接收](evidence/reviewed_pilot_20261112/package_acceptance.json)、[模型检查](evidence/reviewed_pilot_20261112/smoke_report.json)、[最终 CPU 检查](evidence/reviewed_pilot_20261112/final_cpu_check.json)、[GPU 协调记录与精确命令](evidence/reviewed_pilot_20261112/segmentation-reviewed-pilot-20261112-r01.json)、[输出读取检查](evidence/reviewed_pilot_20261112/consumer_check/consumer_report.json)。

## 复现

自己的 segmentation checkout / `envs/segmentation-v1`，使用全新输出路径。先核对磁盘、GPU、共享锁，CPU 数据检查：

```bash
export FG_ROOT=/root/autodl-tmp/fashion-grounding
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
SEG_PY="$FG_ROOT/envs/segmentation-v1/bin/python"
"$SEG_PY" scripts/check_segmentation_package.py \
  --package "$FG_ROOT/shared/processed/fp_grounded9_reviewed_v1" \
  --config configs/runs/segmentation/reviewed_pilot_acceptance.json \
  --output "$FG_ROOT/runs/segmentation/reviewed-pilot-data-NEW"
```

CPU 校验成功后，通过 `scripts/with_gpu_lock.py --seconds 180` 包装 `python -m fashion_grounding.segmentation.smoke`，参数与实际协调记录一致，只将 run ID/output 换为新值。模型环境及实现未改变，本轮使用真实包检查，不为这次接收重复全套旧单元测试。

## 给 wkq 的交接边界

从 lcy 分割接入角度，小包格式已可消费，可以继续按当前格式扩充；新版包须独立路径/version/hash，不覆盖此版本。仍保留 image13898/q014 的待修订状态及旧图答案暴露记录，本次没有将待审查询升级 accepted，也没有人工重评源 mask 的语义质量。

wyz 的新包 SigLIP/B1/B2 消费验收仍需由其完成；本轮仅额外用其 reader 检查了本次空预测包格式，不能代替其模型验收。正式 dev AP/权重选择仍等待冻结分割 train/dev 与对应短训练保存点，完整长训练未启动。

[结束盘点](evidence/reviewed_pilot_20261112/postflight.log)：系统盘约剩 27 GiB、数据盘约剩 170 GiB；GPU 0 MiB/0%，无计算进程。本轮 checkpoint 保留在自己的 run 目录，原数据及他人工作区/环境/进程未更改。状态表与 registry 同步，仅追加自己的工程 run，正式性能字段留空。
