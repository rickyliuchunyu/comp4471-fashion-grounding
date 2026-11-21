# 定向 train10000：分割消费方增量验收

2026-11-16，lcy / segmentation。**技术验收 passed，训练对照仅准备、未启动。** 新增 5000 图 / 21924 实例通过现有分割读取器；旧 5k 保留、新增数据跨分区隔离通过本轮增量核对。没有加载模型、运行推理、训练或 smoke。此结论不替代三方正式冻结，也不宣告分割效果提升。

## 版本与开工核对

- 开工 lcy `e6f6f37`；wkq `4b0584ac04ce088084580bcaf946b74575fc6c0c`；wyz `103dcc546714dd7c7866b4259d96a9825d8090ef`；main `7064ebb`。整组远端分支、全部五个 PR、活跃评论/review 及增量交接已核对，与上次快照相比无队友增量。#2/#4 open/draft；#5 open/non-draft，base 为 wkq 分支；#1/#3 closed-unmerged，均未合并，无新讨论。[GitHub 记录](evidence/train10000_acceptance_20261116/github_preflight.json)。
- 接收依据：[wkq 固定交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/4b0584ac04ce088084580bcaf946b74575fc6c0c/docs/handoffs/data_eval/train10000_targeted_20261116.md)。生产源 `aadcd1b5fc37f74630cf842232390932da363c3a`，最终 mirror4 发布包；未接收其已废弃 staging。
- 开始服务器根盘空闲约 26.9 GiB、数据盘约 153.5 GiB；RTX3090 0 MiB/0%，无计算进程。CPU 任务不占 GPU 锁；未动队友 checkout、环境或数据。前后快照见证据目录；结束仍空闲、无 active/running/queued 协调记录。

输入根目录 `/root/autodl-tmp/fashion-grounding/shared/processed/`：新包 `fp_train10000_targeted_v1`，旧包 `fp_train5000_v1`，保护包 `fp_dev250_v1`、`fp_test400_holdout_v1`。

| 新包文件 | SHA256 |
| --- | --- |
| image_manifest.json | `9966383f85e1a4223a261f023eb80359b945815dff42ad4965c66c26225250d0` |
| annotations_coco.json | `780b55fd3965606b41efeee62a7e708fcd45d3bd114b198d70f813f9f93f9f54` |
| SHA256SUMS.json | `572b6d509f6cd759e697f1aae0280ded91776bc81a3126d2699b50c5d99462e3` |

其余输入 pin 和生产审计绑定见 [接收配置](../../../configs/runs/segmentation/train10000_acceptance.json) 与 [真实验收报告](evidence/train10000_acceptance_20261116/acceptance.json)。报告中的 `added_loader` 两个 hash 属于临时“仅新增 5k”元数据视图，**不是正式 10k 训练指纹**；训练必须使用上表整包指纹。

## 最小检查与证据复用

新增风险是新图片/标注、合并时旧记录被改写、以及新增图与保留分区泄漏。旧权重、类别头、训练器、AMP/恢复及 dev 评价算法没有因此失效，不重跑这些验证。

1. **新包身份与旧子集。** 固定三份发布 hash；逐项比较旧 5000 个 manifest/COCO 图片记录和 17775 个标注记录，包括 ID、源字段、mask RLE、框、area、属性、group。新旧成员清单互斥且恰好覆盖整包，八类/mapping 不变。22775 份旧图片/mask 的清单 hash 与原包一致；旧资产实际字节一致性复用已绑定的生产方 preservation audit，不再次读取旧图片或解码旧 mask。
2. **新增分割消费。** 在本角色输出目录临时生成只含新增数据的元数据视图，图片用硬链接引用原文件，保持读取器的路径约束；完成后自动移除视图。调用原 `load_training_package`：5000 张图片实际 hash/RGB/尺寸检查、21924 份 RLE 解码和 mask/框/area 核对、源细类→项目 1–8→内部 0–7 映射通过，crowd_skipped=0。未调用 mapper、模型或 GPU；Detectron2 的 BoxMode 导入不等于加载权重。
3. **隔离。** 新增 5000 个图片身份、来源身份、图组、SHA 均唯一；对旧 train、dev、test 的以上四类交集全部为 0。来源身份使用 dataset/source_split/source_image_id 联合键，避免把不同官方分区的局部数字 ID 误当同一图片。test 只读 manifest 和校验目录，未打开 test 图、mask、标注或评价结果。
4. **复用上游审计。** 校验清单绑定新增源 mask 逐像素一致性（21924 份）、资源 URL 及四方向原图/镜像 pHash≤6 检查（新增对旧/保留集和新增之间 flags=0）。新增成员与采样 sidecar 完全对应，3500 broad +1500 targeted；126 个隔离来源均未进入新包。没有重新提取 pHash、运行 O(N²) 重复搜索、重做原始源转换或抽样看图。

只新增 4 个 CPU 边界测试：旧记录不可改、成员重复/遗漏拒绝、来源命名空间及跨分区碰撞、新图组唯一。全部通过；只解析新增对照配置并核对两组因子相等，未运行全套历史测试。

## 接收内容

下表复用已校验发布审计；图片/实例总数与本轮读取吻合。面积均为原图 mask 像素。

| 子集 | 图片 | 图组 | 实例 | 小目标 ≤1024 | 极小 ≤256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 5k | 5000 | 4976 | 17775 | 2165 | 540 |
| 新增 | 5000 | 5000 | 21924 | 5001 | 1335 |
| 合并 | 10000 | 9976 | 39699 | 7166 | 1875 |

新增目标图命中：小袜 429、小腰带 548、小头饰 547、小眼镜 548、小鞋 1297、包全尺寸 1297（其中小包 182）。七项采集软目标均满足，目标图之间可重叠；这些数量不证明 AP 或边界质量已改善。难形状标签没有来源时仍为 unknown。

**质量边界保持原样：** `audit/visual_audit.json` 写明 reviewer=`Codex visual inspection, not human certification`，160 个类样本及 48 个小目标样本只是抽查；pHash 保守隔离不证明穷尽所有同商品/同人物/裁剪关系。新包 `formal_freeze=false` 不被消费者改写。旧 5k 的用户人工复核确认不自动扩展到本次新增 5k。后续正式冻结的人工处置记录/三方确认仍由相关成员补齐；这不妨碍本轮完成技术验收和训练方案准备。

## 实际执行

源 commit `cd8e8898fa646393dff2b87553d0cb123119e038`；源脚本及读取器 hash 载于报告。运行 `segmentation-train10000-acceptance-20261116-r01`，CPU 硬限 300 秒，退出 0，验收用时 **59.929 秒**，无重试/延长。服务器路径：

`/root/autodl-tmp/fashion-grounding/runs/segmentation/segmentation-train10000-acceptance-20261116-r01`

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
timeout --signal=TERM --kill-after=5s 300s \
/root/autodl-tmp/fashion-grounding/envs/segmentation-v1/bin/python -u \
scripts/accept_segmentation_expansion.py \
  --plan configs/runs/segmentation/train10000_acceptance.json \
  --output /root/autodl-tmp/fashion-grounding/runs/segmentation/segmentation-train10000-acceptance-20261116-r01
```

这是已执行命令记录，输出目录不可覆盖；无输入变化时复用本报告，不重复执行。实际 CPU 接收记录见 [run_metadata](evidence/train10000_acceptance_20261116/run_metadata.json)，新增逻辑测试见 [new_logic_checks](evidence/train10000_acceptance_20261116/new_logic_checks.json)。本任务不是模型实验，不伪造 AP 或增加训练完成行。

下一步按 [5k/10k 受控方案](train10000_control_plan_20261116.md)准备未来训练接线和明确预算。当前 B20k＋800 的选择、18k 候选及 wyz 联调均保持已有版本。
