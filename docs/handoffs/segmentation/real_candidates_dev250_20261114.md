# u3000 完整 dev250 真实候选交付

2026-11-14；lcy / segmentation，Codex 执行。状态：真实 GPU 候选推理、三个 CPU 阈值导出及 wyz 固定接口全量读取均完成；未追加训练、未重算 AP、未进行正式语言阈值选择。

## 结果与运行范围

使用既有 dev AP 选中的 `checkpoint_003000.pth` 对完整 dev250 **逐图推理一次，共 250 次 model forward、1 次权重加载、0 次 optimizer update**。保留原模型与候选解码实现，复用此前 11 项新增导出单测、COCO 八类加载、恢复、数据接收及 dev AP 证据，没有重跑 smoke 或旧测试套件。

缓存覆盖 250/250 图，100 queries/图，共 25,000 queries；保存 **23,011 个非空候选**，按既定规则丢弃 1,989 个空 mask。阈值前缓存没有零候选图片。真实 decoder query ID、原图 RLE、排他 xyxy、类别、分数及稳定顺序均写入缓存；不使用旧 AP 输出位置重建身份，也没有读取 GT 或语言答案。

| score 阈值 | 图片数 | 候选数 / 已验证裁剪数 | 零候选图片 | 平均候选/图 | 导出及接口验证秒数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.3 | 250 | 1,565 | 0 | 6.260 | 81.483 |
| 0.5 | 250 | 1,258 | 0 | 5.032 | 79.230 |
| 0.7 | 250 | 980 | 1 | 3.920 | 79.358 |

0.7 下的零候选图片保留完整 image row 和 `no_candidates` 状态，没有从分母删除。三包均标记 `trial_unselected`，`selected_threshold=null`。候选数/零候选数不是 matched query coverage 或语言准确率，不能据此宣布哪个阈值最好。

GPU 共享锁实际 **190.576 秒**，硬上限 1800 秒、内部预算 1700 秒；三个 CPU 导出与消费者验证合计 **240.194 秒**，另有 `timeout --signal=TERM --kill-after=5s 1800s` 硬上限。所有阶段成功，未重试推理，阈值导出不加载模型。结束 GPU 0 MiB / 0%，无计算进程和 running 协调记录；系统/数据盘约 27/163 GiB 空闲。

## 可直接交给 wyz / wkq 的服务器路径

```text
/root/autodl-tmp/fashion-grounding/shared/candidates/u3000-dev250-20261114-r01/
  delivery_manifest.json
  cache/                 # 完整阈值前缓存
  trial-0.3/candidates_predicted.json
  trial-0.5/candidates_predicted.json
  trial-0.7/candidates_predicted.json
```

共享目录中的 cache/trial 目录为指向不可覆盖运行产物的服务器符号链接，不重复占用图片空间。实际产物完整位于：

```text
/root/autodl-tmp/fashion-grounding/runs/segmentation/segmentation-u3000-dev-candidates-20261114-r01/
```

必须保留该源目录；跨机器交付时复制符号链接所指的完整目录。每个解析后的 trial 包自包含图片、manifest、mask RLE、候选 JSON、解码配置与元信息；下游以 `candidates_predicted.json` 为入口，不把内部 `cache.json` 当成 schema 3 候选包。旧 `preparation-u3000-20261114` 目录保持原样，不再作为最新 payload。

| 产物 | SHA256 |
| --- | --- |
| u3000 权重 | `e56eb7f46fab899d83f668b1878776db50e13cc273c4d4a7fbdc786692444148` |
| 源 dev250 image manifest | `c8c3b24d5a21f93a32060d2fdd9772102bf48b0d6504d60a3466444c7866ecdc` |
| 阈值前 cache.json | `d306f51fae639446645cb4fe8ce655dbb79306dc609882511a780251abfa708e` |
| 0.3 candidates_predicted.json | `4415d7b468ea3fa16f31bc4705880419378e0c5120062d58752ef62600b18a24` |
| 0.5 candidates_predicted.json | `57f4df8c8bef7ea3cecf5d8d912376fb48368013d332ae1f4feef1a468d8e864` |
| 0.7 candidates_predicted.json | `90a12625db17082ae6b3311fea2e1dcf66de5579b7cfa02224f632c694adfc5a` |

导出 manifest 的文件名被规范为 `images/<image_id>.<suffix>`，其文件 hash 因此不同于源 manifest；原 image ID、像素 hash、尺寸、split/group 保留，源 manifest hash 仍在 provenance 中。不要误把文件名规范化当作换图或新分区。

## wyz 接口验证与接收边界

固定 wyz `103dcc546714dd7c7866b4259d96a9825d8090ef` 的 `src/fashion_grounding/grounding/inputs.py`，源码 SHA256 `74153d2e43d34379d9f6b9b0c9537742402010b00c140935ada7191d03cb13a8`。使用真实默认 `CandidatePack(path)`，没有 synthetic opt-in，逐包执行 `load(image_id)` 和 `crop_regions()`。

全量验证 **750 个图片读取记录、3,803 个候选裁剪**，覆盖原图/掩膜 hash、尺寸、原像素几何、类别、候选顺序和空候选语义。每个候选完整 metadata 与缓存对应 query 精确相同；阈值比较沿用 float32 语义，三组 ID 严格满足 0.7 ⊆ 0.5 ⊆ 0.3。CPU 接收脚本未导入 torch，没有执行 SigLIP 编码、查询排序或语言评价。

这是 **lcy 实际运行 wyz 已发布接口的技术验证**；不能写成 wyz 本人已签收或 A/B 正式评价完成。接收方建立 predicted 特征缓存时须使用各包独立 manifest hash/namespace，不复用 GT 或其他阈值的候选序列。

## 版本、命令与证据

整组开工 GitHub 检查：lcy `7e558e3` / #2 draft；wkq `f5787f4` / #4 draft；wyz `103dcc5` / #5（base data-eval）；main `7064ebb`。#1/#3 closed/unmerged，活跃 PR 无评论/审阅增量，队友 handoff/interface 与上一阶段相同，按 commit 复用。语言 accepted dev、M2 没有新增完成证据。服务器仅将自己的干净 segmentation checkout fast-forward 到 `7e558e3`，不修改队友目录或环境。

实际推理源码 **`7e558e365193693c6d6da48b9ca45cb57702e9b3`**；配置 [candidate_u3000_export.json](../../../configs/runs/segmentation/candidate_u3000_export.json)，环境 `/root/autodl-tmp/fashion-grounding/envs/segmentation-v1`。GPU 命令与退出状态完整保存在 [gpu_coordination.json](evidence/real_candidates_20261114/gpu_coordination.json)，对应 [cache.log](evidence/real_candidates_20261114/cache.log)。GPU 命令与上阶段交接示例一致；实际 run ID 为 `segmentation-u3000-dev-candidates-20261114-r01`。

- [完整运行元信息、实际计数和文件指纹](evidence/real_candidates_20261114/run_metadata.json)
- [全组 GitHub / 资源开工快照](evidence/real_candidates_20261114/github_preflight.json)
- [阈值前缓存索引副本](evidence/real_candidates_20261114/cache_manifest.json)（仅证据，原始 mask/图片留在服务器）
- [CPU 导出与接收脚本](evidence/real_candidates_20261114/export_and_accept.py)、[各阈值实际命令及统计](evidence/real_candidates_20261114/offline_receipt.json)
- [共享发布清单](evidence/real_candidates_20261114/delivery_manifest.json)、[发布及资源回执](evidence/real_candidates_20261114/delivery_receipt.json)

运行登记新增一条真实候选推理和三条离线阈值 trial，均 completed；mask AP、语言性能、coverage 等未测字段保持空白。权重、图片、mask 和完整候选 payload 不提交 Git。

## 下一步

wyz / wkq 可接收上述固定包开展下游准备。待 accepted 且冻结的语言 dev 就绪，再按既定 matched query coverage → 更少候选数 → 更高阈值规则选择阈值并做正式评价。当前仅生成 dev250 候选，未生成 train5000 候选，也未启动训练或完整长阶段；后续范围按用户新指令确定。
