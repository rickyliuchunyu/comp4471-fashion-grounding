# 当前分割候选 · 28,318 步

**完整 dev250 固定为 28,318 步权重和 0.3 阈值。** 250 张图共 1,351 个候选，0 张图没有候选。使用同一批 642 条已审核的中英文开发查询，覆盖 606 条（94.39%），旧 50k＋0.3 覆盖 602 条，本次差值为 +4 条。覆盖表示目标在同类一对一 IoU ≥ 0.5 匹配下有候选可用，下游排序准确率另行评价。[机器可读索引](../../../configs/artifacts/segmentation_dev_candidates.json) · [选择回执](evidence/u28318_candidates_20261118/selection/selection.json)

## 替换下游输入

将 `CandidatePack` 或 `--candidates` 指向以下服务器文件。服务器滚动索引位于 `/root/autodl-tmp/fashion-grounding/shared/candidates/SEGMENTATION_DEV_CURRENT.json`。每次运行应记录实际使用的版本和摘要。

```text
/root/autodl-tmp/fashion-grounding/shared/candidates/u28318-dev250-20261118-r01/selected_candidates/candidates_predicted.json
```

| 输入 | 版本或 SHA-256 |
|---|---|
| 候选集 ID | `u28318-dev250-20261118-r01-trial-0.3` |
| 候选 manifest | `829f699293fd4537ec27dfb549943e5592de37fce91ec929f8371e04cb55fed2` |
| 最终 checkpoint | `aaff8d9dd89e5aad4be0d6304eaa12dc0c8bb880395722220fa1f12af33f7809` |
| 阈值前缓存 | `d609a55f01c082eaf68b1f2f2bc403278e07e4b50284fd23000d4501d47219c9` |
| dev 查询 | `ac8d6af9d1b854452627b2f6d72aed58a91bee0642344ac37932f8701fb841d9` |

- **wyz**：沿用现有 `CandidatePack`、裁剪入口及候选格式。为新候选 manifest 建立独立的区域特征缓存、排序预测和 mask/框结果。候选身份同时绑定候选集 ID 和真实 decoder query ID。文本、语言和编码器签名均相同时可复用文本特征。
- **wkq**：在新的 predicted-candidate 评价中绑定新版 manifest 和查询摘要，使用独立输出目录。相同标注、查询和评价协议下可沿用 GT 基线。旧候选的覆盖或排序记录仍属于旧版。
- **查询快照**：`/root/autodl-tmp/fashion-grounding/shared/candidates/u28318-dev250-20261118-r01/language_dev_inputs/queries_accepted.jsonl`。642 条查询来自 321 个中英配对、246 张开发图。dev250 的实际导出保留全部 250 张。当前交付范围为 Fashionpedia dev，接收方独立的 MHP 数据及历史实验沿用各自版本。

## 三档离线比较

| 阈值 | 候选数（250 图） | 匹配覆盖 | 匹配覆盖率 | 几何召回率 |
|---:|---:|---:|---:|---:|
| 0.3 | 1,351 | 606/642 | 94.39% | 96.88% |
| 0.5 | 1,198 | 598/642 | 93.15% | 96.26% |
| 0.7 | 1,070 | 588/642 | 91.59% | 95.33% |

按覆盖查询数最多选择，同分先取查询图上候选数更少者，再取较高阈值。阈值前缓存包含 24,141 个非空候选，保留原尺寸掩膜、真实 query ID、类别、分数和框。缓存入口为 `/root/autodl-tmp/fashion-grounding/shared/candidates/u28318-dev250-20261118-r01/cache/cache.json`。只需调整分数阈值时可直接复用。[完整比较](evidence/u28318_candidates_20261118/selection/comparison.csv) · [导出与下游读取回执](evidence/u28318_candidates_20261118/offline_receipt.json)

## GUI 与运行证据

[打开本地工作台](../../../scripts/segmentation_gui/打开分割工作台.command)。开发集直接显示已选候选包。「我的图片」在项目服务器载入同一权重和解码配置，默认选择自动模式。共享 GPU 空闲时使用 CUDA，忙碌或不可用时使用 CPU 并显示原因。也可选择「仅 GPU」或「仅 CPU」。「仅 GPU」遇忙会提示稍后重试。结果栏显示实际设备，两个入口的默认阈值均为 0.3。

浏览器验收使用同一张开发集以外的公开样例（Fashionpedia 4993，1024 × 680）。自动模式实际使用 NVIDIA GeForce RTX 3090，一次上传返回 HTTP 200，显示 3 个实例及 28,318 步版本。共享锁竞争、回退和释放使用独立临时锁验证。此前的 CPU 上传及缓存版本记录见 [GUI 验收](evidence/u28318_candidates_20261118/gui_receipt.json)。

首次部署时，将已选目录中的 `candidates_predicted.json`、`image_manifest.json`、`images/`、`masks/` 复制至 `outputs/segmentation_gui/releases/u28318-dev250-20261118-r01/`。本机 Python 环境及 SSH 配置沿用[工作台设置](gui_50k_20261118.md)。

2026-11-18 开工时数据盘约 121 GiB 可用，系统盘约 27 GiB 可用，RTX 3090 占用 0 / 24,576 MiB，GPU 锁空闲。已核对全部七个远端分支及五个 PR，wkq 为 `b82fc99`，wyz 为 `2c47c36`。模型与导出配置通过分割提交 `bc46e85` 固定。共享锁只执行一次完整 dev250 候选推理。三档及阈值选择均为 CPU 离线处理。真实消费验证使用 wyz 当前提交的读取器。三档共检查 3,619 个裁剪。既有完整 dev 分割 AP 71.3205 本次直接复用。队友本人签收状态待对方记录。[GPU 运行回执](evidence/u28318_candidates_20261118/coordination.json) · [缓存回执](evidence/u28318_candidates_20261118/cache_run_metadata.json) · [原始 dev AP](full_single_completion_20261118.md)
