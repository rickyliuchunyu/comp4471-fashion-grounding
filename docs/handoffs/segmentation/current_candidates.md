# 当前分割候选 · 组合微调 28,318 步

**2026-11-19，lcy 已正式冻结 S0。** 固定候选集为 `combo-u28318-dev250-20261119-r01-trial-0.3`。下游复现时使用下表的版本化 manifest 和完整摘要。全组 P7 待汇总，wyz 与 wkq 待本人签收。[机器可读冻结条目](s0_freeze_20261119.json)

GUI 开发集、个人图片和下游 Fashionpedia dev250 使用组合权重 `b7146787a1fc`，固定阈值 **0.3**。同一批 642 条已审核中英文开发查询中，新版本覆盖 610 条，上一版 `aaff8d9dd89e` 覆盖 606 条，差值为 +4 条。覆盖按同类一对一掩膜 IoU ≥ 0.5 匹配，衡量下游有多少查询能拿到正确候选。排序准确率由下游单独评价。[机器可读索引](../../../configs/artifacts/segmentation_dev_candidates.json) · [阈值选择回执](evidence/combo_u28318_candidates_20261119/selection/selection.json)

## 接入方式

把 wyz 的 `CandidatePack` 或 `--candidates` 指向以下 manifest，并在下游运行记录中固定候选集 ID 和 SHA-256。滚动索引为 `/root/autodl-tmp/fashion-grounding/shared/candidates/SEGMENTATION_DEV_CURRENT.json`。

```text
/root/autodl-tmp/fashion-grounding/shared/candidates/combo-u28318-dev250-20261119-r01/selected_candidates/candidates_predicted.json
```

| 项目 | 版本或 SHA-256 |
|---|---|
| 候选集 ID | `combo-u28318-dev250-20261119-r01-trial-0.3` |
| manifest | `4daf79030f88bc285d35272cc8e7160e0143d99216b05740a3569f946bb186bd` |
| checkpoint | `b7146787a1fc8f16f4694f2a9d05c294a49090b095de901b63b963f162a640c1` |
| 阈值前缓存 | `beb84fab9ee980b564be2d74966a931fea0b0a54b9613dc3508d474bd9d75132` |
| 642 条开发查询 | `ac8d6af9d1b854452627b2f6d72aed58a91bee0642344ac37932f8701fb841d9` |

- **wyz** 可直接用已验证的 schema 3 读取器。按新版 manifest 建立独立的区域特征、排序预测和输出 mask/框。文本、语言和编码器签名一致时可复用文本特征。
- **wkq** 可沿用相同查询、标注和协议下的 GT 结果。依赖预测候选的结果需绑定新版 manifest 和独立输出目录。此前的 Fashionpedia 与 MHP 实验仍对应各自记录的输入版本。

## 阈值与缓存

冻结的候选解码使用 800 像素短边、1,333 像素长边上限，每个 query 取一个前景类别，每图最多 100 个候选。生成缓存时分数阈值为 0，交付时按 `score ≥ 0.3` 筛选。交付配置的 SHA-256 指纹为 `2c7de35036c9153b13d1009ebdde810ae8bb5b3723d1c8beb1a4cd39bdcba7cb`，与所选 manifest 的 `provenance.decoder_config_sha256` 一致。[导出配置](../../../configs/runs/segmentation/candidate_combo_u28318_export.json) · [阈值选择回执](evidence/combo_u28318_candidates_20261119/selection/selection.json)

| 阈值 | dev250 候选数 | 零候选图 | 匹配覆盖 | 匹配覆盖率 |
|---:|---:|---:|---:|---:|
| 0.3 | 1,320 | 0 | 610/642 | 95.02% |
| 0.5 | 1,181 | 0 | 606/642 | 94.39% |
| 0.7 | 1,065 | 0 | 596/642 | 92.83% |

按覆盖查询数最多选择，同分依次取查询图上候选更少者、阈值更高者。查询来自 321 组、246 张图，三档都包括全部 250 张图。缓存保存 24,111 个非空候选，字段包含真实 `decoder_query_id`、`class_probability`、`mask_quality`、最终 `score`、框及原尺寸 RLE 掩膜。`score` 为前两项的乘积，掩膜质量分是预测前景内的平均 sigmoid 值。后续更改分数阈值可直接读取以下缓存。[三档完整比较](evidence/combo_u28318_candidates_20261119/selection/comparison.csv) · [分数字段核对](evidence/combo_u28318_candidates_20261119/score_receipt.json)

```text
/root/autodl-tmp/fashion-grounding/shared/candidates/combo-u28318-dev250-20261119-r01/cache/cache.json
```

## GUI 与验证

[打开分割工作台](../../../scripts/segmentation_gui/打开分割工作台.command)。开发集直接显示已导出的候选。「我的图片」在服务器加载相同的权重和解码配置，默认优先用空闲 GPU。发布时用一张开发集之外的公开样图上传一次，实际设备为 NVIDIA GeForce RTX 3090。页面显示步数和 checkpoint 短哈希。其他本机部署可将服务器 `selected_candidates/` 的两个 manifest、`images/` 和 `masks/` 复制至 `outputs/segmentation_gui/releases/combo-u28318-dev250-20261119-r01/`。[GUI 验证回执](evidence/combo_u28318_candidates_20261119/gui_receipt.json)

开工时已核对三人的远端分支、五个 PR 和交接，wkq 为 `b82fc99`，wyz 为 `1e0a457`。数据盘可用 120 GiB，RTX 3090 占用 0/24,576 MiB，共享锁空闲。一次 dev250 候选推理加三档离线导出和选择耗时 469 秒，硬上限 900 秒。使用 wyz 此提交的真实读取器验证了三档的 3,566 个裁剪。既有分割 AP 报告沿用原记录。队友本人签收待对方记录。[GPU 记录](evidence/combo_u28318_candidates_20261119/coordination.json) · [下游读取回执](evidence/combo_u28318_candidates_20261119/offline_receipt.json) · [组合权重选模报告](full_combo_completion_20261119.md)
