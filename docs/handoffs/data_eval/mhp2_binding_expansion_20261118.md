# MHP v2 同图同类服饰训练与验证扩充

wkq 在 AutoDL 发布了 520 张新图。每张至少提供一对可区分的同类服饰，两件属于不同人物。新图来自 [MHP v2 官方数据](https://lv-mhp.github.io/dataset)，仅供本项目非商业课程研究，原图和解析 mask 存在数据盘。Fashionpedia 原包的关键清单已按历史哈希复核。完整包组成、数字和文件指纹见[发布统计](mhp2_binding_expansion_20261118_evidence/release.json)和[哈希](mhp2_binding_expansion_20261118_evidence/package_hashes.json)。

| 数据 | 图片 | 有查询的图片 | 中英文查询行 | 其中人工通过 |
|---|---:|---:|---:|---:|
| 本次新增 MHP train | 320 | 320 | 1692 | 0 |
| 合并 train（Fashionpedia + MHP 首包 + 本次新增） | 1655 | 1587 | 5604 | 0 |
| 本次新增 MHP dev | 200 | 200 | 1036 | 0 |
| 合并 dev（Fashionpedia + 本次新增） | 450 | 446 | 1678 | 642 |

### 标注范围

训练和 dev 分别来自 MHP 官方 train 和 val。按原档目录连续取回 1450 与 800 张，核对原档成员 CRC、官方名单及标签图尺寸；排除一张截断 JPEG。按映射后的同类目标恰好两个、不同人物、中心横距至少图宽 17%、每件面积至少全图 0.5% 初筛。对照原图及左右两个目标的 mask 裁片完成 AI 初标。每个目标都有一条中文和一条英文，单张方位对贡献 4 行。颜色可独立区分时再加 4 行，颜色描述中不使用方位。每张仅选一对。左右相对观看者。中文和英文是平行描述，统计时以图或配对为样本单位。所有新增标注仍待人工核验。

| 查询目标类别 | train 方位对 | dev 方位对 |
|---|---:|---:|
| 上衣 | 102 | 69 |
| 裤子 | 95 | 58 |
| 外套 | 89 | 60 |
| 连衣裙 | 13 | 7 |
| 包 | 16 | 4 |
| 半身裙 | 5 | 2 |
| **方位对合计** | **320** | **200** |
| **其中也有颜色对** | **103** | **59** |

全部配对保留来源图、同图两个目标 ID、项目类别、确认的颜色以及独立人员解析来源。每张均可通过独立包的 selected_pairs.json 追溯。颜色和方位分开出题，可比较只改一个词时模型是否改选。采样侧重横向差异清楚的成对目标，颜色和方位的强度不同，上衣、裤子和外套占大多数。此 dev 适合报告该条件下的跨来源识别能力。

### 划分与去重

对照 43,561 个既有图像指纹，比较 SHA-256、原图及镜像的感知哈希（汉明距离 ≤6），同时对新来源内部比较，隔离命中的 75 张。对目检留下的外部候选与全部 100 张 MHP 首包图片另做冻结 SigLIP 2 全图相似度筛查。余弦相似度 ≥0.90 的 19 条关联按连通组处理，涉及首包则保留首包，其他组最多收一张，另排除目视重复场景与边界问题。阈值筛查仍可能漏掉不同机位的同场景。合并 train、dev 与保留 test 的图片 ID、内容 SHA-256 和图像组 ID 两两无交集，仅核验 test 清单。见[来源核验](mhp2_binding_expansion_20261118_evidence/source_checks.json)和[场景记录](mhp2_binding_expansion_20261118_evidence/scene_guard.json)。

### 给 wyz 和 lcy 的入口

服务器目录前缀为 `/root/autodl-tmp/fashion-grounding/shared`。

- wyz 的合并训练包在 `processed/fp_mhp2_binding_train_v2/`。训练读取 `queries_machine_train.jsonl`、`candidates_gt.json`、`annotations_coco.json`、`image_manifest.json`。真实弱监督入口读到 5604 行，逐条解析到唯一 GT 目标。可将此输入与原 Fashionpedia 训练输入作固定种子、相同更新数的对照。新增独立包为 `processed/mhp2_binding_train_v2/`，其中 `queries_increment.jsonl` 提供本轮 1692 行的配对 ID。若另启用旧实验的配对辅助损失，grounding 中 `pair_objective.counter_targets` 的 86 组断言是旧规模，需按当前训练清单配置；普通 GT 弱监督训练可直接读新包。
- 原有人工 dev 继续使用 `processed/fp_language_dev250_reviewed_v1/queries_accepted.jsonl`，642 行。合并归档 `processed/fp_mhp2_binding_dev_v2/` 的 accepted 文件逐字节一致。该归档的 draft 文件同时包含两种审核状态。wyz 可以立即用 **未审核 MHP dev** 做诊断，请单独指向 `processed/mhp2_binding_dev_v2/queries_draft.jsonl`、开启草稿模式，并单列结果。GT 候选在该独立包内；固定分割候选在 `candidates/wkq-mhp2-binding-dev-lcy50k-t03-v2/candidates_predicted.json`。GT 草稿可走 controlled 的 draft_dev；预测候选可通过 grounding.predict 的 draft 读取入口。请分别报告 Fashionpedia 人工 dev、MHP 草稿 dev，以及 GT 候选与预测候选条件。比较纯 M1/M2 时关闭推理规则加分。
- lcy 的固定 50k 分割权重、阈值 0.3、边长 800/1333，仅输入 RGB。推理在独立环境使用 CPU，全部 200 张均已生成候选。统一同类一对一关联要求 mask IoU ≥0.5。共 1891 个候选，0 张没有候选。200 个方位对中，两个目标都召回 130 对、仅一个 57 对、均未召回 13 对。59 个颜色对中对应 42、13、4 对。逐图关联见预测目录的 `matched_pairs.json`。此处数值衡量分割候选覆盖；新排序头训练与语义准确率待 wyz 实验。
- wkq 的新增数据审核包位于 `/root/autodl-tmp/fashion-grounding/runs/data_eval/20261118-mhp2-expansion-v1/` 中的 `review-train.zip` 与 `review-dev.zip`。解压进入对应目录 index.html，可独立作答、逐页导出，或点“答案预览”查看 AI 选择。train 34 页、846 组双语描述，dev 21 页、518 组；后续回收审核时记录是否已看过答案。审核状态、可读性、抽样核验及签名分别见 [审核页核验](mhp2_binding_expansion_20261118_evidence/review_checks.json)、[GT 读取核验](mhp2_binding_expansion_20261118_evidence/reader_gt.json)、[预测读取核验](mhp2_binding_expansion_20261118_evidence/reader_pred.json)。来源抽取、筛选、标注、推理与发布脚本保存在同一运行目录。
