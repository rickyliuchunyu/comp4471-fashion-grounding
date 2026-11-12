# wkq / data_eval：大型数据包与机器语言初标

任务：P4 数据扩充。负责人 wkq；2026-11-12。工作在 AutoDL 的独立 data_eval 工作区和 CPU 环境完成，未使用 GPU、未训练分割或排序模型。固定分支为 `data-eval/p3-baseline-evaluation`，统一 PR #4；不合并 main。

## 当前可交付内容

已发布版本化分割包及语言初标，**不是全量人工验收或正式 benchmark 冻结**。本轮遵循用户要求：Codex 先初标，再由 wkq 审核；训练仍由 lcy/wyz 负责。共享根目录：

`/root/autodl-tmp/fashion-grounding/shared/processed`

| 包 | 图像 | 整件实例 | 语言记录 | 用途 |
| --- | ---: | ---: | ---: | --- |
| fp_train5000_v1 | 5,000 | 17,775 | 0 | 分割训练源 GT |
| fp_dev250_v1 | 250 | 1,008 | 0 | 分割 dev 源 GT |
| fp_test400_holdout_v1 | 400 | 1,599 | 0 | 留出的测试图；不用于开发 |
| fp_language1000_machine_v2 | 1,000 | 4,784 | 3,160 | 1,580 组中英文训练初标 |
| fp_language_dev_draft_v2 | 116 | 490 | 290 | 145 组中英文 dev 草稿 |

每包含 image_manifest.json、annotations_coco.json、candidates_gt.json、图像、RLE mask、源标注子集、映射配置、验证报告、RELEASE.json 和逐文件 SHA256SUMS.json。语言包是对应分割分区的子集，保留全部该图候选，不只保留目标。

发布索引：`fp_data_eval_large_v1/release_index.json`。同目录保存划分、重复组、来源、人工标注待办、视觉抽查及运行记录；实际运行目录：

`/root/autodl-tmp/fashion-grounding/runs/data_eval/20261112-large-data-regrouped`

实现 commit：`6606285bc9127b8dc2845b9b29f9adb9ca397786`。文档提交在其后；发布文件不随后续代码变更自动改写。大数据没有提交到 GitHub。

## 划分及接口

- 官方 train 采集池 6,000 图，发布前 5,000 图；官方 validation 合格采集池 1,137 图，按组取 dev 250 / test 400。seed 2026；不把官方 validation 移入训练。
- SHA256、具体资源的 original_url、含镜像的 63-bit pHash（Hamming <= 6）构建保守组。**网站首页 URL 不作为同图标识**。发现 53 对近重复线索，隔离 10 张跨官方边界的 validation 图；包含旧试点已曝光图片的保护。
- train/dev/test 分别为 4,976 / 248 / 399 组；发布分区的 group/hash overlap 均为 0。所有可疑近重复先同组，不宣称已经识别全部同人物/同商品或裁剪关系，也不宣称所有线索都经过人工裁决。
- 项目类别仍为 1–8，内部训练类别由 lcy 映射为 0–7。train 项目 ID 为原始 ID + 1；官方 validation 派生的 dev/test 项目 image/annotation ID 为原始 ID + 1 + 1,000,000,000，避免源文件之间 ID 碰撞。原始 source_image_id/source_annotation_id 保留。
- 候选的 source_annotation_id 引用**项目 COCO annotation ID**；COCO annotation 内的 source_annotation_id 才引用 Fashionpedia 原始 ID。消费方不能自行假设两者恒差 1。
- 包内 candidate_set_id、manifest/hash 已变化，旧特征缓存不能直接复用。GT 候选不是分割预测；本轮没有生成预测候选或模型成绩。
- 八类 mask 原样来自源标注；部件排除，含 jumpsuit 的图像排除。没有擅自合并可见碎片或补造缺失 mask。

## 标注来源与限制

主要初标来自 CPU mask 内部像素颜色估计、Fashionpedia 局部纹样属性和框几何；**不是 1,000 张逐图完整视觉审核，也不是独立人工标注**。

v1 的 dev 抽查发现紫/粉、深蓝/黑、橙/棕等混淆；v1 查询包未发布。v2 只在保守阈值下生成黑/白/灰/红/黄/绿/蓝，其他色调放弃判断。比较同一项目类别的全部已标注竞争候选；未知对手属性不能证明唯一。空间配对仅限严格条件下的帽、鞋、包、伞，避免把裤腿等误作两件服饰。已知争议图及一个可疑连衣裙图不进入自动语言池。

训练最终 1,580 组：appearance 1,384、spatial 24、combined 172；按源细类标注数计，有竞争候选的为 197 组。数量不等于物理整件身份已经验证；多数仍为简单查询，不能用此包证明复杂指代能力。未填充无依据的困难描述。

抽查了 64 个训练语言目标、31 个 v2 dev 目标；按图修订训练 7 组 / dev 2 组。比如图 35104 原图有两件黑夹克，官方只标了左侧女士的目标；核对完整叠图后改成“画面左边的黑色夹克”。保留 queries_before_visual_revision.jsonl、visual_revisions.json、annotation_evidence.json 和修改原因。Codex 修订仍不等于人工 accepted。

所有新增查询保持 `needs_revision`、reviewer=null、queries_accepted.jsonl 为空。已有小包的 17 组 / 34 条 accepted 不变，不能把新初标统计进它们。

- `queries_machine_train.jsonl`：仅训练语言包非空；只供**明确选择弱标注实验**的 wyz 使用。若现有加载器要求 accepted，不要伪造状态或关闭正式评价门禁；由方法负责人实现明确的实验输入选择，并记录该文件 hash。
- `queries_draft.jsonl`：人工审核输入，dev 不得混入训练。
- dev 250 图中只有 116 图有草稿，其余 134 图待人工补写；现有 dev 草稿也还不能用来报告正式准确率。
- final test 400 图没有自动语言答案；按协议在方法冻结后由人独立撰写、另一人复核。未根据测试图调整颜色规则或查询模板。
- 官方标注可能遗漏画面中的服饰；“只有一个已标候选”不保证自然语言唯一，必须保留人工复核。

## 验证证据

- 93 项 CPU 单元测试通过，包含验证 ID 命名空间、源 mask 保留、重复组、网站首页排除、颜色放弃判断、竞争样本选择、修订保留待审状态等边界。
- 五个包均通过 schema、图像 hash/尺寸、COCO 与 RLE mask、面积/框、候选引用和查询目标校验。
- 五个包均通过独立 pycocotools 源 mask 逐像素一致性及候选裁剪检查。训练/dev/test 的源实例数分别为 17,775 / 1,008 / 1,599；语言子集为 4,784 / 490。
- 另抽查八类各 8 个分割实例，共 64 个；记录小目标和不确定分离区域。抽样不构成全量边界质量保证。
- 35 个审核页面检查了覆盖、无重复/遗漏语义组、最终 query hash、按页存储与导出、payload 不含目标答案。已有 UI JS 行为测试通过；未声称本轮做了真实浏览器逐页渲染验收。
- 发布到共享目录后逐文件核对 SHA256；没有覆盖旧 outputs、修改其他环境或停止他人进程。
- 环境：Python 3.10.8，requirements-data.txt 锁定 numpy 2.2.6 / Pillow 12.0.0 / pycocotools 2.0.11 / requests 2.32.5；quickjs 用于 UI 行为测试。CPU 4 worker 初标，无 GPU。

## 给三位成员的下一步

**lcy**：读取 fp_train5000_v1 + fp_dev250_v1 的 image_manifest / annotations_coco，确认新 ID、类别及同组边界后开展自己的分割训练；输出权重、dev AP、固定预测候选和 hash。本次只准备数据，没有代跑训练。旧小包已存在 lcy 工作区的 reviewed_pilot_acceptance_20261112.md 验收记录；新大包需单独验收。

**wyz**：重新检查大包读取与缓存键。可明确启用 queries_machine_train.jsonl 做第一版弱监督实验，GT 或预测候选条件分别记录；标签/target/category/query_type 不作为文本模型的推理输入。正式 dev 评价等待人工验收查询。用户报告小包接口 OK；本轮没有把旧工作区的 B1 草稿结果冒充新大包验收或训练结果。

**wkq**：优先审核 dev 的 145 组（3 页），并组织补写其余 134 图；随后分批审核训练初标，处理歧义、漏标和困难描述。语言挑战集、独立测试描述、三方冻结及新包消费验收仍未完成，所以 P4 保持 in_progress。

审核入口：`fp_data_eval_large_v1/review/index.html`；可下载同目录 `human_review.zip`，解压后打开 index.html。每页 50 组，训练 32 页、dev 3 页；每页独立导出，不覆盖文件。页面不显示参考目标，重点核对描述唯一性、双语含义及所选 mask；无需逐个检查所有框。返回审核 JSON 后先核对 manifest/query hash 与重复语义组，再做原有审计流程，不能直接把导出当 accepted 标签。

## 复现与留痕

在独立 data_eval 环境运行脚本，所有输出用新目录，不覆盖上述已发布版本：

1. scripts/acquire_large_pool.py：按 seed 获取官方 ZIP 成员，检查 ETag、HTTP range、长度和 ZIP CRC；实际来源见 download_provenance.json。原始 train 标注沿用 grounded-queries/raw/instances_attributes_train2020.json。
2. scripts/build_large_packages.py：构建分组和三分区、源保真转换及完整验证。
3. scripts/build_machine_queries.py：训练 --count 1000；dev --count 250 --allow-fewer，不强行凑数。
4. scripts/apply_query_revisions.py：使用发布包 visual_revisions.json；必须匹配原文本，保留备份及未验收状态。
5. scripts/check_pilot_consumers.py、scripts/render_dataset_audit.py、scripts/render_review_batches.py：来源/裁剪验证、抽样图、盲审页面。
6. 真实运行日志、旧的未发布构建、被替换的串行扫描日志均保留在 runs/data_eval；publish_release.py 和 run_metadata.json 记录发布检查、实现 commit 和实际状态。


## 发布清单指纹

| 包 | image_manifest SHA256 |
| --- | --- |
| fp_train5000_v1 | `82916717a867c1e2fe4ae8b42ea34072e97b3edef9113ecf83a855e390a0c0f3` |
| fp_dev250_v1 | `c8c3b24d5a21f93a32060d2fdd9772102bf48b0d6504d60a3466444c7866ecdc` |
| fp_test400_holdout_v1 | `ca8f5738c18cf800091e87eb36b86e80c9fe465c3de2adb4db7549f474419cf8` |
| fp_language1000_machine_v2 | `5f625542cef0e5933494edb88a1cc3d70bee76bfddd19def64777e7b60bc71d0` |
| fp_language_dev_draft_v2 | `9fd34ca49761b8d3cfb0ee0a9a325ee733593e31480c7ad2664840ec49148dd7` |

训练 queries_draft / queries_machine_train SHA256：`c577a56575e12e1dbf7072a294ef97674dbb4bf3f6f4ed2a8de645ac3177b49d`。

审核 ZIP SHA256：`2857dc0b59e75f32df44aea717ed8b627df358f4c9e17c01d63de6b70b3ca57c`。
