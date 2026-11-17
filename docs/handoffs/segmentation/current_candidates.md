# 当前分割候选：50k＋0.3，可直接接入dev

2026-11-18，lcy交付，Codex整理。**ready_for_downstream_development**：wyz/wkq现在可以接入下面的完整dev250候选，无需等待下一轮分割优化、train候选或仓库合并。接收方本人消费/评价尚未代签；后续实验由各自负责人按已有算力安排执行。

## 直接使用的路径

候选输入（传给`CandidatePack`或`--candidates`）：

```text
/root/autodl-tmp/fashion-grounding/shared/candidates/u50000-dev250-20261118-r01/selected_candidates/candidates_predicted.json
```

本次选择对应的642条dev查询快照（传给`--queries`）：

```text
/root/autodl-tmp/fashion-grounding/shared/candidates/u50000-dev250-20261118-r01/language_dev_inputs/queries_accepted.jsonl
```

交接索引同时放在服务器和GitHub：

```text
/root/autodl-tmp/fashion-grounding/shared/candidates/SEGMENTATION_DEV_CURRENT.json
/root/autodl-tmp/fashion-grounding/shared/candidates/HANDOFF_50K_DEV250_20261118.md
```

GitHub对应[机器可读索引](../../../configs/artifacts/segmentation_dev_candidates.json)。索引是路径/版本清单，不是schema-3候选包；不要直接把索引或`cache/cache.json`传给候选读取器。读到索引后应将其中的固定路径与hash写入各自run metadata，避免后续current指向升级造成版本漂移。[发布回执](evidence/u50000_downstream_handoff_20261118/publication_receipt.json)记录本轮新交接文件的服务器读回校验；原图、mask、权重和既有接口检查直接复用。

## 固定版本与已有结果

| 项目 | 固定值 |
| --- | --- |
| 权重 / 阈值 | 本次10k训练选中的50k / score≥0.3 |
| candidate_set_id | `u50000-dev250-20261118-r01-trial-0.3` |
| 候选manifest SHA256 | `c13af7db0fa5d07a0f62a12a04a138703e7b5a009b4e5a1e3759c3b7a3e05eb4` |
| 查询SHA256 | `ac8d6af9d1b854452627b2f6d72aed58a91bee0642344ac37932f8701fb841d9` |
| 图像 / 候选 | 完整dev250 / 1279个；0张零候选图 |
| 查询范围 | 642条accepted记录，321组中英配对，覆盖246张dev图 |
| 候选覆盖 | 同类IoU≥0.5一对一覆盖602/642＝93.7695%；不是排序准确率 |
| 分割AP | 已有完整dev AP69.0313；本次交接未重评 |
| 格式 | schema 3；类别1–8；原图RGB、原尺寸mask、排他右下角xyxy、真实decoder query ID |

查询使用当前用户授权的开发口径，保留机器初写及wkq审核来源；另9组pending由上游预先排除，未计入642条。开发快照不冒充整包正式人工验收或最终test。wkq已在相同642条查询上完成的GT结果可按原hash继续复用。

## wyz：更新候选输入与区域特征

现有`103dcc5`读取器已在本次三档新包通过750次图片读取/3496个候选裁剪，复用该证据，不重跑整套接口smoke。

1. 在现有排序入口把`--candidates`换成上面的固定路径；`--queries`固定为上述快照，使用新的run ID/output目录。
2. 继续使用既有`FeatureCache`。已发布的`visual_spec`包含候选manifest hash、candidate_set_id和候选序列，所以新包会形成新的区域特征键；不得把旧18k区域特征改名后复用。
3. **相同文本、语言、encoder signature的文本特征可以复用。** `text_spec`不依赖候选版本；无需删除旧缓存或重建整个缓存根目录。编码器/裁剪策略若另行变化，遵循各自既有版本规则。
4. 排序输出只引用当前包的candidate ID；最终mask/框从该包读取。`image-…-query-…`字符串跨权重可能相同，身份必须同时绑定candidate_set_id和manifest hash。

读取方式保持现有接口即可，以下是接入示例，不是本轮新增验证或已执行的排序：

```python
from fashion_grounding.grounding.inputs import CandidatePack
pack = CandidatePack(
    "/root/autodl-tmp/fashion-grounding/shared/candidates/"
    "u50000-dev250-20261118-r01/selected_candidates/candidates_predicted.json"
)
# 延用既有 pack.load(image_id)、crop_regions、排序和输出流程。
```

## wkq：统一新版predicted条件，保留旧结果

现有`evaluate_predictions.py`的输入参数不变：`--candidates`用本包、`--queries`用固定642条快照；`--annotations`仍是：

```text
/root/autodl-tmp/fashion-grounding/shared/processed/fp_dev250_v1/annotations_coco.json
```

GT标注SHA256为`74837c386c4bbd2805f9976fcfd796931b2f6cfa72693b91d2f441bee473c635`。待队友的新排序输出产生后，用新run ID评价；不要把旧18k预测文件改成50k标签，也不要与旧候选条件的B1/B2/M1结果混成同条件横向对照。B0/B1/B2/M1/M2的新predicted对照使用同一固定候选包，不再各自筛选阈值或重排候选。

旧GT结果在查询/标注/规则等身份不变时可复用；旧18k的候选、特征和历史指标保留原版本。新候选覆盖93.7695%不能作为任一排序方法的新准确率。

## 能立即做什么与范围

- 可立即用：dev区域特征更新、已有排序方法的新版predicted开发评价、从选中candidate ID输出原mask/框。
- 本包只含dev250，不含train或test候选；它不是语言训练集。M1/M2训练与test安排由对应负责人推进。
- 50k后续优化只是提案，不阻塞这次交付，也不授权本轮追加分割推理/训练。
- 共享目录包含指向不可变run/input目录的符号链接。服务器内直接使用即可；跨机器拷贝须复制解析后的完整内容，保留图片、mask与manifest相对结构。

## 依据与接收状态

- [候选生成、阈值选择与技术消费证据](u50000_candidates_dev250_20261118.md)
- [分割训练与最终权重](tenk_completion_20261118.md)
- [剩余错误分析及限制](u50000_residual_diagnosis_20261118.md)
- [本轮整组GitHub快照](evidence/u50000_downstream_handoff_20261118/github_preflight.json)

本轮只发布交接入口、机器索引和状态更新，复用既有验证；不改队友checkout、不运行排序/评价、不代签接收。三方分支保持独立，接入数据文件无需合并lcy全部训练代码。接收方后续只需在自己的handoff记录使用的candidate_set_id/hash、query hash及run ID/结果位置；遇到问题记录具体失败入口，不从环境smoke重做。
