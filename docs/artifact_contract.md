# 文件交接契约与完整示例

契约版本 `3`，2026-11-10。适用于图片 manifest、候选包、查询和预测；不要求独立的类别映射、规划配置也改成版本 3。版本 2 的预测示例缺少 query ID 且使用不存在的 mask 占位，已被替换，消费方不能静默混读两版。

本文固定文件层接口；模型张量接口见 [实现规范](implementation_spec.md)。[examples](../examples/README.md) 是可验证的合成格式夹具，不是 Fashionpedia 数据、人工标注成果或模型结果。

## 1. 公共约定

- JSON 使用 UTF-8；JSONL 每行一个同类型记录。文件根对象均有 `schema_version=3`、`illustrative_only`。正式数据必须为 false，不能通过改标志将夹具变为真实数据。
- 文中所有 path、`file_name`、`mask_reference` 均相对**包含该字段的 JSON 文件所在目录**。发布时保留目录结构；拒绝绝对路径、`..` 和凭据 URL。图片 manifest 与 COCO 标注置于包根目录，使 COCO 的数据根目录和 manifest 所在目录一致。共享原始文件可通过只读挂载/链接提供，但接收方必须能解析且核对相同字节；归档时记录依赖，不把机器专属链接当作自包含备份。大数据包可以在共享存储中使用相同内部结构。
- SHA256 是对应文件原始字节的 64 位小写十六进制；不对重新序列化后的对象计算。包一旦发布不可原地修改；变更内容生成新版本、hash 与缓存。
- 图片/GT annotation ID 是正整数；query/candidate/set/run/group ID 是非空字符串。candidate ID 在同一候选包全局唯一，其身份完整键为 `(candidate_set_id, candidate_id)`。关联查询用 `(run_id, query_id)`，不得按文字内容去重。
- 项目类别 ID 1–8；原图 RGB；`image_size_wh=[W,H]`；原图像素框 `xyxy` 右下排他。本项目内部候选框由非空 mask 精确计算，不能用 GT 修正预测框。
- RLE `size=[H,W]`，沿列遍历，counts 从背景段开始。正式交接支持 COCO 压缩 RLE（counts 为字符串）及非压缩 RLE（整数数组），生产方与消费方须用 COCO 兼容工具核对。夹具采用非压缩 RLE，附带检查器不代替正式 COCO 解码器。
- 数据包中的标签只给训练/评价。推理入口显式构造仅含 image/query ID、文字、语言和候选的请求，不把完整标注 dict 传入模型。GT 候选只用于明确标记的诊断。

## 2. 图片与查询

[image_manifest.json](../examples/image_manifest.json) 根字段为 `manifest_id`、`mapping_version`、`images` 以及公共字段。每图包含 `image_id/file_name/sha256/width/height/source_dataset/source_image_id/group_id/split`。正式 split 为 train/dev/test/unused；fixture 仅限合成包。图片 ID、文件和重复组须稳定，同图所有查询跟随图片 split。

[query_record.json](../examples/query_record.json) 是一条查询。正式查询集合为 JSONL；全部字段沿用 [数据协议](data_protocol.md#6-标注记录与复核)，另要求 `schema_version=3`。`query_id` 唯一，中英翻译各有独立 query ID、共用 semantic ID；目标来自同图非 crowd GT。目标不存在、split 不一致或未 accepted 的记录不得进入正式评价清单。夹具的复核状态明确为 illustrative，不可计为 accepted。

## 3. GT 与预测候选包

完整文件：[GT](../examples/candidates_gt.json)、[预测](../examples/candidates_predicted.json)、[零候选](../examples/candidates_empty.json)。

| 层级 | 必需字段与约束 |
| --- | --- |
| 根对象 | 公共字段、`candidate_set_id`、`candidate_source`、`image_manifest:{path,sha256}`、`provenance`、`images` |
| provenance | `producer_run_id/checkpoint_sha256/decoder_config_sha256/annotation_manifest_sha256`；正式预测前三项非空，正式 GT 的 annotation hash 非空，其余不适用项 null；夹具允许全 null 并明确说明 |
| 每图 | `image_id/image_size_wh/status/candidates`；正式成功解码是 ok 或 no_candidates；推理出错为 inference_error，并带 error，不能伪装成正常零候选 |
| 每候选 | `candidate_id/category_id/score/bbox_xyxy/mask_reference/mask_sha256/decoder_query_id/source_annotation_id` |

`candidate_source` 为 gt/predicted/external。内部预测类别必为 1–8，score 为有限分割分数；GT score=null，decoder query=null，source annotation 为真实整数 ID。预测 source annotation=null，不在生产包中放评价匹配答案；内部预测保留 decoder query ID，一条 query 至多一个候选。external 可以类别/decoder query=null，不能因此送入要求八类的内部几何模块。

GT 按 source annotation ID 升序发布。预测按 score 降序、同分按 candidate ID 字典序发布；消费者不得自行重排。特征缓存保存该序列及候选包 hash。`no_candidates` 和 `inference_error` 的 candidates 都为空，但语义不同；错误状态必须传播到该图所有查询。

## 4. 每查询预测记录

文件：[成功](../examples/prediction.json)、[无候选](../examples/prediction_no_candidates.json)、[推理失败](../examples/prediction_inference_error.json)。三份文件是同一查询在**三个不同合成 run** 中的例子，不能拼成一次运行的三条结果。

必需字段：公共字段、`query_id/image_id/query/language/run_id/method_id/split/candidate_source/candidate_set_id/candidate_manifest:{path,sha256}/status/selected_candidate/error`。

| status | selected_candidate | error 与计分 |
| --- | --- | --- |
| success | `{candidate_id,grounding_score}`；ID 必须属于该图候选包 | error=null；success 仅表示有合法输出，不表示选对 |
| no_candidates | null | error=null；候选包该图确实为 no_candidates |
| no_output / rule_unsupported / rejected | null | error=null；核心有目标查询保留为失败 |
| inference_error | null | `{code,message}`，脱敏说明；保留为失败 |

正式 success 的 grounding score 为有限值；B0 无可比较连续分数时可 null。夹具无模型运行，score 为 null。预测文件只引用候选 ID，不重复框/mask，避免两份数据漂移；展示层从已验证候选包解析几何。E1 先把自身输出导出 external 包，再引用其选中项，不与 S0 包混用。

评价器以冻结查询清单做左连接：每个 run 必须覆盖全部 query ID，不接受未知/重复 ID；检查图片、split、方法和候选包 hash。缺记录首先视为运行不完整，补齐或由评价器显式合成 `no_output` 并标明来源，再按完整分母计分，不能只评价存在的成功记录。GT 的 source annotation 映射和预测的同类一对一匹配只在评价器执行。

## 5. 发布与接收

B 发布图片/GT/查询 manifest；lcy 发布预测候选；A/B 的方法输出预测；B 评价。发布顺序是 mask/图片 → 图片 manifest → 候选包 → 预测，避免循环 hash。正式标注 manifest hash 仅放 provenance，不在候选包中倒写引用自己的 hash。

接收方检查版本、hash、ID、尺寸、mask/框一致性、候选序列和失败记录，之后才建立缓存。示例检查命令见 [examples README](../examples/README.md)；正式转换器、schema 验证器、压缩 RLE 与真实图片验证仍由 B 实现，各模型负责人复核输出。
