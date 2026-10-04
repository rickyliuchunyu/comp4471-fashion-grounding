# 数据转换与双语查询标注协议

本文定义数据进入训练、标注和评估前应满足的条件。源类别已依据官方标注查证；目标映射和划分规则是项目设计，转换程序、完整数据下载和查询标注均未完成。机器可读映射见 [source_category_mapping.json](../configs/source_category_mapping.json)。

## 1. 数据来源与监督覆盖

| 数据源 | 已核对的事实 | 项目中的用途 |
| --- | --- | --- |
| [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) | 13 种服装类别，提供实例框与多边形；不提供鞋、包、配饰标签 | 五类服装监督、跨数据源比较 |
| [Fashionpedia](https://github.com/cvdfoundation/fashionpedia) | 27 种主体类别、19 种部件类别，含实例与属性标注 | 八类完整监督，初始双语查询数据 |

2026 年 10 月 4 日读取官方 [Fashionpedia val JSON](https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json)，其中包含 **1,158 张图片和 8,781 条原始 annotations**。后者包括部件，不是八类整件实例数量。过滤后数量尚未知；不能在报告中直接用 8,781 作为目标实例数。图片包可能同时含验证与无公开标签的测试图片，训练与评估只能读取与标注清单交集中的文件。

官方测试标签不一定公开，不承诺在本地得到其 test AP。项目自建的 final test 来自已公开标注且预先隔离的验证数据，应称为“项目保留测试集”，不是官方测试服务器成绩。实际下载后按文件统计数据量，不沿用来源不一致的宣传数量。

## 2. 完整类别映射

项目 ID 为 1–8，训练 ID 为项目 ID 减一；两个源数据集的原始 ID 起点不同，不能直接合并。

### 2.1 DeepFashion2

| 原始 ID | 官方类别名 | 项目 ID 与类别 |
| --- | --- | --- |
| 1 | short sleeve top | 1 top |
| 2 | long sleeve top | 1 top |
| 3 | short sleeve outwear | 4 outerwear |
| 4 | long sleeve outwear | 4 outerwear |
| 5 | vest | 1 top |
| 6 | sling | 1 top |
| 7 | shorts | 2 pants |
| 8 | trousers | 2 pants |
| 9 | skirt | 3 skirt |
| 10 | short sleeve dress | 5 dress |
| 11 | long sleeve dress | 5 dress |
| 12 | vest dress | 5 dress |
| 13 | sling dress | 5 dress |

类别拼写沿用源名称，包括 `outwear`。这些类别定义由 [官方 annotation 说明](https://github.com/switchablenorms/DeepFashion2#annotation) 核对。源监督覆盖只有项目 1–5，未标注项目 6–8 不等于它们在图片中不存在。普通混合监督与拟采用的集合分类损失必须分开比较，详见 [模型设计](technical_design.md#23-混合数据中的缺失类别监督)。

### 2.2 Fashionpedia

下表的 ID 与名字来自上述官方 val JSON；合并方式是项目选择。

| 原始 ID | 官方类别名 | 项目处理 |
| --- | --- | --- |
| 0 | shirt, blouse | 1 top |
| 1 | top, t-shirt, sweatshirt | 1 top |
| 2 | sweater | 1 top |
| 3 | cardigan | 4 outerwear |
| 4 | jacket | 4 outerwear |
| 5 | vest | 1 top |
| 6 | pants | 2 pants |
| 7 | shorts | 2 pants |
| 8 | skirt | 3 skirt |
| 9 | coat | 4 outerwear |
| 10 | dress | 5 dress |
| 11 | jumpsuit | 排除整张含该标注的图片 |
| 12 | cape | 4 outerwear |
| 13 | glasses | 8 accessory |
| 14 | hat | 8 accessory |
| 15 | headband, head covering, hair accessory | 8 accessory |
| 16 | tie | 8 accessory |
| 17 | glove | 8 accessory |
| 18 | watch | 8 accessory |
| 19 | belt | 8 accessory |
| 20 | leg warmer | 8 accessory |
| 21 | tights, stockings | 8 accessory |
| 22 | sock | 8 accessory |
| 23 | shoe | 6 shoes |
| 24 | bag, wallet | 7 bag |
| 25 | scarf | 8 accessory |
| 26 | umbrella | 8 accessory |
| 27–45 | hood、collar、lapel、epaulette、sleeve、pocket、neckline、buckle、zipper、applique、bead、bow、flower、fringe、ribbon、rivet、ruffle、sequin、tassel | 只排除这条部件 annotation |

`accessory` 是明确列举的宽泛类别，包含袜子和雨伞；不是把未知类别自动归入“其他”。保留 `source_category_id/name`，以便检查合并后帽子、手表等小物体的差异。`bag` 包含源类别里的钱包；“钱包”可以作为描述，但八类输出仍是 bag。cardigan/cape 归入外套，这是团队的语义约定，应在报告中说明。

连体裤 jumpsuit 没有合适的八类目标。起始协议排除含 jumpsuit 的**整张图片**，避免删除其 mask 后把可见主体当成背景。记录排除数量和类别分布变化；若导致某些类样本严重不足，须在训练前重新讨论范围并更新映射版本。服饰部件则与完整衣物重叠，只删除部件标注，保留整件衣物。

不自行把左右鞋合并成一双，也不拆分官方将多区域标为同一实例的 mask。query 中“一双鞋”只有在该 annotation 确实表示目标集合时才适用；首阶段优先使用“左侧那只鞋”等唯一实例描述。

## 3. 统一 COCO 表示

实例标注输出采用 COCO 格式：`images`、`annotations`、`categories`。额外字段由框架 mapper 读取，不能指望标准 COCO evaluator 自动处理缺失类别监督。

| 字段 | 类型与规则 |
| --- | --- |
| image.id | 全局唯一正整数；按冻结源清单生成，不由文件遍历随机决定 |
| image.file_name | 数据根目录下相对路径；加入源前缀，防止同名冲突 |
| image.width / height | 实际解码后的原始尺寸，整数 |
| image.source_dataset / source_image_id | 来源名与原始 ID 字符串，保留前导零 |
| image.supervised_category_ids | FP 为 1–8；DF2 为 1–5，供 source aware criterion 使用 |
| annotation.id / image_id | 全局唯一整数及其图片 ID |
| annotation.category_id | 项目 1–8，不是训练 0–7 |
| annotation.segmentation | 合法 COCO polygon 或 RLE；不把多个分量拆成多个实例 |
| annotation.bbox | COCO `[x, y, width, height]`，从有效 mask 重新计算 |
| annotation.area | mask 前景像素数，不是框面积 |
| annotation.iscrowd | 保留原字段；标准评估按 COCO 规则处理 |
| annotation.source_annotation_id / source_category_id | 溯源 ID 与原类别，用于复核 |

多边形可以有多个分量，允许实例间重叠。解码必须使用 `pycocotools` 等 COCO 兼容工具；压缩 RLE 的 counts 导出成 JSON 字符串，记录编码方式。与训练边界框转换时使用排他右下界：$x_{max}=x+width$，$y_{max}=y+height$。

转换不能凭肉眼框位置修复类别，也不能 silently clip 后继续训练。错误进入 `conversion_issues.jsonl`，附源 ID、错误类型和处理决定。无效多边形、空 mask、尺寸不符、未知类、缺图和重复 ID 先隔离；修复后重新生成版本化清单。只有部件的图片排除；保留困难但有效的整件实例，不删除小目标来美化指标。

## 4. 划分与防泄漏

起始 split seed 固定为 2026。保持官方 train/validation 边界，绝不把官方验证图片移动到训练集。各源的可用 validation 在完成排除和分组后，按组约 1:1 分为项目 dev 与 final test，尽量平衡类别；实际数量和各类缺失情况在审查后记录。

1. 先建立图片文件 SHA256、源 ID 和 group ID 清单。完全相同图片不得跨划分。
2. DeepFashion2 同商品 `pair_id` 及可用的 item style 建立组；多商品图片通过共享标识连接成组。只对存在且有效的标识分组，不能把缺失 ID 的图片全部视为一个商品。来源关联字段见 [官方标注](https://github.com/switchablenorms/DeepFashion2#annotation)。
3. Fashionpedia 对可追踪同图/同商品及近重复建立组。pHash 的起始人工复核阈值为 Hamming distance ≤ 6；它只是发现线索，不是充分的重复判定。裁剪和翻转近似图需人工确认，不能承诺已识别所有同人物图片。
4. 组若跨官方 train 与 validation，将冲突的 validation 组隔离并记录数量，不挪入训练。同一 validation 组整体进入 dev 或 test。
5. 生成固定 manifest、类别统计与 SHA256；三位成员复核后冻结。在各 split 内生成查询，中文、英文、改写、属性和配对查询全部跟随原图。

开发集用于 checkpoint、阈值和预先限定超参数选择。最终测试在方法锁定后统一评估，不通过查看失败案例不断调参。若测试数据本身有标注错误，按同一公开修订清单修正所有方法，并保留旧版本结果。

## 5. 查询标注的数量与类型

第一版只在 Fashionpedia 上建立语言集合，因为它提供八类整件监督。DeepFashion2 仍参与分割训练和评估；暂不承诺其语言描述数量。下面是**待执行的工作量目标**，不是现有数据或课程硬性数量要求：

| 阶段 | train 图片 | dev 图片 | final test 图片 | 预期查询数 |
| --- | --- | --- | --- | --- |
| 流程试点 | 100 | 25 | 25 | 最多 600 |
| 第一版目标 | 600 | 150 | 150 | 最多 3,600：2,400 / 600 / 600 |

每图拟标两条语义不同的描述，每条有中文和英文，共四条记录。两条可以描述不同目标，也可以从位置与外观两个角度描述同一个目标；不得为了凑数强迫无外观区分的图片产生颜色描述。配对翻译共用 `semantic_query_id`，反事实左右描述共享 `contrast_set_id`。实际数量不足时如实记录，不用重复改写冒充新的图片或实例。

分层采样目标为覆盖全部八类，并让约一半的语义描述来自有两个及以上同类实例的图片。这些条件可能受源数据限制，先审查再调整。每组都记录图片数、目标数、语义描述数和语言记录数；中英文翻译不是独立视觉样本。

| query type | 合格例子 | 必须确认的内容 |
| --- | --- | --- |
| category | “这条裤子” / “the trousers” | 图片只有一个符合该类的有效实例 |
| spatial | “右边的包” / “the bag on the right” | 在同类实例中位置唯一，以图片视角为准 |
| appearance | “红色条纹上衣” / “the red striped top” | 颜色与条纹可见，且能与其他候选区分 |
| combined | “左侧黑色外套” / “the black coat on the left” | 类别、位置和外观共同指向一个目标 |

不自动从类别猜颜色、材质和图案。Fashionpedia 属性可以作为待核对线索，但并不等于任意颜色或材质描述已有可靠标注。核心查询只用可见且一致的属性；“丝绸”“真皮”等不能仅从外观确定的材质暂不纳入。

## 6. 标注记录与复核

[query_record.json](../examples/query_record.json) 是人工格式示例；真实记录至少包含下列字段：

```text
query_id: 唯一字符串
semantic_query_id: 中英文同义配对 ID
contrast_set_id: 反事实描述组 ID 或 null
image_id / target_instance_id: 转换后 COCO 整数 ID
source_dataset / split: 来源与冻结划分
query / language: 完整文字与 zh/en
query_types: category/spatial/appearance/combined，可多标签
target_category_id: 1–8，仅标注与评估使用
same_category_instance_count: 全图有效非 crowd 实例中的目标类别计数
spatial_reference: image_view
annotation_version / annotator / reviewer / review_status
```

先由一人写描述，再由另一人只看图片和描述独立选目标，随后比对 annotation ID。第二人选错、多个目标都符合、掩膜本身不可靠或翻译改变含义时，标为 `needs_revision`；两人确认一致才标 `accepted`。原 query 和修改原因保留在复核记录，不用训练或测试结果指导重写难例。

真实非 crowd 标注候选决定描述的唯一性。查询中不含 annotation ID、文件名或标签坐标；类别、query type、目标 ID 和复核字段不送入文本模型。语言对照除 query 文字外读取同一候选集合，不能为 L1 提供人工类别而让 L2 自行预测。

## 7. 转换完成后的检查产物

正式训练前须生成可复核的数据报告，至少包括：

- 源文件/原始标注哈希、映射版本、转换代码 commit 和处理日期。
- 排除前后各源各 split 的图片数、八类实例数、部件数、crowd 数及错误原因统计。
- mask 面積的 small/medium/large 分布、每图实例数与同类多实例数量。
- group overlap、文件 hash overlap 及近重复人工复核记录。
- 每源每类随机可视化至少 20 个实例；不足 20 个则全部检查。
- 查询覆盖、语言配对、同类负例、复核比例及未通过数量。

硬性检查包括：所有 ID 唯一、mask 非空且尺寸正确、框与 mask 一致、目标存在、query split 等于 image split、accepted 记录有不同的 annotator/reviewer。通过这些检查才能把 `mapping_implemented` 和 `queries_reviewed` 标为 true。当前两者均为 false。
