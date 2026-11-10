# B0/B2 共享查询解析约定

版本 `rules-v1-draft`，待实现、待 dev 冻结。唯一词表为 [rule_parser.json](../configs/rule_parser.json)。B 负责词表与拟议 `data/query_rules.py`，A 复核并调用同一解析函数；B0/B2 评分仍分别放在各自模块。不得复制两份词表后独立修改。

## 1. 输入与输出

拟议接口 `parse_query(query: str, language: str, lexicon: dict) -> ParsedQuery`。只输入文字和语言，不接收人工类别、query type 或目标。返回：

```text
lexicon_version, normalized_text
category_id: 1..8 或 null
direction: left/right/top/bottom 或 null
category_status: unique/missing/conflict
direction_status: unique/missing/conflict
unsupported_operator: bool
unconsumed_spans: 未被类别、方位、语气词或标点消耗的文字片段
```

统一 NFKC、casefold、压缩空白，保留原始查询给 SigLIP 2。英文匹配单词边界，禁止在 `bright` 中匹配 `right`。中文全词表最长片段优先、同长按起始位置，已匹配字符不可再分配给短词，例如“连衣裙”优先于“裙”，“钱包”优先于“包”。跨类别的多个有效匹配为 conflict，同类别别名重复仍为 unique；多个不同方位为 conflict。unsupported operator 必须先检测，不能剥掉“不”后按肯定句解释。

配置中的 filler 只消耗完整词/合法中文片段，不能粗暴删除任意子串；未列出的修饰留在 unconsumed_spans。初始词表仅列已定义支持范围，不意味着覆盖全部自然语言。细类词映射到八类后不提供预测细类能力，例如 hat/watch 都归 accessory，应在 B0 局限和覆盖统计中说明。

## 2. 两种方法如何使用

- **B0：**类别须 unique、无冲突方位、无 unsupported operator、无未消耗修饰。无方向时仅当该类候选恰好一个才选；有方向时取该类框中心极值，同坐标并列不按 ID 硬选。否则输出 rule_unsupported（或零候选时 no_candidates）。
- **B2：**类别和方向都 unique，且无 unsupported operator 时，对该类别加入既定方位分数；颜色等 unconsumed 修饰不移除，仍由完整文字的 B1 处理。条件不满足时规则分数全零，保留 B1 结果，不将解析失败变成整条推理失败。
- GT 诊断使用候选包中的 GT 类别，预测条件使用预测类别。两者都不读查询标签中的 target_category_id。

## 3. 共同验收样例与冻结

| 查询 | 解析要求 | B0 / B2 行为 |
| --- | --- | --- |
| 左边的包 / the bag on the left | 类别 7，left | B0 选左端；B2 加左方位分 |
| 右边的红色包 / the red bag on the right | 类别 7，right；保留颜色片段 | B0 不支持；B2 仍用完整查询并加方位分 |
| 连衣裙 / the dress | 类别 5，无方向 | B0 仅唯一类候选可选；B2 等于 B1 |
| bright bag | 类别 7，无方向，bright 未消耗 | 不得误判 right |
| 左边和右边的包 | 方位 conflict | B0 不支持；B2 方位项为零 |
| 不是左边的包 / not the bag on the left | unsupported operator | B0 不支持；B2 方位项为零，仍保留完整文本 |
| the bag and the shoe | 类别 conflict | B0 不支持；B2 方位项为零 |

实现时将这些案例落成公共检查，A/B 使用相同版本和 hash。词表只能用 train/dev 修订，变更记录覆盖范围与原因；P7 冻结后 test 不再加词。报告 B2 可解析比例、实际存在该类候选的比例以及实际非零规则项的比例，避免将单候选零分误称为有效规则作用。当前没有解析器运行结果。
