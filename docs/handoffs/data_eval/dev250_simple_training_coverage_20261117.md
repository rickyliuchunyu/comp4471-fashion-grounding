# dev250 v4 与固定训练集的文字监督覆盖

按中英semantic_query_id去重；同一组可含多个特征。train=1580组/1000图，dev=330组/250图。此表统计明确文字提及，不是视觉属性真值或模型能力评价；不用于更新推理规则词表。详细按目标类别和逐查询缺口见包内training_text_coverage.json。

| 特征 | train语义组 | train图片 | dev语义组 | dev图片 |
| --- | ---: | ---: | ---: | ---: |
| black | 772 | 557 | 89 | 75 |
| white | 148 | 135 | 45 | 44 |
| gray | 136 | 118 | 17 | 17 |
| blue | 86 | 78 | 32 | 31 |
| red | 55 | 47 | 33 | 29 |
| green | 8 | 6 | 11 | 11 |
| pink | 0 | 0 | 11 | 11 |
| purple | 0 | 0 | 7 | 7 |
| orange | 0 | 0 | 7 | 7 |
| yellow | 1 | 1 | 7 | 7 |
| brown | 0 | 0 | 13 | 13 |
| gold | 0 | 0 | 3 | 3 |
| silver | 0 | 0 | 2 | 2 |
| light | 3 | 3 | 36 | 35 |
| dark | 1 | 1 | 36 | 35 |
| multicolor | 0 | 0 | 9 | 9 |
| left | 98 | 98 | 8 | 8 |
| right | 98 | 98 | 7 | 6 |
| middle | 0 | 0 | 2 | 2 |
| lower | 0 | 0 | 1 | 1 |
| stripe | 115 | 104 | 22 | 20 |
| check | 85 | 77 | 7 | 7 |
| dots | 41 | 39 | 3 | 3 |
| floral | 103 | 96 | 6 | 6 |
| pattern | 104 | 97 | 39 | 39 |
| leopard | 0 | 0 | 3 | 3 |
| camouflage | 0 | 0 | 1 | 1 |
| denim | 0 | 0 | 8 | 8 |
| hooded | 0 | 0 | 4 | 4 |
| long_sleeve | 0 | 0 | 20 | 20 |
| short_sleeve | 0 | 0 | 16 | 16 |
| sleeveless | 0 | 0 | 23 | 23 |
| strapless | 0 | 0 | 5 | 5 |
| cropped | 1 | 1 | 4 | 4 |

185组dev触发全局缺失、全局稀少（1—9组）或目标类别缺失中的至少一项。阈值仅用于筛查。保留真实dev语义，不为了覆盖率隐藏缺口；后续训练补标只能在原train分区独立选图进行。本轮未改训练标签、未训练。

训练文件SHA256：c577a56575e12e1dbf7072a294ef97674dbb4bf3f6f4ed2a8de645ac3177b49d
dev文件SHA256：970dc962ae93484ec80cfe00af74969b88d5bb74c06e56f28e3d17f804da8489
