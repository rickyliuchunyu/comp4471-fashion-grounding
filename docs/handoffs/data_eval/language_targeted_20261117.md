# 语言训练定向增量与独立 dev 困难集

日期：2026-11-17；执行人 wkq / data_eval；固定分支 `data-eval/p3-baseline-evaluation`，PR #4。状态：远端 CPU 数据准备与实际消费入口检查通过；AI 逐图初标，人工审核待完成；没有训练、缓存编码或 GPU 使用。最终代码版本见本文件 Git 历史；执行时基线为 `6eb7b923d32142e2510ffc87e6adf0e549a8f743`，未提交脚本的实际 SHA 在证据中。

## 最终交付

共享根目录：`/root/autodl-tmp/fashion-grounding/shared/processed/`。

| 包 | 图片 / 有查询图片 | 语义组 / 中英文记录 | 用途 |
| --- | ---: | ---: | --- |
| `fp_language1149_targeted_draft_v1` | 1149 / 1148 | 1648 / 3296 | 清理后原 train1000 + 新增149图，实验性弱标注训练 |
| `fp_language_dev39_challenge_draft_v1` | 39 / 39 | 39 / 78 | 独立定向 dev 困难集，人工审核后单独评价 |
| 原 `fp_language_dev250_reviewed_v1` | 250 / 246 | 321 / 642 accepted | 主 dev，保持不变；原9组 pending保持不变 |

训练增量仅149组/298条，不把翻译算作额外图片。原1000图的2998条查询逐记录保留，原本1张没有合格查询的图片仍未补造答案。新包所有训练记录仍为 `needs_revision`、reviewer=null；`queries_machine_train.jsonl` 是明确允许实验性机器标注训练的输入，`queries_accepted.jsonl` 为空。不能声称已完成人工验收。
新 dev 的 `queries_machine_train.jsonl` 和 accepted 都为空，禁止送入训练或未审核就正式计分。

原 train5000、train10000、train1000、主 dev250、test400 的划分和资产未修改。新增训练图均属于已发布的官方 train 10k 包；没有增加新的分割监督或改写任何 mask。
新 dev 来自原 unused 官方 validation，不是从 test400 取图。50张被实际看过的候选中，39张保留、11张质量隔离；全部50张在 `dev_exposure_registry.json` 登记为开发曝光，未来不得放入 final test 或训练。

## 初标过程与实际覆盖

以颜色、图片视角位置为主，保留明显的常见外观；不使用材质猜测、精细工艺或“拿在手上/穿在身上”等人体关系。每个候选看原图与目标 mask 裁剪；同图其他区域用于判断唯一性。没有用模型预测替代目标标注，没有把图文模型准确率用于挑选这些图片。
新训练先按同源类别竞争、鞋、包、小配饰定向抽200图；补筛20图；最终149图保留、71图隔离。新dev筛50图，39图保留。数量是质量筛选后的实际值，不为达到200/50而接受含糊标注。

发现大量同源类别“多实例”来自单件裤子左右腿、外套左右片、袖口或只有包带的标注。保留这些官方实例在候选/分割数据中的原样，仅不为不合格目标编写整件语言监督；不宣称官方分割标注整体错误。
对5条空间/竞争关系警报额外放大原图和全部同源类别 mask：隔离3条歧义、纠正2条左右文字。临时1152包已移入运行目录 `staging1152-superseded`，不作为下游输入；最终只使用1149包。

| 新训练查询类型 | 语义组 |
| --- | ---: |
| 外观 | 84 |
| 颜色/外观 + 空间 | 33 |
| 纯空间 | 32 |

149组中，110组（73.83%）存在同项目类别候选竞争，89组存在同源细类别候选竞争。**这些是候选标注数量指标，不能等同于真实独立整件数量或已验收困难度。**
新增目标：bag56、accessory40、shoes31、top16、dress4、outerwear1、skirt1；裤子没有合格新增目标。原八类训练覆盖保留，本增量有意偏向弱项，不是八类均衡扩充。
新dev有25/39组同项目类别竞争、23/39组同源类别竞争；外观24、combined11、spatial4。包含bag11、top5、shoes13、accessory10。
没有伪造反事实对或独立人工自然改写；当前 contrast_set_id / paraphrase_group_id 保持 null。

## 去重、源保真与验证

- 新增训练相对原1000图，image ID、group、SHA均无重叠；相对dev/test无跨区重叠。
- 独立重算保护集10k train + 固定dev/test + 所有unused validation的连通分量，使用SHA、资源级来源URL、原图/镜像全部四方向pHash Hamming<=6。最终新训练/新dev冲突均0。没有改写原冻结group。pHash是保守线索，不能保证所有裁剪/同商品身份已识别。
- `validate_package` 全包通过；原1000记录、annotations和候选顺序逐项保留；旧发布文件校验/索引留证。
- 使用 wyz 工作区固定 `103dcc546714dd7c7866b4259d96a9825d8090ef` 的只读 `scripts/check_grounding_data_release.py`，在其现有 grounding-v1 runtime 运行，输出只写wkq运行目录；未改动该环境或工作区。
- 两包共7122个SHA条目、5908个候选（5708+200）实际图像裁剪、候选/COCO/原始源mask三方逐像素检查通过；中英文配对、目标引用、split均通过。
- 实际 `weak_trial.context` 加载3296条train查询，目标字段不进入推理Query；36条单候选查询沿用原包，训练入口应按既有策略跳过其排名loss，仍保留统计。
- 142项CPU测试通过，新增覆盖缺审核决定、缺双语、人体关系、无原因剔除、资产碰撞等。
- 训练6页/149组、dev2页/39组的载荷覆盖及目标隐藏验证通过；另行答案页与初标目标逐项一致。未执行浏览器交互自动化，不将载荷检查冒称UI人工验收。

这只是技术消费验证，由wkq执行，不替wyz/lcy签收；没有训练收益、分割AP或新模型成功率可报告。

## 给 wyz：接着训练什么

1. 基线对照使用 `fp_language1000_visual_draft_v3`，扩充组使用 `fp_language1149_targeted_draft_v1`。两组均读 `queries_machine_train.jsonl`、`candidates_gt.json`；比较清理后的1000 vs1149，不能把旧machine_v2与新扩充包的差异全部归因于数量。
2. 现有 `fashion_grounding.grounding.weak_trial --phase cache/train --weak-label-trial` 的数据入口已实际检查可读取新包。新查询/候选manifest改变，按实际cache key核查，使用新的run ID和输出目录；GPU缓存需协调同一GPU锁。
3. 同编码器、候选条件、种子、优化器和预登记训练预算；至少单独报告总更新数，避免图片更多带来的更新增加被混作数据效果。没有在本轮替你选择epoch或训练预算。
4. M1仅外观，新增左右文字不能替代原图几何；空间改进仍需M2或B2。M1/M2用同一训练/评价清单。旧M1权重本轮没有更新。
5. 主dev仍用已审核642条；新39图困难集待wkq复核后**另报**，不要混成更大自然分布dev并宣称泛化率提高。最终test保持封存。
6. 如果后续启用预测候选适配，只使用训练图预测候选，记录漏检排除率；M1/M2保持同一协议。此适配本轮未执行。

## 给 lcy：需要接什么

当前分割训练仍使用原 `fp_train10000_targeted_v1`。这149张是其中的语言增量，不需要重新制作分割标注。
如wyz需要训练预测候选，可按1149包的image_manifest导出固定权重/阈值的训练预测，并登记hash；不要用GT修正预测mask。
新dev39的端到端候选可在人工审核后，沿用已锁定或明确登记的分割权重/阈值另行导出；不混用不同候选版本。队友已有训练和GPU进程本轮未改动。

## 人工审核入口

服务器运行目录：
`/root/autodl-tmp/fashion-grounding/runs/data_eval/20261117-language-targeted-expansion-r02`

- `train-review-final/index.html`：仅新增149组，共6页；不要求重新审核原1000图。
- `dev-review/index.html`：新增39组，共2页。
- 每页单独导出JSON。正常时选择对应候选并标一致；不唯一、mask不符合整件或翻译不一致，标needs_revision并备注。
- 两个目录均有独立“AI初标答案预览”；看过答案后请注明曝光，不能冒称独立盲审。
- 原主dev的9组pending仍单独处理，本轮未替用户裁决。

本机副本位于 `D:/COMP4471/language_expansion/train-review-final` 与 `dev-review`。旧 `train-review` 临时预览已注明废弃。

## 复现与证据

代码：`scripts/prepare_language_expansion.py`、`scripts/publish_language_expansion.py`、`scripts/render_language_increment_review.py`。
运行目录保留初选/补选清单、全部270张看图决定、空间复核图、源dev50包、去重脚本/报告、构建日志、消费检查、review HTML、最终hash与元信息。复现最终包使用已冻结的selection与decisions文件，输出到新的版本目录，不覆盖现有发布包。

实际最终构建参数：
- train：source=`shared/processed/fp_train10000_targeted_v1`，base=`fp_language1000_visual_draft_v3`，selection=`train-selection-complete.json`，decisions=`train-decisions.json`，split=train。
- dev：source=`<run>/dev50_source`，selection=`dev-selection.json`，decisions=`dev-decisions.json`，split=dev；无base。
- 入口：`PYTHONPATH=src <envs/data_eval/bin/python> scripts/publish_language_expansion.py --source ... --base ... --selection ... --decisions ... --output <new-version> --split train`；dev省略base。实际输出及检查见 `train-build-final.log`、`dev-build.log`。
- 审核入口：`PYTHONPATH=src <envs/data_eval/bin/python> scripts/render_language_increment_review.py <package> <new-review-folder>`。
- 独立消费入口在grounding工作区：`PYTHONPATH=src <envs/grounding-v1/bin/python> scripts/check_grounding_data_release.py --packages <train> <dev> --output <new-report>`。

[小型证据目录](language_targeted_20261117_evidence/)保存发布索引/hash、schema/分布、实际消费和弱训练入口检查、曝光登记及运行元信息。没有把图片、完整查询、权重或缓存提交Git。数据准备未往模型实验registry写虚构结果。

下一步：wkq审核新增dev和训练草稿；wyz可先按明确弱标注模式开展训练对照；lcy沿用现有分割训练计划。训练与审核可并行，最终正式结论仍须以合格评价清单为准。
