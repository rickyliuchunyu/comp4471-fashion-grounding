# train1000 颜色与空间优先：逐图初查 v3

任务：P4 / wkq / data_eval，2026-11-17。状态：remote_cpu_checked / ready_for_human_review。
固定分支：`data-eval/p3-baseline-evaluation`，PR #4；代码版本见本文件 Git 历史。没有合并 main，没有训练或占用 GPU。

## 交付

新包：`/root/autodl-tmp/fashion-grounding/shared/processed/fp_language1000_visual_draft_v3`

审核页面和运行证据：`/root/autodl-tmp/fashion-grounding/runs/data_eval/20261117-train1000-standard-audit`，其中 `human-review/index.html` 为入口。30 页独立选择审核，每页最多 50 组；另有 30 页初标答案预览。页面内嵌图片，可离线使用。每页分别导出审核 JSON，保留文件名中的页码。
本机下载目录：`D:/COMP4471/comp4471-fashion-grounding/outputs/train1000-human-review-v3/index.html`。

| 项目 | 数量 |
| --- | ---: |
| 原始图片 / GT 实例 | 1000 / 4784 |
| 有当前语义查询的图片 | 999 |
| 中英文语义组 / 语言记录 | 1499 / 2998 |
| 原描述保留 / 修订 | 1309 / 162 |
| 原查询隔离 / 新替代目标 | 109 / 28 |
| 人工 accepted | 0 |
| 同项目类别多实例语义组 | 198 |

原有1580组减去109组隔离，再增加28组替代目标，得到1499组。对另2个问题目标重新看图后直接修订原语义组，计入162组，不重复计入新增。

八类语义组：top 336、pants 278、skirt 160、outerwear 206、dress 175、shoes 170、bag 72、accessory 102。
query type：appearance 1273、combined 205、spatial 21。中英文是配对翻译，不作为两张独立视觉样本。

## 标注标准与边界

全部1000张原图和原有查询目标裁剪已由Codex做首轮视觉检查；另对39张图的候选裁剪做集中裁决。优先明显颜色和必要的图片视角方位，保留清楚的条纹、格纹、印花等常见外观；移除细微装饰、不能可靠辨认的材质和依赖手/身体关系的描述。没有从dev复制标注，也没有用模型得分决定答案。

初查不能代替人工独立核验。唯一性是本轮视觉判断，并非代码证明；原数据对其他人物存在漏标时，仍按完整画面检查可见竞争目标。语义初标可能仍有颜色、译文或mask误判。所有行保持 `needs_revision`，reviewer为空；`queries_accepted.jsonl`为空。

图片38309的原GT将整件连衣裙错误分为上衣和裙子，没有清楚的替代候选。本次保留原图和全部mask，只隔离该图语义查询，等待人工裁决。其他被隔离的原查询及原因在 `queries_quarantined.jsonl` / `visual_audit.json`，未悄悄删除源标注。

## wkq 怎样审核

打开入口，逐页独立选择描述对应的候选，核对唯一性、翻译和所选mask，不必把所有框都重新标一遍。有问题选“需要修改”，留下原因；每页单独导出JSON。
若想先看Codex效果，入口下半部分有答案预览，显示所选候选及青色mask。看过答案的图片请在审核备注注明“看过答案预览”，该次核验不能算独立盲审。原试点答案曝光记录继续保留。
返回JSON后仍需审计manifest/hash、配对关系和分歧，再发布人工验收版；不要直接覆盖训练答案。

## wyz 如何接着用

1. 使用新包的 `image_manifest.json`、原图、`annotations_coco.json`、`candidates_gt.json` 和 `queries_machine_train.jsonl` 做**机器初标实验**；该查询文件与 `queries_draft.jsonl` 字节相同，显式标记实验用途。
2. 以查询记录实际引用的999张图构建监督，不要假定manifest每图都有语言标签。隔离的218条旧语言记录不得加入训练。
3. 训练输入仍只用query文本和图像候选特征；目标ID等字段只作监督/评价。GT候选诊断与预测候选实验分开，不用GT修正分割预测。
4. 新manifest/query/candidate hash需登记。候选几何、ID、顺序及mask内容没变，但包标识和manifest引用变了；不要跳过缓存版本检查。文本特征需按新查询重新生成。
5. dev继续用 `fp_language_dev250_simple_draft_v4`，同样尚待人工验收。正式泛化结论等待人工验收；若先做机器标签诊断，必须明确标识。模型加载、训练和收益由wyz验收登记，本轮没有代验。

lcy的10k分割包保持不变，本次无需因语言修改重新制作分割标注。

## 覆盖情况与验证集建议

显式文字覆盖按语义组去重计数：绿色由8增至23，粉色0→15，紫色0→8，橙色0→6；黑色仍712组、白色157、灰色130，黄色仅1组。详细前后统计和按类别缺口见 [覆盖报告](train1000_visual_v3_evidence/training_text_coverage.json)。这只是文本提及统计，不证明某视觉特征不存在或模型学不会；也不承诺本轮提升精度。
同类多实例仅198/1499组，和期望困难比例仍有差距。后续更值得定向补充颜色区分、同类竞争与自然改写，而不是简单复制左右鞋模板。

本轮保持已有dev250/test400原划分，**没有扩充dev**。建议先完成dev250人工核验并获得wyz按颜色、空间、同类竞争分组的结果，再决定是否增加独立补充验证集。若扩充，从官方validation中未分配且不与既有train/dev/test重复的图片组另建补充版本；不抽取test，不把补充集不断改写成模型容易答对的描述。新增数量由实际可用去重池和类别缺口决定，不能仅凭更大的图片数保证泛化测量质量。

## 验证与复现

- 126项CPU测试通过，新增训练标注构建的跨图目标、非train输入、未知目标、缺双语、漏审、身体关系及隔离后覆盖等边界测试。
- 新包schema/ID/尺寸/框/mask/目标引用校验通过；4784个源mask逐像素一致，4784候选裁剪检查通过。
- 1000张图片与4784个mask资产字节不变；候选内容/顺序不变，旧image/group/split记录不变。
- 与dev250、test400、dev语言v4的image ID/group/hash/source ID交集均为0；上述旧包完整checksums复核通过。没有查看test图片或生成test描述。
- pHash在本训练子集内检出29246/31540一对候选近重复，均在train；不改写原分组，不冒充独立新样本。不保证已识别所有裁剪或同人物近似图片。
- 30页审核payload未嵌入目标答案；30页答案预览全部目标引用有效，QuickJS DOM桩执行初始显示/前后切换通过。现有审核页导出/选择逻辑有单元测试；本轮未进行完整浏览器人工渲染验收。
- 证据：[发布](train1000_visual_v3_evidence/RELEASE.json)、[验证](train1000_visual_v3_evidence/validation_report.json)、[源mask检查](train1000_visual_v3_evidence/consumer_check_report.json)、[划分保留](train1000_visual_v3_evidence/partition_preservation_report.json)、[视觉决定](train1000_visual_v3_evidence/visual_decisions.json)。原图预览保存在数据盘，不提交Git。

环境：`/root/autodl-tmp/fashion-grounding/envs/data_eval/bin/python`；工作目录 `/root/autodl-tmp/fashion-grounding/workspaces/data_eval`，仅CPU，不改其他成员环境。

核心复现（输出必须选新目录；完整视觉决定和索引在本交接同级evidence目录）：

```bash
export PYTHONPATH=src
PY=/root/autodl-tmp/fashion-grounding/envs/data_eval/bin/python
BASE=/root/autodl-tmp/fashion-grounding
OUT=$BASE/shared/processed/fp_language1000_visual_draft_reproduction
EVIDENCE=docs/handoffs/data_eval/train1000_visual_v3_evidence
$PY scripts/build_train_visual_draft.py --parent $BASE/shared/processed/fp_language1000_machine_v2 --decisions $EVIDENCE/visual_decisions.json --index $EVIDENCE/visual_index.json --dev $BASE/shared/processed/fp_language_dev250_simple_draft_v4 --output $OUT
$PY scripts/check_pilot_consumers.py $OUT
$PY scripts/render_review_batches.py $OUT --visual-train --page-size 50 --output $BASE/runs/data_eval/train1000-reproduction-review
$PY scripts/render_train_answer_preview.py $OUT $BASE/runs/data_eval/train1000-reproduction-review
$PY -m unittest discover -s tests
```

发布SHA256：
- queries_draft / queries_machine_train：`ebde5c2bd4a61b908f19a2bf9e202aa7c10c2c5684154367db708cbcf56dd8c4`
- image_manifest：`1bbff2b4a315b597af195097571022dfed4e432a91ad672e8b3c892e69b24f44`
- candidates_gt：`097f6f8cebb605e25fa6f6ec0250eb197145a88f2d9e09e2252e3a93a239f6a5`
- SHA256SUMS：`b5680a595a743e1a61432d85ecf7ac9f1a7a8371c94e8db7b53a6e1e1f167575`

运行日志在上述run目录：`tests.log`、`consumer.log`、`audit_summary.json`、`review_render_report.json`。源码实现hash已写入RELEASE。数据构建不填模型指标，未改训练实验计划为已完成；状态表同步本交接，真实下游模型消费/人工验收仍待完成。
