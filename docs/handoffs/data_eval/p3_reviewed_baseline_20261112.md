# wkq P2 合格查询与 P3 CPU 实现交接
日期：2026-11-12；分支 data-eval/p3-baseline-evaluation；实现提交 1c62ec8df7cd8b7c47b285bf1905905330091882。
状态：implemented / remote_cpu_checked / training_pilot_diagnostic。不是模型训练完成或正式 dev/test 成绩。

## 输入与发布
独立工作区 /root/autodl-tmp/fashion-grounding/workspaces/data_eval，独立环境 /root/autodl-tmp/fashion-grounding/envs/data_eval。
输入 fp_grounded9_draft_v2 及 wkq 导出审核，审核 SHA256：
71d306081ef9ac229962394c2f9adae0a314b158cd9494e21bb1f9f76d99cc60。
原件留在 runs/data_eval/20261112-grounded-human-review/review.json，不上传公开仓库。

共享合格包 /root/autodl-tmp/fashion-grounding/shared/processed/fp_grounded9_reviewed_v1。
17 语义 / 34 双语 accepted，8 张图有合格查询；包保留全部9图/59原候选，1语义/2双语 pending。
其中5组/10条是旧图再审，review_mode 明确保留 previously_exposed_training_rereview；全部仍是机器辅助训练数据，不是独立人工测试描述。
发布门槛：人工明确选择、与初标一致、唯一性/翻译/mask三项全真、文字未改、无备注；否决/改写不自动升级。
旧6组分歧历史不倒写，也不将新查询通过称为旧记录裁决完成。

## mask 核查
q014 / image13898：源 annotation354 转为项目355，mask面积4206，xyxy=[309,286,346,515]。
源标注和审核页均存在该候选，解码后确实只覆盖外套间的一条窄内搭；原图与精确mask对照见包内 mask13898.jpg。
用户反馈无法找到相应mask，记录在 pending_review.json。暂缓该组，不擅自将用户拒绝改成通过，也不重画/合并源mask。

## 实现及接口
- data/query_rules.py：parse_query(query, language, lexicon)，NFKC/casefold，英文词边界、最长非重叠匹配、冲突/否定/残留文本。
- 唯一词表 configs/rule_parser.json，rules-v1-pilot。只针对已知训练描述补充中性画面词“画面 / image / side”；未增加颜色规则，未 dev 冻结。
- baselines/category_position.py：select_target(query, language, candidates, lexicon, image_status)，只依据文字与候选类别/框；未知修饰、多候选纯类别、方向并列明确失败。
- data/candidates.py：schema/hash/图片尺寸/mask/框/候选顺序/重复ID/decoder及来源检查。
- evaluation/matching.py：按score降序、ID稳定排序，同类非crowd、未占用GT最高maskIoU>=0.5，一对一；GT同IoU按annotation ID。
- evaluation/metrics.py：GT oracle、predicted/conditional accuracy、coverage、几何recall、IoU、Success、macro、语言/类别/细类/类型/候选数/同类数/面积分组和配对统计。
- external 候选只报几何指标，不强称内部选择准确率。缺预测、重复/未知query、混run/method、来源hash不一致均拒绝评分；空候选/错误保留分母。
- CLI 显式 train diagnostic 开关；推理构造字段白名单，不将target ID、人工类别、query type传给B0。
- 不实现 B2 评分或改 wyz 的模块；共享解析的残留外观词供B2保留完整文本使用。

## 运行与验证
运行目录 /root/autodl-tmp/fashion-grounding/runs/data_eval/20261112-p3。
74 个独立测试通过；包括原35项、新解析/B0/手算评价31项、文件流水线与发布门槛8项。
覆盖正确、漏检、错类重叠、重复预测、同分/同IoU、0.5边界、选错仍算实际IoU、空候选/上游错误、缺记录、NaN、答案隔离、hash篡改、再审记录及禁止test自动发布。
59 mask与源逐像素相同、59裁剪有效，pip check通过；发布84个文件hash核对通过。未用GPU、未改他人环境。
B0-train-gt-pilot-r01：34条查询/8图，12 success且全部选对，22 rule_unsupported；oracle=12/34=0.352941，macro=0.25。
分母包含全部不支持查询。这是小型训练试点GT诊断，不是泛化成绩；中文/英文均为6/17。
B1/B2、真实预测候选、外部系统尚无实际运行；预测/external逻辑目前只经过手算合成测试。bootstrap/多seed汇总、COCO AP、计时与大规模性能不在本次已完成范围。

## 可运行命令
从上述工作区执行：
```bash
PY=/root/autodl-tmp/fashion-grounding/envs/data_eval/bin/python
PACKAGE=/root/autodl-tmp/fashion-grounding/shared/processed/fp_grounded9_reviewed_v1
RUN=/root/autodl-tmp/fashion-grounding/runs/data_eval/20261112-p3
$PY -m unittest discover -s tests -v
$PY scripts/validate_pilot.py "$PACKAGE"
# output目录必须是新的，不覆盖已有r01
$PY scripts/run_b0.py --candidates "$PACKAGE/candidates_gt.json" --queries "$PACKAGE/queries_accepted.jsonl" --lexicon configs/rule_parser.json --output "$RUN/B0-reproduce-r02" --run-id B0-train-gt-pilot-r02
$PY scripts/evaluate_predictions.py --queries "$RUN/B0-reproduce-r02/queries.jsonl" --predictions "$RUN/B0-reproduce-r02/predictions.jsonl" --candidates "$RUN/B0-reproduce-r02/inputs/candidates_gt.json" --annotations "$PACKAGE/annotations_coco.json" --output "$RUN/B0-reproduce-r02/metrics.json" --allow-train-diagnostic
```
run_b0导出的inputs包含独立可解析的图片/候选文件，预测路径相对predictions.jsonl，避免机器绝对路径。
evaluate_predictions可接其他方法同schema输出；必须使用相同冻结query清单，缺记录报运行不完整。
published包不可原地写报告；check_pilot_consumers会写报告，只对run副本执行。

## lcy 接收步骤（待本人实际验收）
1. 同步本分支/PR代码；在自己的segmentation工作区/环境读取共享包，不改wkq包。
2. 核对 SHA256SUMS.json、类别映射1..8→0..7、annotations_coco.json实例及原图尺寸。
3. 跑自己的最小数据加载/前向/反向/保存恢复检查；GPU使用按团队排队执行。
4. 记录实际版本、加载实例数、mask/框和运行结果；本轮wkq的COCO检查不等同lcy模型验收。

## wyz 接收步骤（待本人实际验收）
1. 在自己的grounding工作区读取 candidates_gt.json 和 queries_accepted.jsonl，保持候选顺序并检查新candidate_set_id/hash。
2. 用原图RGB紧框裁剪，验证SigLIP2特征、标准logits与缓存键；仅GT接口小批测试，不擅自宣称训练完成。
3. B2调用同一个parse_query；未知颜色等留在完整原文，不删词，不读target类别/query type。
4. 按schema3导出B1/B2全部34条预测，明确失败和run/method/hash；交给同一评价CLI。
5. 回填实际消费验收记录。本轮未替他们执行模型检查，也未发送外部消息。

## 阶段边界
P2：合格训练查询版本已发布，1组待修订，消费方验收仍待完成。
P3：wkq解析/B0/评价核心已实现并CPU验证；B1/B2和成员交叉复核未完成，因此整体in_progress。
P4分区/扩充/全量重复审查、P5模型训练均未完成；不做可选扩展。
