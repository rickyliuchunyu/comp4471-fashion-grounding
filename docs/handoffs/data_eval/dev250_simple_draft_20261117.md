# dev250 颜色与空间优先精简版（2026-11-17）

负责人 wkq / data_eval；固定分支 data-eval/p3-baseline-evaluation。本文件的提交历史定位本次代码。全部实现、数据修改和验证在 AutoDL 的 data_eval 独立工作区，CPU 执行；没有训练。

## 当前交付

- 包：/root/autodl-tmp/fashion-grounding/shared/processed/fp_language_dev250_simple_draft_v4
- 审核：/root/autodl-tmp/fashion-grounding/runs/data_eval/20261117-dev250-simple-v4/human_review/index.html
- 本机审核入口：D:/COMP4471/comp4471-fashion-grounding/outputs/dev250-human-review-v4/index.html
- 输入与证据：同一 run 下 decisions.json、text_revisions.txt、review_page_validation.json；原图与候选预览沿用 20261117-dev250-language/sheets/01.jpg 至 50.jpg。
- 当前有效查询：330 中英同义组 / 660 条，覆盖 250 图、248 图片组、1008 GT 实例。
- 对 v3 的331组逐组检查：262组改写、68组保留、1组隔离。连同v3的2组隔离，目前隔离3组/6条。
- 所有主查询仍为 needs_revision，reviewer=null，accepted=0，formal_freeze=false。这是待人工复核的 dev 草稿，不能用于训练或冒充正式 dev 成绩。

## 标注标准与目标唯一性

重新查看全部50张预览表（每表5张原图及全部GT候选裁剪）。颜色和画面空间优先；保留明显条纹、格纹、波点、花纹、牛仔、连帽、袖长等常见可见外观。删去流苏、蕾丝、门襟、细小装饰、手持、穿在里面、肩上、人体姿势等依赖细节或身体关系的表达。不修改源mask或通过来源属性虚构视觉特征。

例：原“喝饮料的人穿的浅蓝色牛仔夹克”改为“浅蓝色夹克”；原“搭在手臂上的浅棕色外套”改为“浅棕色外套”；同色双鞋仍保留画面左右；三块手表保留颜色/中间位置。image1000015113的深色蓬裙描述改为“深色连衣裙”，与整件dress mask一致。

当前 appearance=312、combined=13、spatial=5；38组有同项目类别多个GT候选。空间约束只用于需要且清楚的情况，不强行为单目标添加方位。这个dev仍然以外观查询为主，不能宣称已有充分空间挑战覆盖。没有为了文本多样性制造细节，也没有为了匹配训练词汇删掉真实颜色。

simplification_audit.json保存全部331组旧记录、新文本、原因、预览表和视觉判定的兼容候选ID；发布前要求非隔离查询只有一个兼容目标。唯一性来自Codex看图初判及结构检查，不是程序能证明的视觉事实，最终仍需wkq独立看整图，注意背景人物/镜像/未标注干扰物。所有目标ID及候选顺序稳定。

新增隔离：旧q00035，image1000016614 / target1000002926，“黑色头饰”与头发盘发难分，暂不进入正常审核页。该图仍由黑色上衣和深红色裙子查询覆盖。v3隔离的旧q00048、q00097及mask碎片疑点继续保留，未代替分割负责人裁决。

## 训练覆盖检查

核查实际训练文件 fp_language1000_machine_v2/queries_machine_train.jsonl：1000图、1580语义组、3160条双语机器弱标注。统计合并中英同义组，不能把3160条当3160个独立样本。完整数量、分项目类别数量及逐查询缺口见包内 training_text_coverage.json，仓库摘要见 [覆盖表](dev250_simple_training_coverage_20261117.md)。

- 黑772、白148、灰136、蓝86、红55；左98、右98；条纹115、格纹85、波点41、花卉103（均为语义组，特征可重叠）。
- 绿8、黄1；粉/紫/橙/棕/金/银均0。中间/下方均0。
- 牛仔、连帽、长袖/短袖/无袖、无肩带的明确训练文本均0。虽然是常见可见特征，目前文字监督仍缺。
- 185个dev语义组触发“全局无覆盖/1至9组稀少/目标项目类别无覆盖”中的至少一项。1至9只是审计提示阈值，不代表10组就能学好。
- 此为**文字监督覆盖**，不证明训练图片没有这些特征，也不否认SigLIP预训练知识；没有运行模型，不宣称模型一定失败或改善。
- 训练/dev图片ID和group无交叉；训练文件哈希前后不变，测试集未参与分析。
- 后续若补训练标注，应从固定train图片中寻找这些常见概念及同类干扰例，再独立看图标注/人工审核。不能把dev图片或答案搬入train，也不应用dev成绩反复改词表。本轮只报告缺口，未扩写训练标签。

## 人工复核操作

打开v4的index.html，14页：前13页每页25组，最后5组。每组读中英、独立选择青色mask候选、核查目标唯一/翻译一致/mask完整。正常选完成，备注可空；看不清、两件都符合、颜色不准确或mask有问题就选需要修改并写原因。不要仅依据候选分类判断，应同时看整图。

每页单独导出，共14个JSON。新导出绑定v4 manifest与查询hash；v3页面、旧导出及旧暂存不能直接作为v4审核。页面不嵌入机器答案；既往看过答案与否仍应如实登记。隔离3组不混入正常330组审核，原因及原文在包内。人工导出比对及分歧裁决后才发布accepted版本。

## 给队友

wyz：可以验收v4读取/缓存键；训练仍用原train包。dev待人工接受后使用正式版本，按颜色、空间、常见外观及覆盖缺口分组报告，GT排序与预测候选结果分开。当前没有充分空间查询覆盖，不要将330组当空间泛化已充分验证。

lcy：本次无需重训分割，仍用原分割train/dev/test。新增头饰疑点仅作源mask待裁决记录，没有改动候选集合。

## 验证与限制

117项unittest通过（原110项 + 7项双语计数/分区/源hash/唯一性输入边界测试）。新包1277个清单文件校验通过；原图片/mask逐文件相等、annotations相等、候选ID/顺序相等、manifest图片/group/split相等。所有250图仍有查询。训练hash前后相同。

14页实际HTML载荷检查通过：330组无重复、覆盖250图、无目标答案字段、manifest/query hash正确。已有JS交互单元测试通过；未执行真实浏览器排版自动化测试。

mapping_config仍继承whole8-v1-draft、mapping_implemented=false；不能据此声称正式freeze或三方消费验收已完成。包内old290_visual_audit.json、old_query_decisions.json、visual_annotation_specs.json是v3历史溯源；当前文本看queries_draft.jsonl，当前逐组改动看simplification_audit.json。

## 复现

在data_eval工作区使用独立环境。以下构建/渲染命令已实际执行；重放时必须换成不存在的新输出目录，脚本拒绝覆盖。

~~~bash
BASE=/root/autodl-tmp/fashion-grounding/shared/processed
RUN=/root/autodl-tmp/fashion-grounding/runs/data_eval/20261117-dev250-simple-v4
PY=/root/autodl-tmp/fashion-grounding/envs/data_eval/bin/python
$PY scripts/revise_dev_queries.py --parent "$BASE/fp_language_dev250_visual_draft_v3" --decisions "$RUN/decisions.json" --train "$BASE/fp_language1000_machine_v2/queries_machine_train.jsonl" --output "$BASE/fp_language_dev250_simple_draft_v4"
PYTHONPATH=src $PY scripts/render_review_batches.py "$BASE/fp_language_dev250_simple_draft_v4" --output "$RUN/human_review" --page-size 25 --visual-dev
PYTHONPATH=src $PY -m unittest discover -s tests
~~~

## 校验值

- v3输入queries: 8e7f4893139a1218e21a64ca7ecb73b471455916227628761da4b29816486512
- v4 image_manifest: 98957cc700fb40957f94f76c9c367874f5959eae932ad86f7130d4cceed2936a
- v4 queries_draft: 970dc962ae93484ec80cfe00af74969b88d5bb74c06e56f28e3d17f804da8489
- v4 candidates_gt: 805fc98b071ac9513e6baf03155e79493a3cc464b9aea90629c659c3ced2255b
- v4 SHA256SUMS: 3fba75573f8d19f60bb59f08905dcacd2b1a374410ccda041854b0364b802274
- 原train查询未变: c577a56575e12e1dbf7072a294ef97674dbb4bf3f6f4ed2a8de645ac3177b49d
