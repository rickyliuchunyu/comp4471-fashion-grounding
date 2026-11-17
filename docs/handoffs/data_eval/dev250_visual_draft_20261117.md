# dev250 逐图语义草稿与人工复核交接（2026-11-17）

负责人：wkq / data_eval。代码在 AutoDL 独立工作区实现与运行；只使用 CPU，未训练、未使用 GPU。
固定分支：data-eval/p3-baseline-evaluation；本次提交可通过本文件的 Git 历史定位。

## 交付与状态

新包：`/root/autodl-tmp/fashion-grounding/shared/processed/fp_language_dev250_visual_draft_v3`。
审核页面：`/root/autodl-tmp/fashion-grounding/runs/data_eval/20261117-dev250-language/human_review/index.html`。
证据/逐图预览/输入规格：同一 run 目录。全部 50 张预览表（每表5图，含原图及候选mask）由 Codex 逐图查看，属于机器视觉初标与机器核查，**不是人工验收**。

- 覆盖原 dev 全部 **250 图、248 图片组、1008 GT 实例**；原116图已有草稿，剩余134图本次补齐。
- 主审核集 **331 组中英同义配对，662 条记录**。250图均有至少一组。
- 旧290条（145组）全部有逐组追溯：72组改写（144条）、71组保留（142条）、2组隔离（4条）。
- 新增188组，含原134张空白图片和已有图片上的新增明确目标。旧有效143组 + 新188组 = 331组。
- accepted=0，formal_freeze=false；所有主查询 needs_revision，作者为 Codex 逐图初标，reviewer=null。
- 本包是 **dev 待审草稿，不用于训练，不冒充正式评价集**。训练集和测试集未改。

## 描述与唯一性

使用可见颜色、袖长、领型、局部图案、服装轮廓、人物位置及穿着层次。看不清的波点/格纹等改用更可靠特征，不从来源属性直接推断材质。不按模型结果修改查询。旧ID/旧文本/改写原因均保留。

中文/英文分别有290种不同文本（旧版各57种）；仍保留有用的短描述，不为词面去重制造不自然同义句。331组中 appearance=276、combined=50、spatial=5。同项目类别有多个候选的38组，**不代表其余图没有未标注背景干扰**，也不代表已构成充分语言挑战集。该统计不能当准确率或泛化收益。

八类目标组数：top75 / pants36 / skirt18 / outerwear36 / dress92 / shoe10 / bag24 / accessory40。类别不均衡，后续评价需报告分组分母。

唯一性为 Codex 视觉初判，仍需 wkq 在整图中独立选择目标；尤其注意背景人物、镜面反射、细小目标和遮挡。保留248组划分，视觉重复图不当成250个独立样本。旧contrast_set_id保留在审计里，主草稿清空，避免把改写后文本冒充已验证反事实对；正式D1/D2还未验收。

## 已发现的问题

- 旧q00048：image 1000018407 / mask1000002999，来源标裙子，但视觉上像系腰衣物，整件类别/边界存在疑问。原查询隔离；该图新增明确手表目标1000003001。
- 旧q00097：image1000015784 / mask1000000647，短裤仅露小部分，原格纹说法无法可靠确认，隔离；该图另有可见上衣目标。
- 原图1000010012裤子mask6228/6231、1000017656夹克2198/2199、1000012010外套7896/7897、1000017959包3981/3982有源mask碎片/部位分离疑点。后四组mask短号应加1000000000。本次没有合并、重画或删除原mask，使用其他更明确目标。
- 以上均是看图发现的待复核问题，不宣称已经由分割负责人裁决。

## 人工审核

打开 human_review/index.html：14页，每页25组，最后6组。每组看中英文，独立选青色mask候选，再勾选唯一性、翻译一致、mask完整三项；正常选“已选目标，完成核对”，备注可空。有问题选“需要修改”，说明多目标/看不清/描述错/mask分裂等，可填写修改后的中英文。

每页单独导出JSON，保留14个文件，返回wkq的data_eval流程比对。未审核的记录不会自动接受。页面不嵌入机器目标答案，避免选择受答案提示影响；候选mask仍全部展示供选择。既往是否看过答案未完全掌握，页面“未登记既往答案曝光”不等于保证独立盲审。

导出格式沿用review_format_version=2，绑定新manifest_id和queries_draft SHA256。旧版290条对应的页面/导出不能直接冒充本版审核结果。改写文本、分歧和mask疑点要再裁决后才能发布accepted。两组被隔离的旧查询不进入正常331组页面，完整原文与原因在审计文件中。

## 给 wyz / lcy

wyz 可先验证读取新包和所有query ID/中英配对；不得把dev查询并入训练，不要把662条当人工已验收标签。正式dev评估等wkq发布accepted版本，GT排序与预测候选端到端结果分开。候选集合ID、顺序、mask字节未变化，但新image_manifest ID/hash和查询hash已变化，必须按缓存键规则检查，不能只按目录名混用缓存。

lcy 不需因本次语言版本重训分割。分割继续使用此前train10000包和原dev/test；本次疑似mask碎片作为待复核清单，不修改其现有数据。

## 验证和复现

原父包fp_dev250_v1的1270个清单文件、旧语言包fp_language_dev_draft_v2的622个清单文件校验通过。新包图片/mask逐文件比对父包，annotations/source/图片划分、group、候选ID与候选顺序保持。mapping_config沿用父包：whole8-v1-draft、mapping_implemented=false，正式freeze仍未完成；本次没有改写映射事实或宣称消费方验收。继承conversion报告描述原转换，不是新语言消费验收。

110项unittest通过，含新增8项dev边界测试与已有审核页面JS交互测试。14页实际载荷检查：覆盖331组/250图，无重复semantic ID，无答案目标字段，查询hash一致。未做真实浏览器排版自动化测试；不能以JS stub测试替代浏览器验收。产物重放与全包校验通过，见run/final_verification.json、review_page_validation.json及包内validation_report.json。

```bash
cd /root/autodl-tmp/fashion-grounding/workspaces/data_eval
PY=/root/autodl-tmp/fashion-grounding/envs/data_eval/bin/python
BASE=/root/autodl-tmp/fashion-grounding/shared/processed
RUN=/root/autodl-tmp/fashion-grounding/runs/data_eval/20261117-dev250-language
# 输出必须是尚不存在的新目录；不要覆盖已发布v3
$PY scripts/build_dev_visual_review.py --parent "$BASE/fp_dev250_v1" --old-package "$BASE/fp_language_dev_draft_v2" --specs "$RUN/visual_annotations_final.json" --decisions "$RUN/old_query_decisions.json" --output "$BASE/NEW_VERSION"
PYTHONPATH=src $PY scripts/render_review_batches.py "$BASE/NEW_VERSION" --output "$RUN/NEW_REVIEW_DIRECTORY" --page-size 25 --visual-dev
PYTHONPATH=src $PY -m unittest discover -s tests -q
```

## 校验值
- image_manifest.json: `df5f74f840e987cd5b0288c032b2a4b7e3cd7f3d0e7af7b0aabf9b0c6fe06bab`
- annotations_coco.json: `74837c386c4bbd2805f9976fcfd796931b2f6cfa72693b91d2f441bee473c635`
- candidates_gt.json: `033f1fe441c6df765830570939298c03151ddccf912075ebf3dff99aa25ca4b4`
- queries_draft.jsonl: `8e7f4893139a1218e21a64ca7ecb73b471455916227628761da4b29816486512`
- old290_visual_audit.json: `af2594136398f0512d932e3f4d555d77b2bd7a85f6ad09b162375cf8394633e1`
- SHA256SUMS.json: `0182d0685de87aa3077bf23630563aa0eb42cadce89f2c10714764c88d6b2887`
