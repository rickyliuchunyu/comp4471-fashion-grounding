# wkq 看图初标与人工审核包

日期：2026-11-12；角色 data_eval；任务分支 data-eval/grounded-query-pilot。
基于 dc24f2dcde9a967a28bfc825a10ca85bd50713df，接续远端试点交接；本轮状态 ready_for_human_review，P2 仍 in_progress。

## 目的与实际结果
用户要求机器先标注、wkq 再审核。任务是“图片 + 完整指代表达 → 一个整件服饰实例及已有 mask”，不只是将类名翻译。
新增训练数据筛选与看图初标流程、不可覆盖的草稿包构建器、隐藏参考答案的离线复核页，以及保留修改文字/质量检查/答案暴露信息的审计。
查询语义 ID 不包含目标 ID；审核页只嵌入图片、所有候选和文字，不含参考目标/目标类别/query type。原始和修订文字均保留，审计不自动生成 accepted。

发布包 fp_grounded9_draft_v2：9 图、59 个候选、18 组语义查询、36 条双语记录、accepted=0。
6 appearance、6 spatial、6 combined；12 组来自同源细类真实双实例（包或外套），含 3 对左右反事实。
目标类别仅 bag/outerwear/top/dress；不是八类语言覆盖完成，不是正式训练集或测试集。

## 看图筛查发现
自动筛出的 9 张新图中，6 张疑似多个 annotation 分别覆盖同一件衣服的部分区域：
7074、9713（裤腿），42143（同一上衣），45411（同一裙装的分离区域），48755（同一毛衣），49055（同一外套）。
这些图从本批语言包隔离，source masks 未修改，也未擅自合并；需后续商定整件实例策略。
因此“同类 annotation 数 >= 2”只能筛候选，不能直接声称真实视觉竞争。这是人工/视觉审核的重要作用。
最终新增图为 4360、9080（两包）、21398（穿着和抱着的两件外套）；另外保留 6 张旧分歧图做再次审核。
没有为凑数量复制左右鞋样本。

旧分歧建议：13898、15909、42973 保留目标补充可见描述；11778、29705、50126 建议撤下难辨认的旧内搭查询并新建外套查询。
全部是 proposal_pending_wkq，旧目标和人工选择保留，不能说旧人工选择已被判错。
这 6 张旧图均记为 prior_answer_exposure，不计独立盲审；新查询通过不自动裁决旧记录。

## 位置与复现
独立环境：/root/autodl-tmp/fashion-grounding/envs/data_eval。
工作区：/root/autodl-tmp/fashion-grounding/workspaces/data_eval。
运行目录：/root/autodl-tmp/fashion-grounding/runs/data_eval/20261112-grounded-queries。
共享包：/root/autodl-tmp/fashion-grounding/shared/processed/fp_grounded9_draft_v2。
原始源标注 SHA256：f6af94fb8712bd8170a49c857e842e1227e6806c4f33f32340db0c849133e006。
输入 selection/download provenance、视觉预览、annotation_spec.json、tests.log、environment.txt 与发布 hash 保存在运行目录。
机器初标明细同存 [grounded_query_spec_20261112.json](grounded_query_spec_20261112.json)，不包含真实姓名或原人工备注。

在工作区执行：
```bash
PY=/root/autodl-tmp/fashion-grounding/envs/data_eval/bin/python
RUN=/root/autodl-tmp/fashion-grounding/runs/data_eval/20261112-grounded-queries
export PYTHONPATH=src
$PY -m unittest discover -s tests -v
$PY scripts/validate_pilot.py "$RUN/fp_grounded9_draft_v2"
# 新输出目录必须不存在；不要覆盖已发布包
$PY scripts/build_query_review.py --spec "$RUN/annotation_spec.json" --output "$RUN/fp_grounded9_draft_rebuild"
$PY scripts/render_query_review.py "$RUN/fp_grounded9_draft_rebuild" --output "$RUN/fp_grounded9_draft_rebuild/review.html"
# wkq 导出文件到位后才运行，输出用新的文件名
# $PY -m fashion_grounding.data.review --package "$RUN/fp_grounded9_draft_v2" --review <export.json> --output <new-audit.json>
```

## 验证
35 个测试通过：原数据边界 11、复核审计 13、包构建/无答案载荷 6、JS 交互 5。
JS 测试在独立环境 quickjs 1.19.4 运行，检查缺选项阻止完成、候选切换清空检查、编辑/未审核行导出、拒绝保留建议目标、导航恢复。
这是 DOM stub 行为验证，不是浏览器视觉/真实下载自动化测试。测试依赖单列 requirements-data-review-test.txt。
59 个 mask 与官方源逐像素一致；59 个裁剪有效；COCO/schema/hash/引用检查通过。批内 pHash 阈值 <=6 无候选重复对，不代表全量/跨 split 去重完成。
仅 CPU，未使用 GPU；未改他人环境或停止他人进程；仅在 data_eval 环境安装测试依赖。

## 下一步与限制
wkq 打开离线 review.html 审核并导出。逐项比对目标、编辑、唯一性/翻译/mask 检查，处理分歧后另发 accepted 版本。
最终测试仍须人独立撰写；本批 authorship 明确 machine_authored_after_visual_inspection_pending_human_review。
P3 共享规则解析器/B0/统一模型评价器仍未实现；本轮审计不是模型评价器。
P4 正式分区、全量转换和近重复分组尚未完成；模型消费方验收未完成。
本批 v1 为未发布的中间构建，保留原输出；只交付 v2，不改旧 fp_pilot20_v2。
发布沿用用户已确认的 LE-brawl 授权，远端提交/bundle，本机 GCM 仅用于传输，不复制凭据到服务器。
