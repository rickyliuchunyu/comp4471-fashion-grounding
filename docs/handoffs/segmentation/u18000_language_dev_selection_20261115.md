# u18000 当前语言 dev 的离线 coverage 与阈值选择

2026-11-15；lcy / segmentation，Codex执行。状态：**按当前用户明确授权完成开发集选择，选定score阈值0.3**。本轮不再因旧发布文件的审核标记阻塞使用；保留原来源与审核字段，不虚构独立人工审核或队友签收。

## 用户授权及固定输入

用户原话：**“现在的语言dev可以直接用”**。据此固定服务器当前发布的`fp_language_dev_draft_v2/queries_draft.jsonl`原始字节，用于本轮开发集coverage/阈值选择，见[授权记录](evidence/u18000_coverage_20261115/authorization.json)和[消费者固定版本回执](evidence/u18000_coverage_20261115/freeze.json)。

- 116张dev图、145组中英配对、290条query记录；这是完整分割dev250中的语言子集，不是250张图均有语言查询。
- 查询SHA256：`fa1419b5799fb6c5c5d0b49dc76f8014300b0c59ce7361007627c2da8fd1c395`。
- 上游仍为290条`needs_revision`、reviewer=null、accepted文件为空、RELEASE formal_freeze=false；这些文件保持原样。机器初标/视觉修订来源不改名为独立人写。
- 消费者记录`purpose=user_authorized_language_dev`、`development_freeze=true`、`formal_freeze=false`，原始查询复制到独立输入目录，未改变任一条记录。授权日期按日记录，另有实际记录时间，不伪造用户消息的精确时间。
- 语言子集图片ID、像素hash、尺寸、split/group与已固定完整分割dev一致；目标ID、类别、非crowd、双语同目标及查询分母由现有验证器检查。没有重新检查图片像素、训练数据、COCO加载或稳定性。

本轮依据是`user_authorized_development`，允许继续实际开发联调和选阈值；不是新增独立审核证明。后续数据改版时保留本次结果和分母，不用新文件覆盖旧证据。

## 实际结果与选择理由

固定同一u18000权重、原解码器及[已交付三个候选包](u18000_candidates_dev250_20261115.md)。复用wkq的分数降序、同类、IoU≥0.5、贪心一对一关联；每图全部非crowd GT参与，不能只拿被查询目标匹配。所有查询保留分母，评估关联不写回候选。

| 阈值 | 覆盖查询 / 290 | Matched coverage | 忽略类别的几何recall | 116张查询图候选总数 | 平均候选/查询图 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **0.3** | **258** | **88.9655%** | 91.0345% | 651 | 5.6121 |
| 0.5 | 256 | 88.2759% | 91.0345% | 566 | 4.8793 |
| 0.7 | 250 | 86.2069% | 89.6552% | 505 | 4.3534 |

选择理由为`most_covered_queries`：0.3覆盖数唯一最大，未使用候选数量或高阈值的并列规则。相比0.5，多覆盖2条双语记录，即1个语义查询组；这不构成语言模型准确率提升或统计显著性结论。

中文/英文分别覆盖129/145、128/145、125/145；两种语言引用相同目标，不能把290条当作290个独立目标样本。三个阈值在116张查询图上均无零候选图片，但仍有目标不在匹配候选中；有候选不等于目标被覆盖。0.3下32条未覆盖，26条连几何IoU≥0.5候选也没有，另6条几何可达但未通过同类一对一关联；不把后者全部定性为分类错误。

表内651/566/505只计有查询的116张独立图片，各图不因多条查询重复计数。完整dev250三个包仍为1360/1182/1032个候选，未做新导出。Mask AP=63.5792是此前分割成绩；本轮coverage不是AP，也不是排序准确率。

## 下游入口

```text
/root/autodl-tmp/fashion-grounding/shared/candidates/u18000-dev250-language-selection-20261115-r01/
  delivery_manifest.json
  selection.json
  selected_candidates/candidates_predicted.json
  language_dev_inputs/queries.jsonl
  language_dev_inputs/user_authorization.json
```

以上为服务器符号链接及清单；保留其指向的运行/输入目录，跨机器复制解析后的实际文件。完整运行目录为`runs/segmentation/segmentation-u18000-language-dev-20261115-r01`，输入快照位于同级`segmentation-u18000-language-dev-20261115-r01-inputs`。

- 选定包：`u18000-dev250-20261115-r01-trial-0.3`；manifest SHA256=`dfb06d6ca67b77901dcb25107d5dcedde7fdc2a761d4a66257522400dd64b32d`。
- selection.json SHA256=`bd9f89515fc90faa9ebf7d0c3fd0211c80cf27a994c168c1548aeb8a56be6c82`，状态`selected_for_user_authorized_development`。
- 权重SHA256=`207e2dbe1dcc8b9d0ef936d683d25af33c210347e34506914bbf6dad3426ee01`，cache SHA256=`a1fcda4a5e7af7f6bf591ec9a6078b205135f66f55f7354d95dafac0b1931772`。

原trial包及原delivery_manifest不改写，所以其历史`trial_unselected/selected_threshold=null`保留；**当前开发选择以本次独立selection.json为准**。wyz/wkq可据此使用18k+0.3及同一份固定查询开展开发集联调。消费者须携带本次授权依据和数据来源，不能自行填reviewer或把needs_revision改成accepted来通过旧默认入口。GT/旧权重/其他阈值的特征缓存不可混用。

## 最小实现变化及实测范围

在现有`select_threshold`增加显式`--user-authorized-dev`路径，要求调用标志、独立授权记录、精确query hash、用途和确认信息同时匹配；默认accepted路径仍保持独立review与正式freeze要求。两条路径共用原split/目标/类别/分母/hash和匹配规则；不修改公共schema验证器或wkq matcher。

新路径在run metadata、逐阈值结果及selection中写明授权依据和原始review_status分布，防止开发结果被描述成新的accepted发布物。仅运行新增的3项CPU测试，0.062秒通过：显式授权及hash绑定；train/错误目标类别与默认accepted门禁；实际选择输出保留来源且原查询字节不变。复用原10项匹配/选择测试、候选包及wyz消费验证，均未重跑。

执行源码`bb398225ef34de478c39ea70e18eaa137f9679ab`；[配置](../../../configs/runs/segmentation/candidate_u18000_coverage.json)。实际CLI启用`--user-authorized-dev --budget-seconds 300`，外层`timeout --signal=TERM --kill-after=5s 360s`，只在CPU运行，CUDA_VISIBLE_DEVICES为空，单CPU线程；进程阶段 **7.044秒**成功退出，无重试或延长。0模型推理、0权重加载、0优化器更新、0旧smoke，未计算排序成绩或使用test。

开工整组GitHub：lcy`8e9a2b7`，wkq`f5787f4`、wyz`103dcc5`、main`7064ebb`；队友分支/交接及PR讨论无增量，#2/#4 draft，#5 open/non-draft，#1/#3 closed-unmerged，未合并。自己的服务器checkout由`00b408d`干净快进到执行源，不修改队友目录。GPU开工/结束均0MiB/0%，无计算进程/active记录；本轮不申请GPU锁。

- [选择结果](evidence/u18000_coverage_20261115/selection.json)、[三档CSV](evidence/u18000_coverage_20261115/comparison.csv)
- [0.3逐查询及分组](evidence/u18000_coverage_20261115/coverage-0.3.json)、[0.5](evidence/u18000_coverage_20261115/coverage-0.5.json)、[0.7](evidence/u18000_coverage_20261115/coverage-0.7.json)
- [实际命令/退出码/计时](evidence/u18000_coverage_20261115/execution_receipt.json)、[run metadata](evidence/u18000_coverage_20261115/run_metadata.json)
- [GitHub快照](evidence/u18000_coverage_20261115/github_preflight.json)、[资源及原始发布状态](evidence/u18000_coverage_20261115/server_preflight.json)、[结束资源](evidence/u18000_coverage_20261115/postflight.json)
- [共享交付清单](evidence/u18000_coverage_20261115/delivery_manifest.json)、[18份轻量证据hash](evidence/u18000_coverage_20261115/evidence_manifest.json)、[传输校验](evidence/u18000_coverage_20261115/transfer_receipt.json)

registry记本轮选择及三档coverage，AP/排序准确率字段留空；查询原文、图片、mask、权重及特征留服务器。下一步直接推进选定包的最小排序联调和接收反馈；完整250图语言覆盖、M2、端到端成绩和最终test仍各自按证据推进，不把本次选择冒充这些阶段完成。
