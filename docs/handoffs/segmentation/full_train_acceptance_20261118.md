# 全量 Fashionpedia 分割训练包增量接收

2026-11-18，lcy / segmentation。**新增 32,476 张通过 CPU 分割输入验收，合并包共 42,476 张、42,452 组、152,848 个实例。** 全量包正式冻结前，仍需裁决 5 个同时触及旧训练图和保护集的近重复组件。相关的 123 张新候选已全部隔离。本次只做数据接收，训练、推理和 smoke 均为零。

| 分割训练数据 | 图片 | 独立组 | 实例 |
| --- | ---: | ---: | ---: |
| 原 `fp_train10000_targeted_v1` | 10,000 | 9,976 | 39,699 |
| 本次新增 | 32,476 | 32,476 | 113,149 |
| `fp_train_full_eligible_v1` | 42,476 | 42,452 | 152,848 |

官方 train 来源账逐图覆盖 45,623 张。10,000 张原样保留，32,476 张新增，3,147 张排除或隔离，待处理 0 张。5 个历史组件的处置另行登记。服务器包位于 `/root/autodl-tmp/fashion-grounding/shared/processed/fp_train_full_eligible_v1`。发布描述 SHA256 为 `7567710f2d8446fd106a646051de2e6a4748c31f22252dab07940fbf234e1d35`。[wkq 的发布索引](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/a6dd946f4873523d8835b3915af2c60addbc40be/docs/handoffs/data_eval/full_train_20261118_evidence/published.json)

## 保留、隔离和真实读取

- 旧 10,000 张 manifest / COCO 图片记录、39,699 条 COCO 标注逐项相等。旧、新成员清单互斥，并集等于发布包，实际 CPU 增量视图与合并包的新增记录完全相同。旧图和 mask 共 49,699 个资产摘要逐一对应；沿用 wkq 发布时复制后逐文件计算的 SHA 和旧像素验收。未重新解码旧图或旧 mask。[保留与隔离回执](evidence/full_train_acceptance_20261118/release_check.json)
- 新增记录对旧 10k、dev250、test400、全部 50 张已曝光图（39 张开发图、11 张隔离图）的项目图片 ID、带官方 split 的源身份、group 和 SHA256 交集全部为零。校验了这些分区在保护指纹表中的身份及绑定，复用发布方对 10,761 份保护指纹及新增代表之间的资源 URL 和原/镜像四组合 pHash 距离 ≤6 核对，保留代表触发数为 0。test 仅读取保护元数据。[发布后审计](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/a6dd946f4873523d8835b3915af2c60addbc40be/docs/handoffs/data_eval/full_train_20261118_evidence/postpublish_check.json)
- 新扫描发现 5 个含旧训练图及 dev/test/曝光图的连通组件。逐组排除的新增数为 67、18、6、7、25，合计 123。五组中的旧 train/dev/test/曝光图命中合计分别为 23/5/3/2。这些关系可能经中间图片传递，实际同源关系仍待判读。建议数据负责人先裁决这 5 组，并记录保留、隔离或换版的决定，再做全量正式冻结。[wkq 的组件明细](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/a6dd946f4873523d8835b3915af2c60addbc40be/docs/handoffs/data_eval/full_train_20261118_evidence/frozen_bridge_clues.json)
- 用原 `load_training_package` 对发布方提供的增量视图实际读取一次，历时 353.77 秒。32,476 张图片通过哈希、RGB 转换与尺寸核对，113,149 份 mask 通过 RLE 解码和 bbox/area 核对。八类从 1–8 映射为 0–7，crowd 跳过 0 个。复用绑定到增量 manifest 和 COCO 标注的 113,149 份官方源 mask 像素一致性检查。[分割读取回执](evidence/full_train_acceptance_20261118/added_reader.json) · [源保真证据](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/a6dd946f4873523d8835b3915af2c60addbc40be/docs/handoffs/data_eval/full_train_20261118_evidence/source_fidelity.json)

## wkq 回报的覆盖

面积按原分辨率前景像素统计。新增小目标（≤1,024）共 11,189 个，分布在 8,706 个独立组；其中极小目标（≤256）2,508 个，分布在 2,106 个独立组。合并后相应实例数为 18,355、4,383。[八类及源细类覆盖表](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/a6dd946f4873523d8835b3915af2c60addbc40be/docs/handoffs/data_eval/full_train_20261118_evidence/coverage.json)

| 弱细类 | 新增全部实例 | 新增小目标 | 其中新增极小目标 |
| --- | ---: | ---: | ---: |
| 袜 | 1,324 | 646 | 323 |
| 腰带 | 4,497 | 676 | 152 |
| 鞋 | 31,790 | 5,675 | 967 |
| 头饰（源聚合类） | 2,105 | 1,005 | 482 |
| 手表 | 2,417 | 1,768 | 385 |
| 包／钱包 | 4,630 | 0 | 0 |

全量中的 215 个小包（含 23 个极小包）全部来自旧 10k。复杂边界方面，新增 560 个带官方 cutout 属性的实例，鞋和包在该属性上均为 0。新增 mask 中 33,155 个有多个前景分量，18,171 个有内孔，两类可以重叠。wkq 的 12 张定向 AI 看图记录分三层，各 4 张；鞋孔洞层 2 张可见细带及间隙，稀疏包层 2 张可见肩带，其中 1 张可见袖子隔开包体和带子，多分量层明确由外部遮挡造成的为 0 张。鞋实例 96852 的源 mask 疑似覆盖部分皮肤，已留原记录。几何统计与这些定向案例均不能给出全包真实肩带、遮挡的发生率。[边界记录与逐例观察](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/a6dd946f4873523d8835b3915af2c60addbc40be/docs/handoffs/data_eval/full_train_20261118_evidence/boundary.json)

## 整组进度与复现

wkq 通过 `a6dd946` 发布全量包和数量账，PR #2 原请求下尚无 wkq 的单独回帖。wyz 的两个分支已更新至 `b5e3729`，本人已在 642 条审核 dev 查询上使用 50k＋0.3 的 1,279 个候选。PR #2、#4 为 draft，#5 的 base 为 wkq 的工作分支，#1、#3 已关闭且未合并。main 为 `7064ebb`。[开工 GitHub 快照](evidence/full_train_acceptance_20261118/github_preflight.json) · [wkq 全量交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/a6dd946f4873523d8835b3915af2c60addbc40be/docs/handoffs/data_eval/full_train_20261118.md) · [wyz 新版候选联调](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/b5e372925dcd6f581fb1a0c8f65bd5dee5dde3cd/docs/handoffs/grounding/u50000_dev_20261118.md)

同步前再次核对分支，wkq 的 PR #4 已推进至 `307e48e`。新增提交复核了 wyz 的 50k 结果，并给出加方位分的 M2 在审核 dev 上严格选对 512/642；此次变更只涉及排序评价，分割包仍对应 `a6dd946`。[wkq 最新交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/307e48e53137ee21f2dd7820fcfbaadcaca10243/docs/handoffs/data_eval/spatial_prior_dev_20261118.md)

两份 CPU 检查脚本及复用的分割读取器对应 `a1943e8`。输入和命令保存在服务器 `/root/autodl-tmp/fashion-grounding/runs/segmentation/full-train-acceptance-20261118-r01` 及[执行索引](evidence/full_train_acceptance_20261118/run_metadata.json)。运行前数据盘可用 130 GiB，RTX 3090 显存占用 0 MiB，GPU 利用率 0%。数据、指纹及逐图标注保留在服务器；此处仅发布版本、脚本和小型回执。
