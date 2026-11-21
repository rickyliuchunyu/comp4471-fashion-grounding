# 全量分割训练包的历史组裁决与开发曝光清单

2026-11-18，lcy / segmentation 接收 wkq 的逐组裁决，并为 `fp_train_full_eligible_v1` 补齐开发曝光保护清单。**全量训练包仍为 42,476 张，5 组均保留旧划分，123 张新增候选继续隔离。146 张开发曝光图均列入训练和测试禁用清单。** [机器可读清单](evidence/full_train_protection_20261118/protected_exposures.json) · [分割侧核对回执](evidence/full_train_protection_20261118/verification.json)

## 五组裁决

wkq 的 Codex 图像完整性复核比较了 23 张旧训练图与 10 张保护图（5 dev、3 test、2 开发曝光），计 75 对。关联经其他图片间接形成；wkq 在此范围内未发现跨分区同源。分割侧对照原组件、实际端点配对、旧接收回执与发布 SHA，并将 50k checkpoint 摘要与既有选模记录绑定。[wkq 逐组判据](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/07b0125a14db02e9b448c097d40df491c70d2c7d/docs/handoffs/data_eval/historical_bridge_adjudication_20261118.md) · [逐对机读记录](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/07b0125a14db02e9b448c097d40df491c70d2c7d/docs/handoffs/data_eval/full_train_20261118_evidence/historical_bridge_adjudication.json)

| 组件（原序） | 端点比较对数 | 保持隔离的新增图 | 原划分 |
| --- | ---: | ---: | --- |
| 1 | 56 | 67 | 保留 |
| 2 | 15 | 18 | 保留 |
| 3 | 1 | 6 | 保留 |
| 4 | 1 | 7 | 保留 |
| 5 | 2 | 25 | 保留 |

裁决后训练成员变更为 0。就这五组重复风险而言，旧 10k 选出的权重 `checkpoint_050000.pth` 可作模型初始化，SHA256 为 `1ed0f913602746e027ae84ec9a84c98a80306b34820a824d39ddafba1671208a`。wkq 单独记录了仅为辨重查看的 3 张 test 原图；这些图继续留在 test400，未进入开发登记的 146 张。wkq 将复核来源标为 AI，分割侧此次只用元数据。[50k 选模记录](evidence/tenk_completion_20261118/final_checkpoint.json)

## 两批曝光保护

| wkq 登记批次 | 保留作草稿 dev | 看过后质量隔离 | 全部保护 |
| --- | ---: | ---: | ---: |
| 第一批挑战图 | 39 | 11 | 50 |
| 第二批方位配对图 | 74 | 22 | 96 |
| 去重合计 | 113 | 33 | **146** |

合并清单逐图保存项目 ID、官方 validation 源键、group、文件 SHA256、登记批次和原处置。后续分割训练或新增测试分区应使用这份清单保护全部 146 张和关联组。两份源登记表的路径及 SHA 已写入清单。原发布包内的 50 张登记继续作为发布时快照，新增 96 张由本补充记录绑定。[合并清单](evidence/full_train_protection_20261118/protected_exposures.json)

- 第一批 50 张逐图关联全量包已固定的保护指纹，39 张保留图与原挑战 dev manifest 对应。
- 第二批逐条对照服务器 96 条看图处置和 410 张已筛查图片的 manifest。保留的 74 张与实际草稿 dev manifest 完全对应，其余 22 张均标为质量隔离，源键、组和 SHA 均可追溯。复用 wkq 以 43,237 份保护指纹实施的资源 URL、文件 SHA、分组及原／镜像 pHash 筛查。[wkq 筛查交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/07b0125a14db02e9b448c097d40df491c70d2c7d/docs/handoffs/data_eval/direction_pairs_20261118.md) · [上游隔离回执](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/07b0125a14db02e9b448c097d40df491c70d2c7d/docs/handoffs/data_eval/direction_pairs_20261118_evidence/separation.json)
- 全部 146 张对实际全量 train 和 test400 的项目 ID、带官方 split 的源身份、group、文件 SHA256 八项交集均为 0。分割发布 SHA 与旧 CPU 接收一致。上述两项分割数据依赖已闭合；全组三方正式冻结另行登记。[发布时点的增量接收](full_train_acceptance_20261118.md)

## 追溯与复现

清单 SHA256 为 `6cecceaf3215830f2cb88626f97fa8ee5b29ec5049d9824c1187aedb94fd86fd`。服务器副本与核对回执位于 `/root/autodl-tmp/fashion-grounding/runs/segmentation/full-exposure-protection-20261118-r01/`。在可访问相同服务器元数据的环境，将原接收的 `published.json`、`release_check.json` 和 50k 的 `final_checkpoint.json` 放入新输出目录，执行下列命令。脚本从指定 Git 对象读取 wkq 的两批登记和裁决；输入文件摘要列在核对回执中。[核对脚本](../../../scripts/record_full_train_exposure_protection.py)

```sh
python scripts/record_full_train_exposure_protection.py \
  /root/autodl-tmp/fashion-grounding \
  /root/autodl-tmp/fashion-grounding/workspaces/data_eval "$OUTPUT_DIR"
```

整组 GitHub 开工时 main 为 `7064ebb`，lcy 为 `8168b80`，wkq 的 `07b0125` 发布裁决，wyz 两个分支均为 `aa8ab46`，已发布方位训练增量与颜色诊断。#2 与 #4 为 draft，#5 指向 wkq 分支，#1 和 #3 已关闭。本轮只接收与分割有关的保护信息，未运行训练、推理、重复 smoke 或读取测试图像。[开工快照](evidence/full_train_protection_20261118/github_snapshot.json) · [wyz 开工时交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/aa8ab465af4d3fe969d3c08cbc05154baf921117/docs/handoffs/grounding/direction_increment_20261118.md)

推送前复核，wyz 两个分支和 PR #5 已更新至 `c2fe196`，新增方位监督曝光与空间分支对照。wkq 的裁决 commit 与分割包身份一致。[wyz 最新交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/c2fe196398de0e4589c0713896459b46be3e17d0/docs/handoffs/grounding/spatial_supervision_20261118.md)
