# wyz：B 数据消费验收、B1/B2 和 M1 弱标注试训

日期：2026-11-12。角色 grounding / wyz；分支 `grounding/siglip-baseline-interfaces`。本记录取代旧 grounding 交接中的“共享解析/accepted 尚未交付”现状描述，旧证据保留。

## 结论与边界

- 用户指定的 [B P3 小包交接](../data_eval/p3_reviewed_baseline_20261112.md) **通过 A 的实际消费验收**：图像/候选/查询 → SigLIP 2 → schema-3 预测 → B 统一评价全部跑通。
- [大型数据交接](../data_eval/large_dataset_20261112.md) 的六包全文件校验、A 逐图裁剪、源 mask 一致性、查询引用和分区检查通过。不是仅检查文件存在。
- B0 重跑复现 12/34；B1/B2 在相同 34 条 accepted **train** 查询上完成 GT 诊断；不是 dev/test 成绩。
- 机器训练集完成 M1 seed 42、固定 5 epochs 的弱标注流程试训；不是人工复核训练实验，不做 dev 选模，不声称泛化提升。
- 正式 P4 数据冻结、M1/M2 三种子公平比较、预测候选实验、E1、最终测试仍未完成。M2 未实现/未训练；当前 M1 入口限定弱训练试验，不代替后续正式训练/统一加载入口。

## 版本与环境

B 公布代码 `f5787f4` 已合入本人 checkout，未修改 B checkout。数据验收及 reviewed B1/B2 源码为 `260fc671ffa1f4e905fffde03af76f09423e31bf`；M1 特征/训练为 `f39083aa934424ed4d33bcdf9531c1e3122217bc`，两者实际运行记录均为 clean 工作区。交叉验收脚本的提交见本文件 Git 历史。

沿用 [独立图文环境与模型证据](siglip_baselines_20261112.md)：Python 3.10.8、torch 2.1.2+cu121、NumPy 1.26.4、transformers 4.51.3。SigLIP 2 revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`，768 维，完整文本、官方 FixRes RGB 裁剪、冻结编码器，缓存版本 `siglip-v1`。精确权重 SHA、processor 和环境签名在各 run metadata。

环境验收分开记录：

- B 的数据环境运行合并后的 111 项测试，全部通过（包含当时 18 项 grounding 接口测试）。
- grounding 环境后来运行 24 项专项测试，全部通过，含新增 6 项 M1 测试：零残差、padding 不影响分数、手算 loss、单候选排除、优化、保存恢复下一 update 完全一致。
- 直接在 grounding 环境运行早期全套 110 项时，5 项大包构建测试因 NumPy 1.26 缺少 `bitwise_count` 失败，5 项 UI 测试因无 quickjs 跳过。这是环境分工限制：大包**构建**使用 B 的 NumPy 2 环境；已发布大包的全部**消费**在 grounding 环境通过。没有升级共享 base 或 B 环境。

## B 数据验收

包根目录 `/root/autodl-tmp/fashion-grounding/shared/processed/`。

| 数据包 | 图片 | GT 实例/裁剪 | 查询状态 |
| --- | ---: | ---: | --- |
| fp_grounded9_reviewed_v1 | 9 | 59 | 34 accepted，2 待修订 |
| fp_train5000_v1 | 5000 | 17775 | 无语言查询 |
| fp_dev250_v1 | 250 | 1008 | 无语言查询 |
| fp_test400_holdout_v1 | 400 | 1599 | 无语言查询；只做数据完整性检查 |
| fp_language1000_machine_v2 | 1000 | 4784 | 3160 needs_revision；accepted=0 |
| fp_language_dev_draft_v2 | 116 | 490 | 290 needs_revision；accepted=0 |

语言包是 train/dev 子集，不能把表中图片数直接相加为独立样本数。共核对 32,574 个发布哈希条目及 25,715 份候选/COCO/官方源 mask 三方一致性，计数包含子集重复；所有图像实际解码、RGB 裁剪成功。检查 project/source ID 偏移、类别映射、bbox/area、完整候选引用、中英文成对与训练查询 allowlist。图片/掩码未写回数据包。

train/dev/test 三对之间 image_id、图片 SHA、group_id 交集均为 0；两个语言子集的图片元数据与所属主分区完全相同。没有重新人工检查所有近重复图片，也未运行最终测试模型。

小包的 1 组/2 条争议保持待修订，没有自动提升为 accepted。这里只验接口、审计状态与引用，不替代人工语义正确性审核。

非阻塞元数据问题：包内 mapping_config 仍保留规划模板的 `mapping_implemented=false` / `queries_reviewed=false`；实际转换与查询状态应以 COCO、发布记录及逐条查询为准。建议 B 在下一版修正模板状态，当前已冻结发布文件未改写。

## B0/B1/B2 真实训练小包诊断

运行目录：`/root/autodl-tmp/fashion-grounding/runs/grounding/acceptance-20261112-r01/{B0,B1,B2}`。
输入固定为小包 `queries_accepted.jsonl`，SHA `dda04845dfa71962a86bc94fccc0ff6d580dc27561a3001a777e9e4c94ae111a`；candidate SHA `44969ca41c595d0a0c4ad32321ddceef677d0754b16d09eda6693e10b89be30f`。34 条查询来自 8 张图；包包含 9 图。

| 方法 | 选对 / 34 | oracle selection | 英文 / 17 | 中文 / 17 | 宏类别准确率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 12 | 35.29% | 6 | 6 | 25.00% |
| B1 | 18 | 52.94% | 10 | 8 | 66.67% |
| B2 | 22 | 64.71% | 11 | 11 | 75.00% |

B0 的 22 次 rule_unsupported 全部保留分母。B1/B2 各 34 个合法输出、0 推理错误。B2 lambda=1 是本次运行前固定的**诊断值**，没有用 train 调参；正式 lambda 仍须在 accepted GT dev 的约定网格选定。共享词表 SHA `987935b92f8ff329b3a794d66913abb33515ada98b705b6966bd09b1afac1609`；语法适用、匹配类别、非零规则贡献均为 18/34，已分别记录，singleton/并列不虚报贡献。

独立控制检查通过：完美 GT 选择 accuracy/IoU=1；全部 no_output accuracy/IoU=0 且仍为34分母；缺预测、未知候选、查询元数据篡改均拒绝。B1/B2 强制禁用编码调用，纯缓存重放 34+34 条预测及分数逐项完全一致。B 的评价器没有依赖 CUDA。

## M1 弱标注短试训

使用 `queries_machine_train.jsonl`，SHA `c577a56575e12e1dbf7072a294ef97674dbb4bf3f6f4ed2a8de645ac3177b49d`，全为 train；使用全部 1000 图、3160 条中英文机器描述，不读取 dev/test 查询。

结构严格为设计中的 `[v,t,v*t] → 256 → GELU → Dropout(.1) → 1` 外观残差；末层零初始化，初始分数保留 B1。590,337 个可训练参数，优化器不含编码器参数。AdamW lr=3e-4、weight_decay=.01、batch=32、seed=42、固定5轮、490 updates；36 条单候选查询不计 loss，仍计训练拟合统计。

- 同项目类别竞争查询 410/3160（12.97%），低于规划约一半；不重复简单样本伪造困难覆盖。
- 3160 条描述仅 290 个不同“完整文本+语言”组合，模板重复明显；其中5个已有缓存，新增285个文本缓存。
- GPU 特征提取内部约43.12秒，含进程启动/模型加载的锁持有约50.22秒；已释放 GPU。排序头在 CPU 训练，训练/写盘/重放计约4.97秒，不是端到端延迟基准。
- 初始同训练标签拟合 2467/3160 = 78.07%，最终 2653/3160 = 83.96%；多候选训练集 CE 0.52377 → 0.40221。**这些标签来自机器，且同一训练集被优化，不能作为泛化准确率。**
- 每轮检查点保存 model、AdamW、epoch/update 与 Python/NumPy/torch RNG；无 scheduler/AMP，字段为 null。最后检查点重新加载得到完全相同分数；合成测试验证恢复后下一优化步完全一致。
- 无 dev 选模、无早停，直接用预定第5轮；seeds43/44、M2、正式 dev/test 没有运行。
- 输出 `weak_train_predictions.jsonl` 为标准 schema-3，但其查询未经审核，B 正式评价入口应继续拒绝。

运行根目录：
`runs/grounding/M1-weak-features-20261112-r01`、
`runs/grounding/M1-weak-train-20261112-r01`。
最终 `epoch_05.pt` SHA `ed320169fafc0218ed99ce42cc003c2af7416853972c38db05960e856b888050`。不提交权重/缓存/原始查询。

## 复现与证据

所有命令在本人工作目录 `/root/autodl-tmp/fashion-grounding/workspaces/grounding`，每次明确设置 `PYTHONPATH=src`，解释器为 `/root/autodl-tmp/fashion-grounding/envs/grounding-v1/bin/python`；GPU 调用经 `scripts/with_grounding_gpu_lock.py`。不沿用上次 SSH 环境。

- 数据：`scripts/check_grounding_data_release.py --packages <上述六包绝对路径> --output <新的报告路径>`。实际完整命令、输入 hash 和结果见 [验收证据](acceptance_evidence_20261112.json) 的 data.command。
- 小包：`scripts/run_b0.py`、`python -m fashion_grounding.grounding.predict --method B1/B2`；每个 run 的 resolved_config、run_metadata 存有参数/命令。评价使用 `scripts/evaluate_predictions.py --allow-train-diagnostic`，完整查询/预测/candidates/annotations 参数见证据。
- 交叉核验：`scripts/check_grounding_pilot.py --package <小包> --runs <acceptance运行根> --cache <shared/features/grounding/siglip-v1> --lexicon configs/rule_parser.json --output <新报告>`；精确已执行命令在证据 pilot.command。
- 弱试训：`python -m fashion_grounding.grounding.weak_trial --phase cache/train --weak-label-trial ...`；两次完整实际命令见证据 weak_features.command 和 weak_train.command。cache GPU 限时600秒；train CPU，固定5轮。
- 测试日志与所有结果位于 `runs/grounding/acceptance-20261112-r01/`；各脚本拒绝覆盖已有输出，复现必须使用新 run ID/路径。

## 交给 B / lcy 的内容

B 可直接读取上述 B1/B2 的 `predictions.jsonl`、`evaluation.json`、`run_metadata.json`，核对全部34查询；报告、规则覆盖率、缓存复现和数据验收结果齐全。下一步优先人工审核 dev 的290条、补齐其余134张 dev 图的描述，修订模板重复和同类干扰不足。不要把 needs_revision 改成 accepted 以绕过入口。

lcy 可依据 A 的大包消费证据继续对接预测候选；预测候选尚未由 A 验收。当前仍严格区分 GT 与 predicted 缓存/实验。下游对本次 A 实现的 PR 复核：**尚未收到**；不代替其他成员签字，不合并 main。
