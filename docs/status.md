# 当前进度与下一步

**全量分割单线训练已准备，尚未启动（2026-11-18）。** 从 50k 只加载模型参数，42,476 张基础图各一次，穿插 14,160 次小目标定向取图。半程 14,159 步、结束 28,318 步各评完整 dev250，与缓存的 50k 比较选权重。5 小时总硬上限。真实 CPU 采样排程已固定。[配置、验证与执行入口](handoffs/segmentation/full_single_line_20261118.md)

**全量分割训练包（2026-11-18）**　wkq 已裁决 5 个历史关联组，原划分保留，123 张新增候选继续隔离。42,476 张训练图的成员清单保持原版本。lcy 已将两批 146 张开发曝光图的保护记录与实际 train/test 清单核对，训练和测试交集均为零。[裁决与保护清单](handoffs/segmentation/full_train_protection_20261118.md) · [原增量接收](handoffs/segmentation/full_train_acceptance_20261118.md)

保留备选（2026-11-18，非当前扩充前置）：[50k之后的分割优化方案](handoffs/segmentation/u50000_optimization_plan_20261118.md) **planned，未实现/未启动**。该备选拟在现有10k上做75%均匀＋25%弱细类/小目标采样单因素A/B，同一50k模型参数起点、各新增5000updates，末端完整dev；拟每组60分钟硬限。整体AP/AP75不退步并要求针对性提升，当前50k＋0.3交付保持。边界监督另作有条件第二项，不同时改动。

当前交付（2026-11-18）：[50k＋0.3候选接入说明](handoffs/segmentation/current_candidates.md) **ready_for_downstream_development**。完整dev250的1279个候选及642条查询快照已固定，机器索引和交接说明同步到服务器共享目录；wyz/wkq可立即更新各自候选输入，不必等待分割继续优化。新候选区域特征使用新hash键，相同文本/语言/编码器的文本特征可复用。wyz 和 wkq 已各自发布基于 50k 候选的 dev 结果。[wyz 评价](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/b5e372925dcd6f581fb1a0c8f65bd5dee5dde3cd/docs/handoffs/grounding/u50000_dev_20261118.md) · [wkq 复核](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/307e48e53137ee21f2dd7820fcfbaadcaca10243/docs/handoffs/data_eval/spatial_prior_dev_20261118.md)

前次完成（2026-11-18）：[50k离线残余错误分析](handoffs/segmentation/u50000_residual_diagnosis_20261118.md) **completed**。复用现有AP预测与候选缓存，1008 GT全量分析；固定0.3小目标88/138匹配.5、40/138匹配.75，剩余21低分过滤、24局部几何、3仅异类达标、2无明显重叠。袜6/17、腰带30/47；弱图重采样A/B仅为后续建议；当前优先交接固定候选，未准备或启动新训练。仅CPU诊断与8例看图，0推理/训练/AP重算/smoke；50k＋0.3候选保持。

前次完成（2026-11-18）：[50k完整dev候选与固定阈值](handoffs/segmentation/u50000_candidates_dev250_20261118.md) **completed**。仅一次250图候选推理，阈值前23871个非空候选；三档0.3/0.5/0.7全dev候选1279/1155/1062，当前642条accepted开发查询覆盖602/592/580，选定 **50k＋0.3，93.7695%**。总448.390秒/900秒硬限，3496个新候选裁剪通过；0训练/0AP重评/0smoke/0排序，GPU与锁已释放。新包已发布，旧18k保留；队友本人接收未代签。

前次完成（2026-11-18 00:25 HKT核查）：[10k训练与最终选择](handoffs/segmentation/tenk_completion_20261118.md) **completed**，源0541df4；11月17日23:30正常结束，10epoch/50000成功updates，总6小时6分26秒。五次完整dev AP63.2350/66.3351/66.7770/68.5897/**69.0313**，最终选50k，比旧C5k AP+4.3356、AP_small+12.0599、AP75+4.9937。GPU0MiB、进程退出、锁释放；14次AMP溢出均恢复，0跳过成功更新。七类高于旧最佳，dress略降0.6576；旧下游候选未替换。本轮只复用报告、确认最终权重并同步，0新增训练/推理/smoke。

本次启动前准备（2026-11-17）：[10k分段执行准备](handoffs/segmentation/tenk_execution_preparation_20261117.md)完成，代码`6b5fc8c`，15项新增CPU案例通过，未部署或启动训练。最多10等效epoch/50000新updates，每2epoch完整dev250；统一总硬截止、每段训练/dev/选择合计上限、失败即停、连续3次不超已有最佳早停，按完整dev AP与C5k缓存基线选择。拟总8小时/每段2小时，四项授权标记仍false；模型加载/新训练/推理/旧smoke均0。原model-only/cosine、负裁剪诊断和全部数据/工程验收复用。新10k性能仍未知；C5k20k AP64.6957和旧候选保留。

历史完整数据提案（已由当前10k路线替代，2026-11-17）：[省时完整数据方案](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/0b8a0a88119d4e68f9386e11b88b94d968e0d8d2/docs/handoffs/segmentation/full_data_short_plan_20261117.md)已基于原20k步日志核实。C5k实际40000次曝光/8400.824秒，100epoch×示例4万图约233小时只是旧吞吐外推。推荐C5k20k热启动先1epoch、达标才第2epoch，拟每轮3小时/总6小时；真实N待发布，短配方契约/model-only初始化/硬截止接线尚未实现，当前只做计划，0训练/推理/smoke。历史100epoch入口不作为此次可执行方案。

历史完整数据接收准备（非当前10k阻塞，2026-11-17 15:25 HKT）：[完整Fashionpedia发布缺口与分割接收准备](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/0b8a0a88119d4e68f9386e11b88b94d968e0d8d2/docs/handoffs/segmentation/full_fashionpedia_intake_20261117.md)已整理，继续现有R50，暂不启动训练。100epoch入口位于独立分支，代码ee7db25，21项新增CPU案例复用；真实完整包N/hash、增量接收及冻结未齐，ready_for_training=false。原定向10k验收脚本不是完整包通用接收器，待实际增量格式发布后只适配受影响范围。C5k结果直接引用，D10k保持中断；0新训练/推理/smoke。GPU0MiB/0%，锁空闲，约150.6GiB可用，未部署新代码。

最新（2026-11-17）：[5k/10k对照结果与中断](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/1ec972363271b5ba70be565de71d1b638dc7477c/docs/handoffs/segmentation/expansion_control_20261117.md) **C5k completed，D10k interrupted，整对未完成**。C5k全链路8674.208秒；10k/20k完整dev AP55.8283/64.6957，选C5k20k（较历史B20k+0.9027）。D10k在159步收到主动停止，0 checkpoint/0 dev AP，不能判断扩充收益；没有自动恢复或延长。GPU已释放，共享锁空闲。执行源dc2fba4，旧工程/数据检查复用，无重复smoke。

上一轮（2026-11-16）：[5k/10k 证据复用与统一硬超时入口](handoffs/segmentation/expansion_execution_preparation_20261116.md) **implemented / preparation_checked，未启动训练**。最终代码`e38b836`，服务器准备源`8b522a2`；新增11个CPU案例本地10项、服务器Linux专属1项通过；真实5k/10k证据组合读取通过，未打开图片或解码mask。两组各20k成功更新，保存的10k/20k各评完整dev250；每组单一3小时截止时间、整对最多6小时，不自动延长/重启，失败不继续下一组。默认仅准备；训练/推理/旧smoke均为0。

上一轮（2026-11-16）：[定向 train10000 增量验收](handoffs/segmentation/train10000_acceptance_20261116.md) **passed**，CPU 59.929秒/300秒硬限。实际消费新增5000图/21924实例；旧5000图/17775实例记录保持，新增对旧train/dev/test的身份/group/hash交集均为0。整包10000图/39699实例；源像素/近重复和旧资产字节检查复用已绑定生产证据，未重复smoke、推理或训练。[5k/10k受控方案](handoffs/segmentation/train10000_control_plan_20261116.md)及相同配方配置已准备，尚未实现组合验收凭据的评价接线/有界launcher，未启动训练；技术接收不冒称三方正式冻结。

上一轮（2026-11-16）：[B20k 800→1024 对照](handoffs/segmentation/resolution1024_20261116.md) **completed**，总137.859秒/900秒硬限，一次完整dev250新推理、0训练/0smoke。1024 AP=62.2498，低于复用800的63.7930；保留B20k＋800/1333。小目标严格匹配35→49、薄边界改善，但袜子/腰带仍弱。wkq最新 `4b0584a` 已发布定向10k，ready_for_lcy_review；本轮未消费或验收。wyz `103dcc5`、main `7064ebb` 无增量。

上一轮：2026-11-15；[min_scale 0.1→0.5 单因素对照](handoffs/segmentation/min_scale_trial_20261115.md) **completed**：同一18k→20k完成2000次成功更新，一次完整dev250零错误。按整体AP选B20k **63.79295**，较复用A20k +0.23342、较旧最佳18k +0.21379。总958.213秒/2700秒硬限，3次AMP溢出均按原策略重放；未延长、未重启、未重复smoke或A推理。执行源`e73ef4f`。

**上一轮min_scale只完成自己的分割对照和权重选择。** B的AP_small=10.75248，较旧最佳仅+0.01356，AP75较旧最佳−0.05038，小目标/边界问题未明显改善。现有候选及[最小B1联调][u18000-pipeline]仍对应18k＋0.3（178/290选对、Success@0.5=73.1034%），未因新最佳自动替换缓存或重做联调。前轮[分布分析][train-distribution]和[定向10k需求][train-expansion]保留，wkq已发布扩充包，lcy本轮已完成增量技术接收；数据、排序及全组评价由相应负责人推进。

## GitHub 同步快照

11 月 18 日检查全部七个远端分支及五个 PR。开工时 wkq 为 07b0125、wyz 为 c2fe196；提交前分别推进至 ae08c8d 和 2c47c36。#2、#4 为 draft，#5 指向 wkq 分支，#1、#3 已关闭且未合并。[本轮快照](handoffs/segmentation/evidence/full_single_line_20261118/github_snapshot.json)

| 成员 / GitHub | 已推送分支与 commit | PR / 最新交接 |
| --- | --- | --- |
| lcy / rickyliuchunyu | 开工 af48a30，已准备全量分割单线配置并验证 56,636 个取图位置 | [PR #2](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/2)、[配置与采样回执][full-single] |
| wyz / MayI2004 | 两个 grounding 分支均为 2c47c36，已发布配对目标函数对照 | [PR #5](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/5)、[本人交接][wyz-pair-objective] |
| wkq / LE-brawl | data-eval 分支 ae08c8d，已发布同类服饰、颜色与方位监督审计 | [PR #4](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/4)、[本人交接][wkq-binding] |

队友交接链接固定到本次读取的 commit，避免将未合并内容误认为当前 checkout 已包含。下次开工须重新检查远端增量，流程见 [AGENTS.md](../AGENTS.md)。

历史快照（2026-11-12 17:05 HKT）：当时所有队友分支head与前次一致，四份PR均无review/comment。服务器当时只有 `fp_pilot20_v2`（20图）、`fp_grounded9_draft_v2`（9图）、`fp_grounded9_reviewed_v1`（9图）三个train包；34条accepted是训练语言查询。该次未启动AP、checkpoint选择或训练，见[就绪核查](handoffs/segmentation/freeze_readiness_20261112.md)；随后完成[reviewed_v1接收][reviewed-acceptance]。这不是本日S0运行或当前大包状态。

## 1. 阶段状态

原范围审查核对服务器时只有三个 train 包，四个跨分支解析器/B2 CPU 案例通过；这些是历史证据。责任修正时读取 wkq `f5787f4` 大包交接，此前已完成 train/dev CPU 输入接收；本轮 wyz 消费证据也已齐，用户已确认当前交付数据经过人工复核，消费者固定版本与限定 dev 实验已完成。具体范围、责任修正和原代码审查的适用提交见[全组主线审查](handoffs/team_scope_audit_20261112.md)。上文 17:05 是历史快照，不再作为最新发布状态。

状态：`not_started` 无执行证据；`in_progress` 有明确活动与产物；`blocked` 已开始但受阻；`ready_for_review` 待接收方验收；`done` 满足该阶段放行条件。文档已写不等于该阶段 done。

| 阶段 | 当前状态 | 已有证据 | 尚缺的放行证据 | 主责 |
| --- | --- | --- | --- | --- |
| P0 接口确认 | in_progress | [文件契约](artifact_contract.md)、[合成示例](../examples/README.md)、[解析约定](rule_parser_contract.md) | 三人消费方复核与实际读写器检查 | 全员，B 汇总 |
| P1 环境与工作区 | in_progress | [lcy 工程检查](handoffs/segmentation/pilot_adapter_20261112.md)、[COCO 八类加载/五图推理](handoffs/segmentation/coco_weights_20261112.md)、[200-update 与跨进程恢复](handoffs/segmentation/stability_200_20261112.md)；[wyz 图文环境与 GPU 编码][wyz-handoff]；wkq 独立 CPU 环境；[lcy AP/候选读取实测][lcy-ap] | 全组交付的接收方复核与环境复现验收；wyz 环境只读复用 base torch，base 变化需重验；非长训练验收 | lcy 协调，全员 |
| P2 真实 GT 小包 | in_progress | [原包 20 图/145 GT](handoffs/data_eval/remote_pilot_review_20261112.md)，lcy 加载与 wyz 145 个裁剪消费通过；[新审核包 9 图/59 候选/18 语义查询/36 双语记录][wkq-handoff]，35 测试、59 mask 与源一致 | [新版 reviewed_v1][wkq-p3] 已有 17 组/34 条 accepted；1 组/2 条 pending；lcy 新包加载/前反向/保存恢复已通过，wyz 实际消费、B1/B2 与统一评价已通过 [103dcc5][wyz-latest]；5 组/10 条 accepted 为已暴露答案的训练再审，非独立测试 | B，全员标注 |
| P3 基线与评价器 | in_progress | B0/B1/B2/M1/M2已实现；wyz复算wkq六组评价，B2按GT dev选lambda=2；同642条查询、18k预测候选 | 单seed、同dev选模；多种子及冻结后的独立测试未完成 | A/B |
| P4 数据扩充冻结 | in_progress | 全量 42,476 张的分割增量接收、5 组历史裁决与 146 张曝光保护均已登记。642 条审核语言 dev 已留档 | 三方正式冻结及尚待处理的语言审核 | B，全员 |
| P5-S 分割训练 | in_progress | 10k 训练已选 50k，dev 候选固定 0.3。全量单线配置及真实取图顺序已准备 | 全量训练、两次完整 dev 评价与选模待后续启动 | lcy |
| P5-L 排序训练 | in_progress | [M1/M2在1000/1149图的四组seed42对照][wyz-current]已完成，每组490updates；M2几何有效但未观察到稳定收益 | 其他种子、人工复核训练监督及冻结后的独立测试仍缺；本轮lcy不代做 | A |
| P6 联调与对照 | in_progress | 已有 B1 联调及六组评价，wyz 已用 50k 候选评价 642 条审核 dev 查询，wkq 已复核并发布 M2 方位加分对照 | 多种子空间收益、公平完整对照、E1 与 demo 仍待完成 | 全员分责 |
| P7 方法锁定 | not_started | freeze 清单 | 权重/词表/阈值/数据及最终运行清单 | B 汇总，全员核对 |
| P8 最终测试 | not_started | 评价协议 | 完整预测、指标与失败记录 | 全员 |
| P9 报告与演示 | not_started | 写作分工；此行指最终交付 | 最终实验对应报告、演示与渲染检查 | 全员 |
| P10 归档与提交 | not_started | 归档要求 | 复现索引、课程回执、授权算力收尾 | 全员 |

本表不判断已有 milestone 文件或课程提交状态，最终报告与 milestone 是不同交付物。

## 2. 每人的下一项交付

| 负责人 | 下一项任务 | 可立即做 | 依赖/阻塞 | 交付位置与证据 |
| --- | --- | --- | --- | --- |
| lcy | 全量单线训练已准备 | 复用 50k、146 张保护清单、固定采样顺序及 dev250 协议 | 等待训练启动指令 | [配置及取图回执][full-single] |
| wyz（A）/ grounding | 已发布配对目标函数对照 | 保留原 M2 为参考，整图配对指标无改善 | 草稿审核及独立 test 待相应负责人推进 | [本人交接][wyz-pair-objective] |
| wkq（B）/ data_eval | 已发布同类服饰、颜色与方位监督审计 | 训练划分内的探索草稿及冻结模型探测已有记录 | 独立审核与评估见本人计划 | [本人交接][wkq-binding] |

无 GPU 不阻塞 CPU 接口开发。使用者可用 lcy/wyz/wkq 自报身份；对应角色已绑定，不根据作者顺序推断全名或账号。三人的工作区与代码分支继续遵守 [团队说明](team_workflow.md)。

## 3. 已确认与待确认

| 事项 | 当前值/状态 | 证据或补齐责任 |
| --- | --- | --- |
| 研究与暂定分工 | documented | README、三份角色卡和 workflow |
| 文件交接例及检查器 | locally_checked，仅合成夹具 | [规范补齐交接](handoffs/contract_readiness.md) |
| 模型代码/数据转换器 | 复用现有训练/AP/cache/export/coverage入口；本轮仅新50k配置与有界编排，新产物通过消费检查 | 当前50k固定候选交付完成；历史B1联调仍对应18k，未重跑排序或最终test |
| 实验结果 | 50k AP69.0313；新固定候选0.3覆盖602/642=93.7695%，0.5/0.7为592/580；全dev候选1279/1155/1062 | AP沿用已完成结果；coverage非排序准确率，642查询分母不同于历史290；旧18k包保留 |
| GPU/环境/付费预算 | 本轮候选/三档导出/选择448.390秒/900秒硬限，未延长或重试；10k训练历史保持 | 11月18日00:52：GPU0MiB/0%，无计算/活动记录，锁空闲，数据盘145.08GiB；[结束快照](handoffs/segmentation/evidence/u50000_candidates_20261118/postflight.json) |
| 成员简称与角色 | lcy / wyz（A）/ wkq（B）已确认 | [绑定记录](handoffs/team_aliases.md) |
| GitHub 账号与服务器访问 | lcy：rickyliuchunyu；wkq：LE-brawl；wyz：MayI2004；lcy SSH 已验证 | lcy/wkq 对应 PR 作者；wkq 授权记录、wyz 本人登录/接受邀请见各自交接。两位队友远端运行已有记录，本次未独立验证其各自 SSH 凭据/权限 |
| 公共权重 revision 与数据 manifest | 分割源码、pilot manifest、COCO 文件 hash，以及 SigLIP 2 revision/权重 hash 已固定；两模型实际加载有证据 | [分割来源记录](../envs/segmentation/upstream.json)；[SigLIP 2 来源与校验][wyz-handoff]（镜像传输、官方 API 核对）；train/dev 指纹及用户人工复核确认已登记消费者 freeze；原生产方历史 RELEASE 不改写 |
| 课程实际截止与提交回执 | 未重新核验 | 提交负责人查课程系统；历史日期不是当前确认 |

## 4. 范围检查后的优先顺序

全量包与 50k 源权重现已绑定到分割单线训练配置，后续可在新的授权下启动；当前交付仍为旧 50k＋0.3。[训练准备][full-single] · [保护清单][full-protection]

**责任修正：reviewed_v1 接收时重复安排两步训练/保存恢复，是 Codex 未充分复用已有证据的决策问题，不是用户要求不当或 lcy 让项目跑偏。** 后续执行 AGENTS.md 的验证复用规则：具体变化使原证据失效时才补最小检查；没有新增风险就推进下一项交付。新包不自动触发训练 smoke；该次责任修正仅修改文档，未运行模型或训练验证。

后续大包接收按该规则只执行 CPU 输入检查（52.41 秒），模型加载/训练/恢复均为 0。此结果不等于正式冻结或 dev AP，也不依赖 wyz 的语言 dev 审核完成。

## 5. 更新规则

每次开工先按 AGENTS.md 检查整组远端分支、PR 和最新交接，再确定自己的下一步。每次工作交接在同一 PR 更新相关阶段/个人任务和证据链接，注明实际未验证项；跨分支汇总须标明来源 commit、未合并状态及是否仅引用对方验证。状态与证据不一致时以可复核产物为准，先修正文档；不覆盖他人的完成记录。阶段 done 需对应 workflow 放行条件，训练开始不等于完成，代码完成不等于 GPU 验证。

历史交接文件保留，状态表只展示当前摘要。公开索引不放密码、私钥、token 或私有下载凭据；服务器地址与访问方式由受控渠道交接。

[wyz-handoff]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/66c0da7e066c1235ee618ea7015171e73b82c914/docs/handoffs/grounding/siglip_baselines_20261112.md
[wkq-handoff]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/75bf2ef50f80bc939c35ba15afe6dc6efb15f3b1/docs/handoffs/data_eval/grounded_query_review_20261112.md

[lcy-ap]: handoffs/segmentation/dev_ap_20261112.md
[wkq-p3]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/66dd07c56fb02ae90af7a12de3524650d2fe6455/docs/handoffs/data_eval/p3_reviewed_baseline_20261112.md

[reviewed-acceptance]: handoffs/segmentation/reviewed_pilot_acceptance_20261112.md

[wkq-large]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/f5787f4ac7b6941abdadc43f9fbdc666eba93925/docs/handoffs/data_eval/large_dataset_20261112.md

[large-inputs]: handoffs/segmentation/large_input_acceptance_20261112.md

[freeze-prep]: handoffs/segmentation/freeze_preparation_20261112.md
[wyz-latest]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/103dcc546714dd7c7866b4259d96a9825d8090ef/docs/handoffs/grounding/accepted_baselines_and_weak_trial_20261112.md

[real-dev]: handoffs/segmentation/real_dev_ap_20261112.md

[low-ap]: handoffs/segmentation/low_ap_diagnosis_20261112.md

[postfix1000]: handoffs/segmentation/postfix1000_dev_ap_20261112.md

[resume5000]: handoffs/segmentation/resume5000_dev_ap_20261113.md

[dev-analysis]: handoffs/segmentation/dev_error_analysis_20261114.md
[candidate-prep]: handoffs/segmentation/candidate_u3000_preparation_20261114.md

[candidate-export]: handoffs/segmentation/candidate_export_20261114.md

[real-candidates]: handoffs/segmentation/real_candidates_dev250_20261114.md

[language-ready]: handoffs/segmentation/language_dev_readiness_20261114.md

[offline-coverage]: handoffs/segmentation/offline_coverage_20261115.md

[s0-preparation]: handoffs/segmentation/s0_20000_preparation_20261115.md

[s0-run]: handoffs/segmentation/s0_20000_20261115.md

[s0-recovery]: handoffs/segmentation/s0_recovery20000_20261115.md

[small-analysis]: handoffs/segmentation/small_object_analysis_18000_20000_20261115.md

[u18000-candidates]: handoffs/segmentation/u18000_candidates_dev250_20261115.md

[u18000-selection]: handoffs/segmentation/u18000_language_dev_selection_20261115.md

[u18000-pipeline]: handoffs/segmentation/u18000_b1_pipeline_20261115.md

[train-distribution]: handoffs/segmentation/train5000_distribution_20261115.md
[train-expansion]: handoffs/segmentation/train10000_expansion_request_20261115.md

[wkq-dev250-v4]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/c514a4fc1d0ee3d28ba0dc6ac4fb4c6d11dee4e0/docs/handoffs/data_eval/dev250_simple_draft_20261117.md

[wkq-reviewed-dev]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/6eb7b923d32142e2510ffc87e6adf0e549a8f743/docs/handoffs/data_eval/reviewed_dev_eval_20261117.md

历史 wkq 增量见 [b24515b 语言定向扩充交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/b24515bcda576fe97b354ca9b48e647079831df3/docs/handoffs/data_eval/language_targeted_20261117.md)。当时使用 642 条审核 dev 完成分割候选与阈值选择。

[wyz-current]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/64ed72794f0a432c8a1aa811cea18cdefbf9fdba/docs/handoffs/grounding/spatial_controlled_20261117.md

[full-intake]: handoffs/segmentation/full_train_acceptance_20261118.md
[wkq-full]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/a6dd946f4873523d8835b3915af2c60addbc40be/docs/handoffs/data_eval/full_train_20261118.md
[wyz-50k]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/b5e372925dcd6f581fb1a0c8f65bd5dee5dde3cd/docs/handoffs/grounding/u50000_dev_20261118.md
[wkq-spatial]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/307e48e53137ee21f2dd7820fcfbaadcaca10243/docs/handoffs/data_eval/spatial_prior_dev_20261118.md

[full-protection]: handoffs/segmentation/full_train_protection_20261118.md
[wkq-bridges]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/07b0125a14db02e9b448c097d40df491c70d2c7d/docs/handoffs/data_eval/historical_bridge_adjudication_20261118.md
[wyz-direction]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/aa8ab465af4d3fe969d3c08cbc05154baf921117/docs/handoffs/grounding/direction_increment_20261118.md
[wyz-spatial]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/c2fe196398de0e4589c0713896459b46be3e17d0/docs/handoffs/grounding/spatial_supervision_20261118.md

[full-single]: handoffs/segmentation/full_single_line_20261118.md
[wkq-binding]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/ae08c8de8491256724123f5f2b126b0a7faa74b2/docs/handoffs/data_eval/binding_supply_20261118.md
[wyz-pair-objective]: https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/2c47c3609080a5a1f35914a0ed2269fcba47817d/docs/handoffs/grounding/pair_objective_20261118.md
