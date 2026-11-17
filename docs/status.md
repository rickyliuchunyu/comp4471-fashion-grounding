# 当前进度与下一步

当前交付（2026-11-18）：[50k＋0.3候选接入说明](handoffs/segmentation/current_candidates.md) **ready_for_downstream_development**。完整dev250的1279个候选及642条查询快照已固定，机器索引和交接说明同步到服务器共享目录；wyz/wkq可立即更新各自候选输入，不必等待分割继续优化。新候选区域特征使用新hash键，相同文本/语言/编码器的文本特征可复用。接收方本人运行与评价未代签；本轮仅交接，0训练/推理/smoke。

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

11月18日01:30 HKT已复核整组分支、全部5个PR和活跃讨论，见[最新快照](handoffs/segmentation/evidence/u50000_downstream_handoff_20261118/github_preflight.json)。读取时本分支c8433ea；wkq b24515b、wyz103dcc5、main7064ebb及历史full-data分支均未变，队友交接按相同commit复用。PR #2/#4仍draft，#5依赖#4，#1/#3关闭未合并。PR2旧正文不作为最新候选入口，以本分支置顶交接及固定索引为准；未重复失败的正文编辑权限尝试。

| 成员 / GitHub | 已推送分支与 commit | PR / 最新交接 |
| --- | --- | --- |
| lcy / rickyliuchunyu | 开工c8433ea；50k＋0.3候选及642查询快照可立即接入；本次提交见Git历史 | [PR #2](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/2)；[当前候选交接](handoffs/segmentation/current_candidates.md) |
| wyz / MayI2004 | `grounding/siglip-baseline-interfaces` @ `103dcc5` | [PR #5](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/5)；[实际数据接收、B1/B2 与 M1 交接][wyz-latest] |
| wkq / LE-brawl | `data-eval/p3-baseline-evaluation@b24515b`：保留642条审核dev，新增149训练图/39挑战图的语言草稿，未新增分割监督；完整train未发布 | [PR #4](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/4)；[最新交接][wkq-reviewed-dev]；原#1/#3关闭未合并 |

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
| P3 基线与评价器 | in_progress | wyz已有B0/B1/B2/M1；wkq在同一642条审核dev上完成固定B1/B2/M1的GT及预测候选评价，复用18k/0.3 | B2仍固定lambda=1，未正式选模；M2及公平消融未完成，非最终test | A/B |
| P4 数据扩充冻结 | in_progress | 分割5000/250固定、定向10k增量技术接收通过；642条语言dev已有审核，保留机器初标来源 | 三方正式冻结与另9组语言待处理；完整train另属历史扩充需求，非当前已接收10k准备的依赖 | B，全员 |
| P5-S 分割训练 | in_progress | 10k训练/五点dev/选50k完成；新完整dev250候选、阈值选择及下游接入索引完成，固定50k＋0.3，覆盖602/642=93.7695% | 约定train候选交付及接收方本人复核仍未闭合；本轮仅dev，不代签全阶段 | lcy |
| P5-L 排序训练 | in_progress | [M1 seed42机器弱标签试训][wyz-latest]已完成；本轮固定语言dev可用于明确标记的开发对照 | M2实现及M1/M2三种子公平对照、实际dev选模尚未完成；使用查询须保留本次授权依据及原始来源 | A |
| P6 联调与对照 | in_progress | lcy历史B1最小联调；wkq已有642查询上GT/18k预测B1/B2/M1六项开发结果；50k输入已就绪，尚无新版排序结果 | M2/空间消融、E1、公平对照和demo未完成；队友下游本人接收待确认 | 全员分责 |
| P7 方法锁定 | not_started | freeze 清单 | 权重/词表/阈值/数据及最终运行清单 | B 汇总，全员核对 |
| P8 最终测试 | not_started | 评价协议 | 完整预测、指标与失败记录 | 全员 |
| P9 报告与演示 | not_started | 写作分工；此行指最终交付 | 最终实验对应报告、演示与渲染检查 | 全员 |
| P10 归档与提交 | not_started | 归档要求 | 复现索引、课程回执、授权算力收尾 | 全员 |

本表不判断已有 milestone 文件或课程提交状态，最终报告与 milestone 是不同交付物。

## 2. 每人的下一项交付

| 负责人 | 下一项任务 | 可立即做 | 依赖/阻塞 | 交付位置与证据 |
| --- | --- | --- | --- | --- |
| lcy | 优先交接当前50k＋0.3候选，按队友具体接口反馈维护本角色产物 | 固定候选/642查询/版本hash及接入说明已发布，可立即使用 | 本人签收及约定train候选后续安排；优化A/B提案暂缓，不阻塞当前dev接入，不接管排序 | [当前候选交接](handoffs/segmentation/current_candidates.md) |
| wyz（A）/ grounding | 接收wkq审核dev/新增弱标签交接，推进本人M2与公平对照 | 主642查询、1149图弱训练及新50k＋0.3 dev候选均有交接；本人分支103dcc5未更新 | 上游消费不等于wyz本人验收；M2、B2选模、多种子仍待本人完成，本轮lcy不代做 | [wkq评价][wkq-reviewed-dev]、[wyz交接][wyz-latest]、PR #5 |
| wkq（B）/ data_eval | 按本人最新交接继续新增语言数据审核与评价维护 | 1149图/3296弱训练描述及独立39图困难dev已发布；主642查询不变，新50k候选可用于后续predicted评价 | 新39图dev待审核，原9组pending保留；完整分割包需求保留历史，不阻塞当前10k准备 | [最新交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/b24515bcda576fe97b354ca9b48e647079831df3/docs/handoffs/data_eval/language_targeted_20261117.md) |

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

保持Fashionpedia八类/R50与当前50k＋0.3交付。50k残余错误分析已完成：小目标低分过滤与局部形状并重，匹配小目标外溢较常见，袜/腰带弱而眼镜较强。下一项仅建议准备现有10k内弱图重采样受控A/B，保留原完整图增强；收益待验证，不自动训练或改阈值。已有负裁剪/1024推理对照复用，不重跑旧检查；队友数据和排序仍归本人。

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

最新wkq增量：[b24515b语言定向扩充交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/b24515bcda576fe97b354ca9b48e647079831df3/docs/handoffs/data_eval/language_targeted_20261117.md)。本轮引用其当前642查询dev，仅完成本人50k分割候选与阈值选择，不接新语言训练任务。
