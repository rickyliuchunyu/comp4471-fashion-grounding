# 交接记录入口

最新状态：[按用户要求停止5k/10k](segmentation/expansion_user_stop_20261117.md)：15:08已停止D10k及整组控制器、GPU和锁释放。C5k已完成20k，组内dev AP64.695685；产物保留，无自动恢复，完整数据100epoch未启动。下文运行描述是此前快照。

最新实现：[完整Fashionpedia / R50 100-epoch执行接线](segmentation/full100_runtime_20261117.md)：独立入口默认准备，每epoch保存/每2epoch完整dev，显式恢复复用有效结果；21项新增CPU案例通过。全量数据冻结仍待补，新GPU执行未验证；原5k/10k按原上限运行，未修改/重跑。团队最新wkq `6eb7b92` 的642条审核dev及B1/B2/M1评价已增量核对。

历史准备：[完整Fashionpedia / Mask2Former-R50 100-epoch](segmentation/full_fashionpedia_100ep_preparation_20261117.md)：冻结清单、配方、50个完整dev评价节点及恢复方案；纯CPU准备入口和9项新增检查通过。实际全量N/hash/冻结与长训练接线待补，未启动训练/推理、未连接服务器，不影响原5k/10k运行。

最新：[5k/10k受控对照](segmentation/expansion_control_20261117.md) 已按用户授权启动；两组各20k、10k/20k完整dev、每组总硬上限3小时，失败即停不延长。执行源dc2fba4，无重复smoke，结果待完成。

最新：[5k/10k证据复用与统一硬超时入口](segmentation/expansion_execution_preparation_20261116.md)已实现，11个新增CPU案例及真实metadata证据组合通过。默认不执行；每组20k、保存的10k/20k完整dev、每组3小时，失败停止且不延长。本轮0训练/推理/旧smoke。

历史：[定向train10000技术接收](segmentation/train10000_acceptance_20261116.md)通过，CPU59.929秒；新增5000图/21924实例，旧5k保留及跨分区隔离通过。[5k/10k配对方案](segmentation/train10000_control_plan_20261116.md)已准备，训练入口接线尚待完成，未启动训练/推理/smoke，不冒称人工全面复核或三方冻结。

历史：[B20k 800→1024 单尺度对照](segmentation/resolution1024_20261116.md) 已完成，137.859秒/900秒硬限。1024 AP62.2498低于800的63.7930，保留B20k＋800；严格小目标匹配35→49。0训练/0smoke，GPU释放。wkq10k待lcy接收，本轮未消费。

历史：[lcy min_scale 单因素对照结果](segmentation/min_scale_trial_20261115.md)：15分58秒完成，B20k完整dev250 AP=63.7930，高于A 0.2334点、高于旧最佳0.2138点，按整体AP选B。小目标/边界改善仍有限；旧18k候选保留。未延长、重启或重复smoke，GPU已释放。

历史：[lcy min_scale 单因素续训准备](segmentation/min_scale_preparation_20261115.md)：同一18k→20k、A组复用、B仅结束评完整dev250、45分钟总硬限；11项新增CPU检查通过，未启动训练/推理或重复smoke。

历史：[lcy train5000分布与扩充判断](segmentation/train5000_distribution_20261115.md)完成；2,165小目标中95.43%为鞋/配饰，不优先盲目翻倍长训。[给wkq的定向10k需求](segmentation/train10000_expansion_request_20261115.md)已整理、待本人接收；未代构建数据，0推理/训练/旧smoke。

最新：[lcy最小B1联调](segmentation/u18000_b1_pipeline_20261115.md)已完成，290查询均输出原mask/框，选对178/290；不改队友方法，不代表全组对照完成。

最新：[用户授权当前语言dev的18k阈值选择](segmentation/u18000_language_dev_selection_20261115.md)已完成，116图/290查询选0.3，coverage=258/290；原查询来源保留，当前开发联调可继续。

最新：[18000步完整dev250候选交付](segmentation/u18000_candidates_dev250_20261115.md)：一次推理保存阈值前缓存，三个trial包及wyz读取/裁剪技术验证完成，待接收方本人反馈；未训练、未重复smoke。[旧3000步三包](segmentation/real_candidates_dev250_20261114.md)保留原版本。[此前语言dev核查](segmentation/language_dev_readiness_20261114.md)为历史快照；本次未做语言评价或阈值选择，仍须accepted冻结查询。

每项实现或验证交接新建 `<role>/<task-id>.md`；角色子目录由首次实际交接时创建。共享文档交接可以直接放本目录。不要把下面的模板当作已完成记录。

成员对应见 [简称绑定](team_aliases.md)。全组范围与最新依赖审查见 [2026-11-12 主线审查](team_scope_audit_20261112.md)。

历史受污染权重记录（不得续训）：[固定 train/dev 上的真实 AP 与 checkpoint 选择](segmentation/real_dev_ap_20261112.md)：按用户人工复核确认接收；100 updates、两次完整250图dev已完成，选第100步AP=0.4585，未启动完整长训练。

历史准备记录：[分割冻结补齐与限定 dev 实验准备](segmentation/freeze_preparation_20261112.md)：三方技术接收已齐，补充每类 20 例抽样及人工近重复复核材料，实际人工复核与冻结待完成，未启动模型。

新增：[lcy 新 train/dev 大包 CPU 输入接收](segmentation/large_input_acceptance_20261112.md)，复用既有模型验证，未重跑训练 smoke。

当前记录：[wkq 远端接口与复核审计](data_eval/remote_pilot_review_20261112.md)、[接口与状态规范补齐](contract_readiness.md)、[分割角色服务器盘点](segmentation/server_inventory.md)。新增：[lcy 分割真实输入与 GPU 工程检查](segmentation/pilot_adapter_20261112.md)、[COCO 权重校验与五图推理](segmentation/coco_weights_20261112.md)、[训练入口与200-update稳定性](segmentation/stability_200_20261112.md)、[候选消费与 AP/选择接入](segmentation/dev_ap_20261112.md)、[冻结分割数据就绪核查](segmentation/freeze_readiness_20261112.md)、[wkq 新版训练小包的 lcy 接收验收](segmentation/reviewed_pilot_acceptance_20261112.md)。项目进度汇总见 [状态表](../status.md)，完整阶段条件见 [workflow](../workflow.md)。

## 复制使用的模板

```text
任务 ID / 当前 P 阶段：
角色与实际成员：
任务 / 分支 / PR / commit（本次提交可写“见本文件 Git 历史”）：
状态：planned / implemented / locally_checked / gpu_verified / evaluated
上游包、版本、hash 与接收检查：
改动文件、接口/词表版本：
环境名称与依赖记录：
实际执行命令（不可运行则明确写出）：
检查结果与日志/指标位置：
失败或尚未验证的内容：
下游接收角色、产物位置与下一步：
实际接收方复核结果（未复核就写未复核）：
状态表与运行登记更新：
```

运行人的实现状态与阶段状态分别记录；P0 接口文档 locally_checked 不等于三名成员已验收。评估结果必须关联 [运行元信息](../run_records.md)。不记录已做实验的叙事时间线，保留复现所需版本和证据即可。
