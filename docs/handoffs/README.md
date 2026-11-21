# 交接记录入口

**S0 分割冻结（2026-11-19，lcy）**　权重、候选解码、0.3 阈值及 dev250 manifest 已登记。开工核对时，分割 PR #2 的交付提交为 `341f431`，wkq 的 PR #4 为 `b82fc99`，wyz 的 PR #5 为 `1e0a457`。wyz 已在 `69f3d1f` 签收并复现 610/642，wkq 评价端签收及全组 P7 汇总待办。[排序端签收](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/69f3d1f0a61688470741f95d51686f23dbe2ce48/docs/handoffs/grounding/frozen_combo_dev_20261119.md) · [冻结条目](segmentation/s0_freeze_20261119.json) · [下游取包](segmentation/current_candidates.md)

**组合分割训练已完成（2026-11-19）。** 05:38 HKT 按完整 dev250 整体 mask AP 选中新增 28,318 步的末程权重 `b7146787a1fc`。整体 AP 71.8911（+0.5706），小目标 AP 28.7762（+5.3303）。[完整比较与权重位置](segmentation/full_combo_completion_20261119.md)

**上一轮全量分割训练（2026-11-18）。** 当轮选用 28,318 步权重。 完整 dev250 的整体 mask AP 为 71.3205，比旧 50k 高 2.2892 点。小目标 AP 从半程 20.5696 回升至 23.4459，比旧 50k 高 0.3377 点。[三份报告与最终权重](segmentation/full_single_completion_20261118.md)

**全量分割训练包（2026-11-18）**　wkq 已裁决 5 个历史关联组，原划分保留，123 张新增候选继续隔离。42,476 张训练图的成员清单保持原版本。lcy 已将两批 146 张开发曝光图的保护记录与实际 train/test 清单核对，训练和测试交集均为零。[裁决与保护清单](segmentation/full_train_protection_20261118.md) · [原增量接收](segmentation/full_train_acceptance_20261118.md)

**当前交付（2026-11-19）**　GUI 的开发集、个人图片及下游候选均使用组合权重 `b7146787a1fc` 和阈值 0.3。完整 dev250 共 1,320 个候选，覆盖 610/642 条已审核查询（95.02%）。个人图片默认优先使用空闲 GPU。[交接路径与阈值比较](segmentation/current_candidates.md)

前次完成（2026-11-18）：[50k离线残余错误分析](segmentation/u50000_residual_diagnosis_20261118.md) **completed**。复用现有AP预测与候选缓存，1008 GT全量分析；固定0.3小目标88/138匹配.5、40/138匹配.75，剩余21低分过滤、24局部几何、3仅异类达标、2无明显重叠。袜6/17、腰带30/47；弱图重采样A/B仅为后续建议；当前优先交接固定候选，未准备或启动新训练。仅CPU诊断与8例看图，0推理/训练/AP重算/smoke；50k＋0.3候选保持。

前次完成（2026-11-18）：[50k完整dev候选与固定阈值](segmentation/u50000_candidates_dev250_20261118.md) **completed**。仅一次250图候选推理，阈值前23871个非空候选；三档0.3/0.5/0.7全dev候选1279/1155/1062，当前642条accepted开发查询覆盖602/592/580，选定 **50k＋0.3，93.7695%**。总448.390秒/900秒硬限，3496个新候选裁剪通过；0训练/0AP重评/0smoke/0排序，GPU与锁已释放。新包已发布，旧18k保留；队友本人接收未代签。

历史预算提案：[完整数据先1epoch、达标才第2epoch](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/0b8a0a88119d4e68f9386e11b88b94d968e0d8d2/docs/handoffs/segmentation/full_data_short_plan_20261117.md)。已有日志支持0.21秒/图量级；示例4万图含评价首轮2.43–2.88小时，最多两轮4.87–5.76小时，拟每轮3小时/总6小时。复用C5k20k权重，不重跑COCO初始化训练；新短方案接线和真实完整包仍待补，未启动训练，未重复smoke。

历史完整数据准备：[完整Fashionpedia分割发布需求与接收准备](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/48873ec6e6de7b311519be5eb0767fa8f0316d69/docs/handoffs/segmentation/full_fashionpedia_intake_20261117.md)已同步到独立`segmentation/full100-runtime@48873ec`。完整包实际N/hash、增量接收及冻结待补；100epoch入口已有21项CPU检查记录，本轮复用，不启动训练。C5k结果与D10k停止状态保持，完整数据接线未合并到本结果分支。

历史：[5k/10k对照结果与中断](segmentation/expansion_control_20261117.md)：C5k20k完成，10k/20k AP55.8283/64.6957，选20k；D10k159步主动中断，整对未完成。每组3小时上限未延长，GPU已释放，无重复smoke；不自动重启。

历史：[5k/10k证据复用与统一硬超时入口](segmentation/expansion_execution_preparation_20261116.md)已实现，11个新增CPU案例及真实metadata证据组合通过。默认不执行；每组20k、保存的10k/20k完整dev、每组3小时，失败停止且不延长。本轮0训练/推理/旧smoke。

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
