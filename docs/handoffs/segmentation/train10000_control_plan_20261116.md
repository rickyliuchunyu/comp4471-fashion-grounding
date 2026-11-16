# 5k / 定向 10k 分割受控训练方案（未启动）

2026-11-16，lcy / segmentation。数据已通过 [增量技术验收](train10000_acceptance_20261116.md)。方案、配置及[证据复用/硬超时入口](expansion_execution_preparation_20261116.md)现已准备并通过新增CPU检查，**没有启动或排队 GPU 任务**。用户已固定每组3小时上限，但仍要求本轮先不训练；预算配置不等于启动授权。

## 要回答的问题

在相同计算预算和分割配方下，加入 wkq 新增的独立训练图片，是否改善小目标漏检、弱类和形状精度，同时维持整体 AP？新增 3500 broad +1500 targeted，因此比较“定向扩充”整体收益，不能把结果解释为纯数据量效应。只使用 Fashionpedia 八类；dev250/test400 不进入训练。

## 配对设计

| 因子 | C5k | D10k-targeted |
| --- | --- | --- |
| 数据 | 原 fp_train5000_v1 | 旧 5k 严格保留的 fp_train10000_targeted_v1 |
| 初始化 | 同一已校验官方 COCO 权重；seed42 初始化八类头 | 完全相同 |
| 成功 update / batch | 20000 / 2 | 20000 / 2 |
| 图片曝光数 / 名义 epoch | 40000 / 8 | 40000 / 4 |
| 输入与增强 | LSJ 768，min_scale=0.5，max_scale=2，水平翻转 | 完全相同 |
| 优化 | AdamW 1e-4、backbone×0.1、warmup500、12k/16k 各衰减×0.1；沿用 AMP/clip/恢复策略 | 完全相同 |
| 完整 dev | 10000、20000 步，各 250 图/1008 GT | 完全相同 |
| 推理 | FP32、800/1333、upstream instance top100、无语言阈值 | 完全相同 |

初始 optimizer/scheduler/sampler/RNG 均按相同规则新建；每组内允许按现有严格检查恢复，不跨数据集恢复 sampler。固定成功 update 数，AMP 重试不记成额外 update；不因 10k epoch 少而自动加倍步数。20k 是共同预算，不承诺两组已收敛。

**为何需要新 5k 对照：** 已有 B20k 是先用 min_scale=0.1 训练到 18k，再用 0.5 训练 2k，且有历史分阶段 LR 预算。直接拿它与全程 min_scale=0.5 的新 10k 比较会混入配方/训练轨迹差异。新 C5k 是必要的科学对照，不是重复 smoke；旧 A/B、18k/20k、800/1024 预测和诊断全部复用为历史参考，不再推理。

这里选择已有官方 COCO 初始化入口，使两组冷启动路径完全一致。现有 `--checkpoint` 是 COCO 适配入口，不能把 B20k 伪装成官方 COCO；`--resume` 会拒绝数据 hash/sampler 不一致。若以后改为八类权重 warm-start，须另定两组共同初始化及重置规则，不能只给 10k 重置优化器、修改 checkpoint 元数据绕过检查。

## 如何判断有效

主要终点固定为 **两组 20000 步完整 dev mask AP 差值**；10k 步同预算对照用于观察学习过程。每组在事前固定的 10k/20k 两个点内按 AP 最大选择，平手取更早；跨数据组另外生成比较报告，不把不同训练数据/config 的报告塞进严格的组内 checkpoint 选择器。

预先记录的实用筛选线（百分点，不是统计显著性或预期收益）：D10k−C5k 的整体 AP≥+0.5、AP_small≥+1.0、AP75≥−0.2；skirt/outerwear/bag/accessory 各类 AP 不下降超过 1.0。未通过则报告具体取舍，不自动替换当前权重或继续加步数。AP75 只是较严格重叠指标，不能单独叫“边界已经修好”。

从这四份新 dev 预测直接复用现有诊断入口，报告所有小目标的漏检、分类、IoU≥0.5/0.75 匹配数及固定 1 原图像素内边界 IoU；同时列共同匹配子集，防止换匹配对象造成均值错觉。袜子、腰带、头饰、眼镜、鞋、包保留源细类与分母；小包 dev 仅 2 例、部分尺寸×类别只有 1 例，必须给计数，不能据此宣称普遍显著改善。薄边界诊断不是标准 Boundary AP，不用 GT 修正预测。

即便 D10k 胜过 C5k，还须与复用的当前 B20k＋800（AP=63.79295）比较后决定是否值得替换部署权重。单 seed42 结果只支持这一轮开发决策；不自动追加多种子。若 AP_small/漏检改善但 AP75/边界无改善，结论只能是增加数据帮助了检测，边界问题仍待单独处理。

## 配置、预算与已完成接线

- [C5k 配置](../../../configs/runs/segmentation/expansion_control_train5000.json)、[D10k 配置](../../../configs/runs/segmentation/expansion_control_train10000.json)、[相同 dev 配置](../../../configs/runs/segmentation/expansion_control_dev250.json)、[结构化方案](../../../configs/runs/segmentation/expansion_control_plan.json)。配置已通过现有读取器解析，两个训练配置仅 manifest/annotation hash 不同，[检查结果](evidence/train10000_acceptance_20261116/config_check.json)。
- **已按用户要求固定每组总硬上限 3 小时、两组累计最多 6 小时**，顺序独占共享 GPU。每组训练内部上限 10000 秒，两个 dev 点合计内部上限 600 秒，剩余用于启动/归档；外层单一截止时间必须覆盖全链路。超时保留完整已写 checkpoint，标记 incomplete，不自动重启/延长、不把某组较少步数与另一组 20k 宣称公平终点。建议每 5k 保存完整恢复点，dev 只评预定 10k/20k，避免额外选择机会。
- **有界入口已实现，默认仅准备。** `scripts/run_expansion_control.py` 按两组同一源 commit、共享锁、每组单一硬截止时间执行训练→保存的10k/20k评价→选择。CPU进程案例已验证共同deadline、子进程树终止及失败停止；真实3小时训练尚未执行。详见[入口交接](expansion_execution_preparation_20261116.md)。
- **评价证据组合已接通。** 原 `reuse_eval_package` 保持同train指纹限制；新增 `--input-validation-bundle` 将原dev报告与10k验收、接收计划和运行记录按hash绑定。真实两组 metadata/hash 核对通过，不重做 dev 图片/mask 审计，不篡改旧报告。训练器自身正常建立读取记录不是另起模型 smoke。
- 正式冻结名义所需的新数据人工处置/三方记录由数据负责人和相应成员完善；保留本轮技术验收及生产方原始审核来源，不代签。以后明确启动时再核最新 GitHub、资源、磁盘、共享锁，确认输入仍为本次版本；无需重新做已通过的 COCO/头适配/稳定性/AMP/恢复 smoke。

本轮未改类别损失权重、采样器、网络、训练尺寸或分割候选阈值；未替 wkq 构建数据、未替 wyz 训练排序。数据与权重保持服务器版本，本 PR 仅提交配置、小型验收证据和本角色代码/交接。
