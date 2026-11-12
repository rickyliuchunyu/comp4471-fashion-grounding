# 固定 train/dev 上的限定训练、真实 AP 与 checkpoint 选择

日期：2026-11-12；执行者：Codex（lcy 工作区）；PR #2。运行源码固定为 `b02f6e9d05b6a59297b64a4ad78494db45c75dd4`。本轮目标是完成新 train/dev 上的真实评价与离线选择，不重跑已通过的环境、权重、200-update 或恢复 smoke；不启动完整长训练。

## 数据接收依据与版本

用户最新明确确认“现在发的就是经过人工复核的，你先用”，覆盖此前答复。按此确认接收并固定 `fp_train5000_v1` / `fp_dev250_v1`，见 [本次消费者冻结记录](segmentation_train_dev_freeze_20261112.json)。该记录保留用户确认原话、包指纹、三方技术证据及小阶段授权，不虚构逐对复核结论或其他成员签名，不修改原发布文件的历史 `formal_freeze=false`，不变更语言查询审核状态或最终 test 冻结。

train 为 5,000 图 / 17,775 GT，dev 为 250 图 / 1,008 GT；八类、源 Fashionpedia。开工仅核对 metadata/checksum-manifest 指纹仍与 [已通过接收](large_input_acceptance_20261112.md) 一致；沿用 [53 对分组、10 张隔离与 160 例抽样](freeze_preparation_20261112.md)。训练/评价入口自身必要的数据完整性检查正常执行，没有另外跑一轮验收脚本。

整组 GitHub 增量无变化：wkq=`f5787f4`、wyz=`103dcc5`、本轮前 lcy=`457cc6d`；PR #2/#4 draft、#5 non-draft，均 open/unmerged，无新增评论。队友实现不合并、不重验；本次只快进自己的服务器 checkout。

## 实际运行与边界

服务器根：`/root/autodl-tmp/fashion-grounding`，使用原 `envs/segmentation-v1`，未安装/升级依赖。开始前系统盘约 27 GiB、数据盘约 166 GiB 空闲；3090 24 GiB GPU 0 MiB / 0%，无计算进程及 running 协调记录。训练和评价分别使用已有 `scripts/with_gpu_lock.py`，上限 300 / 600 秒，实际命令记录在证据中的协调 JSON。

- 训练：`runs/segmentation/segmentation-real-dev-train-20261112-r01`；配置 `configs/runs/segmentation/s0_dev_selection_100.json`。官方 COCO 初始化、seed42、batch2、LSJ768、AMP scale1，保留 S0 的 LR、500-update warmup、clip=.01；100 updates 后停止，保存第 50/100 步。
- 评价：`runs/segmentation/segmentation-real-dev-ap-20261112-r01`；配置 `configs/runs/segmentation/dev_ap_250_bounded.json`。分别评价完整 dev，官方 query×class top100 实例解码、FP32、短边800/最长边1333、原图 mask；不应用语言候选阈值。
- 选择：已有 `segmentation.select_checkpoint` 核对报告/预测/checkpoint hash及配置/源码/数据可比性，按 dev mask AP 最大、并列较早 update。输出权重引用，不额外复制权重。

训练已完成 100 updates / 200 个不同图片曝光（不是遍历完整 5,000 图的一轮）。loss 从 407.9111 到 43.4760，范围 29.8887–950.1073；前/后10步均值 400.1402 / 53.8930，仅描述不同 minibatch 上的训练轨迹。全部 loss/梯度有限、0 跳步、0 空目标增广图，backbone 与分类头均更新。裁剪前 norm 范围 341.3677–33456.0508，裁剪后最大 .010000002，AMP scale 始终1。峰值 allocated/reserved 为 4.37/4.97 GiB。没有新增恢复重放；旧恢复验证继续复用。

## 结果与证据

状态：**completed / evaluated**。真实 dev AP、两 checkpoint 比较与离线选择均完成；本轮不是完整 S0。

| update | dev 图片 / 原始预测 | mask AP | AP50 | AP75 | AP-small | AP-medium | AP-large |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 250 / 25000 | 0.100682 | 0.219023 | 0.111050 | 0.053489 | 0.148173 | 0.458256 |
| 100 | 250 / 25000 | 0.458512 | 1.086099 | 0.377752 | 0.000000 | 0.542031 | 0.728348 |

AP 均为 0–100 标度。两次 250/250 图片完成、0 推理错误，evaluation optimizer updates=0、模型指纹保持不变、queries_read=false。

**选择结果：第 100 步，dev mask AP=0.45851211941110814。** checkpoint SHA256：`30f5bf4f5178c83ce420b8396b9f616ceaa2e03be1c28aef69950f15c6482c1d`；原文件为上述 train 目录下 `checkpoint_000100.pth`。选择记录位于 `runs/segmentation/segmentation-real-dev-selection-20261112-r01/selection.json`，本地摘要见 [selection.json](evidence/real_dev_20261112/selection.json)。仅在这两个短跑保存点之间选择，不是最终发布权重。

| 类别 | update50 AP | update100 AP |
| --- | ---: | ---: |
| top | 0.000500 | 0.013310 |
| pants | 0.007807 | 0.113404 |
| skirt | 0.001560 | 0.000000 |
| outerwear | 0.033943 | 0.109653 |
| dress | 0.018924 | 0.712204 |
| shoes | 0.012133 | 0.006816 |
| bag | 0.723567 | 2.709268 |
| accessory | 0.007023 | 0.003441 |

评价峰值 allocated/reserved 为 2.45/9.31 GiB（reserved 是分配器保留量）。GPU 锁实际持有：训练 96.21 秒、评价 235.41 秒，共 331.61 秒；各自未超过 300/600 秒。完整评价内部计时 230.65 秒；update100 报告的 elapsed 是累计值，不能当作第二个 checkpoint 单独耗时。

结束盘点：系统盘约 27 GiB、数据盘约 165 GiB 可用，GPU 0 MiB / 0%，无计算进程和 running 协调记录。

证据入口：[运行元信息/实际命令/文件 hash](evidence/real_dev_20261112/run_metadata.json)、[训练报告](evidence/real_dev_20261112/train/train_report.json)、[逐步训练日志](evidence/real_dev_20261112/train/metrics.jsonl)、[50步 AP](evidence/real_dev_20261112/update_000050/evaluation_report.json)、[100步 AP](evidence/real_dev_20261112/update_000100/evaluation_report.json)、[结束盘点](evidence/real_dev_20261112/postflight.json)。原始 50,000 条 RLE 预测和模型留在服务器，Git 只保存报告/hash/小日志。

registry 保留 14 条 plan 与既有 9 条工程运行，新增 1 条限定 train run + 2 条真实 dev checkpoint 评价 run，三条均 completed；只在真实 dev 两行填 mask_ap，训练及旧工程行不填性能。完成后核对下载证据的文件 SHA 与远端索引一致、报告覆盖数和选择结果一致，没有重跑模型、测试套件或 COCO 评分。

## 停止条件与后续范围

本轮限于两份 checkpoint 的真实 dev 比较，100 updates 尚在 500-update warmup 内，不代表完整 S0 收敛或最终性能。不根据本轮低 AP 自动加步数、调 warmup、改 seed 或开启长训练。当前没有语言候选阈值选择、生产候选导出、SigLIP 再验、语言指标或 test 模型评价。

后续推进正式 S0 预算与固定候选交付时直接复用本轮接收、评价及选择证据；不得再次安排相同 smoke。完整长训练仍未授权启动。
