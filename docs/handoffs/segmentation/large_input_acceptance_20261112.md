# lcy 分割大包接收：仅 CPU 输入检查

日期：2026-11-12 19:01 HKT。执行者：Codex（lcy 工作区）。状态：input_compatibility_passed；不是正式冻结、训练或 dev 性能评价。

## 本次新增问题与复用证据

wkq `f5787f4` 已发布新 train/dev，dev 使用原 ID + 1 + 10^9，与旧 train 小包不同。需要确认新文件指纹、官方分区、类别/ID 和现有分割读取器兼容。复用已有 COCO 加载/八类适配、200-update 稳定性、跨进程恢复和 AP/选择实现验证，不重新加载模型、前反向或保存恢复。

开工检查：GitHub lcy=`efac766`、wyz=`66c0da7`、wkq=`f5787f4`，main=`7064ebb`；PR #2/#4 open/draft/unmerged，未发现新 PR，读取的 PR 评论为空。大包说明见 [wkq 固定提交交接](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/f5787f4ac7b6941abdadc43f9fbdc666eba93925/docs/handoffs/data_eval/large_dataset_20261112.md)。

服务器执行前：系统盘约 27 GiB、数据盘约 167 GiB 可用；GPU 0/24576 MiB、0%，无计算进程。使用现有 segmentation-v1 Python 3.10.8；CUDA_VISIBLE_DEVICES 为空，CPU 单线程读取，不占 GPU 锁。

## 结果

| 输入 | 图片 | 实例 | 现有读取器结果 |
| --- | ---: | ---: | --- |
| fp_train5000_v1 | 5,000 | 17,775 | 全部图片 hash/RGB/尺寸、COCO mask/框/面积通过；项目 1–8 映射内部 0–7；crowd_skipped=0 |
| fp_dev250_v1 | 250 | 1,008 | 全部图片 hash/RGB/尺寸、COCO mask/框/面积通过；十亿级图片/实例 ID 保留并正确引用 |

- train 来源均为 official_train2020；dev 均为 official_val2020。图片/实例项目 ID 与各自源 ID、offset 一致。
- train/dev 的 image_id、source_image_id、SHA256、group_id 交集均为 0。这里只确认已发布分组隔离，不代替未知近重复的人工审查。
- 固定发布 image manifest 与 checksum manifest 指纹；核对分割消费者实际使用的 image manifest、COCO、mapping、RELEASE 元数据，并由现有读取器校验 5,250 张图片和 18,783 个 COCO 实例几何。
- 没有重验生产方全部候选文件、源 mask 逐像素对照或 93 项测试；这些证据按 wkq 发布记录复用，不冒称本轮重跑。
- 52.41 秒；模型加载=0、optimizer updates=0、checkpoint 操作=0。未读取语言查询或最终测试包样本。

数据根目录：`/root/autodl-tmp/fashion-grounding/shared/processed/`。本次输出：`runs/segmentation/large-input-acceptance-20261112-r01/acceptance.json`（相对项目服务器根目录）。

## 可复核证据

- [接收报告](evidence/large_inputs_20261112/acceptance.json)：输入与源码 SHA256、逐类计数、分区交集、范围及剩余条件。
- [实际运行脚本](evidence/large_inputs_20261112/accept_large_segmentation_inputs_20261112.py)：仅组合现有读取器；包含固定数据指纹及不可覆盖输出目录。脚本作为本次证据归档，不要求无变化时复跑。
- 服务器读取器提交为 `dd64016`；schema.py、segmentation/data.py、segmentation/evaluation.py 的内容 hash 与本轮本地 `efac766` 完全一致。后续提交仅补证据，不能将其冒称原运行源码。
- 实际命令：`/root/autodl-tmp/fashion-grounding/envs/segmentation-v1/bin/python /root/autodl-tmp/fashion-grounding/runs/segmentation/env-20261112/accept_large_segmentation_inputs_20261112.py`。

本项是跨 train/dev 的输入接收，不是某个方法/seed/split 的训练或评价运行，按数据验收记录留在本交接和报告；不新增伪 S0 实验行，registry 的正式指标保持空白。未修改模型源码、训练配置、共享包或队友工作区。

## 放行边界与下一步

**lcy 的分割输入兼容性通过。** release_index 和两包 RELEASE 均仍为 `formal_freeze=false`，split_plan 为 `versioned_split_pending_team_acceptance`；本次不替全组宣布冻结。

下一步只需闭合分割分区的接收/近重复复核及冻结记录，然后使用匹配冻结 train 的 checkpoint 做真实 dev AP/选择小阶段。已有 pilot 100/200 步权重仍不能作为正式选择输入。完整长训练保持未启动。

分割 dev 的监督来自源 GT，不依赖语言描述审核；wyz 的语言 dev 人工验收是另一条条件，不能因此把分割准备工作全部阻塞。测试包继续留出。相同版本的本次输入检查已通过，后续直接引用本报告，只有具体变化使结论失效时才补受影响检查。
