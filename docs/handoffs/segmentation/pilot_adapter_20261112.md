# lcy：真实分割输入、候选接口与 GPU 工程检查

日期：2026-11-12（香港）。角色：segmentation / lcy。P0/P1 的一个可验证小阶段；**implemented + CPU checked + bounded GPU smoke verified**，不是正式 S0 训练或性能结果。P1/P2 仍 in_progress，P5-S 未开始。

分支：`segmentation/pilot-adapter-v1`。GPU 通过时源码 commit：`3220ae6ccfde86ddb6214ead396dea4e4ffae6e2`；交接文档版本见本文件 Git 历史。分支已纳入 wkq 的 `dc24f2d`（[PR #1](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/pull/1)），保留其数据代码和状态，不改写历史 proposal。

## 实际完成

- 建立独立课程 checkout、Mask2Former 源码副本和 `envs/segmentation-v1`。环境依赖从旧 base 复制；旧 editable 项目路径仅在副本内禁用。没有升级其他成员的依赖。
- 接收 wkq 的 `shared/processed/fp_pilot20_v2`：全部 20 张真实 Fashionpedia train 图、145 个实例通过 hash、RGB 解码、尺寸、mask/框/面积检查；全部 145 个实例可转换成 Mask2Former 输入。
- 项目 1–8 显式映射到训练 0–7，no-object=8；crowd 不进训练。本包无 crowd，八类实例数依次为 16/8/8/16/7/40/16/34。完整语言查询集不构成本次分割检查的前置条件。
- 官方 R50 模型构建、Xavier 分类头、严格 COCO state 适配函数、独立语言候选解码与 schema 3 导出。公共读写/RLE 复用 wkq 的 `data/schema.py`。
- 共享 GPU `flock` 包装器：进程/显存与 running 记录检查，180 秒超时、独立 run 目录和占用记录。额外实测持锁时第二个入口拒绝启动。它不能约束绕开包装器的其他进程，队友需要采用同一个锁。
- 本地与服务器各 27 项测试通过（wkq 19 + segmentation 8）；原有合成契约检查和 2 项检查器测试本地通过。服务器重新执行 wkq GT 包验证也通过。

证据：[数据检查](evidence/pilot_adapter_20261112/data_check.json)、[27 项服务器测试](evidence/pilot_adapter_20261112/unit_tests_final.log)、[GT 包验证](evidence/pilot_adapter_20261112/upstream_validation.log)、[模型检查报告](evidence/pilot_adapter_20261112/smoke_report.json)。

## GPU 结果与边界

固定随机初始化、seed=42、单图 batch、短边 256/长边至多 384、完整 R50/100 queries/10 次预测/官方点采样损失。两次 optimizer updates；第一个 update 后保存模型、optimizer、scheduler、AMP scaler、Python/NumPy/torch/CUDA RNG 与 update index，重建模型和优化器再恢复，第二个 update 完成。AMP 初始 scale=128，梯度检查有限；这不是正式 LSJ768、200-update 稳定性或持续训练验证。

| 项目 | 实测 |
| --- | --- |
| 输入图片 ID | 1643、3165；均为 train |
| 两次总 loss | 137.5679、94.7770；图片不同，不解释成收敛趋势 |
| 恢复位置 | update 1 → 完成 update 2 |
| 模型检查段耗时 | 7.8565 秒，不含前置全包 CPU 加载 |
| Peak CUDA allocated / reserved | 1,668,814,848 / 2,768,240,640 bytes（约 1.55 / 2.58 GiB） |
| CUDA 扩展数值核对 | 前向误差 ≤2.23e-16，三组梯度误差 ≤4.45e-16（double 小张量） |
| 原图预测导出 | 阈值 0.5，0 个候选，合法 `no_candidates` |
| 导出性质 | `illustrative_only=true`，真实图片上的随机模型工程输出 |

**非空 mask 导出仅由合成数值测试验证。** 本次没有非空真实模型候选的质量证据，也没有标准 COCO AP 或语言评价。保存/恢复成功不代表长训练数值稳定，随机模型不可交给队友当作固定 S0 候选。

所有尝试保留独立日志和 run 行：

| run 后缀 | 状态 | 证据范围 |
| --- | --- | --- |
| r01 | failed | 旧 CUDA 扩展缺少 3090 的 sm_86 kernel；非有限梯度，未完成更新 |
| r02 | failed | 扩展重编译后，数值检查代码传入非连续 grad_output，被上游 CUDA 检查拒绝 |
| r03 | failed | 前向/梯度数值对照通过；随机模型默认 AMP scale=65536 产生非有限梯度 |
| r04 | completed | 数值检查修正、AMP 初始 scale 显式设为 128；两次更新、恢复和空候选导出通过 |

运行均限 180 秒，失败未隐藏；`experiments/registry.csv` 原 14 条 plan 保留，新增 4 条明确标为 P1 engineering smoke 的 run，所有性能指标为空。各 run 的 metadata 和日志位于本目录的 [evidence](evidence/pilot_adapter_20261112)。

## 环境、版本和输入

- [环境实测](evidence/pilot_adapter_20261112/environment.json)：Python 3.10.8、torch 2.1.2+cu121、torchvision 0.16.2+cu121、Detectron2 0.6、NumPy 1.26.4；RTX 3090 24 GB、CUDA Toolkit 12.1。venv 使用 base 的解释器/标准库，依赖文件独立复制，不声称跨机器可完整重建。
- 官方源码固定 `9b0651c6c1d5b3af2e6da0589b719c514ec0d69a`，独立 clone 的 tracked 文件干净。
- 旧扩展仅“可导入”不等于可运行；在独立源码/环境按 `TORCH_CUDA_ARCH_LIST=8.6` 重编译。新 binary SHA256 见环境记录；旧镜像扩展未替换。
- [官方 Model Zoo](https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/MODEL_ZOO.md) 的 COCO instance R50 model `47430277_2` / `model_final_3c8ec9.pkl` 已定位，见 [upstream.json](../../../envs/segmentation/upstream.json)。**文件未下载、SHA256 未锁定、真实预训练加载未验证**；分类头适配约束目前仅合成 state 测试。
- 输入 manifest SHA256：`1b1ec9f25c37cd7b910dfe2882cc33637be37f3fba32e6f7b0c32592b570f111`。
- 输入 COCO annotations SHA256：`96f8edfce9b224992979cc282d6dc960849835cb06f0f83f5ad6e2ef225a4883`。
- r04 checkpoint SHA256：`3ce27cfc9fae65bce166d3d49a832508a2c1e836ae950cb8168b1e41b2468f2d`；随机工程 checkpoint，仅保存在服务器 run 目录。

服务器工作区：`/root/autodl-tmp/fashion-grounding/workspaces/segmentation`。日志/配置/恢复点：`runs/segmentation/segmentation-smoke-20261112-r04`；环境日志：`runs/segmentation/env-20261112`。数据、权重和认证信息不入 Git。

开工时系统盘/数据盘可用约 4.6/50 GB；结束检查约 27/24 GB。本次另观察到其他会话正在迁移 legacy outputs，未干预该进程，不能将系统盘变化归因于本次分割工作。本任务未移动或删除旧 outputs；仅清理自己创建且失败的 Conda clone 副本，保留失败日志。

## 复现与下一项交付

实际命令见各 run 的 `*-metadata.json`，通用入口和环境操作见 [环境说明](../../../envs/segmentation/README.md)。本轮没有长训练、大数据/权重下载、关机或扩容。

- lcy 下一步：获取并校验公开 COCO 权重，执行真实分类头加载检查；之后对合法 train/dev 分割包明确配置与预算，再做 200-update 稳定性、正式 trainer/恢复、dev AP 与非空预测候选检查。当前 `smoke.py` 不是正式 trainer。
- wyz：可审阅候选 API/序列/原图坐标约定；本轮随机模型包不可作为正式语言比较输入。消费方人工代码复核尚未完成。
- wkq：分割端对试点的工程消费已通过；人工 mask 质量、查询修订与正式划分仍待其流程处理。保留其 6 组分歧、2 组已示答案和 accepted=0 的状态，不替其宣告 P2 done。
- 尚未验证：公开预训练加载、全量数据、正式 train/dev 冻结、长训练/恢复一致性、非空真实候选质量、AP、完整系统、其他成员本机 SSH、最终提交。P1 的通过范围仅限本文件明确记录的小检查。
