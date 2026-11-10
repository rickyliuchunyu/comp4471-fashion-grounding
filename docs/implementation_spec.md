# 工程模块与实现接口

版本 `2026-11-05`。本文是待实现规范，当前没有以下运行入口。模型定义见 [技术设计](technical_design.md)，数据与评估语义分别见 [数据协议](data_protocol.md)、[实验计划](experiment_plan.md)。

2026-11-10 补充：[暂定分工](team_workflow.md) 定义各模块主责，B0 与 B2 分文件避免多人同时修改同一规则模块。以下仍为待实现接口。

## 1. 建议结构

```text
src/fashion_grounding/             # 待创建
  data/schema.py                  # 公共读写和验证，data_eval 维护
  data/convert.py                 # FP 八类转换，保留源细类
  data/splits.py                  # 去重、分组和冻结 manifest
  data/queries.py                 # 双语、改写、唯一目标和复核检查
  segmentation/train.py          # Mask2Former 官方 trainer 适配
  segmentation/predict.py        # 固定候选解码与原图 mask
  encoders/siglip.py              # 对应 revision 的双编码器和 processor
  baselines/category_position.py # B0，data_eval 负责
  grounding/rules.py              # B2 方位分数，grounding 负责
  grounding/heads.py              # M1 残差与 M2 空间网络
  grounding/train.py              # 冻结特征上的排序训练
  external/predict.py             # E1 独立系统适配，不读取 GT
  evaluation/matching.py          # 同类一对一关联
  evaluation/metrics.py           # IoU、选择、覆盖、配对与分组统计
  evaluation/latency.py           # 首次与缓存场景
  pipeline.py                    # 最小接通，segmentation 协调
  demo/app.py                    # 图片、文字、叠图与错误状态，data_eval 负责
```

不新增主线 source-aware criterion。X5 若启用，另建两源一致的五类前景训练/评估配置，验证后再登记。

## 2. 数据与张量接口

```python
@dataclass
class Candidate:
    candidate_id: str
    category_id: int | None       # 内部 S0 为 1..8；外部系统可为 None
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    mask: np.ndarray              # 原图 bool[H,W]
    decoder_query_id: int | None

def predict_instances(image_rgb, model, config) -> list[Candidate]: ...
def encode_regions(image_rgb, candidates, encoder, processor) -> np.ndarray: ...
# float32[N,D]，按 candidate 顺序，标准图像输出后 normalize

def encode_queries(queries, encoder, processor) -> np.ndarray: ...
# float32[B,D]，标准文本输出后 normalize，超限明确报错

def geometry_features(candidates, image_size_wh) -> np.ndarray: ...
# float32[N,12]，同类 rank 使用平均名次；单实例 0.5

def rank_candidates(text, visual, geometry, valid_mask, heads, config): ...
# float32[B,N_max]；padding 为 -inf，先处理空候选

def select_target(image_rgb, query, candidates, encoders, heads, config) -> dict: ...
# 不读取 target ID、GT 类别、query type 或复核信息
```

CPU 图片为 uint8 RGB，由各模型的 processor 独立处理。bbox 使用右下排他像素坐标，mask 必须与原图尺寸一致。D 从 checkpoint 读取并用前向核对，图文 D 必须相同；不得继续硬编码 384/1024。M1 第一层为 3D→256，M2 文本空间层为 (D+64)→128→1。

同图多个查询可复用 visual/geometry，valid_mask 必须与候选顺序一致。训练目标是该图片候选的局部索引，不是全局 annotation ID。N=1 不进入排名训练但保留评价；N=0 返回 no_candidates。失败输出 selected_candidate=null，不伪造 mask 或概率。

## 3. 特征缓存与运行记录

图片缓存键包含图片 hash、候选来源、分割 checkpoint/hash 或 GT annotation manifest/hash、解码参数、候选顺序、encoder revision、processor 版本、裁剪/背景处理参数。文本缓存键包含完整原文、语言处理/翻译版本、tokenizer revision 和长度策略。

缓存冻结且归一化的标准图文向量、候选元信息及几何向量；不缓存正在训练的残差/空间输出。E1 缓存单独命名。记录实际 D、数值范围、非零范数，核对 B1 的重建 logits 与模型标准前向一致。

每个 run_id 对应独立目录，保存 resolved_config、environment、源码 commit、权重来源与 hash、数据 manifest hash、训练日志、dev/test 指标、全查询预测及失败状态、latency 和错误案例。final test 仅在锁定方案后生成。checkpoint 保存 optimizer/scheduler/scaler 和随机状态；恢复日志记录起始 update，不拼接不同训练。

只将小型配置、指标与证据索引提交 GitHub。图片、权重、缓存和大输出按 [AutoDL 目录规划](team_workflow.md#7-autodl-与多-codex-同时工作) 组织，实际实例和目录仍待创建与验证。

## 4. 实现检查顺序

| 顺序 | 检查 | 所需证据 |
| --- | --- | --- |
| A 数据 | 小批量正确映射、mask 解码、源细类、分区不重叠 | 统计和原图叠图 |
| B GT 基线 | GT 裁剪下 B0/B1/B2、完整查询、失败计分 | 固定样本预测与 logits 一致性 |
| C 分割 | 200 updates 稳定性、保存恢复、单图合法候选 | 日志、显存与速度 |
| D 小网络 | 初始残差为零、梯度只到 heads、padding 无影响 | 梯度与小数据优化记录 |
| E 评估 | 漏检、错类重叠、重复候选、空输出、配对统计 | 手算可核对的固定案例 |
| F 外部系统 | 完整查询、多输出选择、无 GT 提示 | 固定 E1 配置与全量预测 |
| G 端到端 | 新图+新文字到叠图，首次/缓存计时 | 完整调用记录 |

这些检查尚未运行。小数据过拟合只验证数据流和优化，不能证明泛化。环境实测后再提供安装与训练命令；复用公开模型时锁定基础代码、checkpoint 和 processor，而不是拼接任意最新版本。

## 5. 可选扩展的接口边界

具体功能与评估见 [扩展计划](extension_plan.md)。X3 通过显式区域输入配置和缓存键区分普通裁剪、背景抑制、上下文裁剪；候选 ID 与评价关联保持固定。演示可复用候选、冻结特征和真实分数组成，但不得显示未经校准的正确概率。

交互纠正需另存会话状态与支持的操作类型；拒绝、多目标需独立标注和输出 schema 版本。实现前登记接口与数据变更，不把集合目标直接塞入当前唯一目标索引。上述模块尚未创建，现有单轮单目标接口继续有效。
