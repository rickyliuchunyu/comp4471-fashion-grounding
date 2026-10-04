# 工程模块与实现接口

本文将 [模型设计](technical_design.md) 和 [数据协议](data_protocol.md) 转成后续开发接口。以下目录、类型、函数和伪代码都是**实现规范**，对应运行代码尚未创建。当前仓库不能按本文直接训练；负责人仍全部待定。

## 1. 建议的代码结构

```text
src/fashion_grounding/             # 待创建
  data/convert.py                 # 两源转换、类别映射、质量检查
  data/splits.py                  # 分组、去重、划分 manifest
  data/queries.py                 # 查询加载与复核约束
  segmentation/mapper.py         # 训练 ID 与 source coverage 字段
  segmentation/criterion.py      # source aware loss 扩展
  segmentation/train.py          # 官方 trainer 的项目配置与日志
  segmentation/predict.py        # 固定候选解码、原图 mask
  encoders/visual.py              # DINO mask pooling 与坐标变换
  encoders/text.py                # BGE dense features
  grounding/rules.py              # L1
  grounding/heads.py              # L2/L3
  grounding/train.py              # 缓存特征上训练小网络
  evaluation/matching.py          # 候选与真实实例关联
  evaluation/metrics.py           # 选择准确率、IoU、分组统计
  evaluation/latency.py           # 分阶段时间与缓存场景
  demo/app.py                    # 图片、查询、叠图
```

先实现通用数据和候选接口，再并行开展人类团队的分割、查询与语言开发；具体人员之后决定。预训练模型下载、数据转换和排序训练分别有独立入口，不把三个大模型全部常驻在训练进程中作为起点。

## 2. 接口类型与张量

以下为拟采用的 Python 类型约定。CPU 数据原图采用 uint8 RGB，模型适配器各自预处理；候选 mask 在 CPU 上为 bool，始终是原图 $H\times W$。

```python
@dataclass
class Candidate:
    candidate_id: str             # image_id + decoder query index
    category_id: int              # 项目 1..8
    score: float                  # 分割分数
    bbox_xyxy: tuple[float, float, float, float]
    mask: np.ndarray              # bool[H, W]
    decoder_query_id: int

def predict_instances(image_rgb: np.ndarray, model, config) -> list[Candidate]: ...
def encode_regions(image_rgb: np.ndarray, candidates, encoder, config) -> np.ndarray: ...
# 输出 float32[N, 384]
def encode_queries(queries: list[str], encoder, config) -> np.ndarray: ...
# 输出 float32[B, 1024]；超 token 限额返回明确错误
def geometry_features(candidates, image_size_wh) -> np.ndarray: ...
# 输出 float32[N, 12]
def rank_candidates(query_features, visual_features, geometry, valid_mask, heads): ...
# 批量 logits float32[B, N_max]，padding 位置为 -inf
def select_target(image_rgb, query: str, candidates, encoders, heads, config) -> dict: ...
# 仅返回候选 ID、rank score、状态，不访问标注
```

geometry 与 visual 必须共享同一候选顺序。训练 batch 可包含不同图片的查询，pad 到该 batch 最大候选数；`valid_mask[B,N_max]` 区分真实候选与 padding。目标索引指向该图片自己的候选，禁止把同一个整数位置当作全局 annotation ID。

数据加载时核验 `N>0`、目标唯一、所有候选 mask 尺寸一致；训练排除 `N=1` 的 ranking loss。推理 `N=0` 返回 `no_candidates`；不能对全是 `-inf` 的行 softmax，否则会得到 NaN。真实候选诊断使用非 crowd 整件实例，排序稳定采用 annotation ID；每个 candidate 内部可保留评估映射，但绝不送入 heads。

## 3. Source aware loss 的伪代码

```python
# logits: [B, Q, 9]；各 decoder prediction 独立做 Hungarian matching
# matched[b,q] 与 target_class[b,q] 由 matcher 得到
for b in batch:
    allowed = [5, 6, 7, 8] if source[b] == "deepfashion2" else [8]
    for q in queries:
        if matched[b, q]:
            loss[b, q] = cross_entropy(logits[b, q], target_class[b, q])
            weight[b, q] = 1.0
        else:
            loss[b, q] = logsumexp(logits[b, q]) - logsumexp(logits[b, q, allowed])
            weight[b, q] = 0.1
loss_class = sum(weight * loss) / sum(weight)
# matched mask BCE/Dice 和 auxiliary losses 按模型设计处理
```

正式代码应矢量化，示例只说明语义。标准 `CrossEntropyLoss` 的 no object weight 无法独自实现集合标签；必须新增 criterion 并确认 mapper 的 source 信息实际进入每个输出层。AMP 下 logsumexp 至少用 float32；不能只写映射文件就声称已经解决混合监督。

## 4. 候选缓存与训练产物

图片缓存键拟由以下内容共同生成 SHA256：图片文件 hash、分割 checkpoint hash、候选解码参数、DINO checkpoint/revision、预处理版本、pooling 方法和输入尺寸。真实候选缓存将分割权重字段替换为 annotation manifest hash。只按图片文件名缓存不够。

缓存保存 candidate ID/order、类别、框、mask 引用、原始视觉向量和几何向量。文字缓存按完整原文、tokenizer/model revision、max tokens 和预处理生成键。排序头参数变化不改变 frozen feature 缓存，但一定要重新计算投影与分数。存储时不随意把 uint8 mask、有损 JPEG 叠图和原始 RLE 混用。

每次运行至少产生以下本地文件，不将 checkpoint 或大缓存上传 GitHub：

```text
outputs/<run_id>/
  resolved_config.json            # 完整合并配置，包含实际 override
  environment.json                # Python/torch/CUDA/GPU 与依赖、源码 commit
  data_manifest_sha256.txt
  checkpoint_sources.json         # URL、revision、下载文件 SHA256
  train_log.jsonl
  metrics_dev.json
  metrics_test.json                # 方法锁定后才生成
  predictions.jsonl               # 全查询成功与失败，不只保留好例子
  latency.json
  error_cases/                     # 标注与候选的可视化
```

训练中每 2,000 segmentation updates 或每个 grounding epoch 评估 dev。保存 last、best 与 optimizer/scheduler/AMP scaler/random states 以支持恢复。恢复后记录起始 update/epoch，不能将重新开始训练的日志拼成一条连续曲线。

## 5. 最小实现的验收顺序

| 顺序 | 必须完成的行为 | 证据 |
| --- | --- | --- |
| A 数据 | 32 张审查样本正确解码映射、mask、框和 source coverage | 统计 JSON 与原图叠图 |
| B 分割 | 小数据执行 100 updates；loss 有限，checkpoint 可恢复，单图有合法输出 | 日志、恢复记录、预测文件 |
| C 规则 | 正确选择同图左/右目标，未知属性保留 unsupported 状态 | 固定例子的输入输出 |
| D 特征 | mask pooling 坐标对齐，384/1024/12 维正确，padding 不参与 loss | 特征统计与坐标检查 |
| E 排序 | 小规模数据能降低训练 loss，反向梯度仅到小网络 | 梯度与 loss 日志 |
| F 评估 | 人工设计案例与公式一致：正确选择、错选、漏检、重复候选、空输出 | 指标核对记录 |
| G 演示 | 新图片与完整查询返回目标叠图；失败状态可见 | 完整调用记录 |

上述是之后实现时的验收要求，当前没有运行这些检查。小数据过拟合只说明数据流与优化能工作，不能当作泛化效果；正式实验使用冻结划分。

真正需要测试的是类别 ID 偏移、集合损失、mask 缩放、padding、匹配冲突与缺失候选计分等容易改变研究结论的部分。简单文案和目录占位不需要为每一行写测试。单 GPU 环境版本必须实测兼容后再锁定；当前不提供未经验证的安装命令。

## 6. 可重复性与使用边界

固定 Python/NumPy/PyTorch 随机种子，保存 data loader 和 sampler 配置；记录是否使用非确定 GPU 算子。seed 相同不自动意味着逐位相同结果。正式训练不得修改 dev/test 的任何标注或缓存以迎合某个方法。

演示输入只包含图片和文字；标注文件仅评估器可读。GitHub 保存实现、配置、小型指标与说明；共享存储保存数据、权重与日志。对当前文件的阅读不需要 GPU，实际运行需 A–G 检查通过后补充精确命令。
