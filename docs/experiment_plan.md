# 实验设计与评估规则

本计划把效果主张转成可复现的对照。所有实验均未运行；起始参数是待验证设计，详见 [training_plan.json](../configs/training_plan.json)。划分与数量见 [数据协议](data_protocol.md)，方法公式见 [模型设计](technical_design.md)。

## 1. 待检验的假设

| 问题 | 假设与可能失败的原因 | 需要的证据 |
| --- | --- | --- |
| 混合数据是否有用 | 服装监督可能改善五类；域差异与缺标注也可能降低八类表现 | 各源各类 AP；普通/缺标注损失对照 |
| 区域文本学习是否有用 | 投影头可能学会外观语义；小数据可能只够学类别 | 同候选规则对照；外观和同类多实例组 |
| 空间输入是否有用 | 可能改善左右查询，也可能使模型依赖位置捷径 | L2/L3 消融与反事实查询 |
| 缓存是否有用 | 同图新查询可跳过图片编码；首次处理仍需完整流程 | 分阶段延迟、峰值显存 |

这些是研究假设，不能预先写成提升。课程没有必须达到的 AP 或毫秒门槛。

## 2. 固定实验条件

每次运行关联代码 commit、完整配置、数据/查询 manifest SHA256、映射版本、checkpoint 来源及 hash、环境、seed 和唯一 run ID。输出不得覆盖，失败也登记。负责人仍留空。

训练使用官方 train；dev/final test 来自冻结的官方 validation 分组子集。语言方法使用同一查询版本、分割权重和候选解码器。方法选择只看 dev，final test 在方案锁定后统一评估保留方法。

分割起始预算为有效 batch 2、20,000 optimizer updates，即 40,000 次图片抽样，不等于 40,000 张不同图片。混合采样可重复，各源曝光量不同，必须分别记录。主对照统一 batch、updates、augmentation、optimizer 与初始化；不把更长训练预算归因于数据策略。

## 3. 分割矩阵

| ID | 训练数据与改动 | 控制变量 | 优先级 |
| --- | --- | --- | --- |
| S1-FP | Fashionpedia 八类，普通损失 | R50、公开初始化、20k updates | 必做八类基线 |
| S1-DF2 | DeepFashion2，五类头 | 相同骨干与预算 | 共享五类诊断 |
| S1-MIX | 两源 1:1 图片采样，八类，source aware loss | 与 FP 相同八类结构 | 必做 |
| S4-NAIVE | MIX 改成普通八类 no object 损失 | 源采样、seed、预算不变 | 必做监督对照 |
| S2 | MIX 源比例改为 FP:DF2=3:1 | 损失和其他设置不变 | 可选 |
| S3-RES | 固定权重，推理尺寸 640/768 | 精度、阈值、候选规则相同 | 必做，无需重训 |
| S3-AMP | 固定权重与尺寸，float32/AMP 推理 | 相同图片与候选阈值 | 必做，无需重训 |

DF2 的五类头与八类头不同，须注明。八类主比较是 FP 与 MIX；DF2 单源只解释共享五类域差异。分割起始仅 seed 42，不能据此声称统计稳定；预算允许时对相应对照都追加 seed。

FP 上报告八类 mask AP、AP50、AP75、AP-small、逐类 AP 及共享五类汇总。DF2 上评估类别显式限定 1–5；其鞋、包、配饰缺真值，不能解释成完整八类结果。source aware loss 可能增加这三类假阳性，FP 上须增加 precision/recall 或每图 false positives 与可视化。

COCO mask AP 使用 IoU 0.50–0.95、步长 0.05 的标准汇总及 maxDets=100，面积分组沿用 COCO。没有正例的类别记为不可用，按 evaluator 处理，不手填零。[官方 COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)

## 4. 语言矩阵

| ID | 输入与参数 | 目的 | 优先级 |
| --- | --- | --- | --- |
| L1 | 中英文词表、候选类别与中心 | 类别/方位规则基线 | 必做 |
| L2 | 384 维视觉、1024 维文本、可学习投影 | 检验跨模态学习 | 必做 |
| L3 | L2 + 12 维几何、文本条件空间头 | 显式空间消融 | 必做 |
| L4 | L3 的 mask pooling 改为扩框裁剪 CLS | 区域表示与开销 | 可选 |
| L5 | 固定 L3，互换左右配对文字或去掉修饰词 | 检查输出是否随文字变化 | 必做诊断，无需重训 |

L2/L3 固定编码器、投影、数据、温度选择范围和预算，分别使用 seed 42/43/44。L2/L3 不做人工类别硬过滤；L1 也不读取人工类别和 query type。只新增空间头才能把差值解释为空间输入作用。

默认用真实 mask 训练小网络。预测候选适配阶段 B 暂不计入核心实验；若加入，L2/L3 都执行并单列结果。首轮失败先查数据、梯度和负例，不立即解冻大模型扩大变量。

每种方法都在两套相同候选上评价：

- **Oracle candidates**：真实非 crowd 整件实例，诊断文字选择能力。
- **Predicted candidates**：固定分割器预测，评价实际流程。漏检查询不能删除。

总体结果外，报告 N=1/N≥2、同类别实例数 1/≥2、类别、方位、外观、combined、中文/英文组。query type 可重叠，不相加当成总量。唯一类别成功不能证明会处理同类歧义。

## 5. 预测与 GT 的一对一关联

以下关联只在评估器执行，推理不能读取结果：

1. GT 仅保留八类非 crowd 整件实例。预测按分割分数降序，同分按 candidate ID 字典序。
2. 对当前预测，仅考虑同类别且未匹配的 GT，计算原图 mask IoU。
3. 最大 IoU ≥ 0.5 时关联该 GT；同 IoU 取最小 annotation ID，并占用它。否则不匹配。
4. 所有语言方法复用同一关联；重复预测不能都获得同一个 GT 的正确标签。

这是项目定义的 confidence greedy matching，用于目标选择准确率。它不是训练用 Hungarian matching，也不是替换标准 COCO AP evaluator。候选解码、分割分数和匹配阈值都固定后再比较语言方法。

## 6. 指标定义与误差分解

设测试共有 T 条查询，真实目标 mask 为 M，选中候选 mask 为 M_hat：

$$
\operatorname{IoU}=\frac{|M_{\mathrm{hat}}\cap M|}{|M_{\mathrm{hat}}\cup M|},\qquad
\operatorname{mIoU}=\frac1T\sum_{t=1}^{T}\operatorname{IoU}_t.
$$

无候选、拒绝、空 mask 和推理失败保留在 T 内，IoU 为零。选错目标仍计算实际交并比，不凭标签人为清零。类别错但 mask 重合可能有非零 IoU，因此同时报告类别约束的选择准确率。

| 指标 | 定义 |
| --- | --- |
| Oracle selection accuracy | 选中 GT annotation ID 等于目标 ID 的查询比例 |
| Predicted selection accuracy | 所选候选经上述关联对应目标 ID 的比例，无关联算错 |
| Candidate coverage | 目标 ID 出现在任何候选关联中的查询比例 |
| Conditional selection accuracy | 只在有匹配目标候选的查询中计算准确率，仅作诊断 |
| Mean target mask IoU | 全部查询的最终 mask IoU 平均，包含失败零分 |
| Success at IoU 0.5 | 全部查询中最终 mask IoU ≥ 0.5 的比例 |
| Macro category accuracy | 各有样本类别准确率的等权平均 |

固定上述关联时：

$$
Acc_{\mathrm{pred}}=Coverage\times Acc_{\mathrm{conditional}}.
$$

coverage 为零时 conditional accuracy 记 N/A。这将分割漏检与有目标时的排序错误分开。Success 只衡量 mask 几何，predicted accuracy 还要求类别/关联正确，两者不同并不必然是错误。

中英文同义配对另报选择一致率：两条均有输出且选择同 candidate ID 的配对比例，拒绝算不一致。一致不代表正确，两种语言可能一起选错。左右反事实组报告两条都正确的比例，不只检查输出改变。

## 7. 统计与 checkpoint 选择

grounding 每 seed 报指标，汇总 mean/std，不只挑最好 seed。预登记 seed 42 的 L2/L3 差值可用 paired bootstrap 求 95% 区间：按图片带回所有查询，重采样 1,000 次，bootstrap seed 42。中英及改写相关，不能按独立 query 采样；小组注明图片数与区间，不把几条样本的百分比差异写成确定优势。

分割每 2k updates 评估固定 dev，八类按 FP dev mask AP 选 checkpoint，DF2 诊断按 DF2 dev 五类 AP。grounding 每 epoch 按 oracle dev selection accuracy 早停，patience 5；其与预测候选表现的差异也要解释。若改变选择规则，final test 前登记并对 L2/L3 同时改变。

## 8. 延迟与显存

相同 GPU、batch 1、尺寸与候选下，分解计时：分割、DINO 区域编码、BGE 文字编码、投影/排序、mask 与叠图输出。使用 wall clock，GPU 阶段起止同步；异步 kernel launch 时间不是完整执行时间。

起始微基准为模型加载后预热 20 次、测量 100 次，报告 mean/p50/p95 与 peak memory；记录原图/缩放尺寸、候选数、文本长度和软件版本。单张图重复只是微基准，另在测试图片分布报告实际响应。

| 场景 | 工作内容 |
| --- | --- |
| 首次图片 | 分割 + 区域编码 + 新文字编码 + 排序 + 输出 |
| 同图新查询 | 图片/候选缓存 + 新文字编码 + 排序 + 输出 |
| 同文字重复 | 图片/文字缓存 + 投影与排序 + 输出 |
| 冷启动 | 模型加载与初始化，单列 |

纯计算时间排除下载、磁盘读取和网络；实际演示另报含上传/解码/叠图的 wall time。如果模型不能同时驻留而需重载，首次图片必须计入必要重载，并区分离线实验和在线演示。不能把离线缓存的响应说成完整三模型首次推理速度。

## 9. 结果与错误分析

至少形成三张带实际数量的结果表：各源各类分割 AP；L1/L2/L3 的 oracle/predicted 结果；首次与缓存延迟。未运行格子留空或 N/A，不填目标数字。

错误归为漏检、mask 边界、top/outerwear 混淆、同类选错、颜色/图案失配、空间词失配、中英不一致、标注歧义。每例展示 GT、全部候选、查询、选择与分数。用面积、来源、遮挡、同类数量支持原因，不只展示成功叠图，也不把所有失败归咎于文字模型。

在 [registry.csv](../experiments/registry.csv) 登记系列和具体运行，如 S1-FP-seed42、L3-oracle-seed43。系列计划行保留；新运行填写负责人、配置、哈希、状态与结果。空白表示未运行或不适用，不代表零。

## 10. 优先级与进度

先数据→分割→L1→L2，再 L3 和公平对照。主实验稳定后做可选 S2/L4。

| 时间 | 主要交付 |
| --- | --- |
| 10 月 4–5 日 | 确认技术范围、Proposal 和数据访问，分工待定 |
| 10 月 6–25 日 | 划分/转换、分割 smoke check、S1-FP、语言标注试点、L1 |
| 10 月 26–11 月 5 日 | MIX/NAIVE 与首轮 L2/L3，Milestone 真实初步结果 |
| 11 月 6 日 | 提交 Milestone |
| 11 月 7–22 日 | 语言 seeds、必要消融、错误分析 |
| 11 月 23–27 日 | 锁定方法、统一 final test 与计时 |
| 11 月 28–12 月 4 日 | 报告、引用、补充材料 |
| 12 月 5 日 | 提交 Final report |

日期是建议安排，格式与截止见 [课程要求](course_requirements.md)。GPU smoke check 后估算成本；预算不足先统一缩小全部分割对照的 updates 并记录，再删可选实验，不用不公平训练预算换取提升。
