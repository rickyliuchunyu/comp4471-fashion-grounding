# 模型结构与训练方法

版本 `2026-11-05`。以下是待实现设计与待验证起始值，不是运行配置或测得结果。规划参数见 [training_plan.json](../configs/training_plan.json)，调整理由见 [方案记录](revision_20261105.md)。

## 1. 任务与候选

给定图片 I 与完整查询 q，分割器产生 C(I)={(c_i,b_i,M_i,p_i)}。语言模块为候选排序并返回一个实例的原有 mask 和框。这是 candidate-based referring instance selection，不是任意开放词汇掩膜生成；无法恢复漏掉的目标。核心只处理明确指向一个整件实例的描述。

B0/B1/B2/M1/M2 固定使用相同候选。GT 候选只在训练与诊断使用，实际推理不能读取真实目标、query type 或标注类别。

## 2. 分割器的结构

拟以公开 COCO instance segmentation 预训练的 Mask2Former R50 初始化，保留骨干、像素解码器和 Transformer，重建八类分类头。下载时记录确切 checkpoint URL、文件 SHA256 和基础代码 commit；当前没有下载或训练权重。

| 部分 | 拟采用的设置 | 在系统中的作用 |
| --- | --- | --- |
| ResNet 50 | res2 至 res5 多尺度特征 | 提取图像特征，微调 |
| MSDeformAttnPixelDecoder | 输出 stride 4 mask features，通道 256 | 融合多尺度像素信息 |
| Masked Transformer decoder | 100 个 queries，hidden 256，8 个 attention heads | 每个 query 表示一个潜在实例 |
| 分类头 | 每个 query 输出 9 个 logits | 8 类与 no object |
| mask embedding | 每个 query 输出 256 维向量 | 与像素特征点积产生 mask logits |

以上结构参数依据 [官方 R50 配置](https://github.com/facebookresearch/Mask2Former/blob/main/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml)。其中 `DEC_LAYERS=10` 包含初始 query 预测及 9 层 decoder 后的预测，不能解释成 10 层 attention。它是公开架构参数，不是本项目已经完成的训练设置。

设像素特征为 $F\in\mathbb{R}^{256\times H_m\times W_m}$，query 的 mask embedding 为 $a_j\in\mathbb{R}^{256}$，则：

$$
\ell_j(x,y)=a_j^\top F(:,x,y),\qquad \widehat M_j(x,y)=\sigma(\ell_j(x,y)).
$$

decoder 将上一轮 mask 缩放到当前特征尺度，用其阈值区域限制 cross attention；这就是 masked attention。它描述的是分割内部的注意力机制，与本项目另行训练的语言选择模块不同。[官方 decoder 实现](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py)

### 2.1 八类分类头与预训练加载

项目导出类别 ID 为 1–8，框架内部为 0–7，no object 为 8。COCO checkpoint 的分类头形状与九输出头不同，加载时仅允许跳过这个不匹配的分类头，并打印 missing/unexpected keys；其余权重必须逐项确认加载成功。新分类头拟使用 Xavier uniform 权重与零 bias，不手工把 COCO 的 shirt 等词映射成项目类。

正式微调所有分割参数，骨干学习率是其余层的 0.1 倍。冻结 SigLIP 2 双编码器的决定只适用于后面的语言阶段，不代表 Mask2Former 也冻结。

### 2.2 实例匹配与损失

100 个 query 与图片内 $G$ 个真实实例通过 Hungarian matching 建立一对一匹配。拟保留官方的分类、mask BCE 和 Dice 匹配代价：

$$
C_{jg}=2[-p_j(c_g)]+5L_{\mathrm{BCE}}(\ell_j,M_g)+5L_{\mathrm{Dice}}(\ell_j,M_g).
$$

分类匹配代价是负类别概率，不能和训练交叉熵混为一式。对匹配到的实例计算 mask 损失，未匹配 query 没有 mask 正例；分类损失处理匹配及未匹配 query。Dice 的概念形式为：

$$
L_{\mathrm{Dice}}=1-\frac{2\sum_x\widehat M(x)M(x)+\epsilon}{\sum_x\widehat M(x)+\sum_x M(x)+\epsilon}.
$$

正式实现保留官方点采样和 smoothing 细节，而不是另外写一个不等价的全图版本。官方使用 sigmoid BCE 与 Dice；本方案不把它写成 focal loss。[官方 matcher](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py)、[官方 criterion](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py)

每轮预测的损失拟设为 $2L_{cls}+5L_{BCE}+5L_{Dice}$，并保留 auxiliary predictions 的监督。采样点数 12,544、oversample ratio 3、importance ratio 0.75 和 no object 权重 0.1 沿用官方配置。框损失不单独增加：输出框由掩膜计算，Mask2Former 并非本方案中的框回归检测器。


## 3. 已对齐图文编码器与区域输入

主模型为 [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224)。冻结两侧编码器并置于 eval/no_grad 模式。读取标准图像/文本特征接口，分别 L2 normalize，确认特征形状 [N,D] 与 [B,D] 相同且非零。D 从实际 checkpoint 配置和前向输出验证，不沿用原方案的 384/1024 维。

基线区域输入使用原图候选框的紧致 RGB 裁剪，再由同 revision 的官方 processor 预处理；不额外扩框或使用 GT mask 修正预测裁剪。记录整数裁剪、边界裁切和小目标策略。可选 X3 使用同一裁剪，将候选 mask 外背景替换为固定中性背景；不得把 patch token 任意平均后默认当成已对齐的图像向量。

文本保留完整原文，使用对应 processor/tokenizer 的固定长度和 padding 配方；读取 checkpoint 支持的上下文长度，超限时明确报错与复核，不静默截断。中文直接编码；翻译推理另立实验，不混入原生双语结果。

定义 s0_i = exp(a) * dot(v_i,t) + b，其中 a/b 为 checkpoint 的 logit_scale/logit_bias；先核对与标准模型 logits 的数值一致性，再缓存或单独提取向量。a/b 核心实验冻结，s0 的排序不变性不能代替数值检查。Sigmoid 分数不是校准后的目标正确概率。

## 4. 基线与可训练排序模块

### B0：类别和方位规则

B0/B2 使用 [共享解析契约](rule_parser_contract.md) 和 [唯一词表](../configs/rule_parser.json)，由 B 维护、A 复核；目前为待实现起始词表，未 dev 冻结。明确列举中英文类别/细类别名和方位词。单纯类别且唯一时选择该实例；左右上下按同候选类别的框中心极值选择（GT 条件用 GT 类别，预测条件用预测类别）。多类别冲突、位置并列、未知词或不支持的外观修饰返回明确失败状态，计入总分母。词表在 dev 阶段冻结，推理不读取人工 query type。

### B1：零样本区域文本匹配

使用完整查询，返回 argmax s0_i，不训练，不作类别硬过滤。候选为空时失败。

### B2：零样本加简单方位规则

保持 B1 的完整文本和候选。仅当共享解析器确定一个类别和一个方位且没有否定/集合等不支持操作时，对该候选类别加入位置分数（GT/预测条件分别取其包内类别）；其他类别与无法解析的查询加零。右/下为 2*rank-1，左/上取其相反数，单实例为零。不删除颜色等修饰词，也不因规则无法解析而丢弃 B1 分数。

s_i = s0_i + lambda * r_i，lambda 仅在 GT dev 的 [0,1,2,5,10] 中按 oracle selection accuracy 选择一次，同分取最小 lambda；选定后冻结并用于 GT/预测两种条件。此方法只有规则权重选择，没有训练排序网络；必须报告规则覆盖率。

### M1：轻量外观残差排序

输入 [v_i,t,v_i*t]，维度为 3D，经 Linear(3D,256)、GELU、Dropout(0.1)、Linear(256,1)，得到 delta_i；最后一层权重和 bias 初始化为零，初始输出保留 s0。M1 分数 s_i = s0_i + delta_i。

这是一种明确的轻量 residual ranker 起点，不重新随机投影两个编码器以丢掉已有对齐。若改变结构，需更新配置并对所有相关对照采用相同定义；参数量由实际 D 与 requires_grad 统计，不虚构固定值。

### M2：加入文本条件空间项

几何向量为 12 维：

```text
g_i = [x1/W, y1/H, x2/W, y2/H,
       cx/W, cy/H, width/W, height/H,
       mask_area/(W*H), log((width+eps)/(height+eps)),
       rank_x, rank_y]
```

rank 在同候选类别内计算为 0–1 次序；单实例取 0.5，坐标并列使用平均名次，不能用 ID 构造虚假的左右差异。GT 条件用 GT 类别，预测条件用预测类别，eps=1e-6。

空间网络为 g:12→64→64，隐藏层 GELU；与完整文本向量 t 拼接，使用 (D+64)→128→1 的非线性网络得到 h_i。末层同样零初始化。M2 分数为 s0_i + delta_i + h_i。M1/M2 数据、外观结构和训练预算相同，M2 仅新增空间分支。

### 排序损失与训练候选

同图候选构成 softmax，目标 k 的损失为 L=-log(exp(s_k)/sum_j exp(s_j))。padding logits 为 -inf，不包含在分母中。N=1 时损失恒零，因此不参加排序训练但保留评估；N=0 单独处理，不能对全 -inf 行求 softmax。

强调同图同类竞争样本，目标约一半采样查询来自同类多实例，实际不足时记录比例，不通过重复简单鞋子样本制造困难覆盖。双语及改写均跟随图片划分。水平翻转若用于语言训练，必须同步变换 mask/框/空间文字和目标关系；核心缓存训练默认不使用几何增强。

阶段 A 用 GT 候选训练。可选阶段 B 只用训练划分预测候选适配 M1/M2，按 [实验计划](experiment_plan.md) 的关联规则找目标，漏检样本不计算该训练 loss，并报告排除率；最终测试仍保留漏检查询。两种方法必须同时执行或同时不执行阶段 B。

## 5. 起始训练参数

| 项目 | 分割 S0 | 排序 M1/M2 |
| --- | --- | --- |
| 数据 | Fashionpedia 八类，约 5k–10k 图片 | 约 800–1,200 训练图片的人工复核查询 |
| 参数 | 微调分割模型 | 冻结 SigLIP 2，仅训练小网络 |
| Optimizer | AdamW，LR 1e-4，backbone 0.1 倍 | AdamW，LR 3e-4 |
| Weight decay | 0.05 | 0.01 |
| Batch | 有效 2 图片 | 32 查询，变长候选 padding |
| 初始预算上限 | 20k optimizer updates | 30 epochs，patience 5 |
| 选择 checkpoint | dev mask AP | oracle dev selection accuracy |
| 种子 | 起始 42 | 42、43、44 |
| 输入 | LSJ 768，scale 0.1–2.0，AMP | 官方 224 processor 与对应文本配方 |

分割 warmup 500 updates，从 LR 0.001 倍开始；12k/16k updates 乘 0.1，full-model gradient norm clip 0.01。以上是项目起始预算，非完整官方配方的复现。先测 200 updates 的稳定性、显存与速度再定正式预算。显存不足先 microbatch 1 加累计 2，再考虑尺寸 640；正式对照统一预算和条件。所有步数以 optimizer updates 计。

编码器与分割器可分进程缓存，无需同时常驻 GPU。checkpoint revision、processor、具体依赖和显存仍待验证。

## 6. 外部系统与可选扩展

E1 优先使用 [SAM 3](https://github.com/facebookresearch/sam3)，需检查权重访问与运行环境。否则在正式测试前登记替代为 [Grounding DINO + SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2)。完整描述输入，默认取置信度最高的非空实例，固定 tie-break；不使用 GT 框、点或最佳 IoU 挑选输出。原生中文与固定翻译流程如都测试，分别命名和计时。概念的全部实例分割不自动等于唯一实例指代能力。

扩展的实施范围与优先级见 [扩展计划](extension_plan.md)：优先细化 X3 区域输入对照，并复用既有语言诊断；交互纠正、拒绝与多目标仅为独立可选任务，不修改本文的单目标输出和排序损失。

X1 保留 DINOv2 ViT-S/14 + BGE-M3 投影对照，可参考历史提交的实现设计；将其应用到当前相同数据、候选和预算后才构成有效对照。X2 数据效率，X3 mask 背景抑制，X4 英文 Marqo-FashionSigLIP，X5 DeepFashion2 共有五类分割实验，详见实验计划。

X5 的 FP-only 与 FP+DF2 都使用相同五类头、匹配训练预算、初始化和评估类别。两源都只把共有五类定义为前景，其他类别不进入前景标注和 AP；这是五类任务的范围选择，不是保留八类头却缺少三类正监督。这套实验不能直接冒充八类 S0。当前不实现八类 source-aware partial-label loss。

## 7. 从分割输出到候选

COCO AP 评估保留官方 instance inference；它从 query 与类别组合中取 top k，分数结合类别概率与前景区域的 mask 质量。官方 panoptic 阈值不能直接当作 instance filter 使用。[官方 inference](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py)

语言候选拟单独使用以下固定规则：每个 query 在八个前景类别中取一个最高概率类别，mask logits 大于零为前景；按同样定义计算分割分数，去掉空 mask 与分数小于 0.5 的候选，按分数排序最多保留 100 个。每个 query 至多一条候选，不额外做 mask NMS，以免丢掉相邻同类实例。保留原 query ID 以便追踪重复预测。

这个阈值只在开发集允许以 0.3、0.5、0.7 比较一次，按 [实验选择规则](experiment_plan.md#2-固定条件与选择规则) 最大化 matched candidate coverage，同分优先更少候选，再取更高阈值；选择后固定给所有语言方法。没有候选时返回 `no_candidates`；max rank score 不是已校准正确概率，不据此声称可靠拒绝。AP 评估输出与语言候选是两套明确记录的解码规则，不能把过滤后候选的结果冒充官方 COCO AP。


## 8. 来源与主张

[SigLIP 2](https://arxiv.org/abs/2502.14786) 已有图文与多语言预训练，适合成为起点，但本项目上的中文、服饰外观与位置效果尚未验证。公开模型不是团队原创。报告贡献是复核数据、轻量适配及公平消融、误差分解和可复现证据；不预设 M2 必须超过 B2 或 E1。
