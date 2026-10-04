# 模型结构与训练方法

本文定义课程项目拟实现的技术方案。官方模型的结构和类别事实已经查证；区域文本匹配、缺失类别监督、超参数和消融是本团队的设计，尚未实现或验证。起始配置见 [training_plan.json](../configs/training_plan.json)，接口见 [implementation_spec.md](implementation_spec.md)。

## 1. 任务的数学定义

给定 RGB 图片 $I\in\mathbb{R}^{H\times W\times3}$ 和一条完整描述 $q$，分割器产生候选集合：

$$
\mathcal C(I)=\{(c_i,b_i,M_i,p_i)\}_{i=1}^{N}.
$$

$c_i$ 是八类之一，$b_i$ 是原图坐标框，$M_i\in\{0,1\}^{H\times W}$ 是实例掩膜，$p_i$ 是分割置信度。语言模块计算 $s_i=f(I,M_i,b_i,c_i,q)$，选出 $i^*=\arg\max_i s_i$，返回这个候选的类别、框和掩膜。

本方案属于 **candidate based referring instance selection**：语言模块对已有实例排序，掩膜来自分割器。它没有根据文字重新生成掩膜，也不能恢复分割器漏掉的实例。报告应使用这个准确的任务定义，避免把实例选择写成任意开放词汇分割或服饰部件理解。

核心查询明确对应一个整件实例；无目标、多目标和复杂人物关系暂不纳入主要测试。左右以图片坐标为准。训练标签是目标 annotation ID；推理时只接收图片、查询和预测候选，不能读取这个 ID。

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

正式微调所有分割参数，骨干学习率是其余层的 0.1 倍。冻结 DINOv2/BGE-M3 的决定只适用于后面的语言阶段，不代表 Mask2Former 也冻结。

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

### 2.3 混合数据中的缺失类别监督

DeepFashion2 只覆盖本项目的前五类，没有鞋、包和配饰标注。若直接用普通八类损失，图片中出现而未标注的包可能被当成 no object，这会引入错误的负监督。源类别覆盖见 [数据协议](data_protocol.md)。

安全基线 S1-FP 使用 Fashionpedia 八类。混合训练拟增加 **source aware partial label classification**，这是团队的待验证扩展，不是官方 Mask2Former 的现成功能：

- 匹配到真实实例的 query：使用其真实类别交叉熵。
- Fashionpedia 未匹配 query：使用普通 no object 交叉熵。
- DeepFashion2 未匹配 query：允许标签集合 $U=\{\text{shoes},\text{bag},\text{accessory},\varnothing\}$，惩罚落入集合外的概率。

$$
L_j^{DF2}=-\log\sum_{c\in U}p_j(c)
=\operatorname{logsumexp}(\ell_j)-\operatorname{logsumexp}(\ell_{j,U}).
$$

这里 $\ell_j$ 表示分类 logits。实现使用 logsumexp 保证数值稳定；内部 $U=[5,6,7,8]$。用 $w_j=1$ 对匹配 query 加权、$w_j=0.1$ 对未匹配 query 加权，分类损失为 $\sum_jw_jL_j/\sum_jw_j$，跨 batch 统一归一化。相同规则应用于每个 auxiliary prediction 的匹配结果；mask 损失和 matcher 不变。

该扩展只减少错误的类别负监督，**不会提供缺失物体的正 mask 监督**。它也可能增加鞋、包和配饰的假阳性，需要比较 S4 的普通混合损失与集合损失，并单独报告这些类的 AP 和假阳性。对数据中其他漏标不能自动宣称已解决。

## 3. 从掩膜提取视觉特征

拟冻结 `dinov2_vits14`，使用 patch tokens，而非把全图 CLS token 复制给每个实例。ViT S 的 token 维度为 384，hub 模型的 patch size 为 14。[官方 backbone](https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/backbones.py)、[ViT 实现](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py)

拟先保持长宽比，将长边缩放到 518，再只在右侧和底部补齐到 14 的整数倍。缩放、padding 和 mask 使用同一个坐标变换；图片按 RGB、0–1 输入并使用 ImageNet mean/std 归一化。padding 像素不参与区域池化。DINOv2 预处理与分割器的预处理独立，不能把 Mask2Former 的 0–255 输入原样传入。

设 patch 特征为 $F_p\in\mathbb{R}^{384}$，将变换后的实例 mask 通过 area pooling 缩到 patch 网格，得到每个 patch 内的前景比例 $\alpha_{ip}\in[0,1]$：

$$
v_i=\frac{\sum_p\alpha_{ip}F_p}{\sum_p\alpha_{ip}+10^{-6}}\in\mathbb{R}^{384}.
$$

这保留了非矩形掩膜，避免框里的背景占据主要权重。它仍有局限：小配饰可能只覆盖少量 patch，token 也含有上下文，不能当成纯物体颜色读数。权重和为零时，拟用原图框向外扩展 20% 宽高的裁剪编码 CLS token 作为 fallback；记录发生次数，不返回零向量冒充有效特征。

可选 L4 将所有实例统一改用上述扩框裁剪 CLS 特征，与 mask pooling 比较；其他设置相同。两者都是 384 维，但缓存成本与上下文不同，需要分别报告效果及时间。

## 4. 文本编码与跨模态投影

拟冻结 BGE-M3，仅使用 1024 维 dense embedding，完整保留类别、颜色、位置等文字。BGE-M3 官方提供多语言文本表示；它并非已经与 DINOv2 对齐的图像文本模型。[官方模型说明](https://huggingface.co/BAAI/bge-m3)

本项目短查询起始最大 token 长度为 128，这是团队节省计算的选择。不得静默截断：超限查询单列复核；中英文都使用同一文本编码器。分词器、模型 revision 和预处理版本进入缓存键。

拟训练两个投影头，使用 GELU 和 dropout 0.1：

```text
P_visual: Linear(384, 256) → GELU → Dropout → Linear(256, 256)
P_text:   Linear(1024, 512) → GELU → Dropout → Linear(512, 256)
z_i = L2_normalize(P_visual(v_i))
u   = L2_normalize(P_text(t))
```

因此模型的可学习部分是投影头与下文的空间排序头。两个大编码器在 `eval()` 与 `no_grad()` 下离线编码，梯度不会流入它们；训练中的 dropout 仅作用于小网络。缓存原始 384/1024 维向量，不能缓存仍在更新的投影输出。

按上述 Linear 层均带 bias、温度固定计算，视觉投影有 164,352 个参数，文本投影有 656,128 个参数，L2 总计 **820,480**。L3 再增加 46,209 个空间参数，总计 **866,689**。这些是结构推导值，后续须与 `requires_grad` 参数统计一致；不包括冻结编码器和单独微调的分割器。

使用分离编码器是为了冻结特征、缓存和双语文字实验，并不意味着比预训练图文联合模型更强。BGE-M3 的文本检索预训练也不保证理解服饰颜色；约 900 张查询图片能否足够训练对齐是本项目的重要不确定性。若 L2 只能区分类别，应如实报告并分析同类负例与数据规模，不能把 L1 的位置规则成功解释成跨模态学习成功。

## 5. 三种语言方法

### 5.1 L1 类别与方位规则

词表映射八类的中英文类别词。单纯类别描述只在该类别唯一时选中；存在多个同类时返回 `ambiguous_query`。识别左、右、上、下及其英文对应词后，在该类候选中比较中心位置，选择相应极值。若含不支持的颜色、图案修饰，返回 `unsupported_query`，评估按失败计，不能删掉修饰后猜一个。

规则中的多类别冲突、相同位置和未知类别都保留状态。L1 不读取人工 query type 或目标标签，这些字段只用于评估分组。

### 5.2 L2 只使用区域与文本表示

$$
s_i=\frac{z_i^\top u}{\tau},\qquad
P(i\mid I,q)=\frac{\exp(s_i)}{\sum_{j=1}^{N}\exp(s_j)},\qquad
L_{rank}=-\log P(k\mid I,q).
$$

温度 $\tau$ 起始固定为 0.07。开发集只允许在预先登记的 0.03、0.07、0.1 中选择，最终测试不参与调参。L2 在完整候选集合上排序，不做类别词硬过滤；类别区分也必须由学习实现。

训练负例首先是**同图其他实例**。只有一个候选时，该交叉熵恒为零，无法训练排序，因此这类记录不进入 ranking loss，仍保留在评估中。拟使至少一半训练 minibatch 的查询来自同类多实例图片；若实际标注不足，降低采样目标并记录实际比例。仅增加跨图异类负例不能替代同类负例。

### 5.3 L3 加入文本条件的空间排序

为每个实例计算 12 维几何向量：

```text
g_i = [x1/W, y1/H, x2/W, y2/H,
       cx/W, cy/H, width/W, height/H,
       mask_area/(W*H), log((width+eps)/(height+eps)),
       rank_x, rank_y]
```

`rank_x/rank_y` 在**同预测类别**内按中心排序后归一到 0–1，单实例取 0.5；位置相同按 candidate ID 稳定排序。训练真实候选时用真实类别，预测候选时用预测类别。`eps=1e-6`；所有框必须是正面积且在原图范围内。

$$
e_i=\operatorname{MLP}_{12\to64\to64}(g_i),\quad
r_i=\operatorname{MLP}_{320\to128\to1}([e_i,u]),\quad
s_i=\frac{z_i^\top u}{\tau}+r_i.
$$

两个空间 MLP 的隐藏层使用 GELU。这里使用有非线性的文本条件网络；单独给所有候选加相同的文本线性项会在 softmax 中抵消，不能学习“左边”与“右边”的区别。L3 与 L2 使用相同文字、候选和视觉特征，只新增这一项。

空间特征可能使模型靠位置或类别捷径获得较高总体准确率。必须分别报告外观组、方位组和同类多实例组，并检查反事实查询：对同一图片将“左边的包”改为“右边的包”，目标应随文字改变。它是诊断，不能预先写成已具备的能力。

## 6. 两阶段训练与候选分布

阶段 A 在人工真实实例 mask 上训练排序头，固定分割器与两个编码器，不进行端到端联合训练。这样先测量是否能学会文字与区域的关系，并且缓存易复用。

阶段 B 是可选的预测候选适配：只用**训练划分**的分割预测建立训练输入；按评估中的同类、IoU 至少 0.5 一对一规则关联目标。目标没有匹配候选时不计算这条 ranking loss，但必须报告排除比例。最终测试保留全部查询，缺失目标算失败。阶段 B 必须对 L2 和 L3 同时执行或同时不执行，避免比较不同训练条件。

分割器训练图上的预测可能比新图更好，阶段 B 仍可能有分布差异；预算允许时可做训练集内部交叉预测。不使用开发或测试标注适配参数。阶段 A、阶段 B 和 oracle 测试是不同概念：oracle 测试仅诊断选择能力，最终效果以预测候选结果为准。

## 7. 起始超参数与计算预算

下面是尚未做显存与收敛检查的项目起始值，不是推荐值已经验证。修改必须保存新配置，不能只改终端参数。

| 项目 | 分割微调 | 缓存特征后的排序头 |
| --- | --- | --- |
| Optimizer | AdamW | AdamW |
| 学习率 | 其他层 1e-4，骨干 1e-5 | 3e-4 |
| Weight decay | 0.05 | 0.01 |
| Batch size | 2 张图片，单 GPU | 32 条查询，候选 padding |
| 预算 | 20,000 optimizer updates | 最多 30 epochs |
| LR schedule | 500 updates warmup；12k/16k 乘 0.1 | 开发集早停，patience 5 |
| 输入 | LSJ 输出 768，scale 0.1–2.0 | DINO 长边 518；文本最大 128 tokens |
| 精度 | AMP；损失稳定部分用 float32 | 投影与排序头 float32 |
| 随机种子 | 起始 42 | 42、43、44 |
| 选择 checkpoint | 固定开发集 mask AP | 固定开发集 oracle 选择准确率 |

分割 warmup 拟从基础 LR 的 0.001 线性升至基础 LR；其余周期是分段常数。拟沿用 full model gradient norm clip 0.01。768/batch 2/20k 的设置是对 [官方训练基线](https://github.com/facebookresearch/Mask2Former/blob/main/configs/coco/instance-segmentation/Base-COCO-InstanceSegmentation.yaml) 的项目调整，**不是复现其 1024/batch 16/50 epoch 的完整训练配方**。报告须说明有限训练预算可能限制效果。

若 768/batch 2 显存不足，先 batch 1、累计两个 microbatches，再验证有效 batch 2 的损失归一化；仍不足再使用 640，并统一对照条件。warmup、scheduler 和预算以 optimizer updates 计，不随累积步数重复计数。先测 200 个 updates 的速度和峰值显存，再估算各实验成本；不预先承诺 GPU 小时或性能数字。

## 8. 从分割输出到候选

COCO AP 评估保留官方 instance inference；它从 query 与类别组合中取 top k，分数结合类别概率与前景区域的 mask 质量。官方 panoptic 阈值不能直接当作 instance filter 使用。[官方 inference](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py)

语言候选拟单独使用以下固定规则：每个 query 在八个前景类别中取一个最高概率类别，mask logits 大于零为前景；按同样定义计算分割分数，去掉空 mask 与分数小于 0.5 的候选，按分数排序最多保留 100 个。每个 query 至多一条候选，不额外做 mask NMS，以免丢掉相邻同类实例。保留原 query ID 以便追踪重复预测。

这个阈值只在开发集允许以 0.3、0.5、0.7 比较一次，选择后固定给所有语言方法。没有候选时返回 `no_candidates`；max rank score 不是已校准正确概率，不据此声称可靠拒绝。AP 评估输出与语言候选是两套明确记录的解码规则，不能把过滤后候选的结果冒充官方 COCO AP。

## 9. 项目贡献应如何表述

拟完成的贡献是八类跨源标注统一与监督处理、可复核的双语实例描述数据、冻结表示上的区域文本排序与空间消融，以及分割误差和选择误差分开的评估。它们是否带来提升由实验回答。

公开 Mask2Former、DINOv2、BGE-M3 均不是团队原创。把三个模型串联、或只演示“左边的包”，不能单独作为充分实验结论。最终报告应保留普通混合训练失败、外观描述学习不足等负结果，并用 [实验计划](experiment_plan.md) 中的公平对照支持结论。
