# 第3000步候选交付准备

日期：2026-11-14。segmentation / lcy，Codex执行。状态：**prepared_not_exported**。本轮授权仅使用已有预测离线分析和准备交付；没有模型推理、smoke、训练或正式候选导出。

机器可读清单：[candidate_u3000_preparation_20261114.json](candidate_u3000_preparation_20261114.json)。准备配置：[candidate_u3000_preparation.json](../../../configs/runs/segmentation/candidate_u3000_preparation.json)。服务器共享准备目录：`/root/autodl-tmp/fashion-grounding/shared/candidates/preparation-u3000-20261114/`，只含准备说明、清单和配置，不含 `candidates_predicted.json`。

## 已固定的交付输入

- 权重：`/root/autodl-tmp/fashion-grounding/runs/segmentation/segmentation-resume5000-train-20261113-r01/checkpoint_003000.pth`。
- SHA256：`e56eb7f46fab899d83f668b1878776db50e13cc273c4d4a7fbdc786692444148`；采用已有完整dev选择证据，不重新加载、校验张量或复制权重。
- 模型运行源码 `b61d78f2d279645caade7adf5c7390c617bc6770`，沿用服务器 `envs/segmentation-v1`。配置中的源码hash固定现有 `predict_instances`、`decode_candidates` 和 `export_package` 实现。
- 输入保持 `fp_train5000_v1` / `fp_dev250_v1`；manifest hash见机器清单。只有dev存在本轮原始AP预测，train没有这份权重的预测缓存。test不在交付准备范围内。
- 候选解码：RGB、短边800/最长1333、FP32；每query取一个前景类、原图mask logits>0、去空mask、分数为类别概率乘mask质量、最多100、无NMS；稳定排序与真实decoder query ID按schema 3。
- `score_threshold=null`、`selected_threshold=null`。0.5仍只是未来明确诊断的起始值，没有冒称经过dev选择；正式0.3/0.5/0.7选择需要冻结的accepted语言dev。

## 现有AP预测为什么不能直接改名交付

已有 `update_003000/predictions.json` 是官方query×class top100结果，只保存image ID、类别、合成分数和二值mask；没有decoder query ID、完整类别概率或mask logits。top100已经丢弃部分query/class组合。

不能把输出序号伪造为query ID，也不能凭相同二值mask推断同一query；不同query可能有相同mask，同一query也可能出现多个类别。合成分数不能恢复完整logits。因此本次保留原始AP产物供诊断，不伪造“一query一候选”的生产包，不计算或填写正式语言matched coverage。

## wyz / wkq 接收约定

已复核wyz `103dcc5` 的 `src/fashion_grounding/grounding/inputs.py`：`CandidatePack(path).load(image_id)` 和 `crop_regions(image, candidates)`；接口未变，沿用此前消费验收，不重复模型检查。

正式包仍应包括图片manifest、`candidates_predicted.json`、原图RLE mask及hash、图片字节或明确共享依赖、解码配置与运行metadata。`candidate_source=predicted`、`illustrative_only=false`、`source_annotation_id=null`，候选ID为 `image-{image_id}-query-{decoder_query_id:03d}`，按score降序及ID字典序排列。GT匹配/诊断叠图另存，不能进入候选payload。零候选与推理错误分开保留。

准备清单不是CandidatePack输入；下游暂不据此创建特征缓存。未来真实包产生后，A核对新包裁剪、顺序与缓存键，B核对几何及评价引用；只补受新payload影响的CPU消费检查，不重做已通过的环境/COCO/训练恢复smoke。此前小包接口通过不等于3000步生产包已获A/B验收。

## 尚缺的具体工作

1. 当前已有可调用解码/导出函数，但 `evaluate --export-diagnostic-candidates` 仅支持试点诊断，不是正式候选CLI；正式编排入口尚未实现。本轮没有通过修改标志绕过这个限制。
2. 未来获得相应推理授权后，对所需train/dev图各做一次3000步前向，保存带真实query ID、阈值前分数/二值mask的候选缓存；三档过滤可复用缓存，不为每个阈值重复推理。训练、test和阈值选择仍各自遵守授权及输入条件。
3. wkq交付accepted语言dev及固定query hash后，按既定matched query coverage规则选一次阈值，锁定最终包版本，再由A/B接收。

当前可交付的是固定权重引用、配置、接口说明及缺口清单。**正式候选payload、阈值选择、下游新包验收均未完成。**
