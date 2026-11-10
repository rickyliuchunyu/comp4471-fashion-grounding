# 合成文件交接夹具

本目录全部示例为 `illustrative_only=true`。8×6 PPM 图由两个彩色矩形组成，使用虚构的 bag 类标签，仅检查数据结构，不表示真实服饰或标注成果。mask 文件本身是标准 RLE 对象，合成来源由引用它的包声明。

依赖顺序：

1. [图片 manifest](image_manifest.json) 引用 [合成图片](fixtures/two_regions.ppm)。
2. [GT 候选](candidates_gt.json)、[预测候选](candidates_predicted.json)、[零候选](candidates_empty.json) 引用图片 manifest 和两份 mask。
3. [查询](query_record.json) 指向 GT 左侧实例；[成功预测](prediction.json) 引用预测候选。
4. [无候选](prediction_no_candidates.json)、[推理失败](prediction_inference_error.json) 为独立 run 的失败示例；全部分数均非实验结果。

从仓库根目录运行，Python 3.10+，仅标准库：

```bash
python scripts/check_contract_examples.py
python scripts/test_contract_examples.py
```

检查器验证这套夹具的字段、引用/hash、非压缩 RLE、原图框、查询目标、候选顺序与失败状态；也检查规划登记表和共享规则配置。测试脚本在临时目录修改副本，验证合法夹具和八种错误输入，保留仓库原文件。它们不是通用数据转换器、训练程序、正式评价器或环境安装检查。完整语义见 [文件契约](../docs/artifact_contract.md)。
