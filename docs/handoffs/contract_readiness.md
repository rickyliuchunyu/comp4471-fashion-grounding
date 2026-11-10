# 共享接口与状态规范补齐

- 当前阶段：P0；共享文档任务，未启动模型训练。
- 执行：由 Liu Chunyu 请求 Codex 完成；不是 A/B 的完成声明。
- 代码版本：见本文件所在提交及 Git 历史。
- 状态：locally_checked；仅文档、规划配置和合成文件契约。
- 产物：[文件契约](../artifact_contract.md)、[示例](../../examples/README.md)、[共享解析](../rule_parser_contract.md)、[运行记录](../run_records.md)、[状态表](../status.md)。
- 接口变更：交接文件 schema 3；预测改为 query ID + 候选 ID 引用；规划配置/类别映射使用各自版本。
- 本地验证环境：Python 3.13.12，仅标准库；AutoDL 的 Python/依赖组合尚未验证。
- 实际检查：从仓库根目录运行 `python scripts/check_contract_examples.py` 与 `python scripts/test_contract_examples.py`；合法包通过，八种故障副本被拒绝（查询 ID 缺失、候选引用错误、框与 mask 不符、hash 错误、候选乱序、预测泄漏 GT ID、失败带选中项、RLE 长度错误）。Markdown 本地链接/锚点、JSON/CSV、`git diff --check` 另行检查通过。
- 未验证：正式数据转换、压缩 RLE、真实图片、共享 parser、模型前后向、GPU 环境、训练与评价均未验证；没有生成模型指标。
- 下游：B 实现正式读写器/共享 parser，A/Liu Chunyu 复核消费接口。三方实际接收验收仍未完成。
- 状态表：P0 in_progress；所有模型实验保持 planned/optional_planned，无运行指标。
