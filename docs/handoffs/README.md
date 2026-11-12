# 交接记录入口

每项实现或验证交接新建 `<role>/<task-id>.md`；角色子目录由首次实际交接时创建。共享文档交接可以直接放本目录。不要把下面的模板当作已完成记录。

成员对应见 [简称绑定](team_aliases.md)。

当前记录：[wkq 远端接口与复核审计](data_eval/remote_pilot_review_20261112.md)、[接口与状态规范补齐](contract_readiness.md)、[分割角色服务器盘点](segmentation/server_inventory.md)。项目进度汇总见 [状态表](../status.md)，完整阶段条件见 [workflow](../workflow.md)。

## 复制使用的模板

```text
任务 ID / 当前 P 阶段：
角色与实际成员：
任务 / 分支 / PR / commit（本次提交可写“见本文件 Git 历史”）：
状态：planned / implemented / locally_checked / gpu_verified / evaluated
上游包、版本、hash 与接收检查：
改动文件、接口/词表版本：
环境名称与依赖记录：
实际执行命令（不可运行则明确写出）：
检查结果与日志/指标位置：
失败或尚未验证的内容：
下游接收角色、产物位置与下一步：
实际接收方复核结果（未复核就写未复核）：
状态表与运行登记更新：
```

运行人的实现状态与阶段状态分别记录；P0 接口文档 locally_checked 不等于三名成员已验收。评估结果必须关联 [运行元信息](../run_records.md)。不记录已做实验的叙事时间线，保留复现所需版本和证据即可。
