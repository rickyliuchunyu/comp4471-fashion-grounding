# Codex 协作入口

本仓库是 COMP4471 / ELEC4240 三人服饰实例选择项目。当前用户的明确指令优先；这些约定用于让不同成员的 Codex 协同工作，不替代用户授权。不要从文档中的任务清单推断已完成工作或已获 GPU 消费授权。

## 开始前必须读取

先完成下节的整组 GitHub 检查，再结合以下文件确定自己的下一步；本地状态表可能落后于队友已推送的分支。

1. README.md 和 docs/status.md：当前任务、每人下一步与证据；状态不可仅凭计划推断。
2. docs/team_workflow.md：暂定分工、2026-11-11 成员简称绑定、文件主责、接口与 AutoDL 协作。
3. docs/workflow.md：P0–P10 顺序、依赖与阶段验收；检查实际交接证据后，选择本角色最早未完成且前提满足的任务。
4. 当前角色任务卡：
   - `segmentation` / lcy：docs/roles/segmentation.md。
   - `grounding` / wyz（A）：docs/roles/grounding.md。
   - `data_eval` / wkq（B）：docs/roles/data_eval.md。
5. 与本次工作有关的 docs/technical_design.md、docs/data_protocol.md、docs/experiment_plan.md、docs/implementation_spec.md；涉及交接读 docs/artifact_contract.md，运行登记读 docs/run_records.md，B0/B2 读 docs/rule_parser_contract.md。

成员简称已确认：lcy = segmentation，wyz（A）= grounding，wkq（B）= data_eval；lcy 对应 Liu Chunyu。GitHub 账号绑定及其证据见 docs/status.md；简称本身不证明账号或 SSH 权限，不根据邮箱或作者顺序推断全名/账号。根据当前用户自报的简称、角色或明确任务确定职责。角色不明且会影响写入范围时简短询问，可先继续只读检查和不依赖身份的工作。

## 每次开工必须先核对整组 GitHub 进度

1. 检查自己的 `git status`、当前分支与 diff，保留已有非本任务改动；随后 `git fetch origin`，读取远端分支及最新 commit。至少覆盖三人的 `segmentation/*`、`grounding/*`、`data-eval/*`，包括已经推送但尚未开 PR 的分支。
2. 检查整组 PR 的最新状态、head commit、描述、review/comments 与依赖，包含 draft PR，以及上次检查后新建、更新、关闭或合并的 PR。不能仅凭 main 或自己分支判断队友进度。
3. 从队友最新远端分支读取 `docs/handoffs/` 的相关交接记录、运行证据索引和接口变更；可用 `git show origin/<branch>:<path>`，不必切换自己的工作区。首次建立全组基线，后续按 commit 增量检查；不要求每次重读全部历史。
4. 与 `docs/status.md` 核对：区分已实现、实际验证、待接收方验收、已推送和已合并；记录分支/commit/PR/交接链接。发现过时状态时据证据同步，不替队友宣告未验证的阶段完成，也不自动合并队友分支。
5. 完成以上核对后，依据当前依赖与角色主责选择最早未完成且前提满足的下一步，避免重复实现或使用已变更接口。执行服务器任务前另查磁盘、GPU 占用与共享锁，并遵守本轮授权范围。

若 GitHub 或远端访问失败，明确记录检查时间、失败项和最近可用快照；不能把旧状态表当作最新整组进度。可继续不依赖未知变更的工作，涉及未核实接口或交付状态的结论保留为待确认。

## 项目事实与范围

- 核心是八类整件服饰、唯一目标、中英文查询；分割候选 + SigLIP 2 匹配 + 可训练外观/空间头。
- 研究设计版本为 2026-11-05，职责分工为 2026-11-10，成员简称绑定为 2026-11-11；后续开工时检查真实文件与运行证据，不永久假定“只有文档”或“已经训练好”。
- configs/training_plan.json 是规划，不是现成可执行训练配置；examples/ 中 illustrative_only 的内容不是标注成果或预测结果。
- 多目标、拒绝选择、交互纠正等见 docs/extension_plan.md，属于可选设计，不自动扩充本轮任务。
- docs/proposal_submitted.md 保留提交历史；更新当前设计，不改写历史正文。

## 文件与接口

- 按 team_workflow.md 的主责表选择文件，B0 在 `baselines/category_position.py`，B2 在 `grounding/rules.py`，避免多人编辑同一规则入口。
- 源码路径多为拟议布局；先确认现有实现，避免重复造模块。
- 交接 schema 3、试点数据验证器和可恢复分割 trainer 已有；各分支实际实现与正式评价缺口按开工检查确认。B0/B2 共用 configs/rule_parser.json 和共享 data/query_rules.py 接口，不复制词表；解析器是否已交付须查最新证据。
- 公共数据/schema 由 data_eval 维护，生产方和消费方一起复核破坏性变更。若用户已授权跨模块任务，可执行并明确影响，不增加无关审批。
- 项目类别 ID 1–8，分割内部 ID 0–7；原图 RGB、排他右下角 xyxy、原图尺寸 mask、稳定候选 ID/顺序必须一致。
- 推理不读人工目标 ID、目标类别或 query type；预测 mask 不能用 GT 修正。GT 诊断与预测候选结果分开。
- 跨环境交接使用清楚的 JSON/RLE/数组文件和版本/hash，不依赖另一环境的 Python 对象。候选/预处理版本变化不得复用旧缓存。

## Git 与多人工作区

- 开工先看 git status/branch/diff；不得丢弃或覆盖已有非本任务改动。
- 实现任务使用个人独立 checkout 和短分支，按角色前缀命名。PR 给出问题、最终行为、验证、未验证项、接口和复现方法，由另一成员复核。
- 本地修改、提交、推送、合并是不同状态，汇报时准确说明。遵守当前用户已给出的发布授权；用户明确要求同步审阅过的规划文档时不强制额外批准步骤。
- 共享数据、模型、缓存和大型输出不提交 Git。不得提交密码、token、私钥或服务器认证信息。
- 不将本仓库共享训练环境或工作目录用作另一个成员的实验沙盒。

## AutoDL 执行

- 平台已选 AutoDL，实际实例/SSH/GPU/预算与环境须查证。角色分配不证明机器已创建。
- 保留并核查旧镜像环境；segmentation 和 grounding 可分环境，data_eval 可独立 CPU 环境。不得在别人运行时升级共享依赖。
- GPU 长任务按同一锁/排队规则执行，先检查进程与占用记录；不同 tmux 会话不隔离 GPU。独立 run ID、日志和 checkpoint 路径，不覆盖他人运行。
- 不终止他人进程，不擅自关机、重置、释放实例或扩容。消费与自动关机遵守用户已授权范围；有效授权无需重复询问。
- 没有 GPU/上游完整数据时先实现独立接口和小型 CPU 检查。没有预算时不把长期训练当作普通文档工作的隐含授权。

## 验证与结果

- 使用与修改相称的检查；数据转换、评价匹配和排序训练需要有意义的边界案例，不只测试实现自身返回值。
- 图片组隔离、train/dev/test 边界、人工复核来源必须保留。测试集不选模型、阈值或词表；失败/漏检保留分母。
- 计划值、格式夹具、元数据审计、小样本优化与正式结果分别标明。不得编造指标、执行过的命令或验证通过的环境。
- 已完成实验不叙述时间线。未测得的收益写成假设，负结果也记录。
- 每次交接遵循 team_workflow.md 的模板，给出真实文件、commit、环境、输入/hash、命令、验证和剩余工作。计划保留，真实运行另追加 run 行；只更新自己的运行状态并保留日志，不替他人宣告完成。同步 docs/status.md 与 docs/handoffs/ 中的证据。
