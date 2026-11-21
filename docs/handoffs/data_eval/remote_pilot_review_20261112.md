# wkq 远端数据接口与复核审计交接

日期：2026-11-12。角色：data_eval / wkq；GitHub 授权账号由用户确认为 LE-brawl。
任务分支：data-eval/remote-pilot-review。状态：remote_cpu_checked；没有执行 GPU 模型或训练。

## 工作区和环境

工作区：/root/autodl-tmp/fashion-grounding/workspaces/data_eval
独立环境：/root/autodl-tmp/fashion-grounding/envs/data_eval
共享接口包：/root/autodl-tmp/fashion-grounding/shared/processed/fp_pilot20_v2
运行证据：/root/autodl-tmp/fashion-grounding/runs/data_eval/20261112-pilot-review

工作区原先不存在，从 main 7064ebb933584f2922607e51ce392e1572fd7dbb 独立克隆后创建任务分支。
Python 3.10.8；numpy 2.2.6、Pillow 12.0.0、pycocotools 2.0.11、requests 2.32.5。
pip check 通过；完整依赖在运行目录 environment.txt。环境、pip 缓存与数据均在数据盘。
没有修改其他成员环境、访问其工作区、清理旧输出或停止进程。本轮 CPU 操作未持有或使用 GPU。

## 产物和真实验证

迁入此前已制作的训练接口包和数据代码，在远端修改 Python 3.10 依赖组合并验证。
新增 data/review.py：校验双语组、manifest、目标引用和人工导出记录；区分分歧、未复核、已展示答案和带备注的选择。
审计不修改目标答案、不自动生成 accepted，不将数字一致当作研究质量合格。

- 20 张官方训练图片、145 个整件 GT 实例，八类齐全。
- 原标注 249 条，排除部件 104 条。全包图像、RLE、框、hash 与引用检查通过。
- COCO API 核对 145 个 mask 与源标注逐像素相同，145 个区域裁剪有效。
- 共享包 218 个清单文件 SHA256 全部复核通过，保留原版不可变字节。
- 单元测试 19 项通过（11 项数据边界 + 8 项复核审计）。
- 仓库原有合成交接检查器通过；没有将合成检查记为真实数据或模型检查。

40 组双语复核记录：32 组 target_agreement_only、6 组 target_disagreement、2 组 answer_disclosed。
accepted_promoted=0；6 组分歧需要重新看图裁决，不能直接判定复核者错误。
描述质量统计：34 组纯类别、6 组空间、0 组外观/组合；当前包仅用于接口调试，不用于证明模型语言能力。
原始复核记录包含个人填写内容，只存数据盘运行目录，不上传 GitHub。公开文档仅使用成员简称。

输入压缩包 SHA256：
c73751215b89d769ccc34cb98a64d454b044a024bf833e5e51849e1a16c01335

## 复现命令

从上述工作区执行（不需要 GPU）：

```bash
PY=/root/autodl-tmp/fashion-grounding/envs/data_eval/bin/python
PACKAGE=/root/autodl-tmp/fashion-grounding/shared/processed/fp_pilot20_v2
RUN=/root/autodl-tmp/fashion-grounding/runs/data_eval/20261112-pilot-review
export PYTHONPATH=src
$PY -m unittest discover -s tests -v
$PY scripts/validate_pilot.py "$PACKAGE"
$PY -m fashion_grounding.data.review --package "$PACKAGE" --review "$RUN/pilot20-human-review.json" --disclosed-image-id 1643 --output "$RUN/review_audit_repeat.json"
$PY scripts/check_contract_examples.py
$PY -m pip check
```

审计拒绝覆盖输出，重复运行要选择新文件名。
check_pilot_consumers.py 会写报告，因此在 run/incoming/fp_pilot20_v2 副本上执行，不在已发布共享包中写入。
测试日志 tests.log、validation.json、review_audit.json、install.log、environment.txt 位于运行目录。
复核命令只输出汇总，详细裁决在 review_audit.json，不作为正式评价结果。

## 当前前置依赖与下一步

P1：wkq 工作区/CPU 环境已验证；不代表其他成员环境已完成。
P2：真实图片/GT 接口包可交接；人工目标分歧、查询设计修订和消费方验收未完成，阶段仍 in_progress。
P3：B0、共享解析与统一模型评价器尚未实现；本次 review audit 是数据复核工具，不是模型评价器。
P4：正式分区、全量数据转换和跨分区重复检查未完成。

lcy 可读取 annotations_coco.json 验证加载；wyz 可读取 GT 候选验证裁剪/编码。
queries_draft.jsonl 仅供 draft 接口测试，queries_accepted.jsonl 仍为空。
下一批语言数据需按协议增加真实同类竞争、外观、空间及组合描述，不能扩充当前单类别模板充数。
人工标注与交叉复核由全员分担；不自动开展可选扩展或长期训练。

## 发布与凭据边界

代码和测试均在远端完成。GitHub 连接器写树接口返回 403，但 LE-brawl 的仓库权限包含 push；本机 Git Credential Manager 的登录账号已独立核验为 LE-brawl。远端生成提交与 bundle，由本机 Git 仅负责传输发布；不把本机 token 或私钥复制到共享服务器。共享包中的 tools/environment 是原本地交接快照，远端运行以当前仓库 requirements-data.txt 和本交接命令为准。
