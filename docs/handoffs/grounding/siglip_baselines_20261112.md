# wyz grounding 第一阶段交接

日期：2026-11-12。负责人：wyz（GitHub：MayI2004）。
分支：grounding/siglip-baseline-interfaces。
状态：remote_cpu_checked + gpu_verified（仅编码器与 B1 草稿接口）。
代码和验证均在独立远端工作区完成。没有正式训练、正式性能评价或最终测试。

## 本阶段完成

- 基于 main 7064ebb 建立独立 checkout，快进到 wkq 已发布的 dc24f2d 公共数据接口；
  没有修改 wkq/lcy 的 checkout。PR 依赖 data-eval/remote-pilot-review，不能只取本分支的新提交而遗漏公共读取器。
- 候选消费层复用 data.schema 的 UTF-8/路径/hash/RLE 工具。
  检查类别、候选顺序、mask 与框、图片尺寸；传入编码器/排序器的 Candidate 不包含 GT annotation ID。
  Query 显式白名单只传 ID、文字、语言、split 和 illustrative 标记。
- 固定版本 SigLIP 2：官方 RGB 紧框裁剪；官方 processor/tokenizer；
  FP32、eager attention、eval/no_grad，冻结参数；完整原文，超长明确失败。
- 不可变冻结特征缓存：图片、候选包/hash/顺序、来源、模型及处理版本均入键；
  读取校验数组 hash、形状、有限值和单位范数。可在同图后续查询复用。
- B1 完整文本排序、schema 3 预测；失败保持查询记录。
  交付包 inputs/ 用硬链接保存不可变上游字节，保持相对路径可解析；禁止原地修改交接包。
- B2 位置评分及调用边界已实现，复用 wkq 的 data.query_rules.parse_query。
  该公共解析器当前未发布，因此 B2 完整自然语言运行尚未验证，程序明确报告前置缺失；
  不复制词表，不用私有解析器伪装完成。
- GPU wrapper 使用现有 coordination/gpu.lock，锁内检查进程/显存与 running 记录，
  记录 grounding/wyz、独立 run ID 和超时；不停止他人进程。

## 环境与固定模型

工作区：/root/autodl-tmp/fashion-grounding/workspaces/grounding
环境：/root/autodl-tmp/fashion-grounding/envs/grounding-v1
使用 Python 3.10.8 venv --system-site-packages，以只读方式复用 base torch 2.1.2+cu121 /
torchvision 0.16.2+cu121；新增依赖只写入 grounding-v1，没有升级 base。
这不是完全脱离 base 的环境，base 变化时须重新验证；完整 pip freeze 位于 setup 运行目录。

transformers 4.51.3，huggingface-hub 0.30.2，tokenizers 0.21.4，
sentencepiece 0.2.0，numpy 1.26.4，Pillow 10.2.0，pycocotools 2.0.11。
pip check 通过。正式环境依赖见 envs/grounding/requirements.txt；不要直接套用 B 的 numpy 2.x 配方到旧 torch 环境。

模型：google/siglip2-base-patch16-224
revision：75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
权重 SHA256：612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b
下载约 1.54 GB，来自 hf-mirror.com，官方 Hugging Face API 独立确认相同 revision、大小与 LFS SHA256。
七个文件逐个记录 SHA256，保存在 snapshot/download_manifest.json。
模型目录：
/root/autodl-tmp/fashion-grounding/shared/weights/grounding/siglip2-base-patch16-224-75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
新增 scripts/download_grounding_model.py 供固定版本获取与验证；
本轮校验入口已运行，完整下载先由等价临时下载流程完成，未为验证该新脚本重复下载。

## 输入与实际证据

输入：shared/processed/fp_pilot20_v2，20 张 Fashionpedia train 图，145 个 GT。
candidate manifest SHA256：
61a0f91ff924c5956c3e870bf548866a80c199fadf515236e70786b8cdfa643b
80 条 queries_draft，accepted=0；均只用于明确标记的工程诊断。
没有利用人工目标 ID/类别/query type 进行推理，没有输出准确率。

- 36 项单元测试通过：原数据/复核 19 项 + grounding 17 项。
  覆盖手算排序、类别内位置/并列、答案字段隔离、RGB 裁剪、候选重复、
  缓存损坏/失效/复用、失败传播、未复核查询门禁和相对路径交付。
- 原合成交接检查器通过，pip check 通过。
- 真实包 145 个候选全部能读取并裁剪。
- GPU 编码检查：2 个真实区域、2 条草稿文字；图文 D=768，
  重建与官方 logits 最大绝对差 1.9073486328125e-06；参数冻结、超长拒绝通过。
  该检查内部耗时约 4.89 秒，peak allocated 1,533,655,552 字节；
  这是工程小样本观察，不是统一延迟/显存基准。
- B1 草稿：80/80 条合法输出、0 推理错误。不能解读为 100% 选择准确率。
- 完整暖缓存重放与原结果相同，重放调用编码器次数为 0。
  随后从干净实现提交 6e3dddeaf6a296573871886e271232d643c078a5 再跑 r02，
  80 条选中 ID 和分数保持一致，metadata 的 working_tree_dirty=false。
- GPU smoke、B1 r01/r02 通过现有共享锁运行，预算上限分别为 180/300/180 秒；
  均正常结束，不存在本角色遗留 GPU 任务。

数据盘证据根：
/root/autodl-tmp/fashion-grounding/runs/grounding/
- setup-20261112-r01/：install.log、tests.log、environment.txt、encoder-smoke.log、B1 日志。
- encoder-smoke-20261112-r01/smoke_report.json：真实编码与裁剪检查。
- B1-draft-20261112-r01/：首次完整草稿预测及 consumer_check.json。
- B1-draft-20261112-r02/：**推荐交给 B 的可复现诊断包**。
- coordination/ 下 grounding-*.json：共享锁、时间预算及真实结束状态。
缓存：shared/features/grounding/siglip-v1。

r01/smoke 在实现提交前执行，其 wrapper 的 git_commit 为上游 dc24f2d，
不能把该基线 commit 当作完整 grounding 运行源码。r02 是干净提交下的规范复跑；
之前的日志/产物保留，不覆盖或补造历史源码快照。

## 复现

从 grounding 工作区执行。CPU 检查：

```bash
export PYTHONPATH=src
PY=/root/autodl-tmp/fashion-grounding/envs/grounding-v1/bin/python
$PY -m unittest discover -s tests -v
$PY scripts/check_contract_examples.py
$PY -m pip check
$PY scripts/download_grounding_model.py --verify-only --output /root/autodl-tmp/fashion-grounding/shared/weights/grounding/siglip2-base-patch16-224-75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
```

GPU 草稿预测（每次换新 run ID 和 output；先取得同一 GPU 锁）：

```bash
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
BASE=/root/autodl-tmp/fashion-grounding
$PY scripts/with_grounding_gpu_lock.py \
  --coordination "$BASE/coordination" \
  --run-id grounding-B1-draft-NEW --output "$BASE/runs/grounding/B1-draft-NEW" --seconds 300 -- \
  $PY -m fashion_grounding.grounding.predict \
  --candidates "$BASE/shared/processed/fp_pilot20_v2/candidates_gt.json" \
  --queries "$BASE/shared/processed/fp_pilot20_v2/queries_draft.jsonl" \
  --snapshot "$BASE/shared/weights/grounding/siglip2-base-patch16-224-75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2" \
  --revision 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2 \
  --cache "$BASE/shared/features/grounding/siglip-v1" \
  --output "$BASE/runs/grounding/B1-draft-NEW" --run-id B1-train-gt-draft-NEW \
  --method B1 --device cuda --batch-size 8 --draft-diagnostic
```

r02 的实际命令数组和源码 hash 存在 run_metadata.json；上面 NEW 是重复运行的命名示例。
正式 accepted 查询不加 --draft-diagnostic；当前 accepted 文件为空，会明确拒绝。
最终测试需团队先完成冻结，再使用 --allow-final-test；本轮没有使用。
B2 仅在共享解析器交接后运行，正式 lambda 必须按 GT dev 协议选择；本轮没有调参。

## 交给 wkq / lcy

wkq：读取 r02/draft_predictions.jsonl 及 run_metadata.json；80 条输出只可验收接口，
不可纳入正式成绩。候选引用为 inputs/candidates_gt.json，相对路径/hash 可验证。
encoder/cache 接口和 B2 评分已有；下一项需要公共 parse_query、accepted 查询和统一评价器。
lcy：grounding 可消费当前 GT 包；后续交接固定预测候选即可走相同读取路径。
共同锁文件已经兼容，不要同时启动不遵守锁的任务。新图 pipeline 的调用可复用
grounding.predict.execute / rank_scores，具体 integration 与各方共同验收。
目前下游人员尚未复核本次交付。

P1：wyz 图文环境与真实编码器检查完成，整体团队阶段仍 in_progress。
P2：wyz 的图片/GT 消费检查完成，人工查询未就绪，整体不标 done。
P3：B1 草稿流程完成；B2 只有评分与接口检查，公共解析及统一正式评价待交接。
M1/M2、E1、正式 train/dev 划分、预测候选端到端效果均未验证，未开展可选扩展。
没有正式成绩，registry 只追加明确标注 draft diagnostic 的运行行，指标留空。

GitHub 已用 MayI2004 本人登录并接受本仓库协作邀请。
发布通过本机 Git 传输远端 bundle；GitHub token 和 SSH 私钥均未复制到共享服务器。
不直接合并 main。
