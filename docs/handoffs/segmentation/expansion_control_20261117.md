# 5k/10k 等预算分割对照（2026-11-17）

lcy / segmentation。本轮用户已明确授权按已准备方案启动两组：同一官方 COCO 初始化、seed42、全程 min_scale0.5、各 20000 次成功 update、batch2；训练后对保存的 10000/20000 权重评完整 dev250，再做组内选择和两组比较。每组从开始到评价/选择结束最多 10800 秒，两组合计最多 21600 秒；失败即停，不自动延长、重启或恢复。

## 启动依据

开工源 `4d60fcd`，wkq 新增 `b986630`、`c514a4f`：完整 dev250 语言视觉初稿和精简 v4（330 中英组/660 条，accepted=0），以及训练文字覆盖分析；未改变本轮分割 train/dev 输入。wyz `103dcc5`、main `7064ebb` 未变。所有远端分支、五个 PR 和活跃讨论已核；#2/#4 draft open，#5 open 且依赖 #4；无合并、新 PR 或新讨论。PR #4 正文仍是旧摘要，以上进度依据其最新 commit 的三份交接，不替队友更新 PR。

wkq 标注了原 dev 若干类别/源 mask 碎片疑点，尚无裁决或修订。本轮两组使用同一原 dev250（250 图/1008 GT），保留这些评价局限；不改标注、删样本或重建数据。语言 v4 不进入本轮训练/评价。

资源快照：RTX3090 0 MiB/0%，无计算进程、无活动协调任务，共享锁可用；数据盘空闲 164682031104 bytes（约153.4 GiB）。新输出和两组 run ID 均不存在。见 [GitHub快照](evidence/expansion_control_20261117/github_preflight.json)、[服务器快照](evidence/expansion_control_20261117/server_preflight.json)。

复用 [上一轮入口准备及11项CPU检查](expansion_execution_preparation_20261116.md)、旧 COCO/八类头/AMP/稳定性/恢复、旧 dev 验收和新10k接收/隔离证据。本轮只登记启动授权和新 run ID，训练/评价科学配置及代码未修改；不重跑准备模式、完整数据审计、测试套件或 smoke。正常 trainer/evaluator 的身份门禁继续执行。

## 执行

输出：`/root/autodl-tmp/fashion-grounding/runs/segmentation/segmentation-expansion-control-20261117-r01`。

- C5k：`segmentation-expansion-5000-20261117-r01`
- D10k：`segmentation-expansion-10000-20261117-r01`

固定 segmentation 环境及干净源码，通过已实现的共享锁/独立进程组/绝对截止时间入口执行；仅本角色 checkout。启动命令（状态与实际时间在运行后追加）：

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/autodl-tmp/fashion-grounding/envs/segmentation-v1/bin/python \
scripts/run_expansion_control.py \
  --output /root/autodl-tmp/fashion-grounding/runs/segmentation/segmentation-expansion-control-20261117-r01 \
  --execute
```

比较规则沿用 [受控方案](train10000_control_plan_20261116.md)，主要终点为同预算20k；10k/20k组内选最大完整dev AP，平手取早。旧B20k AP63.79295仅复用报告比较，不重推理；不自动替换现有候选、wyz缓存或部署权重。这里只完成分割对照，不扩展至语言、排序或test。

## 实际状态

启动前登记，尚无本轮训练或AP结果。后续根据真实 run_metadata、stage_status、日志及评价报告更新；不把启动或准备通过写成实验完成。
