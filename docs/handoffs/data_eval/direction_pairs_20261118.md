# 非鞋训练对与独立方位 dev 增量

wkq / data_eval，2026-11-18。训练和开发增量均为 Codex 看图初标，等待独立人工复核。方向按看图者视角。每张新增图标两个独立实例，每个目标各一条中文、一条英文。两个目标共享 contrast_set_id，各有自己的 semantic_query_id。审核时应能在同一张图上选出两个不同掩码。

| 数据包 | 新增图 | 不同目标 | 新增中英文记录 | 水平 / 垂直图片 |
|---|---:|---:|---:|---:|
| fp_language_direction_pairs_train_v1 | 86 | 172 | 344 | 68 / 18 |
| fp_language_direction_pairs_dev_v1 | 74 | 148 | 296 | 65 / 9 |

## 数据位置和用法

服务器共享根目录 `/root/autodl-tmp/fashion-grounding/shared/processed`。训练使用 `fp_language_direction_pairs_train_v1/candidates_gt.json` 和 `queries_machine_train.jsonl`。新包直接继承 1149 图版的 3296 条记录，共 1235 图、3640 条记录（1234 图有查询）。只做增量统计时读取 `queries_increment.jsonl`。wyz 可用原 M2、种子 42、相同更新预算做对照。同时报告同图一对目标都选对的比例，观察是否只记住了一个方向。

新 dev 包只在 `queries_draft.jsonl` 和 `queries_increment.jsonl` 提供监督草稿。机器训练和人工 accepted 入口均为空。正式复评需要 wkq 审核并冻结查询，lcy 为这 74 张图导出同一检查点和阈值的预测候选。之前 50k/dev250 的预测文件只覆盖原主 dev。保持主 dev 的 642 条历史分母，另报新集、非鞋和鞋三个结果。CPU 已完成数据与消费检查，本次未训练或运行模型评价。

## 覆盖与筛选

训练新增按实际描述对象计：包 31 图、手套 22 图、袜子 16 图、上衣 7 图、发饰 4 图、眼镜 2 图、护腿套 2 图、雨伞 1 图和外套 1 图。新增没有鞋。官方 train 10k 和全量 train 先以同源细类、同一项目服装类、候选框中心与间距预筛，再逐图查看原图与两个源掩码。看了 221 张，保留 86 张。拒绝了同件衣物的左右片、包带/包身拆标、连裤袜的两条腿、遮挡太重或不能确认是两件的目标。图 5724 的左右包分别对应标注 121364 和 121365。

正式 dev 的旧纯方位 10 条覆盖 3 图，旧 39 张困难图中 4 张是纯方位。新 dev 74 张都来自另取的官方 validation，包括 61 张鞋、6 张手套、6 张袜子、1 张护腿套。鞋是旧评估中有直接证据的错误类型。经保护筛查得到的 410 张源图里没有一张含两个官方包实例，当前 dev 增量对双包的覆盖为零。图像和文本分布均为定向选择，不用于估算自然场景中的类别频率。

## 保护、核验与审核

- 用全量分割发布的保护指纹及其新增代表对原 unused-validation 重新检查。按源 SHA、资源 URL、继承组、原图/镜像 pHash 四组合距离 ≤6 判断潜在重复。检查 43,237 条保护指纹；新 dev 与全量 train、原 dev250、dev39、test400、当前语言 train 在 image_id、SHA、group_id 上的交集均为零。相似哈希不能穷尽同商品或同系列视觉重复。
- 新 dev 候选中实际看过 96 张，74 张入选、22 张质量隔离。96 张全部登记开发曝光。加上上次的 50 张，两份登记表共 146 张不同图片。将来运行增量分割保护扫描时读取两份登记。
- 校验所有候选掩码与源标注逐像素一致，按真实消费者裁剪 train 6280 个、dev 379 个候选。对新增组另核验每张图两个目标、每个目标两种语言、原始框中心次序和训练/审核状态。wyz 的弱标训练入口实读 3640 条，供推理读取的查询对象不含目标字段。用于消费校验的 grounding 代码固定在 `b5e372925dcd6f581fb1a0c8f65bd5dee5dde3cd`。
- 自然措辞不保证现有共享解析器全能解析。新 dev 中现有解析器可同时唯一识别类别和方向的条数为中文 134/148、英文 143/148。保留原始描述和完整分母。
- 远端审核入口分别为 `/root/autodl-tmp/fashion-grounding/runs/data_eval/20261118-nonshoe-pairs-v1/review/dev/index.html` 与 `/root/autodl-tmp/fashion-grounding/runs/data_eval/20261118-nonshoe-pairs-v1/review/train/index.html`。每页分别导出审核 JSON；只需审核新增目标。盲审和初标答案预览有独立入口。

选择与看图证据、逐图决定、消费校验记录和发布前核验脚本保存在 `/root/autodl-tmp/fashion-grounding/runs/data_eval/20261118-nonshoe-pairs-v1`。重建时从 data_eval 工作区以 `PYTHONPATH=src` 和本角色环境运行 `scripts/prepare_direction_pairs.py <新运行目录>`，再加 `--supplement` 运行补筛。复制冻结的 `visual_keeps.json` 到运行目录后，分别执行 `scripts/publish_direction_pairs.py <运行目录> train` 和 `dev`。视觉保留索引和原提案 SHA 可用于核对重建的输入。[统计、去重与曝光记录](direction_pairs_20261118_evidence/)
