# Fashionpedia 全量分割训练包交接

wkq / data_eval 已按 [lcy 的全量需求](https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/dd786c74622820ef59ce2f5edc771453c3f25e94/docs/handoffs/segmentation/full_train_request_20261118.md) 发布 `fp_train_full_eligible_v1`，直接父包为 `fp_train10000_targeted_v1`。构建 commit `e4a7021da3d5d1acf56c2a8e043589b5c7a1144c`。新增 32,476 张、32,476 个独立组、113,149 个实例。全包为 **42,476 张、42,452 组、152,848 个实例**。数据方已完成技术检查，当前等待 lcy 的 CPU 接收。

## 来源数量与去重

官方 train 的 **45,623 = 10,000 + 32,476 + 3,147 + 0**（旧包、新增、排除或隔离、待处理）。项目图像 ID 为官方 train 源 ID 加一。`audit/ledger.json` 按官方源图片 ID 逐图登记，按一个主要原因统计。匹配在带官方 split 的身份空间内完成。

| 主要去向 | 官方源图数 |
| --- | ---: |
| 旧10k | 10000 |
| 新增代表 | 32476 |
| 含 jumpsuit | 922 |
| 连通至父包或保护组 | 1375 |
| 新组内多余代表 | 794 |
| 已登记保护关联组 | 54 |
| 无整件实例 | 1 |
| 源 mask 无效 | 1 |

完整扫描使用旧10k、固定dev250/test400、全部50张登记过的开发曝光图及关联组的已存指纹。新增与保护组的关系及新增之间的关系按文件SHA、具体资源URL、继承组和原/镜像四组合63位pHash距离≤6保守连通。接触旧或保护组的新成员全部隔离，其余每组取一个代表。旧数据的组号和实例保留，test仅用保护元数据。全源扫描下，5个连通组件同时触及旧10k与实际dev/test或曝光图，共123张关联新候选已隔离。这些是待裁决线索，不能据此判定历史数据泄漏；见[关联记录](full_train_20261118_evidence/frozen_bridge_clues.json)。哈希方法对同商品和复杂裁剪的识别有局限。完整冲突和处置记录位于包内 `audit/duplicates.json`，保护输入版本位于 `audit/protection_inputs.json`。

## 覆盖

每格依次为 **不同图片 / 独立组 / 实例**。面积使用原始分辨率mask前景像素，行间允许重叠。父包小目标和极小目标实例数已与原发布统计的7166和1875核对。八类与26个源细类的完整三视图见[覆盖报告](full_train_20261118_evidence/coverage.json)。

| 类别和尺寸 | 旧10k | 新增 | 合并全量 |
| --- | ---: | ---: | ---: |
| 八类合计 全部 | 10000/9976/39699 | 32476/32476/113149 | 42476/42452/152848 |
| 八类合计 ≤1024 | 4034/4030/7166 | 8706/8706/11189 | 12740/12736/18355 |
| 八类合计 ≤256 | 1393/1393/1875 | 2106/2106/2508 | 3499/3499/4383 |
| 袜 全部 | 634/634/1120 | 736/736/1324 | 1370/1370/2444 |
| 袜 ≤1024 | 544/544/870 | 420/420/646 | 964/964/1516 |
| 袜 ≤256 | 314/314/449 | 239/239/323 | 553/553/772 |
| 腰带 全部 | 1732/1730/1789 | 4386/4386/4497 | 6118/6116/6286 |
| 腰带 ≤1024 | 689/688/719 | 649/649/676 | 1338/1337/1395 |
| 腰带 ≤256 | 136/136/138 | 151/151/152 | 287/287/290 |
| 鞋 全部 | 5739/5729/11099 | 16427/16427/31790 | 22166/22156/42889 |
| 鞋 ≤1024 | 2074/2073/2733 | 4469/4469/5675 | 6543/6542/8408 |
| 鞋 ≤256 | 408/408/435 | 892/892/967 | 1300/1300/1402 |
| 头饰（源聚合类） 全部 | 1016/1016/1198 | 1824/1824/2105 | 2840/2840/3303 |
| 头饰（源聚合类） ≤1024 | 719/719/875 | 785/785/1005 | 1504/1504/1880 |
| 头饰（源聚合类） ≤256 | 322/322/433 | 311/311/482 | 633/633/915 |
| 手表 全部 | 795/793/799 | 2407/2407/2417 | 3202/3200/3216 |
| 手表 ≤1024 | 626/625/628 | 1760/1760/1768 | 2386/2385/2396 |
| 手表 ≤256 | 194/194/194 | 385/385/385 | 579/579/579 |
| 包/钱包 全部 | 2077/2075/2203 | 4470/4470/4630 | 6547/6545/6833 |
| 包/钱包 ≤1024 | 210/210/215 | 0/0/0 | 210/210/215 |
| 包/钱包 ≤256 | 23/23/23 | 0/0/0 | 23/23/23 |
| 官方 cutout 属性 all | 152/152/153 | 558/558/560 | 710/710/713 |
| 官方 cutout 属性 shoe | 0/0/0 | 0/0/0 | 0/0/0 |
| 官方 cutout 属性 bag | 0/0/0 | 0/0/0 | 0/0/0 |

## 复杂边界的证据

源属性308（cutout）的图、组和实例数见上表。鞋细带、包肩带及遮挡分离区域按具体图像记录计数。此前分层AI抽查了36张图，其中36张进入本次新增成员。下面为该批留存观察中的明确案例，分母为留存AI观察，不据此外推全量发生率。源图类别本身不作为困难语义证据。

| 观察 | 有逐图依据的新增实例 |
| --- | ---: |
| shoe_thin_straps | 1 |
| shoe_open_toe_visible_gap | 1 |
| bag_shoulder_strap | 2 |
| separated_visible_shoe_regions | 1 |

露趾可见空隙单列，不能据此判定鞋的制造工艺。旧10k抽查记录对部分包带或遮挡有定性描述，各语义的完整计数未知。新增原尺寸mask用 OpenCV 4.5.4 RETR_CCOMP 统计连通分量与内孔。下表为自动几何线索，是否为真实肩带、遮挡或镂空需看图判断。旧10k采用原有像素验收。

| 新增几何线索 | 图片 | 组 | 实例 |
| --- | ---: | ---: | ---: |
| multiple_foreground_components_None | 19740 | 19740 | 33155 |
| multiple_foreground_components_23 | 4866 | 4866 | 6356 |
| multiple_foreground_components_24 | 1941 | 1941 | 1968 |
| interior_holes_None | 14303 | 14303 | 18171 |
| interior_holes_23 | 2562 | 2562 | 3185 |
| interior_holes_24 | 948 | 948 | 957 |

另从本轮新采且不在前次候选池中的图片抽查12张，每层4张。鞋孔洞层有2/4张清晰可见细鞋带及带间空隙；人为切割工艺可确认0/4。稀疏包mask层有2/4张能确认肩带或斜挎带，其中1张可见袖子隔开包体和肩带。多分量层4/4具有几何分离，能明确归因为外部遮挡的为0/4。鞋实例96852的前足细带之间疑有源mask覆盖皮肤，已单列记录并保留原标注。叠图保存在包内audit，逐例记录见边界证据。

原来抽查记录中的小目标语义不确定性和局部袖部边界偏粗的疑点仍保留。此次沿用官方实例粒度和原mask。上述AI检查与自动统计不计为人工复核。详见[边界证据与案例](full_train_20261118_evidence/boundary.json)。

## lcy 接收

全量训练包为 `/root/autodl-tmp/fashion-grounding/shared/processed/fp_train_full_eligible_v1`。用 `RELEASE.json` 读取实际父包和数量，核对 `old_subset_membership.json` 与 `added_subset_membership.json` 互斥且并集覆盖全包。用原八类 CPU `load_training_package` 读取 **`/root/autodl-tmp/fashion-grounding/runs/data_eval/20261118-full-train-v1/fp_full_added_v1`** 即可验收增量；检查RGB、尺寸、RLE、框、面积、crowd及1–8到0–7映射。新增原文件32343张为RGB、133张为灰度，已按读取器的RGB转换路径核对灰度图。旧的固定10k计数或采样配额须改为读取本次RELEASE。新增113,149个mask已逐像素核对官方源，证据绑定到增量manifest和COCO标注。独立检查重核源身份、保留记录、哈希和近重复关系。权重与训练安排由lcy另行登记。

| 绑定文件 | SHA256 |
| --- | --- |
| source_annotation_sha256 | f6af94fb8712bd8170a49c857e842e1227e6806c4f33f32340db0c849133e006 |
| mapping_sha256 | ce88a18ade3f6620eb83335b88da2e062847e033b7b75fd9a4266a480400583f |
| manifest_sha256 | 7174024e0b70db3f3911b1f9a7153a240bd228367505e247215edb8b1434a107 |
| annotations_sha256 | 567c2b6e76cdca50bc55682f17f232dedea05b80c836c000c43b2d75fe0c0340 |
| release_sha256 | 7567710f2d8446fd106a646051de2e6a4748c31f22252dab07940fbf234e1d35 |
| checksums_sha256 | 37d64b68185c99926e49bee3bae7a64dcc6b92cffbf4224011766322e4895948 |

[发布索引](full_train_20261118_evidence/published.json)与[发布后核验](full_train_20261118_evidence/postpublish_check.json)。数据、逐图账和完整指纹在服务器数据盘，GitHub 保存实现与小型交接证据。

## 运行

独立环境为 `/root/autodl-tmp/fashion-grounding/envs/data_eval`，run目录为 `/root/autodl-tmp/fashion-grounding/runs/data_eval/20261118-full-train-v1`。运行入口依次为 `full_train_inventory.py download`、`full_train_inventory.py select`、`build_full_segmentation.py`、`review_full_train_boundaries.py`（分层样本看图结果写入 run 目录 visual_observations.json，已发布原始记录见包内 audit）、`publish_full_segmentation.py`、`verify_full_segmentation.py`（均在scripts目录）。用 `c++ -shared -fPIC -O2 scripts/full_train_topology.cpp -I/usr/include/opencv4 -lopencv_imgproc -lopencv_core -o /root/autodl-tmp/fashion-grounding/runs/data_eval/20261118-full-train-v1/topology.so` 编译几何统计。发布目录按版本只读使用。运行仅涉及CPU数据准备，模型效果待后续训练评价。
