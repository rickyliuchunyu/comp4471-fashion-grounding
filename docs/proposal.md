# 自然语言引导的八类服饰实例分割 Proposal 草稿

英文标题：Natural Language Guided Selection of Fashion Instances

课程：COMP 4471 与 ELEC 4240，Fall 2026。

成员：待填写三名成员的姓名及课程要求的身份信息。

提交日期：2026 年 10 月 5 日 23:59，香港时间。

## 英文正文

We propose to investigate natural language guided selection of fashion instances: given an image and a description such as “the bag on the left” or “the red skirt,” our system will return the corresponding garment mask and bounding box. This task is useful for interactive product exploration and requires both accurate instance segmentation and discrimination between objects of the same category. Motivated by prior internship experience, we will rebuild the pipeline from public data and pretrained models, with new implementations and experiments completed by our three-person course team. Our background reading will cover Mask2Former, the DeepFashion2 and Fashionpedia datasets, and visual and textual representation learning using DINOv2 and BGE-M3. We will map public annotations into eight categories: top, pants, skirt, outerwear, dress, shoes, bag, and accessory. Images will be partitioned before generating referring expressions, and we will create and manually review descriptions paired with target instances, including category, spatial, and reliable visual attribute cues. We will fine-tune Mask2Former and compare single-source and balanced mixed-source training. For language grounding, we will first implement a category-and-position rule baseline, then train a region-text alignment or ranking module using frozen visual and text encoders and explicit spatial features. We will examine whether learned matching improves selection of ambiguous instances. Segmentation will be evaluated using COCO mask AP and per-category results; grounding will be evaluated using target-selection accuracy and mask IoU on both ground-truth candidate masks and predicted candidates, with the former treated as a diagnostic. We will report matched comparisons, ablations, inference latency, qualitative overlays, and failure cases, and reserve an untouched image-level test partition for final evaluation. We expect the project to reveal tradeoffs between category coverage, localization quality, and computational cost without presupposing a performance improvement.

## 提交前检查

正文是一个 200 至 400 词的段落。填写三名成员信息，确认团队认可项目范围，并在提交前核对 Canvas 是否另有要求。具体分工尚未确定。旧实习成果的背景和本次重新实现的范围已在正文区分，旧实验指标不作为本课程新结果。

## 背景阅读与引用来源

- [Mask2Former 论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf) 与 [官方实现](https://github.com/facebookresearch/Mask2Former)。
- [DeepFashion2 官方数据集说明](https://github.com/switchablenorms/DeepFashion2)。
- [Fashionpedia 官方说明](https://github.com/cvdfoundation/fashionpedia)。
- [DINOv2 官方实现](https://github.com/facebookresearch/dinov2)。
- [BGE-M3 官方模型说明](https://huggingface.co/BAAI/bge-m3)。
- [课程项目要求](https://course.cse.ust.hk/comp4471/project.html)。
