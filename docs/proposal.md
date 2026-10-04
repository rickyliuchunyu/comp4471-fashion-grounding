# 自然语言引导的八类服饰实例分割 Proposal 草稿

英文标题：Natural Language Guided Selection of Fashion Instances

课程：COMP 4471 与 ELEC 4240，Fall 2026。

成员：待填写三名成员的姓名及课程要求的身份信息。

提交日期：2026 年 10 月 5 日 23:59，香港时间。

## 英文正文

We propose a candidate based system for natural language guided selection of fashion instances: given an image and a Chinese or English description, the system will return the category, bounding box, and mask of one referenced whole object. Unlike category-only segmentation, the task requires distinguishing instances such as two bags through spatial or visible appearance cues. Motivated by prior internship experience, our three-person team will independently rebuild the pipeline using public data, implementations, and pretrained weights, and produce new course experiments. We will unify DeepFashion2 and Fashionpedia annotations into eight categories: top, pants, skirt, outerwear, dress, shoes, bag, and accessory. Since DeepFashion2 does not annotate the last three categories, we will compare a Fashionpedia baseline, naive mixed-source training, and a source-aware classification loss that permits missing categories for unmatched predictions rather than treating them as confirmed background. We will fine-tune Mask2Former R50 while retaining its Hungarian matching and mask BCE/Dice supervision. Referring expressions will be created and independently reviewed on image-disjoint Fashionpedia splits, including bilingual pairs, spatial descriptions, and verifiable appearance cues. For selection, frozen DINOv2 mask-pooled region features and BGE-M3 dense text embeddings will be mapped into a shared 256-dimensional space by trainable projection heads. A within-image ranking objective will use other instances, especially same-category objects, as negatives. We will compare this model with category-and-position rules and with a text-conditioned geometry head using normalized coordinates and relative instance ranks. Segmentation will be measured by COCO mask AP and per-category results. Selection will be evaluated on both ground-truth and predicted candidates, reporting candidate coverage, conditional selection accuracy, and final target-mask IoU to separate missed objects from ranking errors. Controlled ablations, counterfactual left/right queries, bilingual comparisons, latency with and without caching, and failure analysis will test the proposed mechanisms. An untouched final test partition will be reserved before annotation, and reported improvements will depend on measured results rather than assumed gains.

## 提交前检查

正文是一个 200 至 400 词的段落。填写三名成员信息，确认团队认可项目范围，并在提交前核对 Canvas 是否另有要求。具体分工尚未确定。旧实习成果的背景和本次重新实现的范围已在正文区分，旧实验指标不作为本课程新结果。

## 背景阅读与引用来源

- [Mask2Former 论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf) 与 [官方实现](https://github.com/facebookresearch/Mask2Former)。
- [DeepFashion2 官方数据集说明](https://github.com/switchablenorms/DeepFashion2)。
- [Fashionpedia 官方说明](https://github.com/cvdfoundation/fashionpedia)。
- [DINOv2 官方实现](https://github.com/facebookresearch/dinov2)。
- [BGE-M3 官方模型说明](https://huggingface.co/BAAI/bge-m3)。
- [课程项目要求](https://course.cse.ust.hk/comp4471/project.html)。
