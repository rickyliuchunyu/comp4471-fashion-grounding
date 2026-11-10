# AutoDL 服务器初始只读检查

检查日期：2026-11-11（香港日期）。角色 segmentation，由 Liu Chunyu 授权 Codex 通过专用 SSH 密钥执行。P1 in_progress；这是远程环境盘点，不是模型训练验收。连接地址、端口、实例 ID 和认证材料不放公开仓库。

## 已核验的环境

| 项目 | 实际结果 |
| --- | --- |
| GPU | RTX 3090，24576 MiB；检查时 0 MiB 占用、无 GPU 计算进程 |
| 驱动 | 580.105.08；nvidia-smi 显示最高 CUDA 13.0，不是当前 Toolkit 版本 |
| 系统盘 | 30G，已用约 26G，剩余约 4.6G，85% |
| 数据盘 | /root/autodl-tmp，50G，基本为空 |
| Conda | 仅列出 /root/miniconda3 基础环境，无角色独立环境 |
| Python | 3.10.8 |
| PyTorch / torchvision | 2.1.2+cu121 / 0.16.2+cu121，两者导入成功 |
| CUDA runtime 检查 | torch.version.cuda=12.1，cuda.is_available()=True，设备识别为 RTX 3090 |
| CUDA Toolkit | /usr/local/cuda-12.1/bin/nvcc，12.1.105；非交互 shell 的 PATH 未找到 nvcc，但绝对路径可运行 |
| GCC | 11.4.0 |
| Detectron2 | 0.6，detectron2._C 导入成功 |
| 可变形注意力扩展 | MultiScaleDeformableAttention 共享库导入成功 |
| mask 工具 | pycocotools 2.0.11，mask 模块导入成功 |
| 其他已装包 | numpy 1.26.4、scipy 1.15.3、opencv-python 4.10.0.84、timm 1.0.27、ninja 1.13.0 |
| 图文依赖 | transformers、sentencepiece、accelerate 未安装；safetensors 0.8.0、huggingface-hub 1.22.0 已安装；未进行兼容性组合验证 |
| 进程工具 | flock 可找到；本次 shell 中未找到 tmux |

截图另显示 14 vCPU / 90GB 内存和包年包月租期；这些来自用户控制台截图，不冒称本次通过操作系统配额接口独立核验。

## 旧镜像与空间

旧项目目录仍在系统盘，outputs 目录 du 约 23G；保留未删除、未移动。镜像中有 Mask2Former 源码目录，但默认 Python 路径找不到 mask2former 包。本次只确认目录存在与已装扩展能导入，没有加载该源码或核对其 revision，也没有将旧项目代码、数据或结果当课程成果。

目录 du 与 overlay 文件系统 df 的统计口径可能不同，不将各目录数字直接相加推算剩余空间。后续下载与输出优先放数据盘；新环境也需先评估空间。不要在仅剩约 4.6G 的系统盘直接堆放新权重或整套环境。

## 实际检查方法与范围

在 SSH 登录后的服务器执行了以下只读命令或等效 Python 读取：

```bash
nvidia-smi
df -h / /root/autodl-tmp
/root/miniconda3/bin/conda env list --json
/root/miniconda3/bin/python -V
/usr/local/cuda-12.1/bin/nvcc --version
gcc --version
command -v tmux
command -v flock
```

另用 importlib.metadata 读取上表包版本；Python -B 导入 torch、torchvision、detectron2._C、MultiScaleDeformableAttention、pycocotools.mask；检查 torch.cuda.is_available()/get_device_name()。使用 ps 的进程名/资源列和 du 读取进程与目录大小；没有导出环境变量、凭据或旧实验内容。

**没有执行 GPU 张量计算、分割/图文模型前向、反向、训练、保存恢复或性能测试。** 扩展能够导入不等于其 CUDA kernel 已验证。没有安装/升级依赖、删除/搬移文件、建立服务器工作区、下载数据或启动训练，也未关机。

## 下一步交接

- Liu Chunyu：保持旧环境，先规划数据盘中的独立课程 checkout、缓存/输出和 GPU 协调；核对公开分割源码/权重后，用 B 的真实小包验证前向、反向、候选与恢复。
- A：准备独立图文环境，不直接升级旧分割基础环境；固定兼容的 SigLIP 2/processor 依赖再验证模型。
- B：继续公共数据/评价接口及真实小包，不因模型尚未就绪停工。
- 尚未验证：三人账号访问、独立工作区、共享锁包装脚本、模型兼容性和长期训练配置/预算。正式 environment_verified 仍为 false。

[项目状态](../../status.md) 已更新；模型实验仍是计划，没有新增指标。下游成员尚未完成实际接收验收。
