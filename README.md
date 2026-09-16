# GoLLIE Activity Extraction

## 项目目的

本项目使用 GoLLIE-7B 从叙事文本中提取人物的活动、移动及其关联信息，形成结构化出行链，并将结果保存为便于检查和后续分析的 JSONL 文件。

## 代码架构

```text
scripts/
  check_env.py                 # 检查 Python、PyTorch、CUDA 和 GPU 环境
  gollie_runtime.py            # 模型加载、提示构造、推理与结果解析
  test_gollie.py               # 最小推理 smoke test
  extract_epub.py              # 文本提取及重叠切块
  run_activity_extraction.py   # 对指定文本块执行出行链抽取
data/
  raw/                         # 原始资料
  chunks/                      # 切分后的文本块
  outputs/                     # 测试与抽取结果
logs/                          # 本地运行记录
requirements.txt               # Python 依赖及版本
```

`gollie_runtime.py` 使用 Python AST 解析模型输出，不直接执行模型生成的代码。

## 当前环境

- Windows 11 + WSL2 Ubuntu 24.04
- Python 3.10.21
- NVIDIA GeForce RTX 3060 Laptop GPU，6 GiB 显存
- PyTorch 2.7.1 + CUDA 12.8
- Transformers 4.44.2
- Accelerate 0.34.2
- bitsandbytes 0.48.1
- FlashAttention 2.8.3


## 模型

- 模型：[HiTZ/GoLLIE-7B](https://huggingface.co/HiTZ/GoLLIE-7B)
- 固定提交：`d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9`
- 加载方式：bitsandbytes 4-bit NF4、double quantization、FP16 compute

