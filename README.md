# AFD-Nano-vLLM

This repository serves as a learning resource for exploring **nano-vLLM**. It features support for Attention and Feed-Forward Network (FFN) disaggregation on the **Qwen3-0.6B** model.

### 📚 References & Documentation
* [afd-nano-vLLM Design Document](https://my.feishu.cn/docx/D5oPdBjS4oCOUbxQHDsc79sBnkg?from=from_copylink) *(Detailed analysis and architecture for this repository)*

---

## Requirements

- 2x NVIDIA GPUs

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

```bash
python bench.py
```
