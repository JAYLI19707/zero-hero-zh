# Zero to Hero 中文学习笔记

[English](README.md) | [中文](README.zh-CN.md)

本项目为学习 Andrej Karpathy 的神经网络入门课程 “Zero to Hero” 的中文笔记与实践代码。

- 原项目地址：[https://github.com/karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)
- 本仓库包含课程相关的代码实现、实验记录以及中文学习笔记，便于理解和复现课程内容。

## 目录结构

1. [`micrograd/`](micrograd/)
   - micrograd 相关代码与实验
   - 中文学习笔记
2. [`bigram/`](bigram/)
   - bigram 相关代码与实验
   - 中文拆解笔记
   - 数据文件与输出示例
3. [`MLP/`](MLP/)
   - 多层感知机（MLP）相关代码与实验
   - 中文学习笔记
4. [`gpt-2/`](gpt-2/)
   - 使用 PyTorch 从零实现的字符级迷你 GPT
   - Transformer、自注意力、训练和文本生成的完整讲解

## GPT-2 迷你 Demo

打开 [`gpt-2/gpt_dev.ipynb`](gpt-2/gpt_dev.ipynb)，从上到下依次运行 notebook。该 notebook 使用 Tiny Shakespeare 数据集训练一个约 0.21M 参数的字符级 GPT。

安装所需依赖：

```bash
pip install torch matplotlib numpy
```

数据集已包含在 [`gpt-2/input.txt`](gpt-2/input.txt) 中，无需额外下载。

## 学习路径

项目按照从简单的标量自动求导到语言模型的顺序排列：

```text
micrograd -> bigram -> MLP -> mini GPT
```

欢迎学习与交流！
