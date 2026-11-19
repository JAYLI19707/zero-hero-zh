# Zero to Hero: Chinese Learning Notes

[English](README.md) | [中文](README.zh-CN.md)

This repository contains study notes, experiments, and code implementations for Andrej Karpathy's **Zero to Hero** neural network course.

- Original course repository: [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)
- The notes and explanations are primarily written in Chinese to make the course easier to follow and reproduce.

## Repository structure

1. [`micrograd/`](micrograd/)
   - Micrograd implementation and experiments
   - Chinese study notes
2. [`bigram/`](bigram/)
   - Bigram language model implementation and experiments
   - Chinese walkthrough notes
   - Data files and generated examples
3. [`MLP/`](MLP/)
   - Multi-layer perceptron implementation and experiments
   - Chinese study notes
4. [`gpt-2/`](gpt-2/)
   - A character-level mini GPT built from scratch with PyTorch
   - Transformer, self-attention, training, and text generation walkthrough

## GPT-2 mini demo

Open [`gpt-2/gpt_dev.ipynb`](gpt-2/gpt_dev.ipynb) and run the notebook from top to bottom. It trains a roughly 0.21M-parameter character-level GPT on the Tiny Shakespeare dataset.

Install the required packages with:

```bash
pip install torch matplotlib numpy
```

The dataset is included as [`gpt-2/input.txt`](gpt-2/input.txt), so no additional download is required.

## Learning path

The projects are arranged from simple scalar automatic differentiation to language modeling:

```text
micrograd -> bigram -> MLP -> mini GPT
```

欢迎学习与交流！
