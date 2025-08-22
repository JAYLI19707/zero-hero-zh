# Bigram 字符级语言模型实现

## 项目简介

本项目实现了一个基于 Bigram（二元组）的字符级语言模型，用于生成人名。项目展示了从统计方法到神经网络方法的演进过程，是深度学习入门的经典案例。

**代码来源**: [Andrej Karpathy - Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/makemore)

## 项目结构

- `makemore.ipynb` - 主要的 Jupyter Notebook 文件，包含完整的实现
- `names.txt` - 包含人名数据的文本文件（需要单独下载）

## 核心概念

### 1. Bigram 模型

Bigram 模型是一种基于字符前后组合的统计语言模型：
- 统计所有相邻字符对（二元组）的出现频次
- 基于频次计算条件概率 P(下一个字符|当前字符)
- 利用这些概率来生成新的名字

### 2. 数据预处理

```python
# 读取数据
words = open("names.txt",'r').read().split()

# 添加特殊标记
# 使用 '.' 作为开始和结束标记，简化了原来的 '<S>' 和 '<E>'
for w in words:
    chs = ['.'] + list(w) + ['.']
```

### 3. 字符编码

```python
# 字符到索引的映射
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0  # 特殊标记

# 索引到字符的映射
itos = {i:s for s,i in stoi.items()}
```

## 实现方法

### 方法一：统计方法

1. **构建计数矩阵**
```python
N = torch.zeros(27, 27, dtype=torch.int32)

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]  
        ix2 = stoi[ch2]  
        N[ix1, ix2] += 1
```

2. **转换为概率分布**
```python
# 加1平滑处理，避免零概率
P = (N+1).float()
P /= P.sum(1, keepdim=True)
```

3. **生成新名字**
```python
g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    out = []
    ix = 0  # 从起始符号开始
    while True:
        p = P[ix]  # 获取当前字符的概率分布
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:  # 遇到结束符号停止
            break
    print(''.join(out))
```

### 方法二：神经网络方法

1. **One-hot 编码**
```python
import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes=27).float()
```

2. **网络结构**
```python
# 权重矩阵：27个输入特征 -> 27个输出（每个字符的分数）
W = torch.randn((27, 27), generator=g, requires_grad=True)

# 前向传播
logits = xenc @ W          # 计算logits
counts = logits.exp()      # 转换为计数
probs = counts / counts.sum(1, keepdim=True)  # softmax归一化
```

3. **损失函数和训练**
```python
# 负对数似然损失
loss = -probs[torch.arange(num), ys].log().mean()

# 反向传播和参数更新
W.grad = None
loss.backward()
W.data += -70 * W.grad  # 梯度下降
```

## 核心技术要点

### 1. Softmax 函数
```python
# logits -> counts -> probabilities
counts = logits.exp()
probs = counts / counts.sum(1, keepdims=True)
```

### 2. 负对数似然损失
- **目标**: 最大化数据的似然度
- **等价于**: 最小化负对数似然
- **实现**: `loss = -probs[torch.arange(num), ys].log().mean()`

### 3. 梯度下降优化
- 使用 PyTorch 的自动微分
- 手动实现参数更新：`W.data += -learning_rate * W.grad`

## 可视化

项目包含了计数矩阵的可视化，显示字符转换的频率：

```python
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')
plt.show()
```

## 运行结果

### 统计方法生成的名字示例：
```
junide.
janasah.
p.
cony.
a.
```

### 模型评估
- **统计方法负对数似然**: 2.4544
- **神经网络训练后损失**: 2.4560（接近统计方法）

## 学习要点

1. **统计方法 vs 神经网络**: 两种方法在这个简单案例中得到几乎相同的结果
2. **One-hot 编码**: 将离散字符转换为神经网络可处理的连续向量
3. **Softmax**: 将任意实数转换为概率分布
4. **梯度下降**: 通过反向传播优化模型参数
5. **数据平滑**: 加1平滑避免零概率问题

## 依赖要求

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
```

## 参考资料

- **原始教程**: [Andrej Karpathy - Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/makemore)
- **相关视频**: [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)

## 许可证

本项目基于 Andrej Karpathy 的开源教程改编，遵循原项目的许可证。
