# 简易 Tokenizer 实现与 NLP 分词技术详解

本项目演示了如何从零开始构建一个基础的文本分词器（Tokenizer），并详细介绍了自然语言处理（NLP）中常见的分词策略及其优缺点。此外，项目还展示了如何利用 **Tiktoken** (GPT-2/3 的分词器) 以及如何构建用于训练 GPT 模型的 **Dataset** 和 **DataLoader**。

## 📂 项目代码解读 (`tokenizer_subword.ipynb`)

### 1. 手动构建 Tokenizer
#### 1.1 SimpleTokenizerV1 (基础版)
- **功能**：实现了最基本的 `encode` (文本转ID) 和 `decode` (ID转文本)。
- **原理**：基于正则表达式拆分单词，构建词到ID的映射。
- **局限性**：如果输入文本中包含了训练语料里没有的词（例如 "Hello"），程序会直接报错（KeyError）。

#### 1.2 SimpleTokenizerV2 (改进版 - 处理未知词)
- **特殊标记**：
    - `<|endoftext|>`：表示文本结束或文档分隔。
    - `<|unk|>`：表示未知词（Unknown）。
- **改进逻辑**：
    - 在 `encode` 过程中加入了兜底机制：`tokens = [tok if tok in self.str_to_int else "<|unk|>" ...]`
    - **这就是为什么新词表可以解读 "Hello"**：它并不是真的“认识”这个词，而是检测到 "Hello" 不在词表中，将其强制映射为 `<|unk|>` 的 ID（在本项目中是 1131）。

### 2. 使用工业级分词器：Tiktoken (BPE)
- **引入库**：使用 OpenAI 开源的 `tiktoken` 库。
- **模型**：加载了 `gpt2` 的编码方案。
- **优势**：
    - 使用 **Byte Pair Encoding (BPE)** 算法。
    - 能够处理任何文本（无 OOV 问题）。
    - 词表大小通常为 50,257，在效率和覆盖率之间取得平衡。

### 3. 构建 GPT 训练数据集
为了训练 GPT（生成式预训练 Transformer）模型，我们需要将文本转换为模型可读的 **输入-目标 (Input-Target)** 对。

#### 核心逻辑：Next Token Prediction
GPT 的训练目标是：给定前面的 context，预测下一个 token。

- **滑动窗口机制**：
    - 设定 `context_size` (例如 4)。
    - **Input (x)**:  `[token_1, token_2, token_3, token_4]`
    - **Target (y)**: `[token_2, token_3, token_4, token_5]`
    - 这里的 Target 刚好是 Input 向右移动一位的结果。

#### 代码实现 (`GPTDatasetV1`)
- **Dataset**:
    - 将整段文本编码为 Token IDs。
    - 使用滑动窗口截取固定长度的序列。
    - 返回 PyTorch Tensor 格式的数据。
- **DataLoader**:
    - 使用 `create_dataloader_v1` 函数封装。
    - 支持 `batch_size` (批量大小) 和 `stride` (滑动步长)。
    - `shuffle=True` 用于训练时的随机打乱。

---

## 🔍 BPE 迭代过程可视化

为了更好地理解 Byte Pair Encoding (BPE) 算法，我们在代码中实现了一个可视化的 BPE 合并过程。
这个过程展示了如何通过迭代统计最高频的字符对，逐步构建新的 Token。

以下是使用一个玩具语料库 `["lower", "lowest", "newer", "wider", "finest", "older"]` 进行前 5 轮合并的演示：

### 1. 原始状态
所有单词被拆分为单个字符，并在末尾加上 `</w>` 标记：
`l o w e r </w>`
`l o w e s t </w>`
...

### 2. 迭代合并示例
我们统计每一轮出现频率最高的字符对（Pair），并将它们合并。

#### **Iteration 1**
- **统计**: 字符 `e` 出现 7 次，`</w>` 出现 6 次。
- **合并**: 发现 `e + r` 这一对出现了 4 次（例如在 `lower`, `newer`, `wider`, `older` 中）。
- **操作**: 将所有的 `e r` 合并为新 token **`er`**。

#### **Iteration 2**
- **统计**: `er` 现在作为一个独立的 token 参与统计。
- **合并**: 发现 `er + </w>` 这一对出现了 4 次。
- **操作**: 将所有的 `er </w>` 合并为新 token **`er</w>`**。
  - 此时，单词 `lower` 变成了 `['l', 'o', 'w', 'er</w>']`。

#### **Iteration 3**
- **合并**: 发现 `l + o` 这一对出现了 2 次（在 `lower`, `lowest` 中）。
- **操作**: 将 `l o` 合并为 **`lo`**。

... 经过多轮迭代后，常见的词缀（如 `est`, `er`）甚至整个单词都会变成单一的 Token，从而实现了高效编码。

---

## 📚 核心知识点：分词方式演变

在上述代码中，我们构建的是一种**词级（Word-level）** tokenizer。在 NLP 领域，分词方式主要经历了以下几个阶段：

### 1. 常见分词粒度对比

假设原始文本为：`"I love AI."`

| 分词方式 | 切分结果 (示例) | 优点 | 缺点 |
| :--- | :--- | :--- | :--- |
| **Word-level** (词级) | `"I"`, `"love"`, `"AI"` | 1. **语义清晰**：保留单词完整含义。<br>2. **计算成本低**：序列较短。 | 1. **OOV 问题严重**：遇到生僻词（如 "Internationalization"）或新词（"Covid19"）无法处理。<br>2. **词表巨大**：通常需要 100K+ 词表，内存占用高。 |
| **Character-level** (字符级) | `"I"`, `"l"`, `"o"`, `"v"`, `"e"`, `"A"`, `"I"` | 1. **无 OOV 问题**：任何词都能拆解为字符。<br>2. **词表极小**：仅需几十个字符。 | 1. **序列过长**：模型计算成本大幅上升。<br>2. **语义丢失**：单个字符（如 'N'）无法表达 "Neural" 的含义。 |

### 2. Sub-word Tokenization (子词分词)

**这是目前主流的分词方式（如 GPT, BERT 使用）。**

它在 Word-level 和 Character-level 之间取得了平衡：
- **常见词保持整体**：例如 `computer` -> `["computer"]`
- **罕见词拆分为子词**：例如 `computational` -> `["comput", "ational"]`
- **适应新词**：例如 `unhappiness` -> `["un", "happiness"]`

**核心优势**：
- 减少了未登录词（OOV）问题。
- 通过词根（如 "boy"）让模型理解变体（如 "boys"）的含义，保留了语义关联。

### 3. BPE (Byte Pair Encoding) 算法

BPE 是一种具体的 Sub-word 分词算法，广泛用于 GPT 和 Transformer 模型。它通过统计学方法构建词表。

**执行步骤**：

1.  **初始化词汇表**：将所有单词拆分为单个字符。
2.  **统计频率**：统计语料中所有相邻字符对（bigram）的出现频率。
3.  **合并最频繁字符对**：找到频率最高的字符对（例如 'e' 和 's' 经常一起出现），将其合并为一个新的符号（'es'），并加入词表。
4.  **重复迭代**：重复步骤 2 和 3，直到词汇表达到预设大小。

通过这种方式，BPE 能够自动学会像 "ing", "ed", "tion" 这样的常见词缀，从而高效地表示语言。

---

## 🏭 工业界实战：Training Custom Tokenizer (`train_tokenizer.py`)

在实际的大模型（LLM）开发中，我们通常不会手动实现 BPE 算法，而是使用高性能库（如 Hugging Face `tokenizers`）来训练自定义分词器。`train_tokenizer.py` 展示了目前工业界标准的 Tokenizer 训练流程。

### 核心流程解析

1.  **数据流 (Data Pipeline)**
    - 使用 Python Generator (`yield`) 逐行读取大规模 JSONL 数据。这避免了将整个几 GB 的数据集一次性加载到内存中，是处理海量数据的标准做法。
    - 本项目使用的 `pretrain_hq.jsonl` 来自 [ModelScope MiniMind-Dataset](https://www.modelscope.cn/datasets/gongjy/MiniMind-Dataset)。该文件不会提交到 Git，请先下载到 `dataset/` 目录。

2.  **BPE 模型初始化**
    - **ByteLevel Pre-tokenizer**: 使用字节级预分词。这是 GPT-2/GPT-3/Llama 等模型的核心技巧。它直接对 UTF-8 字节进行编码，而不是 Unicode 字符。这意味着理论上它支持人类所有的语言和符号（包括 Emoji），彻底解决了 Unknown Token (`<|unk|>`) 问题。
    - `models.BPE()`: 初始化 BPE 算法模型。

3.  **特殊 Token 管理 (Special Tokens)**
    准备训练语料（就要写成 ChatML 格式）
    - 定义了对话模型必备的控制符：
        - `<|im_start|>`: 对话开始
        - `<|im_end|>`: 对话结束
        - `<|endoftext|>`: 文本结束/填充位
    - 这些 Token 在指令微调（Instruction Tuning）中至关重要，用于区分 System, User, Assistant 的角色。

    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>

    <|im_start|>user
    你好
    <|im_end|>

    <|im_start|>assistant
    你好！很高兴见到你！
    <|im_end|>


4.  **训练 (Training)**
    - `tokenizer.train_from_iterator(...)`: 结合前面的数据流生成器，开始迭代训练。
    - `vocab_size=6400`: 设定词表大小。在实际大模型中，这个值通常在 32k 到 100k 之间（如 Llama 3 是 128k）。

5.  **Chat Template (对话模板)**
    - 代码中包含了一个复杂的 Jinja2 模板 (`chat_template`)。
    - **作用**: 自动将对话历史列表（List of dicts）格式化为模型训练所需的纯文本字符串。
    - **示例**:
      ```python
      # 输入
      messages = [{"role": "user", "content": "Hello"}]
      # 模板渲染后
      "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
      ```
    - 这是现代 Chatbot（如 ChatGPT, Claude）能理解多轮对话的关键配置。

通过这个脚本生成的 `tokenizer.json` 和 `tokenizer_config.json`，可以直接被 Hugging Face `transformers` 库加载使用，完全符合工业界标准。
