
通过从零构建一个名为`micrograd`的微型自动求导引擎，来深入、直观地理解神经网络的训练核心——反向传播算法。

# 第一阶段：核心概念与导数直观理解

- **Micrograd 简介**:
    
    - 它是一个自动梯度（Autograd）引擎，其核心功能是实现**反向传播（Backpropagation）**。
        
    - 反向传播是高效计算损失函数相对于神经网络所有权重和偏置的**梯度**（Derivatives）的算法。梯度指明了调整参数以减小损失的方向。
        
    - 这是 PyTorch、JAX 等现代深度学习框架的数学核心。
        
- **核心演示**:

    - `Value`对象：`micrograd`中的基本数据单元，可以构建数学表达式图。
        
    - **前向传播 (Forward Pass)**：从输入开始，通过计算图计算出最终的输出值。
        
    - **反向传播 (Backward Pass)**：在最终输出上调用`.backward()`，`micrograd`会自动、递归地应用**链式法则（Chain Rule）**，计算出最终输出对每一个输入和中间变量的梯度。
```python
class Value:

    def __init__(self,data,_children=(),_op="",label=""):
 
        self.data=data    #data: 存储节点的值

        self.grad=0.0      #grad: 存储梯度

        self._prev=set(_children) #_prev: 存储节点的父节点

        self._op=_op       #_op: 存储节点的操作符

        self.label=label   #label: 存储节点的标签

        self._backward=lambda:None #_backward: 存储梯度计算的函数


    def __repr__(self):

        return f"Value(data={self.data},grad={self.grad})"
        

    def __add__(self,other):

        other = other if isinstance(other, Value) else Value(other)

        out=Value(self.data+other.data,(self,other),"+")

        return out          


    def __mul__(self,other):

        other = other if isinstance(other, Value) else Value(other)

        out=Value(self.data*other.data,(self,other),"*")
        return out

    def __pow__(self,other):

        assert isinstance(other, (int, float))

        out=Value(self.data**other,(self,),f"**{other}")

        return out
```

        
- **导数的意义**:
    
    - 导数（梯度）衡量的是一种**敏感度**。例如，`a.grad = 138` 意味着，如果输入`a`的值发生一个微小的正向变动，最终输出会以138倍的斜率增加。
        
    - 神经网络训练就是利用这个敏感度信息来微调参数，使得损失函数的输出向着减小的方向移动。
        
- **构建`Value`类**:
    
    - 为了构建计算图，`Value`对象不仅存储数据，还记录了`_prev`（它的前驱节点/子节点）和`_op`（生成它的运算符号）。
        
    - 通过重载Python的`__add__`和`__mul__`等方法，使得`Value`对象可以像普通数字一样进行运算，并自动构建计算图。
        
    - 使用`draw_dot`函数可以可视化这个计算图，清晰地看到数据流。
        

# **第二阶段：手动反向传播与链式法则**

- **这是理解反向传播最核心的部分。**
    
- **梯度初始化**:
    
    - 在`Value`对象中增加`grad`属性，初始为0。
        
    - 对于最终的输出节点（例如`L`），其`grad`被初始化为`1.0`，因为任何变量对自身的导数都是1 (`dL/dL = 1`)。
        
- **手动计算梯度**:
    
    - **从后向前**，逐个节点计算梯度。
        
    - **加法节点 (+)**：像一个“路由器”，它将上游传来的梯度**原封不动地分配**给它的所有输入。因为 `d(a+b)/da = 1`，所以梯度传递时乘以1，保持不变。
        
    - **乘法节点 (*)**：像一个“交换机”。对于 `c = a * b`，`dc/da = b`，`dc/db = a`。因此，上游传来的梯度会分别乘以**另一个输入的值**，再传递给当前输入。`a.grad += L.grad * b.data`，`b.grad += L.grad * a.data`。
        
    - **链式法则 (Chain Rule)**：是这一切的数学基础。如果 `z` 依赖于 `y`，`y` 依赖于 `x`，那么 `dz/dx = dz/dy * dy/dx`。反向传播就是链式法则在整个计算图上的递归应用。
        
    - **梯度累加**: 如果一个变量在计算图中被多次使用，它的梯度需要**累加（+=）**，而不是直接赋值。这是一个非常关键且容易出错的细节。

```python
    def __add__(self,other):

        other = other if isinstance(other, Value) else Value(other)

        out=Value(self.data+other.data,(self,other),"+")

        def _backward():        # 没有 def 就会立即执行梯度计算。

            self.grad+=out.grad

            other.grad+=out.grad

        out._backward=_backward # def 把梯度计算延迟到调用 backward() 时执行

        return out              #确保梯度计算的正确顺序。

    def __mul__(self,other):

        other = other if isinstance(other, Value) else Value(other)

        out=Value(self.data*other.data,(self,other),"*")


        def _backward():

            self.grad+=out.grad*other.data

            other.grad+=out.grad*self.data

        out._backward=_backward

        return out

    def __pow__(self,other):

        """举例：

            乘法: a * b - 两个都是变量，都需要梯度

            a = Value(2.0)

            b = Value(3.0)

            c = a * b → 父节点是 (a, b)

            幂运算: a ** 2 - 只有底数是变量，指数是常数

            a = Value(2.0)

            c = a ** 2 → 父节点只有 (a,),因为指数2不需要梯度

            """

        assert isinstance(other, (int, float))

        out=Value(self.data**other,(self,),f"**{other}")

        def _backward():

            self.grad+=out.grad*other*self.data**(other-1)

        out._backward=_backward

        return out


    def relu(self):

        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')


        def _backward():

            self.grad += (out.data > 0) * out.grad

        out._backward = _backward


        return out

```

        

# **第三阶段：自动化反向传播与构建神经网络 **

- **自动化 `_backward` 函数**:
    
    - 为每一种运算（加、乘、Tanh激活函数等）定义一个局部的`_backward`函数。这个函数知道如何将输出的梯度传播给输入。
        
    - 例如，Tanh的导数是 `1 - tanh(x)²`。它的`_backward`函数就会将上游梯度乘以 `1 - self.data²` 再传递下去。
        
- **拓扑排序 (Topological Sort)**:
    
    - 为了确保反向传播按正确的顺序（从输出到输入）进行，需要对计算图进行拓扑排序。这保证了在计算一个节点的梯度时，所有依赖它的下游节点的梯度都已经计算完毕。
        
- **完整的 `backward()` 方法**:
    
    - 这个方法封装了整个流程：1. 拓扑排序；2. 初始化最终节点的梯度为1；3. 反向遍历排序后的列表，依次调用每个节点的`_backward`函数。
    
```python
    
    def backward(self):

        """

        反向传播方法：从当前节点开始，自动计算整个计算图的梯度

        工作原理：

        1. 使用拓扑排序确保梯度计算的正确顺序

        2. 将当前节点的梯度设为1.0（作为反向传播的起点）

        3. 按拓扑排序的逆序遍历所有节点，调用每个节点的_backward()方法

        只要在任意节点调用.backward()，该节点之前（上游）的所有节点都会被自动计算完梯度！

        这就是自动微分的核心思想：一次调用就能得到整个计算图的梯度。

        如果没有这步只能一个一个节点按._backward(),得到上一个节点的梯度

        """

        topo=[]

        visited=set()

        def build_topo(v):

            if v not in visited:

                visited.add(v)

                for child in v._prev:

                    build_topo(child)

                topo.append(v)

        build_topo(self)

        self.grad=1.0

        for node in reversed(topo):

            node._backward()
```
        
- **构建神经网络模块**:
    
    - **Neuron (神经元)**：一个基本的计算单元，包含一组权重 `w` 和一个偏置 `b`。它对输入执行 `w*x + b` 的线性变换，然后通过一个非线性的激活函数（如Tanh）。
        
    - **Layer (层)**：由多个独立的神经元组成。
        
    - **MLP (多层感知机)**：将多个层按顺序堆叠起来，构成一个完整的神经网络。
```python
import random

class Module:

    """神经网络模块的基类"""

    def zero_grad(self):

        """将所有参数的梯度清零"""

        for p in self.parameters():

            p.grad = 0

    def parameters(self):

        """返回模块的所有参数，基类默认返回空列表"""

        return []


class Neuron(Module):

    """单个神经元类"""


    def __init__(self, nin, nonlin=True):

        """

        初始化神经元

        nin: 输入维度

        nonlin: 是否使用非线性激活函数(ReLU)

        """

        # 随机初始化权重

        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]

        # 偏置初始化为0

        self.b = Value(0)

        # 是否使用非线性激活

        self.nonlin = nonlin

    def __call__(self, x):

        """前向传播计算"""

        # 计算加权和加偏置

        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)

        # 根据nonlin决定是否应用ReLU激活函数

        return act.relu() if self.nonlin else act

    def parameters(self):

        """返回神经元的所有参数(权重+偏置)"""

        return self.w + [self.b]


    def __repr__(self):

        """字符串表示"""

        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

```

---

```python

class Layer(Module):

    """神经网络层类，包含多个神经元""
  

    def __init__(self, nin, nout, **kwargs):

        """

        初始化层

        nin: 输入维度

        nout: 输出维度(神经元数量)

        **kwargs: 传递给神经元的其他参数

        """

        # 创建nout个神经元

        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]


    def __call__(self, x):

        """前向传播"""

        # 每个神经元都处理相同的输入

        out = [n(x) for n in self.neurons]

        # 如果只有一个输出，直接返回值而不是列表

        return out[0] if len(out) == 1 else out

  

    def parameters(self):

        """返回层中所有神经元的参数"""

        return [p for n in self.neurons for p in n.parameters()]


    def __repr__(self):

        """字符串表示"""

        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

```
---

```python

class MLP(Module):

    """多层感知机(Multi-Layer Perceptron)"""

    def __init__(self, nin, nouts):

        """

        初始化MLP

        nin: 输入维度

        nouts: 各层的输出维度列表

        """

        # 构建层尺寸列表: [输入维度, 第一层输出, 第二层输出, ...]

        sz = [nin] + nouts

        # 创建各层，最后一层不使用非线性激活函数

        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

  

    def __call__(self, x):

        """前向传播，逐层处理"""

        for layer in self.layers:

            x = layer(x)

        return x

  

    def parameters(self):

        """返回所有层的参数"""

        return [p for layer in self.layers for p in layer.parameters()]

  

    def __repr__(self):

        """字符串表示"""

        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
```


# **第四阶段：完整的神经网络训练流程 **

- **训练循环 (Training Loop)** 是神经网络学习的核心，包含以下五个步骤：
    
    1. **前向传播 (Forward Pass)**：将输入数据喂给神经网络，计算出预测值 `ypred`。
        
    2. **计算损失 (Compute Loss)**：使用一个**损失函数**（如均方误差MSE）来衡量预测值 `ypred` 和真实目标 `y_true` 之间的差距。损失是一个标量，代表了模型当前的“糟糕”程度。
        
    3. **梯度清零 (Zero Grad)**：**极其重要的一步！** 在每次反向传播之前，必须将所有参数（权重和偏置）的梯度重置为0。因为梯度是累加的，如果不清零，之前的梯度会干扰本次的计算。
        
    4. **反向传播 (Backward Pass)**：调用 `loss.backward()`，计算出损失对网络中每一个参数的梯度。
        
    5. **参数更新 (Update Parameters)**：根据梯度信息更新每一个参数。公式为：`参数.data += -学习率 * 参数.grad`。
        
        - **学习率 (Learning Rate)**：一个超参数，控制每次更新的步长。
            
        - **负号**表示我们希望向着**梯度下降**的方向移动，从而最小化损失。
            
- 通过成千上万次重复这个循环，神经网络的参数会被微调，使得损失逐渐降低，模型的预测能力越来越强。
    

# **总结 **

- **神经网络的本质**: 它们是可微调的复杂数学表达式。
    
- **训练的本质**: 通过**梯度下降**算法，找到一组最优的参数（权重和偏置），使得损失函数最小化。
    
- **核心引擎**: **反向传播**是高效计算梯度的关键，它使得在巨大的参数空间中进行梯度下降成为可能。
    
- **Micrograd 与 PyTorch**: Micrograd 的API设计与PyTorch高度相似，理解了Micrograd的原理，就能更好地理解PyTorch等工业级框架的内部工作机制。

****神经网络是一个巨大的数学表达式，我们通过反向传播计算梯度，再通过梯度下降来最小化损失，最终让这个表达式学会我们希望它完成的任务。**

希望这份笔记能够帮助你拨开神经网络的迷雾，真正理解深度学习的魅力所在。