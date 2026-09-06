# deep learning notes

这本书是我的深度学习笔记，从2026年9月4日开始，我刚刚结束保研，是准大四的状态。

深度学习改变了世界，整个人工智能的大厦建立在深度学习之上，于是我想在正式成为一名严格意义上的研究生之前，把深度学习的基础打扎实了，系统地学一遍。

我们这本书不会涉及到太多的传统机器学习算法，比如SVM,EM等算法，但是学习一些经典的传统机器学习算法是非常有必要的，我们会涉及到一些。

本书适合有一定机器学习或深度学习基础，且对python语法比较熟悉，但是又没有系统地学习和整理这些知识的读者。希望这本书对各大读者有帮助，欢迎各位提交PR、和我沟通等。

- 我的邮箱: zhouhy5523@gmail.com

为了方便起见，本书大写字母符号都代表矩阵，小写字母符号代表标量。

## 一、线性回归

机器学习的监督学习算法分为回归和分类问题。直观上理解：回归问题指的是，给你一个样本向量，预测值是连续的，分类问题指的是，给你一个样本向量，预测值是离散的。

回归问题定义：给定n * (m+1)的矩阵X,代表n个样本，m个特征的样本组成的矩阵；给定n * 1的矩阵Y，代表n个样本的标签组成的矩阵，该矩阵的取值是连续的。请学习X,Y中的规律（回归算法），使得给出一个新的样本x，能预测一个新的y使得和真实正确的y尽量接近。

线性回归是一种回归算法，且是最简单的回归算法。

![线性回归](pics/1.png)

设最终学习到的模型权重W形状是(m+1) * 1,即最终的模型是：

$$
\hat{Y} = X * W 
$$

则损失函数为：

$$
L = \frac{1}{2n}||Y - \hat{Y}||^2=\frac{1}{2n}\sum_{i=0}^{n-1}(Y_i-\hat{Y_i})^2=\frac{1}{2n}\sum_{i=0}^{n-1}(Y_i-X_i*W)^2
$$

求导：

$$
\frac{\partial L}{\partial W_j} = -\frac{1}{n}\sum_{i=0}^{n-1}X_{ij} (Y_i-X_i*W)
$$

由梯度下降法：

BGD:

$$
W_j^{(t+1)}= W_j^{(t)} - \alpha\frac{\partial L}{\partial W_j}\Big|_{W^{(t)}} = W_j^{(t)} + \frac{\alpha}{n}\sum_{i=0}^{n-1}X_{ij} (Y_i-X_i*W^{(t)})
$$

SGD:

$$
W_j^{(t+1)}= W_j^{(t)} + \alpha X_{ij} (Y_i-X_i*W^{(t)})
$$

MBGD:

$$
W_j^{(t+1)}= W_j^{(t)} + \frac{\alpha}{|B|}\sum_{i \in B} X_{ij} (Y_i-X_i*W^{(t)}) \\
\text{where B is a random mini-batch from } [0,n)
$$

上标 $(t)$ 是迭代轮次，求和里的 $W^{(t)}$ 表示用当前这一轮的权重计算预测值。

> 实验1：线性回归算法的具体实现(MBGD)：
>
>`code/linear_regression/linreg_low_level.py`
>
>`code/linear_regression/linreg_high_level.py`

## 二、Softmax回归

Softmax回归是一种多类分类算法。

多类分类问题定义：给定n * (m+1)的矩阵X,代表n个样本，m个特征的样本组成的矩阵；给定n * 1的矩阵Y，代表n个样本的标签组成的矩阵，该矩阵的取值是离散多类的,一共k种。请学习X,Y中的规律（多类分类算法），使得给出一个新的样本x，预测该样本为y类别的概率尽可能大。

![Softmax回归](pics/2.png)

Softmax函数的定义是：

$$
Softmax(j) = \frac{exp(x_j)}{\sum_{i=0}^{k-1}exp(x_i)}
$$

设最终学习到的模型权重W形状是(m+1) * k,即最终的模型是：

$$
\hat{Y} = Softmax(X * W)
$$

Y hat 是一个n*k的矩阵，则损失函数为：

$$
L = -\frac{1}{n}\sum_{i=0}^{n-1} log \hat{Y_{ij}} \\
\text{j is the argmax index of  the one hot encoding of the }Y_i 
$$

按照梯度下降法更新参数W即可。

> 实验2：Softmax回归算法的具体实现(MBGD)：
>
>`code/softmax_regression/softmax_low_level.py`
>
>`code/softmax_regression/softmax_high_level.py`

## 三、多层感知机MLP

![多层感知机MLP](pics/3.png)

没有太多新的原理，趁着MLP的机会谈一下BP算法：



## 四、欠拟合、过拟合、VC维、正则项、权重衰退、丢弃法

当模型容量大于数据复杂度时，过拟合，当模型容量小于数据复杂度时，欠拟合。VC维指的是：对于一个分类模型，VC等于一个最大数据集的大小，不管如何给定标号，都存在一个模型来对它进行完美分类。

防止过拟合的一些手段，比如加正则项。通过限制参数值的选择范围来控制模型容量：

$$
min \space l(W,B) 
\\
subject \space to \space ||W||^2 ≤ \theta
$$

通常不限制B，小的θ意味着更强的正则项。

等价于:

$$
min \space l(W,B)+\frac{\lambda}{2}||W||^2
$$

这就是权重衰退，原因是：

$$
\frac{\partial}{\partial W} (l(W,B)+\frac{\lambda}{2}||W||^2) = \frac{\partial l(W,B)}{\partial W}+\lambda W
$$

时间t的更新你参数为：

$$
W_{t+1}=(1-η\lambda)W_t-η\frac{\partial l(W_t,B_t)}{\partial W_t}
$$

通常ηλ<1,在深度学习中称为权重衰退。

还有一种防止过拟合的方法，叫丢弃法(dropout)，丢弃法将一些输出项随机置0来控制模型的复杂度，要求保持无偏性：对X加入噪音得到X'，我们希望：

$$
E(X') = X
$$

丢弃法对每个元素进行如下扰动：

$$
x_i' = \begin{cases} 0,& \text{with probability p} \\ \frac{x_i}{1-p} , & \text{otherwise}\end{cases}
$$

dropout只在训练时对某些作用层发挥作用，推理时不发挥作用。

![丢弃法dropout](pics/4.png)

> 实验3：Dropout的具体实现：
>
>`code/dropout/dropout.py`

## 五、数值稳定性

当网络深度增加，由于链式法则，可能会出现两个问题：梯度爆炸和梯度消失。

对于梯度爆炸，值可能会超出值域，尤其是16位浮点数最为严重，对学习率也很敏感，太大会导致更大的梯度，太小训练又没有进展。

对于梯度消失，不管如何选择学习率，训练都没有进展。且对于底部层尤为严重，仅仅顶部层训练的较好，无法让神经网络更深。

目标：让梯度值在合理的范围内，例如[1e-6,1e3]，解决方法：

- 将乘法变加法
- 归一化，clip
- 合理的权重初始和激活函数

合理的权重初始方法是：将每层的输出和梯度都看做随机变量，让它们的均值和方差都保持一致。

正向传播,对于第t层的第i个神经元的输出：

$$
E[H_i^t] = 0 
\\
Var[H_i^t] = a
$$

反向传播，对于第t层的第i个神经元的梯度：

$$
E[\frac{\partial l}{\partial H_i^t}] =0 
\\
Var[\frac{\partial l}{\partial H_i^t}] =b 
$$

权重初始化是在合理值区间里随机初始参数。训练开始的时候更容易数值不稳定。使用N(0,0.01)来初始可能对小网络没问题，但不能保证深度神经网络。

对第t-1层的第j个神经元和第t层的第i个神经元之间的权重，假设有：

$$
1.W_{i,j}^t 独立同分布(i.i.d)，那么E[W_{i,j}^t]=0,Var[W_{i,j}^t]=\gamma_t
\\
2.H_j^{t-1}独立于W_{i,j}^t
$$

假设没有激活函数，
$$
H^t = W^tH^{t-1}
$$

则:

$$
E[H_i^t] = E[\sum_j W_{i,j}^tH_j^{t-1}] = \sum_j E[W_{i,j}^t]E[H_j^{t-1}] = 0
$$

正向方差：

$$
Var[H_i^t] = E[(H_i^t)^2]-E[H_i^t]^2 = E[(\sum_j W_{i,j}^tH_j^{t-1})^2] = E[\sum_j (W_{i,j}^t)^2(H_j^{t-1})^2+\sum_{j≠k}W_{i,j}^tW_{i,k}^tH_{j}^{t-1}H_{k}^{t-1}]=\sum_jE[(W_{i,j}^t)^2]E[(H_j^{t-1})^2]=\sum_j Var[W_{i,j}^t]Var[H_j^{t-1}]=n_{t-1}\gamma_t Var[H_j^{t-1}]
$$

由此可推出：

$$
n_{t-1}\gamma_t=1
$$

反向均值和方差：

$$
\frac{\partial l}{\partial H^{t-1}} = W^t\frac{\partial l}{\partial H^t}
$$

则：

$$
E[\frac{\partial l}{\partial H^{t-1}}] = 0 
\\
Var[\frac{\partial l}{\partial H^{t-1}}] = n_t\gamma_tVar[\frac{\partial l}{\partial H_j^t}]
$$

则：

$$
n_t\gamma_t = 1
$$

但是难以满足

$$
n_{t-1}\gamma_t=1 \text{ and } n_t\gamma_t = 1
$$

Xavier初始化使得

$$
\gamma_t(n_{t-1}+n_t)/2 =1
$$

则：

$$
\gamma_t = 2/(n_{t-1}+n_t)
$$

则可以是：

$$
N(0,\sqrt{2/(n_{t-1}+n_t)})
\\
U(-\sqrt{6/(n_{t-1}+n_t)},\sqrt{6/(n_{t-1}+n_t)})
$$

Xavier初始化策略有助于提升数值稳定性，适用于在 0 点附近近似恒等映射（f(0)=0 且 f'(0)=1）的激活函数，典型代表是 tanh。

## 六、卷积神经网络CNN

从全连接到卷积，需要先明白图片特征提取的一些原则：平移不变性和局部性。

![从全连接到卷积](pics/5.png)

很容易有：

$$
H_{i,j} = \sum_{k,l} W_{i,j,k,l}*X_{k,l} + bias
$$

令 k = i + a, l = j + b,V是W的重新索引，

有：

$$
H_{i,j} = \sum_{a,b} V_{i,j,a,b}*X_{i+a,j+b} + bias
$$

由平移不变性（对一个局部图案：X 平移后，跟随图案的窗口内像素值不变；假设窗口在平移前后都完整；我们要求输出值也不变，只是位置跟着平移——要这对任意输入成立，权重必须不随位置变化），得：

$$
V_{i,j,a,b} = V_{a,b}
$$

推导里求和默认窗口总是完整的。真实场景中图像边缘处窗口可能缺了一块，实际做法（补零 padding）意味着边缘位置的"相对结构"被人为改了——严格来说补零会破坏完美的平移等变性，只是实践中影响小，大家接受了。想让等变性严格成立得用循环 padding（把图像当环面）。   

由局部性，得：

$$
H_{i,j} = \sum_{a=-\Delta}^{\Delta} \sum_{b=-\Delta}^{\Delta} V_{a,b}*X_{i+a,j+b} + bias
$$

这其实严格意义上来说，是交叉相关，不是严格意义上的数学卷积。但由于核是学习出来的，翻转与否不影响表达能力，深度学习框架实现的其实都是交叉相关。

> 实验4：交叉相关的具体实现：
>
>`code/cnn/cross_correlation.py`

如果考虑填充和步幅，则输出形状是：

$$
\lfloor (n_h-k_h+p_h+s_h)/s_h \rfloor × \lfloor (n_w-k_w+p_w+s_w)/s_w \rfloor
$$

> 实验5：填充和步幅的具体实现：
>
>`code/cnn/padding_and_stride.py`

如果再考虑多输入通道:

$$
X : c_i × n_h × n_w \\
W : c_i × k_h × k_w \\
b : \text{scalar} \\
Y ：m_h × m_w \\
Y = \sum_{i=1}^{c_i} (X_{i,,} ※ W_{i,,}) + b \\
\text{where ※ is the convolution}
$$

如果再考虑多输出通道：

$$
X : c_i × n_h × n_w \\
W : c_o × c_i × k_h × k_w \\
b : c_o \\
Y ：c_o × m_h × m_w \\
Y_{j,,} = \sum_{i=1}^{c_i} (X_{i,,} ※ W_{j,i,,}) + b_j \\
\text{where ※ is the convolution, } j = 1, \dots, c_o
$$

偏置的维度是 $c_o$——每个输出通道一个标量偏置 $b_j$，广播加到该通道整个 $m_h × m_w$ 的输出上。单输出通道时偏置就退化成一个标量 $b$。

每个输出通道可以识别特定模式，输入通道核识别并组合输入中的模式。每个输入通道有独立的二维卷积核，每个输出通道有独立的三维卷积核。

---

1×1的卷积层的本质：它不识别空间模式，只是融合通道。

![1x1的卷积层](pics/6.png)

等价于：

$$
(n_h×n_w,c_i)为输入的，权重参数为(c_o,c_i)的Linear(c_i,c_o)层
$$

> 实验6：1x1卷积的具体实现以及和全连接层等价的验证：
>
>`code/cnn/one_multiply_one_convolution.py`

---

计算复杂度的计算：

一个输入通道对应一个二维卷积核，单次卷积操作计算量级别为

$$
k_h × k_w
$$

多个输入通道对应一个输出通道，对应多个二维卷积核（整体是一个三维卷积核），则单次卷积操作计算量级别为

$$
c_i × k_h × k_w
$$

每个输出通道都有一个三维卷积核，则多个输出通道的单次卷积操作计算量级别扩张到：

$$
c_o × c_i × k_h × k_w
$$

一共进行了

$$
m_h × m_w
$$

次卷积操作，所以综上，总的乘法计算量级别为：

$$
c_o × c_i × k_h × k_w × m_h × m_w
$$

如果考虑加法，

**第一部分**：注意到涉及到bias的加法是在做完卷积之后进行加bias的时候发生的，一次卷积操作且一个输出通道对应一次。现在我们进行了

$$
m_h × m_w
$$

次卷积操作，

$$
c_o
$$

个输出通道，所以涉及到bias的加法次数应该是:

$$
m_h × m_w × c_o
$$

**第二部分**：加法还有一种，多个输入通道和某个三维卷积核里的自己的一个二维卷积核卷积计算后，还应该做加和，合并成一个输出通道。一个输出通道即为:

$$
(c_i-1) × m_h × m_w
$$

次，一共又有

$$
c_o
$$

个三维卷积核（或者叫输出通道），则次数是：

$$
c_o ×(c_i-1)× m_h × m_w
$$

次。

**第三部分**：加法还有一种，单次二维卷积内部的加法，单次二维卷积是

$$
k_h × k_w -1
$$

次加法，一共进行了

$$
c_o × m_h × m_w
$$

次三维卷积操作，换算成二维是

$$
c_i × c_o × m_h × m_w
$$

次，所以这部分加法的计算量为：

$$
c_i ×c_o × m_h × m_w × (k_h × k_w -1)
$$

考虑所有运算，单层卷积层的计算量为：

$$
c_o × c_i × k_h × k_w × m_h × m_w + m_h × m_w × c_o + c_o ×(c_i-1)× m_h × m_w + c_i ×c_o × m_h × m_w × (k_h × k_w -1) = 2 × c_o × c_i × k_h × k_w × m_h × m_w
$$

综上，单层卷积层的计算级别是:

$$
O(c_o × c_i × k_h × k_w × m_h × m_w)
$$

> 实验7：多输入通道和多输出通道的具体实现：
>
>`code/cnn/multi_input_and_output_channels.py`


