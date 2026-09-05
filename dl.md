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
W_j^{(n+1)}= L - \alpha(\sum_{i=0}^{n-1}X_{ij} (Y_i-X_i*W))
$$

SGD:

$$
W_j^{(n+1)}= L - \alpha X_{ij} (Y_i-X_i*W)
$$

MBGD:

$$
W_j^{(n+1)}= L - \alpha(\sum_{random \space choose \space some \space i \space from \space [0,n)} X_{ij} (Y_i-X_i*W))
$$

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
subject \space to \space ||w||^2 ≤ \theta
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
>`/home/zane/sound/deep-learning/code/dropout/dropout.py`

