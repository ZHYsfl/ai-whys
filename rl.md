# reinforcement learning notes

这本书是我的强化学习笔记，从2026年10月9日开始，我结束了保研，是刚大四的状态。

强化学习和强化学习一样，改变了世界，我会把强化学习放在当前智能的上限的最后一公里的位置上，Yann LeCun 在 NIPS 2016 大会上提出了他如今已是家喻户晓的“蛋糕比喻”：“如果智能是一个蛋糕，那么蛋糕的主体是无监督学习，蛋糕上的糖霜是监督学习，而蛋糕顶部的樱桃则是强化学习 (RL)。

于是我想在正式成为一名严格意义上的研究生之前，把强化学习的基础打扎实了，系统地学一遍。

我们这本书数学基础和代码并重。

本书适合有一定机器学习、深度学习和强化学习基础，且对python语法比较熟悉，但是又没有系统地学习和整理强化学习知识的读者。希望这本书对各大读者有帮助，欢迎各位提交PR、和我沟通等。

- 我的邮箱: zhouhy5523@gmail.com

为了方便起见，本书大写字母符号都代表矩阵，小写字母符号代表标量。

## 一、基础概念

![concepts](pics/24.png)

Agent、Environment、State、Reward、Policy、Action、Done(terminated or truncated)、Episode、Trajectory、Return、StateValue、ActionValue、Advantage、TDError、Markov Property、data buffer、online algorithm、offline algorithm、on policy algorithm、off policy algorithm、Optimal State Value Function、Optimal Action Value Function.

## 二、q-learning

![q-learning](pics/25.png)

q-learning为什么有效的理论推导：

有贝尔曼最优方程：

$$
Q^*(S_t,A)=R_{t+1}+γmax_{A}Q^*(S_{t+1},A)
$$

带α的q-learning更新公式：

$$
Q_{t+1}(S_t,A_t) = Q_t(S_t,A_t)+α[R_{t+1}+γmax_{A}Q_t(S_{t+1},A)-Q_t(S_t,A_t)]
\\
= (1-\alpha)Q_t+\alpha[R_{t+1}+\gamma maxQ_t(S_{t+1})]
$$

定义误差：

$$
e_t = Q_t - Q^*
$$

我们的目标是看$e_{t+1}$和$e_t$的关系。

等式两边同时减去Q*：

$$
Q_{t+1}-Q^* = (1-\alpha)Q_t+\alpha[R_{t+1}+\gamma maxQ_t(S_{t+1})] - Q^*
\\
= (1-\alpha)(Q_t-Q^*)+\alpha[R_{t+1}+\gamma maxQ_t(S_{t+1})-Q^*]
$$

带入贝尔曼最优方程：

$$
e_{t+1} = (1-\alpha)e_t + \alpha[R_{t+1}+\gamma maxQ_t(S_{t+1})-(R_{t+1}+\gamma maxQ^*(S_{t+1}))]
\\
e_{t+1} = (1-\alpha)e_t+\alpha\gamma[maxQ_t(S_{t+1})-maxQ^*(S_{t+1})]
$$

由于

$$
|maxQ_t - maxQ^*| ≤ max|Q_t - Q^*| = |e_t|
$$

再由三角不等式，有：

$$
|e_{t+1}| ≤ (1-\alpha)|e_t| + \alpha\gamma |e_t| 
$$

即为：

$$
|e_{t+1}| ≤ (1-\alpha + \alpha\gamma )|e_t| 
$$

由于

$$
\alpha ∈ (0,1],\gamma ∈ [0,1)
$$

则

$$
(1-\alpha+\alpha\gamma) < 1
$$

即：误差经过q-learning更新公式的迭代变小了。最终Q Table会收敛到最优动作价值。

> 实验33: q-learning的实现：
>
>`code/q_learning/q_learning.py`

## 三、DQN

![dqn](pics/26.png)

> 实验34: DQN 的实现：
>
>`code/q_learning/dqn.py`

## 四、REINFORCE

基于价值的强化学习算法流派先告一段落，我们进入基于策略的强化学习算法流派的学习。

![dqn](pics/27.png)

我们的目标：最大化期望回报 

$$
J(\theta) = \mathbb{E}_{\pi_\theta}[G_t]
$$

也就是要算

$$
∇_θJ(θ)
$$

期望的展开式是：
$$
J(\theta) = \sum_{\tau} P(\tau|\theta) G(\tau) 
$$

这里 $P(\tau|\theta)$ 是这条轨迹发生的概率，它由三部分相乘构成：

$$
P(\tau|\theta) = \underbrace{\rho_0(S_0)}_{\text{初始状态概率}} \cdot \prod_{t=0}^{T} \underbrace{P(S_{t+1}|S_t, A_t)}_{\text{环境状态转移概率}} \cdot \prod_{t=0}^{T} \underbrace{\pi_\theta(A_t|S_t)}_{\text{策略网络概率}} 
$$

我们要求 $\nabla_\theta J(\theta)$：

$$
\nabla_\theta J(\theta) = \nabla_\theta \sum_{\tau} P(\tau|\theta) G(\tau) = \sum_{\tau} \nabla_\theta P(\tau|\theta) G(\tau) 
$$

我们引入一个纯代数恒等式：

$$
\nabla_\theta P = P \cdot \nabla_\theta \log P
$$

把这个恒等式代入：

$$
 \nabla_\theta J(\theta) = \sum_{\tau} P(\tau|\theta) \cdot \nabla_\theta \log P(\tau|\theta) \cdot G(\tau) 
$$

写成期望形式就是：

$$
 \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim P(\cdot|\theta)} [ \nabla_\theta \log P(\tau|\theta) \cdot G(\tau) ] 
$$

我们对刚才那个长长的 $P(\tau|\theta)$ 取 $log$（连乘变连加）：

$$
\log P(\tau|\theta) = \log \rho_0(S_0) + \sum_{t=0}^{T} \log P(S_{t+1}|S_t, A_t) + \sum_{t=0}^{T} \log \pi_\theta(A_t|S_t) 
$$

现在，对 $\theta$ 求梯度 $\nabla_\theta$：

$$
\nabla_\theta \log \rho_0(S_0) = 0
$$

$$
\nabla_\theta \log P(S_{t+1}|S_t, A_t) = 0
$$

（**环境状态转移概率（比如物理规律）不由网络参数决定**）

只剩下策略部分：

$$
\nabla_\theta \log \pi_\theta(A_t|S_t)
$$

所以

$$
\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t|S_t)
$$

把该结果代回期望：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau} \left[ \left( \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t|S_t) \right) \cdot G(\tau) \right] 
$$

因为 t 时刻的动作无法影响 t 时刻之前的奖励（因果律），所以交叉项期望为0。最终只剩下各自时间步的对应关系：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [ \sum_{t=0}^T\nabla_\theta \log \pi_\theta(A_t|S_t) \cdot G_t ] 
$$

下标的替换是因为：策略 $\pi_\theta$ 是决定轨迹分布 $P(\tau|\theta)$ 中唯一包含参数 $\theta$  的部分（环境转移 $P(S'|S,A)$ 和初始状态 $\rho_0$ 都不依赖 $\theta$）。所以，对轨迹分布求期望 $\mathbb{E}_{\tau \sim P(\cdot|\theta)}[\dots]$，在数学上就完全等价于对策略 $\pi_\theta$ 求期望 $\mathbb{E}_{\pi_\theta}[\dots]$。这两个符号指代的是同一个东西。

我们不需要、也没法推出 J 本身。回到J的定义，其对所有可能的轨迹 τ 求和，轨迹的数量那就多了去了， J(θ) 的解析表达式是永远写不出来的。它在坐标系里是一个无穷维的怪物。J的梯度同理。但是我们可以跑一条轨迹，用单次采样（蒙特卡洛）来近似！

我们用跑一条轨迹（蒙特卡洛采样）算出来的值，去估计（估算）真实的梯度：

$$
\nabla_\theta J(\theta) ≈  \sum_{t=0}^T\nabla_\theta \log \pi_\theta(A_t|S_t) \cdot G_t
$$

由此可以反推出Loss(求J的最大值，等价于求-J的最小值):

$$
\text{Loss} = - \sum_{t=0}^T \log \pi_\theta(A_t|S_t) \cdot G_t 
$$

更新公式为：

$$
\theta = \theta + \alpha  \sum_{t=0}^T\nabla_\theta \log \pi_\theta(A_t|S_t) \cdot G_t
$$

> 实验35: REINFORCE 的实现：
>
>`code/reinforce/reinforce.py`

## 五、