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

