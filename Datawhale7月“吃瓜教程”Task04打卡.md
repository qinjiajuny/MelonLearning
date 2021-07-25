_ps:本文为记录参与Datawhale-7月吃瓜教程的学习笔记_

**_pss:文章所有PPT截图来自于：Datawhale吃瓜教程（[https://www.bilibili.com/video/BV1Mh411e7VU](https://www.bilibili.com/video/BV1Mh411e7VU)），记得一键三连~_**

  

**目录**

[Task04 详读西瓜书+南瓜书第5章](#Task04%20%E8%AF%A6%E8%AF%BB%E8%A5%BF%E7%93%9C%E4%B9%A6%2B%E5%8D%97%E7%93%9C%E4%B9%A6%E7%AC%AC5%E7%AB%A0)

[1 神经元模型](#1%20%E7%A5%9E%E7%BB%8F%E5%85%83%E6%A8%A1%E5%9E%8B)

[2 感知机与多层网络](#2%20%E6%84%9F%E7%9F%A5%E6%9C%BA%E4%B8%8E%E5%A4%9A%E5%B1%82%E7%BD%91%E7%BB%9C)

[3 误差逆传播（BP）算法](#3%20%E8%AF%AF%E5%B7%AE%E9%80%86%E4%BC%A0%E6%92%AD%EF%BC%88BP%EF%BC%89%E7%AE%97%E6%B3%95)

[4 全局最小与局部最小](#4%20%E5%85%A8%E5%B1%80%E6%9C%80%E5%B0%8F%E4%B8%8E%E5%B1%80%E9%83%A8%E6%9C%80%E5%B0%8F)

[5 深度学习](#5%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0)

  

Task04 详读西瓜书+南瓜书第5章
===================

1 神经元模型
-------

**定义**：我们在机器学习中所谈论的神经网络一般指的是“人工神经网络”，是机器学习与神经网络两个学科的交叉部分。所谓神经网络，目前用得最广泛的一个定义是“神经网络是由具有适应性的简单单元组成的广泛并行互连的网络，它的组织能够模拟生物神经系统对真实世界物体所做出的交互反应”。

神经网络中最基本的单元是**神经元（neuron）**模型。即上述定义中的“简单单元”。

在生物神经网络中，每个神经元与其他神经元相连，当它"兴奋"时，就会向相连的神经元发送化学物质，从而改变这些神经元内的电位;如果某神经元的电位超过了一个"阔值" (threshold)， 那么它就会被激活，即 "兴奋" 起来，向其他神经元发送化学物质。下图则将上述情形抽象化，也就是著名的“**M-P神经元模型**”。

![](https://img-blog.csdnimg.cn/20210725202951759.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

在这个模型中，神经元接收到来自 n 个其他神经元传递过来的输入信号,这些输入信号通过带权重的连接(connection)进行传递，神经元接收到的总输入值将与神经元的阀值进行比较，然后通过"激活函数" (activation function) 处理以产生神经元的输出。

神经元模型最理想的激活函数是**阶跃（sgn）函数**，它将神经元输入值与阈值的差值映射为输出值“1”或“0”，若差值大于等于零输出1，对应兴奋；若差值小于零则输出0，对应抑制。

然而阶跃函数不连续，不光滑，故在M-P神经元模型中，实际是采用**Sigmoid函数**来作为激活函数的， Sigmoid函数将较大范围内变化的输入值挤压到 (0,1) 输出值范围内，所以也称为挤压函数（squashing function）。

![](https://img-blog.csdnimg.cn/20210725205703234.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

 ![](https://img-blog.csdnimg.cn/20210725210001872.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

  

 把许多个这样的神经元按一定的层次结构连接起来，就得到了神经网络。

> 西瓜书P98左侧小字：![](https://img-blog.csdnimg.cn/20210725205818270.png)
> 
>  10个神经元中每个神经元都有9个连接劝和1个阈值，可见图5.1。

2 感知机与多层网络
----------

**感知机(Perceptron)**由两层神经元组成， 如图 5.3 所示，输入层接收外界输入信号后传递给输出层， 输出层是 M-P神经元，亦称"阔值逻辑单元" (threshold logic unit)。

![](https://img-blog.csdnimg.cn/20210725210148911.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

 ![](https://img-blog.csdnimg.cn/20210725210204229.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

 ![](https://img-blog.csdnimg.cn/20210725210235272.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

 ![](https://img-blog.csdnimg.cn/20210725210245427.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

 ![](https://img-blog.csdnimg.cn/20210725210256727.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

 ![](https://img-blog.csdnimg.cn/2021072521030618.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

 感知机只有输出层神经元进行激活函数处理，即只拥有 一层功能神经元(functional neuron)，其学习能力非常有限。事实上，上述与、或、 非问题都是**线性可分(linearly separable)**的问题。但异或问题时非线性可分问题。

![](https://img-blog.csdnimg.cn/20210725210700914.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

 而要解决非线性可分问题，可考虑使用多层神经元。如图5.5的简单两层感知机来解决异或问题。

![](https://img-blog.csdnimg.cn/20210725210817238.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

在图5.5(a)中， 输出层与输入层之间的一 层神经元，被称为隐层或隐含层(hidden layer)，隐含层和输出层神经元都是拥有激活函数的功能神经元。

**多层前馈神经网络"(multi-layer feedforward neural networks）：**每层神经元与下一层神经元全互连，神经元之间不存在同层连接，也不存在跨层连接。

![](https://img-blog.csdnimg.cn/20210725211058797.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

其中输入层神经元接收外界输入，隐层与输出层神经元对信弓进行加工，最终结果由输出层来输出。换言之，输入层神经元仅是接受输入，不进行函数处理，隐层与输出层包含功能神经元来进行处理工作。

**神经网络的学习过程，就是根据训练数据来调整神经元之间的 "连接权" (connection weight) 以及每个功能神经元的阈值。换言之，神经网络所要"学"到的东西，蕴涵在连接权与阈值中。**

3 误差逆传播（BP）算法
-------------

**误差逆传播（error BackPropagation，简称BP）算法**是具有强大学习能力的多层网络算法中的最杰出代表，也是迄今为止最成功的神经网络学习算法。

![](https://img-blog.csdnimg.cn/20210725211458134.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

 ![](https://img-blog.csdnimg.cn/20210725211516464.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

上图的网络中有 (d + l 十 1) q 十 l个参数需确定:输入层到隐层的 d xq 个权值、 隐层到输出层的 q x l 个权值、 q 个隐层神经元的阈值、 l 个输出层神经元的阈值。![](https://img-blog.csdnimg.cn/20210725211532799.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

 ![](https://img-blog.csdnimg.cn/20210725211543206.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

**BP 是一个迭代学习算法，在迭代的每一轮中采用广义的感知机学 习规则对参数进行更新估计。**

任意参数![v](https://latex.codecogs.com/gif.latex?v)的更新估计式为：

![](https://img-blog.csdnimg.cn/20210725212127991.png)

** BP算法的工作流程：**

![](https://img-blog.csdnimg.cn/20210725212357319.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)  
 

> ** 累积BP算法与标准BP算法：**
> 
> *   标准BP算法每次更新只针对单个样例，参数更新频繁，而且针对不样例进行更新的效果可能出现"抵消”现象
>     
> 
> *   累积BP算法在读取完整个训练集后才对参数进行更新，参数更新频率较低
>     
> 
> 累积误差下降到一定程度之后，进 一步下降会非常缓慢，这时标准 BP 往往会更快获得较好的解，尤其是在训练集 D 非常大时更明显。  
>  
> 
> **缓解BP网络的过拟合：**
> 
> *   **早停(early stopping):** 将数据分成训练集和验证集，若训练集误差降低但验证集误差升高则停止训练，同时返回具有最小验证集误差的连接权和阑值
>     
> 
> *   **正则化（regularization）：**是在误差目标函数中增加一个用于描述网络复杂度的部分
>     

  

4 全局最小与局部最小
-----------

模型学习的过程实质上就是一个寻找最优参数的过程，例如BP算法试图通过随机梯度下降来寻找使得累积经验误差最小的权值与阈值，在谈到最优时，一般会提到局部极小（local minimum）和全局最小（global minimum）。

> 局部极小解：参数空间中的某个点其邻域点的误差函数值均不小于该点的函数值
> 
> 全局最小解：参数空间中所有点的误差函数值均不小于该点的误差函数值

![](https://img-blog.csdnimg.cn/20210725213435510.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

 现实任务中常用于“跳出”局部极小的策略：

2.  以多组不同参数值初始化多个神经网络，按标准方法训练后，取其中误差最小的解作为最终参数
    

4.  使用 "模拟退火" (simulated annealing) 技术
    

6.  使用随机梯度下降
    

8.  遗传算法
    

5 深度学习
------

理论上来说，参数越多的模型复杂度就越高，容量（capability）也就越大，从而能完成更复杂的学习任务。但复杂的模型训练效率低，容易陷入过拟合。而随着计算机算例的提高和训练数据的大幅增加，训练的低效性和过拟合的风险都得到相应的缓解。深度学习（deep learning）正是一种极其复杂而强大的模型。

**增大模型复杂度的两个办法：**

一是增加隐层的数目，二是增加隐层神经元的数目。前者更有效一些，因为它不仅增加了功能神经元的数量，还增加了激活函数嵌套的层数。但是对于多隐层神经网络，经典算法如标准BP算法往往会在误差逆传播时发散（diverge），无法收敛达到稳定状态。

**有效地训练多隐层神经网络的两种方法：**

*   **无监督逐层训练（unsupervised layer-wise training）**：每次训练一层隐节点，把上一层隐节点的输出当作输入来训练，本层隐结点训练好后，输出再作为下一层的输入来训练，这称为**预训练（pre-training）**。全部预训练完成后，再对整个网络进行**微调（fine-tuning）**训练。典型例子是**深度信念网络（deep belief network，简称DBN）**。这种做法其实可以视为把大量的参数进行分组，先找出每组较好的设置，再基于这些局部最优的结果来训练全局最优。
    

*   **权共享（weight sharing）**：令一组神经元使用相同的连接权，典型的例子是**卷积神经网络（Convolutional Neural Network，简称CNN）**。这样做可以大大减少需要训练的参数数目。
    

![](https://img-blog.csdnimg.cn/20210725214158286.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

**深度学习可以理解为是一种特征学习或者表示学习**，无论是DBN还是CNN，都是通过多层处理，逐渐将初始的“低层”特征表示转化为“高层”特征表示，从而使得最后可以用简单的模型来完成复杂的学习任务。

在传统任务中，样本的特征需要人类专家来设计，这称为**特征工程（feature engineering）**。特征好坏对泛化性能有至关重要的影响。而深度学习为全自动数据分析带来了可能，可以自动产生更好的特征。这使机器学习向“全自动数据分析”又前进了一步。

  

* * *
