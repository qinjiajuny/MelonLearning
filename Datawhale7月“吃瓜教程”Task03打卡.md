_ps:本文为记录参与Datawhale-7月吃瓜教程的学习笔记_

**_pss:文章所有PPT截图来自于：Datawhale吃瓜教程（[https://www.bilibili.com/video/BV1Mh411e7VU](https://www.bilibili.com/video/BV1Mh411e7VU)），记得一键三连~_**

**目录**

[Task03 详读西瓜书+南瓜书第4章](#Task03%20%E8%AF%A6%E8%AF%BB%E8%A5%BF%E7%93%9C%E4%B9%A6%2B%E5%8D%97%E7%93%9C%E4%B9%A6%E7%AC%AC4%E7%AB%A0)

[1 基本流程](#1%20%E5%9F%BA%E6%9C%AC%E6%B5%81%E7%A8%8B)

[2 划分选择](#2%20%E5%88%92%E5%88%86%E9%80%89%E6%8B%A9)

[2.1 信息增益](#2.1%20%E4%BF%A1%E6%81%AF%E5%A2%9E%E7%9B%8A)

[2.2 增益率](#2.2%20%E5%A2%9E%E7%9B%8A%E7%8E%87)

[2.3 基尼指数](#2.3%20%E5%9F%BA%E5%B0%BC%E6%8C%87%E6%95%B0)

[3 剪枝处理](#3%20%E5%89%AA%E6%9E%9D%E5%A4%84%E7%90%86)

[3.1 预剪枝](#3.1%20%E9%A2%84%E5%89%AA%E6%9E%9D)

[3.2 后剪枝](#3.2%20%E5%90%8E%E5%89%AA%E6%9E%9D)

[4 连续与缺失值](#4%20%E8%BF%9E%E7%BB%AD%E4%B8%8E%E7%BC%BA%E5%A4%B1%E5%80%BC)

[4.1 连续值处理](#4.1%20%E8%BF%9E%E7%BB%AD%E5%80%BC%E5%A4%84%E7%90%86)

[4.2 缺失值处理](#4.2%20%E7%BC%BA%E5%A4%B1%E5%80%BC%E5%A4%84%E7%90%86)

[5 多变量决策树](#5%20%E5%A4%9A%E5%8F%98%E9%87%8F%E5%86%B3%E7%AD%96%E6%A0%91)

Task03 详读西瓜书+南瓜书第4章
===================

1 基本流程
------

**决策树**是基于树结构来进行决策的，正好对应的是人类面对决策时的处理机制，决策过程的最终结论对应了我们所希望的判定结果。

![](https://img-blog.csdnimg.cn/20210722193751874.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

  

**算法原理**

![](https://img-blog.csdnimg.cn/20210722193324408.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

 决策树与数据结构中的树类似，包含一个根结点、若干内部结点和若干个叶结点；其中：

*   **叶结点：**对应于决策结果
    

*   **其他结点（非根的父节点）：**对应于一个属性测试
    

*   **根结点：**包含样本全集
    

 决策树**学习的目的**是为了产生一棵泛化能力强，即处理未见示例能力强的决策树。

** 基本流程**![](https://img-blog.csdnimg.cn/20210722193412452.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

 在决策树算法中，有三种情形会导致递归返回：

2.   当前结点包含的样本全属于同一类别，**无需划分**
    

4.  当前属性集为空，或是所有样本在所有属性上取值相同，**无法划分**
    

6.  当前结点包 含的样本集合为空，**不能划分**
    

针对“情形2”，把当前结点标记为叶结点，井将其类别设定为该结点所含样本最多的类别，利用的是当前结点的**后验分布**；

针对“情形3”，同样把当前结点标记为叶结点，但将其类别设定为其父结点所含样本最多的类别，利用的是把父节点的样本分布作为当前结点的**先验分布**。

2 划分选择
------

### 2.1 信息增益

**信息熵（information entropy）**：度量样本集合纯度最常用的一种指标。假定当前样本集合DD中第k类样本所占的比例为![p_k(k=1,2,\dots,|\mathcal{Y}|)](https://latex.codecogs.com/gif.latex?p_k%28k%3D1%2C2%2C%5Cdots%2C%7C%5Cmathcal%7BY%7D%7C%29)则D的信息熵表示为：

![](https://img-blog.csdnimg.cn/20210722213746221.png)

Ent(D)值越小，则D的纯度越高。Ent(D)的最小值为0，最大值为![log_{2}|y|](https://latex.codecogs.com/gif.latex?log_%7B2%7D%7Cy%7C)。举例子理解：

![](https://img-blog.csdnimg.cn/20210722214133775.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

严谨的数学证明详见：Datawhale社区的南瓜书的[第4章 决策树 (datawhalechina.github.io)](https://datawhalechina.github.io/pumpkin-book/#/chapter4/chapter4) 的**4.1。**

**信息增益：**假定使用属性a对样本集D进行划分，产生了v个分支节点，v表示其中第v个分支节点，其中第 v 个分支结点包含了 D 中所有在 属性 α 上取值为![a^{v}](https://latex.codecogs.com/gif.latex?a%5E%7Bv%7D)的样本, 记为![D^{v}](https://latex.codecogs.com/gif.latex?D%5E%7Bv%7D),根据上式子可以计算出用属性a对样本集D进行划分所获得的的“信息增益”（information gain）。

![](https://img-blog.csdnimg.cn/2021072221504454.png)

 一般来说，信息增益越大，则意味着使用属性 α 来进行划分所获得的"纯度提升"越大。著名的**ID3 决策树**学习算法就是以信息增益为准则来选择划分属性的。

![](https://img-blog.csdnimg.cn/20210722215414830.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/20210722215426711.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/20210722215439213.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

信息增益越大，意味着在当前条件下，信息的不确定性减小的越多，不确定性越低，在本场景中就是样本的纯度越高。

### 2.2 增益率

为了平衡信息增益准则倾向于可取数目较多的属性这一偏好带来的不利影响，著名的C4.5决策树算法使用**“增益率”（gain ratio）**来选择最优划分属性**。增益率定义为：**

![](https://img-blog.csdnimg.cn/20210722235439976.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)称为属性 α 的"固有值" (intrinsic value) ， 属性 α 的可能取值数目越多(即 V 越大)，则 IV(α) 的值通常会越大。

  

### 2.3 基尼指数

![](https://img-blog.csdnimg.cn/20210722222604895.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

 ![](https://img-blog.csdnimg.cn/20210722222755508.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

  

![](https://img-blog.csdnimg.cn/20210722222612958.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

3 剪枝处理
------

**剪枝(pruning)**是决策树学习算法对付"过拟合"的主要手段.在决策树学习中，为了尽可能正确分类训练样本，结点划分过程将不断重复，有时会造成决策树分支过多，这时就可能因训练样本学得"太好"了，即过拟合.因此可通过主动去掉一些分支来降低过拟合的风险。

### 3.1 预剪枝

**预剪枝（prepruning）**的定义就是在决策树生成的过程中，对每个结点在划分前先进行估计，若当前结点的划分不能带来决策树泛化性能的提升，则停止划分，并将当前结点标记为叶子结点。

预剪枝的**好处**是使得决策树的很多分支都没有"展开“，可以降低过拟合的风险；还显著减少了决策树的训练时间和测试时间的开销。但**坏处**是有有些分支的当前划分虽不能提升泛化性能、甚至可能导致泛化性能暂时下降，但在其基础上进行的后续划分却有可能导致性能显著提高，还有就是预剪枝基于"贪心"本质禁止这些分支展开，会带来欠拟合的风险。

### 3.2 后剪枝

**后剪枝（postpruning）**是发生于决策树生成完成以后，从由底至上考虑每个非叶节点，若将该结点替换为叶节点能够给决策树带来泛化性能的提升的话，则将该结点替换为叶节点。

因为后剪枝是在一棵完整的决策树的基础上从底至上考虑每个非叶节点，通常相对于预剪枝决策树保留了更多的分支，其欠拟合风险很小，泛化性能往往也优于预剪枝决策树，不过训练时间开销会较大。

4 连续与缺失值
--------

到目前为止我们仅讨论了基于离散属性来生成决策树，现实学习任务中常会遇到连续属性，有必要讨论如何在决策树学习中使用连续属性。 最简单的策略是采用二分法(bi-partition)对连续属性进行处理，这正是 C4.5 决策树算法中采用的机制。（划分为：>a or <a ）

  

———————————————————————————————————————————

（待补充......）
===========

### 4.1 连续值处理

### 4.2 缺失值处理

5 多变量决策树
--------
