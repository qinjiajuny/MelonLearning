_ps:本文为记录参与Datawhale-7月吃瓜教程的学习笔记_

Task01——概览西瓜书+南瓜书1、2章
=====================

1 绪论
----

### 1.1 引言

通过根据已有的经验来预测判断西瓜“好坏”与否来引入机器学习的概念。

> **机器学习**：一门致力于研究如何通过计算的手段，利用经验来改善系统自身的性能。

  

### 1.2 基本术语

2.  **数据集（data set）**：一组记录的集合。
    

4.  **示例（instance）/样本（example）**：数据集中的一条对事件或对象的记录。
    

6.  **属性（attrribute）/特征（feature）**：能反映事件或对象在某方面的表现或性质的事情。
    

8.  **属性值（attribute value）**：属性上的取值，书中例子为“青绿” “乌黑” 。
    

10.  **属性空间（attribute space）/样本空间（sample space）/输入空间**：属性长成的空间，例如我们把"色泽" "根蒂" "敲声"作为三个坐标轴，则它们张成 一个用于描述西瓜的三维空间，每个西瓜都可在这个空间中找到自己的坐标位置.由于空间中的每个点对应一个坐标向量，因此我们也把…个示例称为一个 "特征向量" (feature vector).。
    

12.  **学习（learning）/训练（training）**：从数据中学得模型的过程。
    

14.  训练过程中使用的数据称为"**训练数据" (training data)**，其中每个样本称为一个**训练样本" (training sample**), 训练样本组成的集合称为"**训练集" (training set)**。
    

16.  **假设（hypothesis）**：学得模型对应了关于数据的某种潜在的规律，这种潜在规律自身，则称为"**真相**"或"**真实**" **(ground-truth)**，学习过程就是为了找出或逼近真相。
    

18.  关于示例结果的信息，例如"好瓜"，称为**"标记" (label)**; 拥有了标记信息的示例，则称为"**样例" (example)**。
    

20.  **分类（classification）**：欲预测的是离散值，例如"好瓜" "坏瓜"，对只涉及两个类别的"**二分类" (binary classification)任务**，通常称其中一个类为 **"正类" (positive classification）**另一个类为**"反类" (negative classification)**; 涉及多个类别时，则称为**"多分类" (multi-class classificatio）任务**。
    

22.  **回归（regression）**：欲预测的是连续值，例如西瓜成熟度 0.95、 0.37。
    

24.  ![](https://img-blog.csdnimg.cn/2021071316535287.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)
    

>  台大李宏毅老师在其深度学习的课上说过：“_机器学习任务的本质就是找函数关系 f（x）_"

 13.

![](https://img-blog.csdnimg.cn/20210713165728854.png)

14.**聚类（clustering）**：即将训练集中的西瓜分成若干组，每组称为一个"簇" (cluster); 这些自动形成的簇可能对应一些潜在的概念划分，例如”浅色瓜“ ”深色瓜“，甚至”本地瓜“ ”外地瓜“，有助于了解数据内在的规律，能为更深入地分析数据建立基础。

15.根据训练数据是否拥有标记信息，学习任务大致划为两类：**”监督学习（supervised learning）“**和**”无监督学习（unsupervised learning）“**，分类和回归是监督学习的代表，聚类则是无监督学习的代表。

> 监督学习其实就是我们对输入样本经过模型训练后有明确的预期输出，如图像识别分类猫狗中，我们一开始标注好猫和狗，给一张新的图像，预期就是输出其中的猫和狗。结合西瓜的例子，监督学习就是经过模型训练后会分为好瓜或坏瓜。
> 
> 非监督学习就是我们对输入样本经过模型训练后得到什么输出完全没有预期。而非监督学习则会将西瓜聚类为几种我们之前没有明确定义的瓜，如“浅色瓜”“外地瓜”。

**16.泛化（generalization）能力**：学得模型适用于新样本的能力。机器学习的目标是使学得的模型能很好地适用于"新样本"， 而不是仅仅在训练样本上工作得很好，具有强泛化能力的模型能很好地适用于整个样本空间。

### 1.3 假设空间

2.  科学推理的两大基本手段：**归纳（induction）与演绎（deduction）**，前者是从特殊到一般的**"泛化" (generalization)过程**，即从具体的事实归结出一般性规 律;后者则是从一般到特殊的**"特化" (specializatio）过程**，即从基础原理推演出具体状况。"从样例中学习"显然是一个归纳的过程，因此亦称 **"归纳学习" (inductive learning).**
    

4.  归纳学习有狭义与广义之分，广义的归纳学习大体相当于从样例中学习， 而狭义的归纳学习则要求从训练数据中学得概念(concept)，因此亦称为"**概念学习**"或"概念形成”。
    

6.  ![](https://img-blog.csdnimg.cn/20210713170107517.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)
    
    > 65种可能的计算：[西瓜书《机器学习》笔记\-\-假设空间 \- 李林超博客 ~ 个人博客 (lilinchao.com)](https://www.lilinchao.com/archives/909.html)
    

  

### 1.4 归纳偏好（暂时跳过）

* * *

  

2 模型评估与选择
---------

### 2.1 经验误差与过拟合

2.  **错误率（error rate）**：分类错误的样本数占样本总数的比例。在 m 个样本中有 α 个样本分类错误，则错误率 E= α/m。
    

4.  **精度（accuracy）**：精度=1一错误率，即1 一 α/m。
    

6.  **误差（error）**：学习器的实际预测输出与样本的真实输出之间的差异，学习器在训练集上的误差称为"**训练误差" (training error)**或**”经验误差" (empirical error)** ，在新样本上的误差称为"**泛化误差" (generalization error).**
    

8.  **过拟合（overfitting）与欠拟合（underfitting）**：我们实际希望的是在新样本上能表现良好的学习器，而**过拟合**是学习器把训练样本学的“太好了”的时候，把一些训练样本本身的特点当作所有潜在样本都会有的一般性质，导致泛化能力下降。比如说下图（_<机器学习>P24_）中给的训练样本中大多数是由锯齿的形状，学习器就把这一性质当作是所有的叶子都有的性质，当预测没有锯齿的叶子就会输出不是叶子的错误预测。而**欠拟合**则是与过拟合相反，对训练样本的一般性质都尚未学好，只在训练样本中学到绿色是叶子的一般性质，在测试中遇到一棵树时也误以为是叶子。
    

  

![](https://img-blog.csdnimg.cn/20210713170324356.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

5.过拟合常见的情况是学习能力过于强大，把训练样本所包含的不太一般的特性都学到了，欠拟合则是学习能力低下。过拟合是无法避免的，也是机器学习面临的关键障碍，欠拟合是比较容易克服的。

### 2.2 评估方法

**测试集应该尽可能 与训练集互斥，即测试样本尽量不在训练集中出现，未在训练过程中使用过。**

**2.2.1 留出法（hold-out）**

将数据集D划分为两个互斥的集合，其中一个 集合作为训练集 S，另一个作为测试集 T， 即 D=S交T，S并T=空集。在 S 上训练出模型后，用T来评估其测试误差，作为对泛化误差的估计。

训练/测试集的划分要尽可能保持数据分布的一致性，避免因数据划分过程引入额外的偏差而对最终结果产生影响。常见做法是将大约 2/3 ~4/5 的样本用于训练，剩余样本用于测试。

**2.2.2 交叉验证法 (cross validation)**

 将数据集 D 划分为 k 个大小相似的互斥子集,然后每次用 k-1 个子集的并集作为训练集,余下的那个子集作为测试集;这样就可获得 k 组训练/测试集，从而可进行 k 次训练和测试，最终返回k 个测试结果的均值。通常把交叉验证法称为**“k折交叉验证”（k-fold cross validation）。**k的取值最常为10。

![](https://img-blog.csdnimg.cn/20210713210359437.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

**留一法(Leave-One-Ot比，简称 LOO)**，只留一个样本来验证，优势是在大多数情况训练出来的模型与期望评估用数据集D训练出来的模型很相似，即留一法的评估结果较为准确，劣势是数据集较大时的计算开销是巨大的且未必就比其他评估方法准确。

**2.2.3 自助法（bootstrapping）**

为解决留出法和交叉验证法因训练样本规模不同所带来的估计偏差和留一法的巨大的计算开销，提出了**自助法**，**自主采样**亦称“可重复采样”或“有放回采样”，类似于有放回地“摸球”。

![](https://img-blog.csdnimg.cn/20210713220211295.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/20210713220322467.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/20210713220335398.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

自助法的好处是适用于数据集较小，难以有效划分训练/测试集的情况，且能从初试数据集中产生多个不同的训练集，有利于集成学习。不好的地方时改变了初始数据集的分布会引入估计偏差。故在数据集量足够时仍是用留出法和交叉验证法。

**2.2.4 调参与最终模型**

**调参（parame tuning）**：对算法参数进行设定。现实常用的做法是，对每个参数选定一个范围和变化步长，已达到在计算开销和性能估计上的折中结果。

**验证集（validation set）**:模型评估与选择中用于评估测试的数据集。

通俗地理解训练集、验证集、测试集：

*   训练集->平时做的题目
    

*   验证集->时不时的一些小测验
    

*   测试集->最终的考试
    

### 2.3 性能度量

**性能度量（performance measure）**：衡量模型泛化能力的评价标准，模型的“好坏”是相对的，更取决与任务需求。

**2.3.1 错误率与精度**

分类任务中最常用的两种，二分类与多分类任务都适用。错误率是分类错误的样本数占样本总数的比例，精度则是分类正确的样本数占样本总数的比例。

**2.3.2 查准率、查全率与F1**

**查准率（precision）**：又称准确率，如挑出的西瓜中好瓜的比例是多少，Web搜索时检索出的信息有多少比例是用户感兴趣的。

**查全率（recall）**：又称召回率，如所有好瓜被挑了出来的比例是多少，Web搜索时用户感兴趣的信息有多少被检索出来了。

**关系**：两者是一对矛盾的度量，一般来说，一方偏高另一方则会偏低，如将所有西瓜都选上，那么所有的好瓜也必然都被选上了，但这样查准率就会较低;若希望选出 的瓜中好瓜比例尽可能高，则可只挑选最有把握的瓜，但这样就难免会漏掉不少好瓜，使得查全率较低。

对于二分类问题，可将样例根据其真实类别与学习器预测类别的组合划分为**真正例TP(true positive)、假正例FP(false positive)、真反倒TN(true negative)、 假反例FN(false negative)**四种情形。**P=真正例/（预测）正例，R=真正例/（真实）正例。**

  

![](https://img-blog.csdnimg.cn/20210713220428287.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

以查准率为纵轴、查全率为横轴作图，就得到了查准率—查全率曲线，简称 "**P-R曲线**"，显示该曲线的图称为 "**P-R图**" 。

![](https://img-blog.csdnimg.cn/20210713220456265.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

  

*   若一个学习器的 P-R 曲线被另一个学习器的曲线完全"包住" ， 则可断言后者的性能优于前者， 例如图 2.3 中学习器 A 的性能优于学习器 C
    

*   如果两个学习器的 P-R 曲线发生了交叉7 例如图 2.3 中的 A 与 B ， 这时一个比较合理的判据 是比较 P-R 曲线下面积的大小，它在一定程度上表征了学习器在查准率和查全率上取得相对"双高"的比例，由此引入**"平衡点 " (Break-Event Point，简称 BEP)**就是这样一个度量，它是"查准率=查全率"时的取值，例如图 2.3 中学习器 C 的 BEP 是 0.64，而基于 BEP 的比较，可认为学习器 A 优于 B。
    

**F1度量**：

![](https://img-blog.csdnimg.cn/20210713220548222.png)

 在一些应用中，对查准率和查全率的重视程度有所不同.引入F1 度量的一般形式 F1ß， 能让我们表达出对查准率/查全率的不同偏好。

![](https://img-blog.csdnimg.cn/20210713220635334.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NvcmFjYQ==,size_16,color_FFFFFF,t_70)

在n个二分类混淆矩阵上综合考虑查准率和查全率有两种方法：

*   在各混淆矩阵分别计算出查准率和查全率，在计算平均值，得到"**宏查准率" (macro-P)**、 "**宏查全率" (macro-R**)，以及相应的"**宏F1" (macro-F1)**
    

*   先将各泪淆矩阵的对应元素进行平均，得到 TP、 FP、 TN、 FN 的 平均值,，再基于这些平均值计算出"**微查准率"(micro-P**)、 "**微查全率" (micro-R)**和"**微F1" (micro-F1)**
    

* * *

2.33 至2.5暂时跳过
-------------
