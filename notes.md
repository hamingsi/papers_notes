# notes

### 十问

Q1 论文试图解决什么问题？

Q2 这是否是一个新的问题？

Q3 这篇文章要验证一个什么科学假设？

Q4 有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？

Q5 论文中提到的解决方案之关键是什么？

Q6 论文中的实验是如何设计的？

Q7 用于定量评估的数据集是什么？代码有没有开源？

Q8 论文中的实验及结果有没有很好地支持需要验证的科学假设？

Q9 这篇论文到底有什么贡献？

Q10 下一步呢？有什么工作可以继续深入？

## Attention Spiking Neural Networks

### Abstract

#### module

solve the performance gap between SNNs and ANNs，引入了Attention模块，并且编成Multi-dimensional Attention(MA)   temporal dimension, channel dimension, as well as spatial dimension separately or simultaneously

并且模块plug-and-play(即插即用)，提出了MA-SNN的端到端模型

#### dataset performance

数据集主要是DVS128 Gesture/Gait action recognition and ImageNet-1k image classification

与Res-ANN-104相比,, the performance gap becomes -0.95/+0.21 percent and has 31.8×/7.4× better energy efficiency

#### theory

we theoretically prove that the spiking degradation or the gradient vanishing, which usually holds in general SNNs, can be resolved by introducing the block dynamical isometry theory

这个理论提到对谱范数的控制The theory proposes that by controlling the spectral norm of the weight matrices in each layer of the network, it is possible to achieve dynamical isometry. 

块动态等距理论表明，通过正确初始化神经网络的权重，可以在前向和后向传播过程中保持稳定的信号幅度。这有助于缓解梯度消失或爆炸问题，从而使深度网络的训练更加稳定和高效。

### Introduction

将attention模块：apply the attention as an auxiliary unit

#### problems

keep the neuromorphic computing characteristic of SNNs, which is the basis of SNN's energy efficiency，implementing the attention while retaining SNN's event-driven is the primary consideration

SNNs are used to process various applications

二值脉冲binary spiking activity makes deep SNNs suffer from spike degradation  and gradient vanishing

#### advantages of MA-SNN

sparser spiking responses and incurs better performance and energy efficiency concurrently

new spiking response visualization method

solve the degradation problem

可以使用MS-Res-SNN方式训练非常深的Att-Res-SNN

#### Contributions

##### Multi-dimensional Attention SNN

temporal: when channel: what spatial:where

##### Understanding and Visualizing of Attention

###### effectiveness

proper focusing

###### effciency

improvement of sparsity (inhibiting the membrane potentials)

##### Gradient Norm Equality of Att-Res-SNN

梯度范数相等

将注意力添加到MS-Res-SNN，解决一般SNN中的退化问题

