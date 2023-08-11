### Binary Neural Networks: A Survey

#### Abstract

二值化不可避免地会造成严重的信息丢失

###### task

图像分类、目标检测和语义分割

#### Introduction

###### approaches for compressing the deep networks can be classified into five categories

压缩网络就是这5种方法

参数剪枝和量化主要侧重于通过去除冗余/非关键参数或压缩参数空间（例如，从浮点权重到整数权重）来分别消除模型参数中的冗余。

低秩分解应用矩阵/张量分解技术使用小尺寸的代理来估计信息参数。

基于压缩卷积滤波器的方法依赖于精心设计的结构卷积滤波器来降低存储和计算复杂度。

知识蒸馏方法试图提取更紧凑的模型来重现更大网络的输出。

##### 基于量化压缩的方式，BNN可以使用轻量级的XNOR-Bitcount计算代替，实现高效和节能

通过二值化技术，通过将层切换到全精度或1位，可以很容易地验证层的重要性。

如果二值化某些层后性能大幅下降，我们可以得出结论，该层位于网络的关键路径上。

此外，重要的是要找出全精度模型和二值化模型是否以与可解释的机器学习视图相同的方式工作。

##### 揭示模型二值化的行为，并进一步解释模型的鲁棒性与深度神经网络结构之间的联系

在神经网络的前向传播过程中，噪声不断放大，BNN有助于减少这种情况

##### 基于BNN的研究还可以帮助我们分析深度神经网络的结构是如何工作的

###### Bi-Real Net additional shortcuts (Bi-Real) 

前向和反向两方面，前向的时候可以传到更深的层，反向梯度传播的时候避免了梯度消失的情况

基于 BNN 的分析和实验，他们表明神经元的数量比位宽更重要，并且可能不需要在深度神经网络中使用实值神经元，这与生物神经网络原理相似。

此外，减少某一层的位宽来探索其对准确性的影响是研究深度神经网络可解释性的一种有效方法。有许多工作来探索不同层对二值化的敏感性

第一层和最后一层应该以更高的精度作为共识，这意味着这些层在神经网络的预测中起着更重要的作用

##### the nature of binary neural networks

naive binarization without optimizing the quantization function

the optimized binarization including 

minimizing quantization error

improving the loss function

reducing the gradient error

#### Preliminary

the full-precision convolutional neural network
$$
\mathbf{z}=\sigma(\mathbf{w}\otimes\mathbf{a})
$$
BNN forward propagation
$$
Q_{w}(\mathbf{w})=\alpha\mathbf{b_{w}},\quad Q_{a}(\mathbf{a})=\beta\mathbf{b_{a}}
$$
where $b_w$ and $b_a$ are the tensor of binary weights (kernel) and binary activations, corresponding scalars α and β.
$$
\mathbf{z}=\sigma(Q_{w}(\mathbf{w})\otimes Q_{a}(\mathbf{a}))=\sigma(\alpha\beta(\mathbf{b_{w}\odot b_{a}}))
$$
XNOR-bitcount

![fig1](https://github.com/hamingsi/papers_notes/raw/81a68a30e8a7aeb7337a92bb64e5a5adecf7abde/pictures/BNN/fig1.png)

use STE function(sign function vanishes) to BP
$$
\mathtt{clip}(x,-1,1)=\max(-1,\min(1,x)).
$$

#### Binary Neural Networks

##### Naive Binary Neural Networks

$$
w_b=\begin{cases}&+1,&\text{with probability }p=\hat{\sigma}(w)\\&-1,&\text{with probability }1-p\end{cases}
$$

where $\sigma$ is the“hard sigmoid”function:

$$
\hat{\sigma}(x)=\mathsf{clip}(\frac{x+1}2,0,1)=\operatorname*{max}(0,\operatorname*{min}(1,\frac{x+1}2))
$$

##### Optimization Based Binary Neural Networks

Naive Binary Neural Networks inevitably suffer the accuracy loss for the wide tasks

###### Minimize the Quantization Error

the weights in BWN are binarized to {−α, +α}
$$
\min_{\alpha,\mathbf{b_{w}}}\|\mathbf{w}-\alpha\mathbf{b_{w}}\|^{2}
$$
Wide Reduced-Precision Networks (WRPN)  that also minimize the quantization error in a similar way to XNOR-Net, but **increase the number of filters in each layer.**

(HORQ)generates the final quantized activation by a **linear combination of the approximation in each recursive step.**

ABC-Net  that **linearly combines multiple binary weight matrices and scaling factors** to fit the full-precision weights and activations, which can largely reduce the information loss caused by binarization.

two-step quantization (TSQ)
$$
\min_{\alpha,\mathbf{b_w}}\quad\|\mathbf{z}-Q_a\left(\alpha(\mathbf{a}\odot\mathbf{b_w})\right)\|_2^2
$$
PArameterized Clipping Activation (PACT) [74] with a **learnable upper bound** for the activation function.

###### Improve the Network Loss Function

finding the desired network loss function that can guide the learning of the network parameters with restrictions brought by binarization.

distribution loss
$$
\mathcal{L}_{total}=\mathcal{L}_{CE}+\lambda\mathcal{L}_{DL}
$$
where $L_{CE}$ is the common cross-entropy loss for training deep neural networks,$L_{DL}$is the distribution loss for learning the proper binarization

The Apprentice method  trains a low-precision student network using a **well-trained, full precision, large-scale** teacher network, using the following loss function:
$$
\mathcal{L}\left(x;\mathbf{w}^T,\mathbf{b}_\mathbf{w}^S\right)=\alpha\mathcal{H}\left(y,p^T\right)+\beta\mathcal{H}\left(y,p^S\right)+\gamma\mathcal{H}\left(z^T,p^S\right)
$$
where $w^T$ and $b^S_w$ are the full-precision weights of the teacher model and binary weights of the student (apprentice) model respectively, y is the label for samplex, H(·) is the soft and hard label loss function between the teacher and apprentice model, and α, β, γ are the weighting factors, $p^T$ and $p^S$ are the predictions of the teacher and student model, respectively.

###### Reduce the Gradient Error

binarization function (e.g. , sign) and STE 明显不同  [-1,1]以外的范围不会进行梯度更新

Bi-real设计了ApproxSign来替换sign函数
$$
\text{ApproxSign}(x)=\left\{\begin{array}{ll}-1,&\text{if}\quad x<-1\\2x+x^2,&\text{if}\quad-1\leq x<0\\2x-x^2,&\text{if}\quad0\leq x<1\\1,&\text{otherwise}\end{array}\right.
$$

$$
\frac{\partial\text{ApproxSign}(x)}{\partial x}=\left\{\begin{array}{ll}2+2x,&\text{if}&-1\leq x<0\\2-2x,&\text{if}&0\leq x<1\\0,&\text{otherwise}\end{array}\right.
$$

Binary Neural Networks+ (BNN+)直接提出了改进的sign function导数近似，该函数鼓励二进制周围学习权重

good quantization functions in forward propagation

**Differential Soft Quantization (DSQ)** method replacing the traditional quantization function with a soft quantization function
$$
\varphi(x)=s\tanh\left(k\left(x-m_i\right)\right),\quad\mathrm{if}\quad x\in\mathcal{P}_i
$$
where k determines the shape of the asymptotic function, s is a scaling factor to make the soft quantization function smooth and mi is the center of the interval Pi.

IR-Net provided a new perspective for improving BNNs that retaining **both forward and backward information** is crucial for accurate BNNs, and it is **the first to** design BNNs considering both forward and backward information retention

The weight update of BCGD goes by a weighted average of the full-precision weights and their quantized counterparts
$$
\mathbf{w}^{t+1}=(1-\rho)\mathbf{w}^t+\rho\mathbf{b}_{\mathbf{w}}^t-\eta\nabla f\left(\mathbf{b}_{\mathbf{w}}^t\right)
$$

###### Efficient Computing Architectures for Binary Neural Networks

XNOR.AI team, who proposed XNOR-Net [58], successfully launched XNORNet on the cheap Raspberry Pi device

###### Applications of Binary Neural Networks

58× faster convolutional operations and 32× memory savings theoretically

Defensive Quantization (DQ) to defend the adversarial examples for quantized models by suppressing the noise amplification effect and keeping the magnitude of the noise small in each layer

Quantization improves robustness instead of making it worse in DQ models

###### Tricks for Training Binary Neural Networks

aspects including network structure transformation, optimizer and hyper-parameter selection, gradient approximation and asymptotic quantization.

**Adjusting the network structure**

almost all binarization studies have repositioned the location of the **pooling layer**

pool 和 BN很重要  可以将BN放到量化过程之前

Liu et.al. proposed circulant filters (CiFs) and a circulant binary convolution (CBConv) to enhance the capacity of binarized convolutional features

circulant back propagation (CBP) was also proposed to train the structures

**Optimizer and Hyper-parameter Selection**

BN参数需要设置好

**Asymptotic Quantization**

INQ groups the parameters and gradually increases the number of groups participating in the quantization to achieve group-based step-by-step quantization.

 the idea of stepping the bit-width, which first quantizes to a higher bit-width and then quantizes to a lower bit-width

**Gradient Approximation**

an inspiring idea is to align its shape with that of the binarization function

##### Evaluation and Discussions

image classification：CIFAR10  and ImageNet

object detection and semantic segmentation： PASCAL VOC  and COCO

对激活值分布进行正则化

专门设计考虑BNN特征的方法可以获得更好的性能

XNOR-Net 提出了 smooth transition  addition shortcuts scale factors

像这些short cut可能硬件不友好

##### Future Trend and Conclusions

在前向和反向中保留好信息

customizing or transferring binary networks for different tasks, designing hardware-friendly or energy-economic binarization algorithms, etc.

8-bit训练的可能性