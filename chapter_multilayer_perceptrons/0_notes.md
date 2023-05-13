# Notes

### **Gradient of relu at 0**
while it is non-differentiable at 0, in practice, we default to the gradient of leftside of 0, i.e. 0. This might not be theoretically correct but first of all, we might not need it at all during training (unless working with very low precisions), and even if it happens, we can neglect it for one iteration and hope for training in another one.

### **Why use relu**
It's gradient is well-behaved. either passes the gradient or doesn't, without influencing it's value in the positive region. This mitigated the vanishing gradient problem?.

There are also variations of relu, e.g. parametrized relu (pReLU) defined by $\text{pReLU}(x) = \max(0, x) + \alpha \min(0, x)$. This variation still passes some gradient information, even when x lies in the negative region.

Other variations are **GeLU**(Gaussian error linear unit) $x\Phi(x)$ where $\Phi(x)$ is the standard gaussian cumulative distribution function. Also the **swish activation function** $\sigma(x) = x\cdot\text{sigmoid}(x)$ can lead to better performance and accuracy in many applications.

### **Sigmoid** activation function
Or called the squash function, maps every number $(-\inf, + \inf)$ to $(0, 1)$. It was historically used for neuron representation and earlier implementations of NN because it is differentiable everywhere. It can be seen as a special case of softmax. It was used because it has a nice analytical form, but with the start of automatic differentiation for optimizing the training process, it showed some problems and it has been replaced by relu which is much simpler and easier trainable. Sigmoid poses **challenges for optimization** since its **gradient vanishes** for large positive and negative values. **This can lead to plateus that is difficult to escape from**.

$
\text{sigmoid}(x) = \frac{1}{1+\exp(-x)} \\
\frac{d}{dx}\text{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \text{sigmoid}(x)(1-\text{sigmoid}(x))
$

### **tanh**
$\tanh(x) = \frac{1-\exp(-2x)}{1+\exp(-2x)}$ squashes -inf,+inf to -1,+1. derivative similar to sigmoid but peak is higher. around 0 is also linear as in sigmoid. Also vanishing gradient problem.

### **Note: We usually select width of layers as powers of 2**https://d2l.ai/chapter_multilayer-perceptrons/backprop.html
This is **computationally efficient** due to the way memory is allocated and **addressed** in hardware implementations.

## **Forward propagation (pass), Backward propagation (backprop) and computational graph**
Consider a simple two-layer MLP (one hidden layer). Draw the computational graph, and do forward and backward propagation on paper. Not taking notes because it is too much but you can look it up [here](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html).

### **Memory requirements of training using backpropagation**
In training using mini-batch SGD using the backpropagation algorithm, The current values of the parameters are used to obtain the loss, and also store all values of the intermediate variables. In backpropagation, which goes a path from loss to each paramter in the reverse order of forward pass, the intermediate values of parameters are used to obtain gradients and update parameter values. These intermediate values are stored to not be calculated again for the backpropagation. This is why training requires much more memory than inference. It grows linearly with number of exmaples in the batch and depth of network.

## **Numerical Stability**, vanishing and exploding gradients
Consider a deep neural network. Each layer can be written as $h^{(l)}=f_l(h^{(l-1)})$ with $l^{(0)}=X$. Let's assume that each f is a linear transformation. The gradient of loss wrt each layer's parameters can be written as:
$\partial_{W^{(l)}}o = \partial_{h^{(L-1)}}h^{(l)} \times \dots \times \partial_{h^{(l)}}h^{(l+1)} \partial_{W^{(l)}}h^{(l)}$. This id the backprop from last year to layer l. Each of these factors is a matrix of gradients and they are multiplied all together to a point. Each of these matrices used for training can have varying eigen-values, and if too large, they can lead to NaN computations or **exploding gradients**, and if too small, the consecutive multiplications can lead to **vanishing gradients** and essentially no update information can be passed down from a point and lower layers not trained.

One reason for _vanishing gradients_ is poor activation functions such as the sigmoid making training of starting layers very unstable and impossible. remedy can be e.g. using ReLU.

In the case of exploding gradients, sometimes initialization can play a big role. large gradients can cause training unstable (think of jumping from one side of valley to the other.)

One way to mitigate these problems is proper initialization (but also many more).

## Parameter Initialization

### **Xavier** initialization
It builds on the following and a crucial assumption is that each layer is connected to the next only linearly without any non-linear functions, but is still very usefull in practice nevertheless. For each layer, for each neuron we can define $n_{in}$ and $n_{out}$ which are the connected neurons from the previous layer, and connected to the one afterwards. Let's focus on the forward propagation for now. We can write the following:
$o_i = \sum_{j=1}^{n_{in}} w_{ij}x_j$. Let's assume that both the inputs to the layers and weights distributions have mean 0 and some variances (need not necessarily be gaussian, and still have mean and variance defined for it). We can write:

$
\mathbb{E} [o_i] = \sum_{j=1}^{n_{in}} \mathbb{E} [w_{ij}x_j] = \sum_{j=1}^{n_{in}} \mathbb{E}[w_{ij}] \mathbb{E} [x_j] = 0 \\
\text{var}[o_i] = \mathbb{E}[o_i^2]-\mathbb{E}[o_i]^2 = \mathbb{E}[\sum_{j=1}^{n_{in}} w_{ij}^2 x_j^2] - 0 = n_{in} \sigma_{w}^2 \sigma_{x}^2
$

We can do the same analogy for backward propagation where gradient are computed, but this time $n_{out}$ comes into play.

The **Xavier** initialization recommends to choose the initializations of weights such that it accounts for $n_{in}$ and $n_{out}$. In this way, the layer to layer variance scale will be kept to zero, i.e. $n_{in}\sigma_{w}^2 = 1$ and $n_{out}\sigma_{w}^2 = 1$. These are to prevent scaling of the original distribution after propagating through each layer and possibly leading to vanishing or exploding scenarios. With some subtle change, in practice, this becomes:

$
1/2(\sigma_{in}^2 + \sigma_{out}^2) = 1  \rightarrow \sigma=\sqrt{\frac{2}{n_{in} + n_{out}}}
$

## Generalization

### **Early-stopping**
There are some research showing that albeit Deep Networks can arbitrarily any random labeling or noise in the dataset, they tend to first attend and learn the patterns and then overfit to the noise. This motivates, a **early-stopping** technique can be used as generalization technique for DNN. Instead of constraining the value of parameters, here we put a constrain on number of training epochs. A common way to determine stopping criterion is to monitor validation error throughout training, typically by checking once after each epoch of training, and to cut-off training when the validation error has not decreased by more than some small amount $\epsilon$ for some number of epochs. (Think of this, at some point, after learning the important parts of the dataset, the training starts to tend and attend to the noise in the dataset).

### **Dropout**
In classical generalization theory, one way to reduce the generalization gap is to use simpler functions. One notion of simplicity could be constraining the parameter space of the model, for example with weight decay, i.e. the norm of the parameters could be thought of simpler. Another notion for simple functions is smoothness, i.e. small perturbations in input, leads to small perturbations in output.

Dropout has some analogy to this idea, and the standard dropout is to drop some of the neurons at each layer during the forward propagation before computing the subsequent layer. One thing to consider is to let the expectation of a layer be the same, before and after this perturbation. In practice this means (for standard dropout regularization):

$
h' = \begin{cases}
    0 & \text{with probability p}\\
    \frac{h}{1-p} & \text{otherwise}
\end{cases}
$

where h are neurons before dropout and h' afterwards. This way, $\mathbb{E}[h']=\mathbb{E}[h]$. Typically, dropout is dropped at test time. However, some applications use dropout as a mean to evaluate **uncertainty**. It is used at test time for several runs to find and agreement or to esetimate uncertainty.

### **Use log-space to get relative differences**
For example for price prediction, since the difference depends on the item, we use the difference of the log of the target and estimate. This gives a relative difference measure. we can just do MSE on the log values.

### **Caution: Model selection with K-fold cross-validation**
With cross-validation, you can tune hyperparameters for best model selection. But remember that although k-fold cross-validation can be good, doing it for an absurd large set of hyperparameters might turn to be bad and select a hyperparameter set that overfits on the k-fold training procedure but does poorly on test set.

# Not understood