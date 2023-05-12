# Notes

### **What is the underlying assumption about linear regression**
assumes that relationship between features $\bf{x}$ and target $y$ is approximately linear, i.e., that $E[y | \mathbf{x}]$ can be expressed as a weighted sum of the features $\bf{x}$.

$price = w_{area} . area + w_{age} . age + b$ 

 This setup allows that the target value may still deviate from the expected value on account of observation noise. Next we can impose the assumption that such noise is well-behaved, following a Gaussian dist.

### **Gradient Descent vs. SGD vs mini-batch SGD**
multiplication and summation are an order faster than moving memory. Hence matrix-vector product is much faster than vector-vector(SGD - a single example). Also, some layers such as 'Batch Normalization' need more than one sample of data. 

### **Inspiration**
As much as airplanes are inspired by birds (however they are not built same as bird), DL is inspired by neurons.

## Generalization

We need to distinguish between empirical error (training error) $R_{emp}$, a statistic calculated on training dataset, and the _generalizatoin error_ $R$ which is an expectation taken with respect to the underlying distribution. In practice, we don't know the underlying dist. and cannot calculate it on an infinite number of streams. Therefore, we use the same computation as in empirical error, but on the held-out test dataset. **Note** that the error on train set is biased since the model is conditioned on the same set. The **central question in generalization** is then, when should we expect out training error to be close to the population error (and thus the generalization error).

Generalization has also links with _model complexity_. But in general, in deep learning (since models are typically very complex and powerful), all we can say is: **_low training error alone is not enough to certify low generalization error_** and hence test set evaluation has high importance in DL. We must rely on holdout test set to clarify generalization error.

Too confusing but interesting: Karl Popper, an influential philosopher of science, who formalized the criterion of falsifiability. According to Popper: "a theory that can explain any and all observations is not a scientific theory at all! After all, what has it told us about the world if it has not ruled out any possibility?". In short, what we want is a hypothesis that could not explain any observations we might conceivably make and yet nevertheless happens to be compatible with those observations that we in fact make.

### **Model Selection**
Be aware, in model selction, the test set should not be touched, and evaluation set should be used. If for example one trains different models on train data and calculates performance on accuracy, and then chooses the best model, this approach is flawed and prone to overfitting. Usually in books and papers, it could come that test never mentioned or used and only validation (probably refered to as test) is used, because usually a specific method is examined, and could be assumed that a final evaluation on test set will be conducted. But again one should be careful when seeing or using this. **Caution**

### **On L2 or L1 weight decay**
L2 leads to small overall distribution of parameters, while l1 can lead to sparse parameter distribution which can be good in some cases for example for feature selection (the model learns to act on a small number of high impact features)

### **Cross-validation**
When data is scarce, this is used. data is split in k non-overlapping subsets. trainde on k-1 and evaluated on the other, averaged overall or judged individually. Note that, test still shouldn't be used.

# Not understood

### **L2 Loss and Maximum likelihood estimation**
I think if L2 loss is used with normally distributed noise($\epsilon$) in linear regression as given by $Y = Xw + b + \epsilon$, L2 Loss optimization is equivalent to Maximum Likelihood Estimation of $P(Y|X)$. You can derive it by minimizing the $-\log P(Y|X)$. This somehow has connection with L2 loss bad with outlier data, and that noise is not normally distributed. I know L1 better for data with outliers, and apparently to get best of both L2 and L1, one could implement L2 but with clipping large gradient values.

# Ideas