# Notes

### **cross-entropy and its gradients**
In classification, we want probability estimates for each class, because sometimes more than one true class, but also others. To this end, soft-max is used at the last neurons $\hat{y}_i = \text{softmax}(o_i) = \frac{e^{o_i}}{\sum_k e^{o_k}}$. The maximum likelihood estimation of classification in linear, can be written down over data. Since data is **iid**, it is the multiplication of likelihood for each datapoint. The negative log-likelihood should be calculated since dealing with multiplication is hard. 

$
-\log P(X|Y) = \sum-\log P(y_{(i)}|x{(i)}) = \sum l(y^{(i)}, x^{(i)}) \\
l(y^{(i)}, x^{(i)}) = -\sum_{j=1}^{q} y_j\log \hat{y}_j
                    = -\sum_{j=1}^{q} y_j\log \frac{e^{o_j}}{\sum_{k=1}^{q}e^{o_j}}
                    = \sum_{j=1}^{q} y_j \log \sum_{k=1}^{q}e^{o_k} - \sum_{j=1}^{q} y_jo_j
                    = \log \sum_{k=1}^{q}e^{o_k}-\sum_{j=1}^{q} y_jo_j
$

where (i) denotes the i-th example. o_j denotes the j-th **logit**. q is the number of classes.
This is also known as the **cross-entropy** loss. The gradients w.r.t. logits are:

$
\partial_{o_j}l(\hat{y}, y) = \frac{e^{o_j}}{\sum_{k=1}^{q}e^{o_k}} - y_j = \text{softmax}(o_j) -y_j
$

What this means is that gradient steps push the softmax of logits towards the corresponding class value, which is either 0 or 1. Or changing the estimated probability towards the one-hot encoding.

### **How softmax should actually and is implemented**(#LogSumExpTrick)
A problem with naive implementation of softmax is that it can quickly become overflow or underflow, since we are dealing with exponentiation. a single precision float can represent numbers in range $[10^{-38}, 10^{38}$, as a result logit's range would be $[-90, 90]$ and it can easily lead to problems. Another issue is that at some point we have to compute log and if this is 0 or very close to 0, it can lead to _underflow_ and output NaN. As a remedy, one can subtract all logits by their max value, and in this way we get a managed range.

## Genrelization in Classification
Notes written here are very very very vague. just tried to write sth in the hope that I learn sth about it. In summary, there are some theoretical guarantees from statistical learning. some asymptotic and some for finite but more conservative. By generalization, in general we mean how confidence we are about being close to the true mean, by just sampling a finite times of an underlying distribution. These confidences are based on number of samples. rate is $\mathcal{O}(1/\sqrt{n})$ but this would be nonsensical for practical use cases and would require trillions of samples. because they are very conservative. probably better bounds won't be found as stated in the d2l material. VC-dimensions also talks about sth similar. However, in practice, we observe that with much lower samples, we are seeing good generalizations and this observations are what drives DL.

Another example to better understand generalization: a function that can memorize the whole dataset by passing one time over it, cannot do better than random guessing. emphasizing why we talk about generalization and why we need test set.

### **Central Limit Theorem**
We have never access to the underlying prob. dist., we instead have a sample representative $D$ of it. For classification, the estimated generalization error for a fixed function/model f is: $\epsilon _d(f) = \frac{1}{n}\sum_{i=1}^{n} \bf{1}(f(x^{(i)}) \neq y^{(i)})$, and the population error is: 
$\epsilon(f) = \mathbb{E}_{(\bf{x}, y)\sim P} \bf{1}(f(\bf{x}) \neq y) = \int \int \bf{1}(f(\bf{x}) \neq y)p(x, y)dxdy$. **Central limit theorem** tells us that if we posess n samples drawn from a distribution with mean $\mu$ and stddev $\sigma$, as n grows to infinity, the sample average $\hat{\mu}$ approximately tends toward a normal distribution centered at the true mean with stddev $\sigma/\sqrt{n}$. This tells us that as the number of samples grows, $\epsilon _D(f)$ should approach $\epsilon (f)$ at a rate of $\mathcal{O}(1/\sqrt{n})$.

### Hoeffding inequality and _not_ asymptotic guesses
Not much to talk about, it goes too deep, but for example for the case of binary classification, which the underlying distribution of errors is bernouli, and using Hoeffding inequality, it can be shown for finite sample sizes (not asymtotic), in order to lie with 95% confidence within the +-0.001 of the true generalization error, only 15000 examples are needed (lower margin). This is evaluated 10000 for the asymptotic case using central limit theorem. Just to have written it down, Hoeffding inequality says:

$
P(\epsilon _D(f) - \epsilon(f) \geq t) < \exp{-2nt^2}
$

### **VC dimension** and bias-variance tradeoff
Something about guaranteeing that a class of models, converge to the true generalization error with a rate. for example a linear class of models in 3d, can seperate 3 points, but fails for 4 (d+1). These gurantees are very conservative though. But with deep learning, we see that with larger deeper models, we get better generalization across different tasks, which is not what these theories tell. These are about the trade-off between bias and variance. You can have lower dimesion function classes that have bias (further away from the true $\epsilon$) but we know in which variance we lie. on the other hand, with more dimensionality, we get lower bias but higher variance (we can't know for sure how far we might be from the true $\epsilon$) because we have fixed samples and this somehow bounds the variance but with VC dimension theory we need much much much more samples to have the same bound.

---
## **Distribution Shift**

### Covariate shift (most widely studied type)
Labels stay the same, distribution of features/covariates change, i.e. P(y|x) does not change, but P(x) changes. Underlying assumption is that x causes y. Example case, we are given real images of cats and dogs in training set, but predicting cartoon styles of them in test set (or the shifted set - this is toy example)

### Label shift
reverse of before. P(x|y) stays same, but P(y) changes. happens when y causes x, e.g. disease causes symptoms, while we are predicting disease from symptoms. however the distribution of disease can change e.g. a pandemic.

### Concept shift
when the labels change. Fashiom. or diagnostics criteria for mental illness, or brand names change, or job labels change. For example for translation, geography and temporal changes are reasons for concept shifts.

### Non-stationary distributions
shifts over time. e.g. in advertising, a new ipad launches, can change a lot. spam filter, but spammers adapt. recommending santa hats which is good for winter but fails for summer. Here one easy solution is to update the model quickly.

# Not understood

### Regarding [softmax implementation](./0_notes.md/LogSumExpTrick)
I still don't understand the details, but maybe not so much important. Just be aware that this exists. check [this](https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html) and [this](https://en.wikipedia.org/wiki/LogSumExp)

### Skipped section 4.7.3: Correction of Distribution shift

# Ideas