# Notes

## Autoregressive models

Goal is to find statistics for $P(x_t| x_{t-1}, x_{t-2}, ..., x_1)$. A problem is that the conditioned variables, or input features to the model is of variable length, and in this chapter we examine approaches to deal with this not-fixed sequence variables.

* One idea is to only look at a fixed length $\tau$ of the history and therefore get a fixed-length input.

* Latent autoregressive models: maintain some variable $h_t$ as a summary of the past and predict $\hat{x}_t = P(x_t | h_t)$ with $h_t = g(h_{t-1}, x_{t-1})$ ($h_t$ is unobserved, hence latent).

### Stationary dynamics
While $x_t$ changes, the dynamics given the previous observations do not change.

## Markov Models
Whenever we can discard the history beyond some time $\tau$ without any loss in predictive power, we say that the sequence satisfies **Markov condition**, i.e., the future is conditionally independent of the past, given the recent history. When $\tau=k$, we say that the data is characterized by a _k-th order Markov model_.

### Cumulative errors in multi-step predictions
One problem of multi-step prediction (predicting steps ahead of observed data where the previous predictions are used for conditioning and features for the steps ahead), is that the errors accumulate over time. 

## Perplexity
A measure to assess the quality of a language model. It should account for varying length of texts, therefore it needs to be mean. A notion of entropy is used here as following,

$
\frac{1}{n} \sum_{t=1}^n -\log P(x_t | x_{t-1}, ..., x_1)
$

The actual perplexity, is the exponentiation of this (for historical reasons). If a LM predicts always with absolute confidence (P(.)=1) then perplexity is 1 (best case scenario). If the probability of the correct word given the previous ones is 0 in all cases, perplexity is positive infinity (worst-case). As the baseline case, if model predicts uniformly for all words in the vocabulary, the perplexity equals the number of unique tokens in the vocabulary. In fact, if we were to store a sentence without any compression, the vocabulary of tokens would be the best we could do to encode the sequence. Hence it provides a trivial upperbound baseline that any useful model must beat.

## RNN: Recurrent Neural Network
A Markov model with a prefixed number of parameters can only store a limited number of values for tokens conditioned on. For modeling all possible occurences of a vocabuluray of size $\mathcal{V}$, the model requires to store $|\mathcal{V}|^n$ values to consider n-grams. Therefore, if desired to model the language based on longer history, this puts a limit for the model. Therefore it is desirable to model as:

$
P(x_t|x_{t-1} ..., x_1) \approx P(x_t|h_{t-1})
$

where $h_{t}$ is a hidden/latent variable that stores all information about the sequence until time-step t. In general $h_{t-1}=f(x_t,h_{t-1})$.

RNNs are neural networks with hidden states. **Note** that hidden state is different than hidden layers.

### Clipping Gradients

Same as any DNN, when backpropagating through lots of layers that consist of large matrix-matrix multiplications and non-linearities, there is the problem of vanishing and exploding gradients.

One inelegant but ubiquitous is to simply clip the gradients. Downside is we are limiting the speed at which we can reduce the loss. On the bright side, it limits how much we can go wrong in one gradient step. It is done by projecting the gradients on a unit ball based on some clipping hyperparameter.

$
g \leftarrow \min(1, \frac{\theta}{||g||})g
$

Gradient clipping addresses the issue of exploding gradients, but not **vanishing** issue.

# Not understood