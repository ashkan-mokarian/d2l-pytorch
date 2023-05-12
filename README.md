# How to use

- **Error** when `!pip install d2l`: instead do `!pip install d2l==1.0.0a0`

# Skipped but important

- About **Autograd** and **chain rule**: Check [this](./chapter_preliminaries/0_notes.md#automatic-differentiation-chain-rule-forwardpropagation-vs-backwardpropagationnotunderstood-autograd). I don't understand it, but if you spend some time, a very good resource is linked.

# TOC

1. [Introduction](./chapter_introduction/)

    0.  [notes](./chapter_introduction/0_notes.md): Data is important; Major breakthroughs in the last century leading to current state of DL

2. [Preliminaries](./chapter_preliminaries/)

    0. [notes](./chapter_preliminaries/0_notes.md)
    1. [data manipulation and processing.ipynb](./chapter_preliminaries/1_data_manipulation_and_processing.ipynb): Tensor and ndarray, initialization, operation, in-place assignment for memory management, pandas
    2. [derivatives, plots, automatic differentiation](./chapter_preliminaries/2_derivatives_plots_automatic_differentiation.ipynb): some basic matplotlib plots; automatic differentiation aka autograd; usually zero grads, but sometimes useful; how to deal with non-scalar grads; detaching computation; Backpropagation in Python control flow (for, if, dependent on tensor);
    3. [probability, statistics, documentation](./chapter_preliminaries/3_probability_statistics_documentation.ipynb): drawing samples to compute frequenct of events; by _law of large numbers_ and _central limit theorem_, we know for coin toss frequencies converge to true probs and errors should go down with a rate of $(1/\sqrt{n})$; Bayes' theorem; chebyshev inequality; invoke documentatio, help, and docstrings by help() or ?list;

3. [Linear Neural Networks for Regression](./chapter_linear_NN_regression/)

    0. [notes](./chapter_linear_NN_regression/0_notes.md): Generalization and model selection;
    1. [Linear Regression](./chapter_linear_NN_regression/1_linear_regression.ipynb): Some implementation basics for d2l OOP implementation in notebooks; Creating dataset using d2l; linear regression implementation from scratch; Weight decay notes and implmenetation;