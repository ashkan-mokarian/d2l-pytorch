# How to use

- **Error** when `!pip install d2l`: instead do `!pip install d2l==1.0.0a0`

# Very important parts for a quick look

- Generalization in DNN: Weight regularization - weight decay (not so powerful); Early-stopping (essential for noisy datasets); Dropout (adding noise to training to make the problem hard for network);

- Look at the iimplementation of [batch norm](./chapter_modern_cnn/4_batchnorm.ipynb): Contains a good example for coding. Note that the main functionality of the added behaviour of batch norm is seperated from the module definition in a seperate function. This part contains the algorithms and functionality of it. The model book keeping, parameters, lr, moving averages, etc, are seperated from the math inside the module.

# Skipped

- About **Autograd** and **chain rule**: Check [this](./chapter_preliminaries/0_notes.md#automatic-differentiation-chain-rule-forwardpropagation-vs-backwardpropagationnotunderstood-autograd). I don't understand it, but if you spend some time, a very good resource is linked.
- Read more about central limit theorem if you find out. Why does the true average uncertainty tend towards a normal distribution centered at the true mean with rate $\mathcal{O}(1/\sqrt{n})$.

- [Neural tangent kernels](https://papers.nips.cc/paper_files/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html): a good framework for understanding DNN. apparently it relates DNN (parametric model) to non-parametric models (kernel methods). In short, there are more analytical arguments that can be made for non-parametric methods, and by this connection, it can serve an analytical tool for understanding over-parameterized DNN.

- BLEU Score for sequence to sequence evaluation. Described [here](https://d2l.ai/chapter_recurrent-modern/seq2seq.html#evaluation-of-predicted-sequences).

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

4. [Linear Neural Network for Classification](./chapter_linear_NN_classification/)

    0. [notes](./chapter_linear_NN_classification/0_notes.md): cross-entropy loss; A naive softmax implementation can lead to over/under-flow; Central limit theorem; Environments and **distribution shift**;
    1. [linear classification with Fashion-MNIST](./chapter_linear_NN_classification/1_linear_classification_with_fashionmnist.ipynb): Softmax linear regression implementation;

5. [Multilayer Perceptrons](./chapter_multilayer_perceptrons/)

    0. [notes](./chapter_multilayer_perceptrons/0_notes.md): what happens to derivate of relu at 0?; sigmoid; usually number of elements, e.g. layer width are powers of 2, why?; Forward pass, backpropagation, computational graph, and the memory requirements for training using backprop; Numertical stability, Vanishing and Exploding gradients; Early stopping; Dropout; 
    1. [MLP](./chapter_multilayer_perceptrons/1_mlp.ipynb): plots of relu, sigmoid, tanh, scratch and concise implementation for MLP;
    2. [Dropout](./chapter_multilayer_perceptrons/2_dropout.ipynb)
    3. (**Could not make it work. NaN loss**)[House prediction on Kaggle](./chapter_multilayer_perceptrons/3_kaggle_house_prices.ipynb): implementation of house prediction on Kaggle dataset; using pandas for loading csv data, and preprocessing; first checking with simple linear regression model, to see if data processing works and also get a baseline;

6. [Builder's Guide](./chapter_builder_guide/)

    0. [notes](./chapter_builder_guide/0_notes.md): nothing
    1. [IO and saving models on disk](./chapter_builder_guide/1_io.ipynb)
    2. [GPU](./chapter_builder_guide/2_gpu.ipynb): By default, parameters stored on gpu; operation on multiple parameters require them to be on the same device, otherwise cannot conclude where to store result, or where to do the computations;

7. [CNN](./chapter_cnn/)

    0. [notes](./chapter_cnn/0_notes.md): convolution kernel, padding, striding, computation for multi-channel
    1. [LeNet](./chapter_cnn/1_cnn.ipynb): not much

8. [Modern CNN](./chapter_modern_cnn/)

    0. [notes](./chapter_modern_cnn/0_notes.md): Conv _blocks_ instead of Conv _layers_; stem, body, head design pattern; multi-branching in googlenet; Batch Normalization; ResNet and residual connections; Grouped Convolutions to reduce memory and time by dividing the channels into multiple branches; Some general Design concepts for designing CNNs; Final note about **Scalability trumps Inductive biases**, transformer better than CNN;
    1. [VGG](./chapter_modern_cnn/1_vgg.ipynb): Implementation of VGG-11
    2. [NiN](./chapter_modern_cnn/2_NiN.ipynb): " of NiN, takes less memory by using 1x1 in early and intermediate layers and nn.AdaptiveAveragePool2d in the end.
    3. [GoogleNet](./chapter_modern_cnn/3_googlenet.ipynb): not much to see, model too complicated, a lot of parameters for the number of channels which do not give any hinsights. maybe just overal design pattern seems interesting. implementation of inception block, and the modular design of such a large network could also be interesting to look at.
    4. [Batch Norm](./chapter_modern_cnn/4_batchnorm.ipynb): Implementation of batch norm from scratch; batch norm is placed between the conv/FC layer and the consequent non-linearity layer;

9. [RNN](./chapter_rnn/)

    0. [notes](./chapter_rnn/0_notes.md): autoregressision; $\tau -th order markov condition$;
    1. [Markov model](./chapter_rnn/1_markov_model.ipynb): k-step ahead prediction and accumulation of errors problem showcased on a synthetic dataset;
    2. [Language Model](./chapter_rnn/2_language_model.ipynb): Preprossing raw text into sequence data, tokenizer, vacabulary set; Zipf law for n-grams; Dataset sampling strategy ot How to sample train and val datasets from a corpus;
    3. [RNN](./chapter_rnn/3_rnn.ipynb): 

10. [Modern RNN](./chapter_modern_rnn/)

    0. [notes](./chapter_modern_rnn/notes.md): lstm; GRU; 
    1. [LSTM](./chapter_modern_rnn/1_lstm_and_rest.ipynb): lstm; Deep RNN; 
    2. [Encoder Decoder]():

11. [Transformer](./chapter_transformer/)

    0. [notes](./chapter_transformer/0_notes.md): scaled dot-product and additive attention scoring function; Multi-head attention very short description; Self-attention; Positional Encoding; The transformer architecure; 
    1. [attention](./chapter_transformer/1_attention.ipynb): heatmap visualization of attention weights; Nadaraya-Watson regression, a simple example of regression problem using attention pooling; Masked softmax; scaled dot-product and additive attention;
    2. [Bahadanau Attention](./chapter_transformer/2_bahadanau_attention_mechanism.ipynb): nothing too special, just the implementations of seq2seq model using attention but not exactly like trasformers. probably better models to look at in the notebooks below.
    3. [Transformer](./chapter_transformer/3_transformer.ipynb): Multi-head attention, scratch implementation with reshaping convenience functions for parallel computation of all heads; Positional encoding; encoder-decoder implementation for sequence task using transformer self-attention;
    4. [ViT](./chapter_transformer/4_ViT.ipynb): Patch Embedding using conv; ViT implementation;

12. [Optimization Algorithms](./chapter_optimization/)

    0. [notes](./chapter_optimization/0_notes.md): Convexity; Dynamic learning rate; SGD; Momentum; AdaGrad; RMSprop; Adadelta; Adam; LR scheduling;

13. [Computational Performance](./chapter_computational_performance/)

    0. [notes](./chapter_computational_performance/0_notes.md): Imperative vs. Symbolic programming; Asynchronous computation; blockers and barriers between frontend and backend in ML frameworks; Parallelism of GPUs; non-blocking communication; Hardware short introduction; Multiple-GPU training strategies and data-parallelism;
    1. [some code](./chapter_computational_performance/1_computational_performance.ipynb): Nothing special.

14. [Computer Vision](./chapter_computer_vision/)

    0. [notes](./chapter_computer_vision/0_notes.md): Some basic Augmentation methods in `torchvision.transforms`; `Compose[transforms]` for combining multiple augmentations; A particular case of transfer learning: fine-tuning; Anchor box; AnchorBoxes as training data for object detection; Multi-scale object detection; Semantic segmentation (mainly in the notebook explained and not here); transposed convolutions for upsampling; FCN; initializing transposed convolutions using bilinear interpolation; predicting semantic segmentations for images larger than input size; Neural Style Transfer (using a pretrained CNN to update parameters of a synthesized image using backpropagation); 
    1. [Object Detection](./chapter_computer_vision/1_object_detection.ipynb): anchor box implementation based on a list of sizes and ratios;
    2. [SSD](./chapter_computer_vision/2_SSD.ipynb): Single Shot Multibox Object Detection implementation;  
    3. [Semantic Segmentation](./chapter_computer_vision/3_semantic_segmentation.ipynb): Implementation for reading VOC dataset, doing necessary transformation of colors into labels or indices for the label maps, creating dataset and dataloader class; Removing the head of a ResNet18, replacing with a 1x1 layer to get the num_classes channels and adding transposed conv layer to build the whole semantic segmentation network; Not continued because the rest was just training and evaluation;
    4. [Neural Style Transfer](./chapter_computer_vision/4_neural_style_transfer.ipynb): uses a style image to apply the style to a content image. Pretrained VGG is used to extract features. The parameters of the synthesized image is the model and weights to be trained, and VGG is frozen in this scenario. Content loss is the squared loss of features for one of the layers close to ouput, and image also initialized with content image. For Style loss, the Gram matrix of several layers is compared and not the features itself, but just the correlation of their features to match only the style and not the contents. Total variation loss is used for denoising.

17. [Reinforcement Learning](./chapter_rl/)

    0. [notes](./chapter_rl/0_notes.md): Markov Dicision process; Value, Action-Value, Policy functions; Value iteration algorithm;
