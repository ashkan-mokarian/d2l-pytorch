# Notes

CNNs leverage prior knowledge about local spatial structure in the image data. The convolution kernels endow CNNs with far fewer parametrs and also parallelizing on GPU making them efficient for images. For a 1MB picure, if we don't assume _a priori_ structure for our features, and if implemented with a simple MLP, (assume 1000 hidden layer size), only the first layer requires 10^9 parameters, too large for GPU. CNNs systemize the idea of _spatial invariance_ (a cat can be anywhere in the image) exploiting it to learn useful representations with fewer parameters.

### Convolution kernel
Idea behind are:

* Translation invariance: each hidden neuron is the map of one kernel. This does not require every map to be linked to every hidden neuron in the next layer. or rather not to the fixed noeuron but a shift in image also shifts the map.
* Locality: restrict the size of kernels to small size, because each concept is local in space. 

Interleaving these filters/kernels/weights with non-linearity, deeper layers represent more complex and larger aspects of image.

These are the basic assumptions about the model design, turning computationally and statistically infeasible problems into tractable models.

$
[H]_{i,j,d} = \sum_{a=-\Delta}^{+\Delta} \sum_{b=-\Delta}^{+\Delta} \sum_{c} [V]_{a,b,c,d}[X]_{i+a, j+b,c}
$

i, j are pixel coordinates, $\Delta$ is the size of kernel, d represents the filter/map channel, and c are the color channels of input, but could also be considered as channels in deeper levels.

### Receptive fiels
Refers to all elements of the previous layers that may affect the calculation of some element in a layer during the forward propagation.

## Padding and Striding
General formual for output shape with kernel size k, padding p, striding s, and input size n is:

$
\lfloor (n_h - k_h + p_h + s_h)/s_h \rfloor \times \lfloor (n_w - k_w + p_w + s_w)/s_w \rfloor
$

if kernel size is odd, padding set as $p = k - 1$, and stride size divisible by input size, then the size of the output is:

$
(n_h/s_h) \times (n_w/s_w)
$

There are alternatives to padding, but zero-padding is the most famous one. **Moreover** zero padding allows the model to encode implicit position information, simply by learning where the _whitespace_ is.

## **Note** about multiple input channels and convolution kernel operation
when the input channels to a convolutional layer is higher than 1, then the convolutional kernel is of dimensions $c_i \times k_h \times k_w$. **However** the computations are not 3d convolutions, but rather each $c_i$ 2d kernel is convoloved with the corresponding channel and the results of all the channels are summed up together.

Now that I think, it doesn't matter, because it is summation in the end anyways. So to sum up, a conv layer with $c_i$ input channels, $c_0$ output channels, is a kernel with shape $(c_o, c_i, k_h, k_w)$ where the convolution operation is only done on the first three dimensions and the results are concatenated along the last dimension to for the multi-channel output.

**Important**, however, for pooling layers, it does make a difference. Pooling layers keep the channel dimension of input and output the same. they only operate on the spatial dimensions individual per channel.

## $1 \times 1$ convolution kernels
While this does not correlate adjacent pixels in the x-y dimensions, but it correlates along the channel dimension at each pixel. For example this could be used to resize the channel dimensions and establish and learn connections between the layers at each pixel.

# Not understood 