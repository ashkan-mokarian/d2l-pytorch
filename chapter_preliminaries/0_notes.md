# Notes

### **Backpropagation and memory**
For doing automatic differentiation (aka autograd) using backpropagation, we have the parameters of the model which have fixed memory allocated. If we want to compute gradient for each, we also allocate memory for their backprop values, and we keep them fixed. Because in many iterations, if at each step, we would allocate memory and free again, this causes a lot of memory overhead and management, slows down the algorithm, and may risk running out of memory. So with this, now you understand why we need to tell pytorch whether a variable requires extra memory for backprop or not. It is done either at declaration `x = torch.tensor(..., requires_grad=True)` or with method call `x.requires_grad(True)`. There is also `requires_grad_` which doesn't check if the variable is a leaf node. The former one checks and throws an error. Also make sure to zero the gradient buffer at each iteration, otherwise the
new `backward()` call will add to the buffer. This comes handy when gradient of sum of losses needs to be computed.

### **Backpropagation on non-scalar value**
Each framework deals with it differently. Essentially has to be reduced to a scalar. In pytorch, some vector $\bf{v}$, called _gradient_, such that `backward` will compute $\bf{v}^T \partial_{\bf{x}}\bf{y}$ is computed which reduces the gradient to a scalar. `y.backward(gradient=torch.ones(len(y)))`. This one does the same but is faster `y.sum().backward()`.


# Not understood

### **In 2.1.6 Converting to other python objects**
Claims that both tensor and ndarray classes share the same underlying memory and if changed in-place (= using [:] or +=) it changes the other too. But when running this, the memory is different.
```pytorch
A = np.array([1, 2, 3])
B = torch.from_numpy(A)
assert id(A) == id(B)
```
**Answer**: keyword here is underlying memory. They are both python objects, but in the class definition, they have a pointer to a more primitive type, which this could be shared between both, hence underlying memory is shared.
**Note**: do **in-place** assignments so that memory does not dereference in one and eventually lead to memory leak.

### **In ipynb:2_derivate..., Gradient and python control flow** and also at **2.5.4 in d2l.ai**
How flexible can the control flow be? Does this make the training too slow, because each time building a new computation graph, since e.g. a is input and changes? How is the derivative calculated for checking?

## **Automatic differentiation, Chain-rule, Forwardpropagation vs. Backwardpropagation**(#notunderstood-autograd)
I really don't understand the details. Apparently when we have tensor inputs and scalar output as it is in DL, backpropagation is much more effiecient than forwardpropagation. But forwardpropagation does not need to save intermediate values. A very good resource can be found here: [Autograd: Forward vs backward propagation of chain rule](https://mblondel.org/teaching/autodiff-2020.pdf)

### **Exercises 5,6,7,8 remain unsolveled**: [permalink here](https://d2l.ai/chapter_preliminaries/autograd.html#exercises)

# Ideas