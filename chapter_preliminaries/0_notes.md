# Notes

### Dynaic Computation Graph

* **requires_grad only for parameter leaf nodes**: You must tell PyTorch which tensors need gradients using requires_grad=True. You only track for leaf nodes, if you try requires_grad for the other non-leaf nodes, you get error. Leaf nodes are parameters or inputs, and in almost all cases you don't need gradients for inputs, so requires_grad only for parameter leaf nodes.

* **Gradient Accumulation is default behaviour**: By default, .backward() adds gradients to the .grad buffer rather than overwriting them. This is essential for Gradient Accumulation (simulating larger batches) but requires you to call optimizer.zero_grad() or x.grad.zero_()联 at every step to avoid "double-counting" gradients from previous batches.

* **Dynamic computation graph**: PyTorch uses a dynamic computational graph, which is rebuilt every forward pass. However, the .grad buffers for leaf nodes (model parameters) are persistent. Reusing these pre-allocated buffers avoids the overhead of constant allocation/deallocation during millions of training iterations.
The dynamic computational graph is built everytime before the forward pass. it's like a train building the rails as it moves forward. This enables flexibility, e.g. for doing control flows (if-statements) in the graph, or dealing with varying size matrices during computations. also frees up VRAM. This also allows for debugging or adding print statements in the graph. it is fast bcuz c++ and garbage collection backend, but still can do torch.JIT or torch.compile to create static graphs which improves speed in production. but remember that the overhead of cleaning up intermediate buffers and the computation graph is not too much, it just changes a pointer, pytorch keeps the freed of intermediate buffers as caches and do not give it back to the system. </br>
and because it is dynamic, you can use whatever python offers (if, while, ...) and pytorch orchestrates an interplay with heavy matrix computations on gpu, and building the graph on cpu. that's why with dynamic computation graph, one can use python for manipulation or debugging. The only overhead is that gpu computations stalls to send data back to cpu for checks if needed, but this is not too much. for very small models where computations are small, the overhead of the cpu part might become too large in comparison to the gpu part, but for larger models it is like 5% or 10%. but you get the huge advantgage of debugging. The biggest fault of a dynamic graph is probably the optimizations a compiler can do, eg fusing operations and memory management if it knows ahead of time. If too worried about the performance, one can do torch.compile which traces the graph building for some steps, and compiles the graph based on that, and if something unexpected hapens, reverts back again to the dynamic mode.


### Backpropagation on non-scalar value
Each framework deals with it differently. Essentially has to be reduced to a scalar. In pytorch, some vector $\bf{v}$, called _gradient_, such that `backward` will compute $\bf{v}^T \partial_{\bf{x}}\bf{y}$ is computed which reduces the gradient to a scalar. `y.backward(gradient=torch.ones(len(y)))`. This one does the same but is faster `y.sum().backward()`.

### Underlying memory and in-place operations
Claims that both tensor and ndarray classes share the same underlying memory and if changed in-place (= using [:] or +=) it changes the other too. But when running this, the memory is different.
```pytorch
A = np.array([1, 2, 3])
B = torch.from_numpy(A)
assert id(A) == id(B)
```
**Answer**:
False (the ids are not the same), because they are different object, one numpy and the other pytorch objects, with different locations in memory. **BUT** they point to the same underlying memory location. That's why you **should** do in-place operations on them, i.e. A+=1. Otherwise with A=A+1, python does A+1 and allocates new underlaying memory where the new A points to, so the value that B is refering to will not change, and now new memory for A, and they don't point to the same location, and memory leaks, etc.


## Deep Dive in Automatic differentiation, Chain-rule, Forwardpropagation vs. Backwardpropagation
A very good resource can be found here, but a little difficult to understand and needs more time: [Autograd: Forward vs backward propagation of chain rule](https://mblondel.org/teaching/autodiff-2020.pdf)

# Ideas