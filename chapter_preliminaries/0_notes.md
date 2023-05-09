# Notes

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

# Ideas