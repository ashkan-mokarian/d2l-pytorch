# Notes

## **GPU and performance**
Data transfer between devices, and from other memories (e.g. memory on gpu) to main memory (computer ram/cpu ram) is usually very costly. Especially when there is a PIL that makes everything wait for it. So try to limit the amount of data transfer between different memories as much as possible.

Few Large operations always better than lots of small operations. Because usually there are race conditions and different pathes of computation has to wait for others to complete. For example, one large matirx-matrix multiplication much faster than equivalent matric-vector multiplications.

# Not understood