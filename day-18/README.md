| Kernel                      | Time (ms) | TFLOPS |
|-----------------------------|----------:|-------:|
| Naive                       |   357.738 |   0.38 |
| 1D Block Tiling             |    77.428 |   1.78 |
| 2D Block Tiling             |    31.453 |   4.37 |
| Vectorized (BM=64, BN=128)  |    24.381 |   5.64 |
| cuBLAS                      |    21.151 |   6.50 |