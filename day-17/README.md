| Kernel          | Time (ms) | GFLOP/s |
|-----------------|----------:|--------:|
| Naive           |   378.074 |   362.6 |
| 1D Block Tiling |    86.450 | 1,585.9 |
| 2D Block Tiling |    37.144 | 3,691.1 |
| Vectorized      |    35.169 | 3,898.5 |
| cuBLAS          |    22.464 | 6,103.3 |