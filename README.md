- 100 day deep dive into writing GPU kernels.
- Inspired by Umar Jamil's challenge on his Discord.

| Day | Kernel(s) Written | Description |
|:---:|--------------------|-------------|
| 1 | [Vector Addition](day-1/vecAdd.cu), [Naive MatMul](day-1/matmul_naive.cu) | Implemented two simple kernels, Reviewed basic CUDA programming model. |
| 2 | [Row & Col MatMul](day-2/row_col_matmul.cu), [Tiled MatMul](day-2/tiled_matmul.cu)|pmpp ch.4 exercises done |
| 3 | [1D Convolution](day-3/1d_conv.cu)|pmpp ch.5 exercises done, Didn't get much time|
| 4 | [ReLU](day-4/relu.cu), [Leaky ReLU](day-4/leaky_relu.cu), [SWiGLU](day-4/swiglu.cu)|Read pmpp ch.6, Kernels for activation functions|
| 5 |[Corner turning](day-5/corner_turning.cu), [Tiled 2D conv](day-5/tiled_2d_conv.cu) |pmpp ch.6 exercises, read ch. 7|
| 6 |[3D Conv. Basic](day-6/3d_conv_basic.cu), [3D Conv. Constant F](day-6/3d_conv_const.cu), [3D Conv. tiled & const.](day-6/3d_conv_tiled_const.cu) |pmpp ch.7 exercises |
| 7 | [Reduction](day-7/reduction.cu)|Basic Reduction kernel |
| 8 | [Classic Softmax](day-8/softmax_basic.cu), [Reduction Sum](day-8/sum_reduction.cu)|Started pmpp ch.10, 2 basic kernels using reduction.|
| 9 |[Convergent Reduction](day-9/convergentSumReduction.cu), [Shared Memory](day-9/sharedMemorySum.cu), [Hierarchial Reduction](day-9/hierarchialReduction.cu) |finished pmpp ch.10|
| 10 |[Reduction with thread coarsening](day-10/reductionThreadCoarsening.cu) |pmpp ch.10 exercises |
| 11 |[Profiled Tiled matmul NCU](day-11/ncu_profiling.cu) |[12 hours CUDA](https://www.youtube.com/watch?v=86FAWCzIe_4) |
| 12 |[CuBLAS SGEMM](day-12/cuBLAS_example.cu), [CuBLAS-Lt example](day-12/CuBLASlt_example.cu), [CuDNN example](day-12/CuDNN_example.cu) |Read about CuBLAS, CuBLAS-Lt & CuDNN API (Overview) |
| 13 |[Triton fused softmax](day-13/triton_fused_softmax.py) |[Tensor cores evolution](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell) |
| 14 |[mini-vllm](https://github.com/mohitxya/mini-vllm) |Read about nanovllm, started building mini-vllm|
| 15 | | |
| 16 | | |
| 17 | | |
| 18 | | |
| 19 | | |
| 20 | | |
| ... | | |
| 100 | | |