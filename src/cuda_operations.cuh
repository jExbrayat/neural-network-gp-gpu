#ifndef CUDA_OPERATIONS_CUH
#define CUDA_OPERATIONS_CUH

__global__ void matrixMulKernel(const float* A, const float* B, float* C, const int A_rows, const int A_cols, const int B_cols);

__global__ void addBiasToMatrixKernel(const float* matrix, const float* biases, float* result, int rows, int cols);

__global__ void sigmoidKernel(const float* input, float* output, const int rows, const int cols);

#endif