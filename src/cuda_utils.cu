#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"

// Empty constructor
CudaMatrixMemory::CudaMatrixMemory(const int rows, const int cols) {
    memory_size = sizeof(float) * rows * cols;
    cudaMalloc((void**)device_ptr, memory_size);
};


