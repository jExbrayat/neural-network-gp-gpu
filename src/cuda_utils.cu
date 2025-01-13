#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"

CudaMatrixMemory::CudaMatrixMemory(const int rows, const int cols) : rows(rows), cols(cols) {
    memory_size = sizeof(float) * rows * cols;
    cudaError_t err = cudaMalloc((void**)&device_ptr, memory_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(err) << std::endl;
    }
};

CudaMatrixMemory::~CudaMatrixMemory() {
    cudaFree(device_ptr);
}

void CudaMatrixMemory::sendMatrix2Device(const float *carray) {
    cudaMemcpy(device_ptr, carray, memory_size, cudaMemcpyHostToDevice);
}


