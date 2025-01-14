#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"

CudaThrowError::CudaThrowError(cudaError_t error): error(error) {}
void CudaThrowError::throwError(std::string custom_msg) {
    if (error != cudaSuccess) {
        std::cerr << custom_msg << cudaGetErrorString(error) << std::endl;
    }
}

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

void CudaGrid::setKernelGrid(const int blocksize_x, const int blocksize_y, const int rows, const int cols) {
    threads = dim3(blocksize_x, blocksize_y);
    grid = dim3((cols + blocksize_x - 1) / blocksize_x, (rows + blocksize_y - 1) / blocksize_y);
}