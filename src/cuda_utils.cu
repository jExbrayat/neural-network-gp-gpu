#include <iostream>
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

void CudaGrid::setKernelGrid(const int blocksize_x, const int blocksize_y, const int rows, const int cols) {
    threads = dim3(blocksize_x, blocksize_y);
    grid = dim3((cols + blocksize_x - 1) / blocksize_x, (rows + blocksize_y - 1) / blocksize_y);
}


__global__ void sigmoidKernel(const float* input, float* output, const int rows, const int cols) {
    // Compute the global thread index for both x and y dimensions
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within bounds
    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;  // Convert 2D index to 1D
        output[index] = 1.0f / (1.0f + expf(-input[index]));  // Sigmoid function
    }
}

