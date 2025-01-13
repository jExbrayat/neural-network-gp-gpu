#include <functional>
#include <any>
#include <vector>
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

void CudaKernel::setKernelFunction(const std::function<void(std::vector<std::any>)>& func) {
    kernel_function = func;
}

void CudaKernel::runKernel(std::vector<std::any> args) {
    if (kernel_function) {
        kernel_function(args); // Call the assigned function
    } else {
        std::cerr << "Error: No function assigned to execute.\n";
    }
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

