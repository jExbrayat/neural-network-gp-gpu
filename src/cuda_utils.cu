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
    CudaThrowError throwErr(err);
    throwErr.throwError("cudaMalloc failed: ");
}

CudaMatrixMemory::~CudaMatrixMemory() {
    if (device_ptr) {
        cudaFree(device_ptr);
        std::cout << "FREEING MEMORY" << std::endl;
    }
}

void CudaMatrixMemory::sendMatrix2Device(const float *carray) {
    cudaError_t err = cudaMemcpy(device_ptr, carray, memory_size, cudaMemcpyHostToDevice);
    CudaThrowError throwErr(err);
    throwErr.throwError("cudaMemcpy failed: ");
}

/**
 * @brief Allocate host memory into host_ptr and perform cudaMemcpy from device to host.
 * The user need to free the allocated memory in returned host_ptr !
 */
float* CudaMatrixMemory::allocAndSend2Host() {
    // Allocate memory for the host
    float* host_ptr = new float[rows * cols]; // Use new[] for proper cleanup with delete[]
    
    if (host_ptr == nullptr) { // Check for successful allocation
        throw std::runtime_error("Memory allocation failed on host.");
    }
    
    // Copy data from device to host
    cudaError_t err = cudaMemcpy(host_ptr, device_ptr, memory_size, cudaMemcpyDeviceToHost);
    CudaThrowError throwErr(err);
    throwErr.throwError("cudaMemcpy failed: ");

    return host_ptr;
}

void CudaGrid::setKernelGrid(const int blocksize_x, const int blocksize_y, const int rows, const int cols) {
    threads = dim3(blocksize_x, blocksize_y);
    grid = dim3((cols + blocksize_x - 1) / blocksize_x, (rows + blocksize_y - 1) / blocksize_y);
}