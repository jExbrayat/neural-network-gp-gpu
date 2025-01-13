#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <functional>
#include <any>
#include <vector>

class CudaMatrixMemory
{
public:
    // Constructor
    CudaMatrixMemory(const int rows, const int cols);
    // Destructor
    ~CudaMatrixMemory();

    // Class members
    float *device_ptr;
    unsigned int memory_size;
    const int rows;
    const int cols;

    // Method
    void sendMatrix2Device(const float *carray);
    
private:
};

class CudaGrid {
public:
    
    // Set threads and grid dim3 objects
    void setKernelGrid(const int blocksize_x, const int blocksize_y, const int rows, const int cols);

    // Class members
    dim3 threads;
    dim3 grid;
};

__global__ void sigmoidKernel(const float* input, float* output, const int rows, const int cols);

#endif