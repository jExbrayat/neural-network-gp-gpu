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

class CudaKernel {
public:
    // Set the execute function (it can take any number of arguments of any type)
    void setKernelFunction(const std::function<void(std::vector<std::any>)>& func);

    // Call the execute function with arbitrary arguments
    void runKernel(std::vector<std::any> args);

    // Set threads and grid dim3 objects
    void setKernelGrid(const int blocksize_x, const int blocksize_y, const int rows, const int cols);

    // Class members
    dim3 threads;
    dim3 grid;
    
private:
    std::function<void(std::vector<std::any>)> kernel_function; // Store the function
};

__global__ void sigmoidKernel(const float* input, float* output, const int rows, const int cols);

#endif