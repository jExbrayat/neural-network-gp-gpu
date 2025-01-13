#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

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

    // Method
    void sendMatrix2Device(const float *carray);
    
private:
};


#endif