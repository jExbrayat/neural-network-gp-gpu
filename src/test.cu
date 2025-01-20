#include <iostream>


#include "utils.hpp"
#include "cuda_utils.cuh"
#include "debugging_utils.hpp"
#include "cuda_operations.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

int main() {

    int rows = 10;
    int cols = 15;
    float* x = new float[rows * cols];
    for (int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            x[i*cols+j] = i;
        }
    }
    print_carray(x, rows, cols, "print a");
    
    CudaMatrixMemory dx(rows, cols);
    dx.sendMatrix2Device(x);

    int batchsize = 2;
    int rowstart = 2;

    float* dxbatch = dx.device_ptr + rowstart * cols;

    // CudaMatrixMemory Dbbatch(batchsize, cols);

    CudaGrid grid;
    grid.setKernelGrid(4, 4, batchsize, cols);
    transposeKernel<<<grid.grid, grid.threads>>>(dxbatch, dxbatch, batchsize, cols);

    // Dbbatch.device_ptr = dbbatch;

    float* res = new float[batchsize * cols];
    int ressize = batchsize * cols * sizeof(float);
    cudaMemcpy(res, dxbatch, ressize, cudaMemcpyDeviceToHost);
    print_carray(res, cols, batchsize, "print res");

    cout << "end of tests" << endl;
    return 0;
}
