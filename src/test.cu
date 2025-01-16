#include <iostream>


#include "utils.hpp"
#include "cuda_utils.cuh"
#include "debugging_utils.hpp"
#include "cuda_operations.cuh"
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <xtensor/xio.hpp>

using namespace xt;
using namespace std;

int main() {

    xarray<float> axt = xarray<float>{{2, 2, 2}, {1, 1, 1}};
    xarray<float> axt_error = xarray<float>{{2, 2, 2}, {9, 9, 9}};
    xarray<float> bxt = xarray<float>{{1, 1}, {1, 1}, {1, 1}};
    xarray<double> cxt = xarray<double>{{1.0, 1.0}};
    cxt.reshape({2, 1});
    print_shapes(cxt, "cxt shape: ");
    ArrayHandler ah;
    ah.cast_xtarray(axt);
    ArrayHandler bh;
    bh.cast_xtarray(bxt);
    ArrayHandler ch;
    ch.cast_xtarray(cxt);
    
    CudaMatrixMemory a(2, 3);
    a.allocateCudaMemory();
    CudaMatrixMemory b(3, 2);
    b.allocateCudaMemory();
    CudaMatrixMemory c(2, 1); // bias
    c.allocateCudaMemory();
    CudaMatrixMemory d(2, 2); // result
    d.allocateCudaMemory();

    a.sendMatrix2Device(ah.carray);
    b.sendMatrix2Device(bh.carray);
    c.sendMatrix2Device(ch.carray);

    CudaGrid matMulGrid;
    CudaGrid addGrid;
    CudaGrid sigmoidGrid;
    matMulGrid.setKernelGrid(16, 16, a.rows, b.cols);
    addGrid.setKernelGrid(16, 16, a.rows, b.cols);
    sigmoidGrid.setKernelGrid(16, 16, a.rows, b.cols);

    matrixMulKernel<<<matMulGrid.grid, matMulGrid.threads>>>(a.device_ptr, b.device_ptr, d.device_ptr, a.rows, a.cols, b.cols); // w * la, write the result in lo
    addBiasToMatrixKernel<<<addGrid.grid, addGrid.threads>>>(d.device_ptr, c.device_ptr, d.device_ptr, d.rows, d.cols);
    sigmoidKernel<<<sigmoidGrid.grid, sigmoidGrid.threads>>>(d.device_ptr, d.device_ptr, d.rows, d.cols);

    float *resa = a.allocAndSend2Host();
    float *resb = b.allocAndSend2Host();
    float *resc = c.allocAndSend2Host();
    float *resd = d.allocAndSend2Host();

    // Perform computation with xtensor
    xarray<float> xtres = sigmoid(xt::linalg::dot(axt, bxt) + cxt);
    
    // Check computation
    checkCudaComputation(d, xtres, 0.1, "CHECK CUDA COMPUTATION: ");

    // Print results
    print_carray(resd, d.rows, d.cols, "resd: ");
    
    cout << "xtres: " << endl;
    cout << xtres << endl;
    
    cout << "end of tests" << endl;
    return 0;
}
