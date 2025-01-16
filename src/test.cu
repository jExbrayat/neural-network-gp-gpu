#include <iostream>


#include "utils.hpp"
#include "cuda_utils.cuh"
#include "debugging_utils.hpp"
#include "cuda_operations.cuh"
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <xtensor/xio.hpp>

using namespace xt;
using namespace std;

void initializeGaussian(xt::xarray<float>& matrix, int rows, int cols) {
    matrix = xt::random::randn<float>({rows, cols}, 0.0f, 1.0f);
}

int main() {

    int a_rows = 35, a_cols = 40; // Adjust as needed
    int b_rows = 40, b_cols = 45;
    int c_rows = 35, c_cols = 1;
    int d_rows = 35, d_cols = 45;

    // Initialize matrices
    // Matrix A (10x8)
    xt::xarray<float> axt, bxt, cxt, dxt;
    initializeGaussian(axt, a_rows, a_cols);
    initializeGaussian(bxt, b_rows, b_cols);
    initializeGaussian(cxt, c_rows, c_cols);

 
    print_shapes(cxt, "cxt shape: ");
    ArrayHandler ah;
    ah.cast_xtarray(axt);
    ArrayHandler bh;
    bh.cast_xtarray(bxt);
    ArrayHandler ch;
    ch.cast_xtarray(cxt);
    
    CudaMatrixMemory a(axt.shape(0), axt.shape(1)); // Matrix A
    a.allocateCudaMemory();
    CudaMatrixMemory b(bxt.shape(0), bxt.shape(1)); // Matrix B
    b.allocateCudaMemory();
    CudaMatrixMemory c(cxt.shape(0), 1); // Bias
    c.allocateCudaMemory();
    CudaMatrixMemory d(axt.shape(0), bxt.shape(1)); // Result
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
    // sigmoidKernel<<<sigmoidGrid.grid, sigmoidGrid.threads>>>(d.device_ptr, d.device_ptr, d.rows, d.cols);

    float *resa = a.allocAndSend2Host();
    float *resb = b.allocAndSend2Host();
    float *resc = c.allocAndSend2Host();
    float *resd = d.allocAndSend2Host();

    // Perform computation with xtensor
    xarray<float> xtres = xt::linalg::dot(axt, bxt) + cxt;
    
    // Check computation
    checkCudaComputation(d, xtres, 0.001, "CHECK CUDA COMPUTATION: ");

    // Print results
    print_carray(resd, d.rows, d.cols, "resd: ");
    
    cout << "xtres: " << endl;
    cout << xtres << endl;
    
    cout << "end of tests" << endl;
    return 0;
}
