#include <iostream>


#include "utils.hpp"
#include "cuda_utils.cuh"
#include "debugging_utils.hpp"
#include "cuda_operations.cuh"
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace xt;
using namespace std;

int main() {

    xarray<float> axt = xarray<float>{{1, 1, 1}, {1, 1, 1}};
    xarray<float> bxt = xarray<float>{{1, 1}, {1, 1}, {1, 1}};
    xarray<double> cxt = xarray<double>{{1.0, 1.0}};
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
    CudaMatrixMemory c(2, 1);
    c.allocateCudaMemory();

    a.sendMatrix2Device(ah.carray);
    b.sendMatrix2Device(bh.carray);
    c.sendMatrix2Device(ch.carray);

    CudaGrid matMulGrid;
    CudaGrid addGrid;
    CudaGrid sigmoidGrid;
    matMulGrid.setKernelGrid(16, 16, a.rows, b.cols);
    addGrid.setKernelGrid(16, 16, a.rows, b.cols);
    sigmoidGrid.setKernelGrid(16, 16, a.rows, b.cols);

    matrixMulKernel<<<matMulGrid.grid, matMulGrid.threads>>>(a.device_ptr, b.device_ptr, c.device_ptr, a.rows, a.cols, b.cols); // w * la, write the result in lo
    addBiasToMatrixKernel<<<addGrid.grid, addGrid.threads>>>(c.device_ptr, b.device_ptr, c.device_ptr, c.rows, c.cols);
    sigmoidKernel<<<sigmoidGrid.grid, sigmoidGrid.threads>>>(c.device_ptr, c.device_ptr, c.rows, c.cols);

    float *resa = a.allocAndSend2Host();
    float *resb = b.allocAndSend2Host();
    float *resc = c.allocAndSend2Host();

    print_carray(resa, a.rows, a.cols, "resa: ");
    print_carray(resb, b.rows, b.cols, "resb: ");
    print_carray(resc, c.rows, c.cols, "resc: ");

    cout << "end of tests" << endl;
    return 0;
}
